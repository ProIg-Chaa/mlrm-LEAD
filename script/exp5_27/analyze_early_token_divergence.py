#!/usr/bin/env python3
"""Create a compact early-token divergence audit for the rerun matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WINDOWS = [1, 2, 4, 8, 16]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_dir_for(base_dir: Path, phase: str, dataset: str, run: str) -> Path | None:
    matches = sorted((base_dir / phase / dataset).glob(f"{run}_gpu*"))
    for path in matches:
        if (path / "results.jsonl").exists():
            return path
    return None


def rows_by_id(run_dir: Path) -> dict[int, dict]:
    return {int(row["id"]): row for row in load_jsonl(run_dir / "results.jsonl")}


def traces_by_id(run_dir: Path) -> dict[int, dict]:
    return {int(row["id"]): row for row in load_jsonl(run_dir / "token_entropy_full.jsonl")}


def final_answer_marker_position(text: str | None) -> int | None:
    if not text:
        return None
    matches = list(re.finditer(r"answer\s*[:.]", text, flags=re.I))
    if matches:
        return matches[-1].start()
    boxed = text.rfind("\\boxed")
    return boxed if boxed >= 0 else None


def token_texts(trace: dict | None) -> list[str]:
    if not trace:
        return []
    out = []
    for token in trace.get("tokens") or []:
        text = token.get("text")
        if text is None:
            text = token.get("token")
        if text is None:
            text = str(token.get("token_id", ""))
        out.append(str(text))
    return out


def entropy_summary(trace: dict | None, n: int) -> dict:
    if not trace:
        return {"mean": None, "max": None, "soft": 0, "entry_steps": [], "exit_steps": []}
    tokens = (trace.get("tokens") or [])[:n]
    vals = [
        float(t.get("raw_entropy"))
        for t in tokens
        if isinstance(t.get("raw_entropy"), (int, float))
    ]
    return {
        "mean": sum(vals) / len(vals) if vals else None,
        "max": max(vals) if vals else None,
        "soft": sum(1 for t in tokens if t.get("mode") == "soft"),
        "entry_steps": [int(t.get("step")) for t in tokens if t.get("lead_delayed_transition_entry")],
        "exit_steps": [int(t.get("step")) for t in tokens if t.get("lead_delayed_transition_exit")],
    }


def compact_text(text: str, max_chars: int = 180) -> str:
    text = text.replace("\n", "\\n")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def first_windows(trace: dict | None, fallback_answer: str | None) -> dict[int, str]:
    texts = token_texts(trace)
    if not texts and fallback_answer:
        texts = fallback_answer.split()
    return {n: compact_text("".join(texts[:n]) if trace else " ".join(texts[:n])) for n in WINDOWS}


def case_block(
    case_id: int,
    ref_run: str,
    cur_run: str,
    ref_row: dict | None,
    cur_row: dict | None,
    ref_trace: dict | None,
    cur_trace: dict | None,
) -> list[str]:
    lines = [f"### id={case_id}：{ref_run} -> {cur_run}", ""]
    for name, row, trace in [
        (ref_run, ref_row, ref_trace),
        (cur_run, cur_row, cur_trace),
    ]:
        answer = (row or {}).get("model_answer") or ""
        output_tokens = (row or {}).get("output_tokens")
        marker = final_answer_marker_position(answer)
        lines.append(f"- {name}: output_tokens={output_tokens}, first_answer_marker_pos={marker}")
        windows = first_windows(trace, answer)
        ent16 = entropy_summary(trace, 16)
        mean = ent16["mean"]
        maxv = ent16["max"]
        lines.append(
            "  early_entropy_16="
            + (
                "NA"
                if mean is None
                else f"mean {mean:.3f}, max {maxv:.3f}, soft {ent16['soft']}/16, entry {ent16['entry_steps']}, exit {ent16['exit_steps']}"
            )
        )
        for n in WINDOWS:
            lines.append(f"  first{n}: `{windows[n]}`")
    lines.append("")
    return lines


def add_comparison(
    lines: list[str],
    base_dir: Path,
    phase: str,
    dataset: str,
    ref_run: str,
    cur_run: str,
    ids: list[int],
    title: str,
    max_cases: int = 6,
) -> None:
    ref_dir = run_dir_for(base_dir, phase, dataset, ref_run)
    cur_dir = run_dir_for(base_dir, phase, dataset, cur_run)
    if not ref_dir or not cur_dir:
        return
    ref_rows = rows_by_id(ref_dir)
    cur_rows = rows_by_id(cur_dir)
    ref_traces = traces_by_id(ref_dir)
    cur_traces = traces_by_id(cur_dir)
    lines.extend([f"## {title}", ""])
    if not ids:
        lines.extend(["没有可抽样的样本。", ""])
        return
    for case_id in ids[:max_cases]:
        lines.extend(
            case_block(
                case_id,
                ref_run,
                cur_run,
                ref_rows.get(case_id),
                cur_rows.get(case_id),
                ref_traces.get(case_id),
                cur_traces.get(case_id),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    deltas_path = base_dir / "pairwise_deltas.json"
    if not deltas_path.exists():
        raise SystemExit(f"Missing {deltas_path}; run summarize_rerun_early_path_dependence.py first")
    deltas = load_json(deltas_path)
    lines = [
        "# early_token_divergence",
        "",
        "目标：检查 fixed/damaged 样本是否在生成最早期已经出现 wording/trajectory 分叉，并记录第 0/早期 transition 事件、前 1/2/4/8/16 个生成 token 的差异、前 16 token entropy 摘要、答案标记位置与输出长度。",
        "",
        "说明：这里是非推理审计，不替代主表 accuracy；只抽代表样本，避免堆叠完整长输出。",
        "",
    ]

    targets = [
        (
            "phase1_vstar_mechanism",
            "vstar",
            "cot_orign_greedy",
            "initial_transition_only",
            "fixed_ids",
            "VStar：COT wrong -> initial_transition fixed",
        ),
        (
            "phase1_vstar_mechanism",
            "vstar",
            "initial_transition_only",
            "initial_transition_no_to_normal",
            "damaged_ids",
            "VStar：initial_transition correct -> no_to_normal damaged",
        ),
        (
            "phase3_cross_dataset_minimal",
            "mmvp",
            "cot_orign_greedy",
            "initial_transition_only",
            "fixed_ids",
            "MMVP：COT wrong -> initial_transition fixed",
        ),
        (
            "phase3_cross_dataset_minimal",
            "realworldqa_fixed200",
            "cot_orign_greedy",
            "initial_transition_only",
            "fixed_ids",
            "RealWorldQA fixed200：COT wrong -> initial_transition fixed",
        ),
    ]
    for phase, dataset, ref_run, cur_run, field, title in targets:
        group = deltas.get(f"{phase}/{dataset}", {})
        comp = group.get(f"{ref_run}__vs__{cur_run}", {})
        add_comparison(
            lines,
            base_dir,
            phase,
            dataset,
            ref_run,
            cur_run,
            comp.get(field, []),
            title,
        )

    (base_dir / "early_token_divergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {base_dir / 'early_token_divergence.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

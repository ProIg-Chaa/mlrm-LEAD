#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

WINDOWS = [1, 2, 4, 8, 16, 32]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_dir_for(base: Path, phase: str, dataset: str, run: str) -> Path | None:
    root = base / phase / dataset
    matches = sorted(root.glob(f"{run}_gpu*"))
    for p in matches:
        if (p / "results.jsonl").exists():
            return p
    return None


def run_label(path: Path | None) -> str:
    if path is None:
        return "missing"
    return re.sub(r"_gpu\d+$", "", path.name)


def rows_by_id(run_dir: Path) -> dict[int, dict]:
    return {int(r["id"]): r for r in load_jsonl(run_dir / "results.jsonl")}


def traces_by_id(run_dir: Path) -> dict[int, dict]:
    return {int(r["id"]): r for r in load_jsonl(run_dir / "token_entropy_full.jsonl")}


def evaluator_correctness(run_dir: Path, dataset: str) -> dict[int, bool]:
    if dataset == "mmvp":
        rows = load_jsonl(run_dir / "specialized_eval_rows.jsonl")
        if rows:
            return {int(r["id"]): bool(r.get("specialized_is_correct")) for r in rows}
    if dataset == "realworldqa_fixed200":
        rows = load_jsonl(run_dir / "realworldqa_mcq_eval_rows.jsonl")
        if rows:
            return {int(r["id"]): bool(r.get("realworldqa_is_correct")) for r in rows}
    out = {}
    for row in load_jsonl(run_dir / "results.jsonl"):
        out[int(row["id"])] = row_is_correct(row)
    return out


def extract_choice(text: str | None) -> str | None:
    if not text:
        return None
    tail = text[-1800:]
    patterns = [
        r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}",
        r"\\boxed\{\s*\(?([a-d])\)?\s*\}",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"(?:option|choice)\s*\(?([A-Da-d])\)?",
        r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, tail)
        if m:
            return m.group(1).upper()
    region = tail.split("</think>")[-1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-400:])
    return letters[-1].upper() if letters else None


def gold_choice(row: dict | None) -> str | None:
    if not row:
        return None
    ans = str(row.get("answer") or "").strip()
    m = re.search(r"\(?([A-Da-d])\)?", ans)
    return m.group(1).upper() if m else None


def row_is_correct(row: dict | None) -> bool:
    pred = extract_choice((row or {}).get("model_answer"))
    gold = gold_choice(row)
    return bool(pred and gold and pred == gold)


def token_texts(trace: dict | None) -> list[str]:
    if not trace:
        return []
    out = []
    for t in trace.get("tokens") or []:
        text = t.get("token_text")
        if text is None:
            text = t.get("text")
        if text is None:
            text = t.get("token")
        if text is None:
            text = str(t.get("token_id", ""))
        out.append(str(text))
    return out


def compact(text: str | None, max_chars: int = 260) -> str:
    text = text or ""
    text = text.replace("\n", "\\n")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def strip_think_start(text: str | None) -> str:
    text = text or ""
    text = re.sub(r"^\s*<think>\s*", "", text, flags=re.I)
    return text


def answer_marker_pos(text: str | None) -> int | None:
    if not text:
        return None
    boxed = text.rfind("\\boxed")
    markers = list(re.finditer(r"answer\s*[:.]", text, flags=re.I))
    if markers:
        return markers[-1].start()
    return boxed if boxed >= 0 else None


def answer_tail(text: str | None, max_chars: int = 260) -> str:
    if not text:
        return ""
    pos = answer_marker_pos(text)
    if pos is None:
        pos = max(0, len(text) - max_chars)
    return compact(text[pos:], max_chars)


def first_n(trace: dict | None, row: dict | None, n: int) -> str:
    toks = token_texts(trace)
    if toks:
        return compact("".join(toks[:n]), 320)
    text = strip_think_start((row or {}).get("model_answer"))
    return compact(" ".join(text.split()[:n]), 320)


def first_divergence(ref_trace: dict | None, cur_trace: dict | None) -> int | None:
    a = token_texts(ref_trace)
    b = token_texts(cur_trace)
    if not a or not b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def entropy_stats(trace: dict | None, n: int = 32) -> dict:
    toks = (trace or {}).get("tokens") or []
    head = toks[:n]
    vals = [float(t["raw_entropy"]) for t in head if isinstance(t.get("raw_entropy"), (int, float))]
    return {
        "mean": sum(vals) / len(vals) if vals else None,
        "max": max(vals) if vals else None,
        "soft": sum(1 for t in head if t.get("mode") == "soft"),
        "modes": [t.get("mode") for t in head[:8]],
    }


def bucket(values: list[int | None]) -> dict[str, int]:
    keys = ["0", "1-2", "3-4", "5-8", "9-16", "17-32", ">32", "no_full_trace_or_identical"]
    out = {k: 0 for k in keys}
    for v in values:
        if v is None:
            out["no_full_trace_or_identical"] += 1
        elif v == 0:
            out["0"] += 1
        elif v <= 2:
            out["1-2"] += 1
        elif v <= 4:
            out["3-4"] += 1
        elif v <= 8:
            out["5-8"] += 1
        elif v <= 16:
            out["9-16"] += 1
        elif v <= 32:
            out["17-32"] += 1
        else:
            out[">32"] += 1
    return out


def compare_runs(base: Path, phase: str, dataset: str, ref_run: str, cur_run: str, mode: str) -> dict:
    ref_dir = run_dir_for(base, phase, dataset, ref_run)
    cur_dir = run_dir_for(base, phase, dataset, cur_run)
    if ref_dir is None or cur_dir is None:
        return {"missing": True, "phase": phase, "dataset": dataset, "ref_run": ref_run, "cur_run": cur_run}
    ref_rows = rows_by_id(ref_dir)
    cur_rows = rows_by_id(cur_dir)
    ref_correct = evaluator_correctness(ref_dir, dataset)
    cur_correct = evaluator_correctness(cur_dir, dataset)
    ids = sorted(set(ref_rows) & set(cur_rows))
    if mode == "fixed":
        selected = [i for i in ids if not ref_correct.get(i, False) and cur_correct.get(i, False)]
    elif mode == "damaged":
        selected = [i for i in ids if ref_correct.get(i, False) and not cur_correct.get(i, False)]
    else:
        raise ValueError(mode)
    ref_traces = traces_by_id(ref_dir)
    cur_traces = traces_by_id(cur_dir)
    divs = [first_divergence(ref_traces.get(i), cur_traces.get(i)) for i in selected]
    return {
        "phase": phase,
        "dataset": dataset,
        "ref_run": run_label(ref_dir),
        "cur_run": run_label(cur_dir),
        "mode": mode,
        "ids": selected,
        "divergence_steps": divs,
        "bucket": bucket(divs),
        "median_divergence": median([d for d in divs if d is not None]) if any(d is not None for d in divs) else None,
        "ref_rows": ref_rows,
        "cur_rows": cur_rows,
        "ref_traces": ref_traces,
        "cur_traces": cur_traces,
    }


def emit_case(lines: list[str], comp: dict, case_id: int) -> dict:
    ref_row = comp["ref_rows"].get(case_id)
    cur_row = comp["cur_rows"].get(case_id)
    ref_trace = comp["ref_traces"].get(case_id)
    cur_trace = comp["cur_traces"].get(case_id)
    div = first_divergence(ref_trace, cur_trace)
    detail = {
        "id": case_id,
        "question": (ref_row or {}).get("question"),
        "gold": gold_choice(ref_row),
        "ref_pred": extract_choice((ref_row or {}).get("model_answer")),
        "cur_pred": extract_choice((cur_row or {}).get("model_answer")),
        "first_divergence": div,
    }
    lines += [
        f"### id={case_id}",
        "",
        f"- question: {compact((ref_row or {}).get('question'), 220)}",
        f"- gold: `{detail['gold']}`, {comp['ref_run']} pred: `{detail['ref_pred']}`, {comp['cur_run']} pred: `{detail['cur_pred']}`",
        f"- first token divergence: `{div}`",
        f"- output length: {comp['ref_run']} `{(ref_row or {}).get('output_tokens')}`, {comp['cur_run']} `{(cur_row or {}).get('output_tokens')}`",
        f"- answer marker pos: {comp['ref_run']} `{answer_marker_pos((ref_row or {}).get('model_answer'))}`, {comp['cur_run']} `{answer_marker_pos((cur_row or {}).get('model_answer'))}`",
        "",
    ]
    for name, row, trace in [(comp["ref_run"], ref_row, ref_trace), (comp["cur_run"], cur_row, cur_trace)]:
        es = entropy_stats(trace, 32)
        mean = "NA" if es["mean"] is None else f"{es['mean']:.3f}"
        maxv = "NA" if es["max"] is None else f"{es['max']:.3f}"
        lines += [
            f"**{name}**",
            "",
            f"- first32 entropy: mean `{mean}`, max `{maxv}`, soft `{es['soft']}/32`, modes_head `{es['modes']}`",
        ]
        for n in WINDOWS:
            lines.append(f"- first{n}: `{first_n(trace, row, n)}`")
        lines += [
            f"- reasoning opening: `{compact(strip_think_start((row or {}).get('model_answer')), 420)}`",
            f"- final answer region: `{answer_tail((row or {}).get('model_answer'), 300)}`",
            "",
        ]
    return detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    base = Path(args.base_dir)
    out_md = Path(args.out_md) if args.out_md else base / "early_token_divergence_v2.md"
    out_json = Path(args.out_json) if args.out_json else base / "early_token_divergence_v2.json"

    specs = [
        ("VStar: COT wrong -> initial_transition fixed", "phase1_vstar_mechanism", "vstar", "cot_orign_greedy", "initial_transition_only", "fixed"),
        ("VStar: initial_transition correct -> no_to_normal damaged", "phase1_vstar_mechanism", "vstar", "initial_transition_only", "initial_transition_no_to_normal", "damaged"),
        ("MMVP: COT wrong -> initial_transition fixed", "phase3_cross_dataset_minimal", "mmvp", "cot_orign_greedy", "initial_transition_only", "fixed"),
        ("MMVP: initial_transition correct -> no_to_normal damaged", "phase3_cross_dataset_minimal", "mmvp", "initial_transition_only", "initial_transition_no_to_normal", "damaged"),
        ("VisuLogic: COT wrong -> initial_transition fixed", "phase3_cross_dataset_minimal", "visulogic300", "cot_orign_greedy", "initial_transition_only", "fixed"),
        ("VStar timing: step0 correct -> step16 damaged", "phase2_timing_curve", "vstar", "transition_step0", "transition_step16", "damaged"),
        ("VisuLogic timing: step0 correct -> step16 damaged", "phase2_timing_curve_cross", "visulogic300", "transition_step0", "transition_step16", "damaged"),
    ]

    lines = [
        "# Early Token Divergence Analysis",
        "",
        "目的：验证 fixed/damaged 样本是否在生成最早期就出现文本轨迹分叉，从而支持 `early trajectory commitment`。这里不重新跑模型，只读取已有 full token trace、结果文件和 evaluator rows。",
        "",
        "读法：`first token divergence` 是两条输出 token 序列第一次不同的位置。前 0-16 token 内分叉，说明差异发生在 reasoning opening，而不是答案末尾。`first32 entropy` 用于判断早期是否已经发生 soft transition 或熵变化。",
        "",
    ]
    json_out = []
    for title, phase, dataset, ref, cur, mode in specs:
        comp = compare_runs(base, phase, dataset, ref, cur, mode)
        lines += [f"## {title}", ""]
        if comp.get("missing"):
            lines += ["缺少对应 run 目录。", ""]
            continue
        total = len(comp["ids"])
        b = comp["bucket"]
        bucket_text = "; ".join(f"{k}: {v}" for k, v in b.items())
        lines += [
            f"- comparison: `{comp['ref_run']}` -> `{comp['cur_run']}`",
            f"- selected samples: `{total}` ({mode})",
            f"- median first divergence: `{comp['median_divergence']}`",
            f"- divergence buckets: {bucket_text}",
            "",
        ]
        chosen = sorted(
            comp["ids"],
            key=lambda i: (first_divergence(comp["ref_traces"].get(i), comp["cur_traces"].get(i)) is None,
                           first_divergence(comp["ref_traces"].get(i), comp["cur_traces"].get(i)) or 10**9,
                           i),
        )[:6]
        details = []
        for case_id in chosen:
            details.append(emit_case(lines, comp, case_id))
        json_out.append({
            "title": title,
            "phase": phase,
            "dataset": dataset,
            "ref_run": comp["ref_run"],
            "cur_run": comp["cur_run"],
            "mode": mode,
            "selected_count": total,
            "median_first_divergence": comp["median_divergence"],
            "bucket": b,
            "case_details": details,
        })

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(json_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize unified route annotations from token_entropy_full.jsonl."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_mcq_answer(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"\\boxed\{([A-Da-d])\}",
        r"\*\*([A-Da-d])\*\*",
        r"(?:^|\n)\s*([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    last_letters = re.findall(r"\b([A-D])\b", text[-200:])
    return last_letters[-1].upper() if last_letters else None


def load_correctness(run_dir: Path) -> dict[int, dict]:
    rows = {}
    for row in load_jsonl(run_dir / "results.jsonl"):
        pred = extract_mcq_answer(row.get("model_answer") or "")
        gold = (row.get("answer") or "").strip().upper()
        ok = pred is not None and pred == gold
        rows[int(row["id"])] = {
            "correct": ok,
            "pred": pred,
            "gold": gold,
            "output_tokens": int(row.get("output_tokens") or 0),
            "subtopic": row.get("subtopic", "unknown"),
        }
    return rows


def pct(n: int, d: int) -> str:
    return "0.0%" if d == 0 else f"{100.0 * n / d:.1f}%"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def collect(run_dir: Path, baseline_run_dir: Path | None = None) -> dict:
    correctness = load_correctness(run_dir)
    baseline = load_correctness(baseline_run_dir) if baseline_run_dir else {}
    by_group = defaultdict(lambda: defaultdict(Counter))
    sample_rows = []
    total = {
        "route_action": Counter(),
        "route_signal": Counter(),
        "phase": Counter(),
        "suppressed": Counter(),
    }

    trace_path = run_dir / "token_entropy_full.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    for rec in load_jsonl(trace_path):
        sid = int(rec["id"])
        meta = correctness.get(sid, {})
        cur_ok = meta.get("correct")
        base_ok = baseline.get(sid, {}).get("correct") if baseline else None
        if base_ok is True and cur_ok is False:
            delta_group = "damaged"
        elif base_ok is False and cur_ok is True:
            delta_group = "fixed"
        elif cur_ok is True:
            delta_group = "correct"
        elif cur_ok is False:
            delta_group = "wrong"
        else:
            delta_group = "unknown"

        route_counts = Counter()
        signal_counts = Counter()
        phase_counts = Counter()
        suppressed_counts = Counter()
        visual_effective = 0
        visual_candidate = 0
        collapse = 0
        fmt = 0
        answer = 0

        for token in rec.get("tokens") or []:
            action = token.get("route_action") or token.get("mode") or "unknown"
            signal = token.get("route_signal") or "unknown"
            phase = token.get("generation_phase") or "unknown"
            route_counts[action] += 1
            signal_counts[signal] += 1
            phase_counts[phase] += 1
            total["route_action"][action] += 1
            total["route_signal"][signal] += 1
            total["phase"][phase] += 1
            visual_candidate += int(bool(token.get("visual_bias_candidate")))
            visual_effective += int(bool(token.get("visual_bias_effective")))
            collapse += int(bool(token.get("collapse_on_diffuse")))
            fmt += int(bool(token.get("format_cooldown_active")))
            answer += int(bool(token.get("answer_zone_discrete_active")))
            for item in token.get("route_suppressed_by") or []:
                suppressed_counts[item] += 1
                total["suppressed"][item] += 1

        for key, counter in [
            ("route_action", route_counts),
            ("route_signal", signal_counts),
            ("phase", phase_counts),
            ("suppressed", suppressed_counts),
        ]:
            by_group[delta_group][key].update(counter)
            by_group["all"][key].update(counter)

        sample_rows.append({
            "id": sid,
            "group": delta_group,
            "correct": cur_ok,
            "baseline_correct": base_ok,
            "output_tokens": meta.get("output_tokens", 0),
            "route_counts": route_counts,
            "signal_counts": signal_counts,
            "phase_counts": phase_counts,
            "visual_candidate": visual_candidate,
            "visual_effective": visual_effective,
            "collapse": collapse,
            "format": fmt,
            "answer_zone": answer,
        })

    return {
        "run_dir": str(run_dir),
        "baseline_run_dir": str(baseline_run_dir) if baseline_run_dir else None,
        "correctness": correctness,
        "sample_rows": sample_rows,
        "total": total,
        "by_group": by_group,
    }


def counter_table(title: str, counter: Counter, limit: int = 20) -> list[str]:
    lines = [f"## {title}", "", "| item | count | share |", "|---|---:|---:|"]
    total = sum(counter.values())
    for key, value in counter.most_common(limit):
        lines.append(f"| {key} | {value} | {pct(value, total)} |")
    lines.append("")
    return lines


def render(report: dict) -> str:
    samples = report["sample_rows"]
    correct = sum(1 for s in samples if s["correct"] is True)
    lines = [
        "# Unified Route Summary",
        "",
        f"- run: `{report['run_dir']}`",
        f"- baseline: `{report['baseline_run_dir']}`",
        f"- samples: `{len(samples)}`",
        f"- accuracy: `{correct}/{len(samples)} = {pct(correct, len(samples))}`",
        "",
    ]

    groups = Counter(s["group"] for s in samples)
    lines.extend(["## Sample Groups", "", "| group | samples | mean_len | mean_format | mean_collapse | mean_visual_eff |", "|---|---:|---:|---:|---:|---:|"])
    for group, count in groups.most_common():
        subset = [s for s in samples if s["group"] == group]
        lines.append(
            f"| {group} | {count} | "
            f"{mean([s['output_tokens'] for s in subset]):.1f} | "
            f"{mean([s['format'] for s in subset]):.1f} | "
            f"{mean([s['collapse'] for s in subset]):.1f} | "
            f"{mean([s['visual_effective'] for s in subset]):.1f} |"
        )
    lines.append("")

    lines.extend(counter_table("Route Actions", report["total"]["route_action"]))
    lines.extend(counter_table("Route Signals", report["total"]["route_signal"]))
    lines.extend(counter_table("Generation Phases", report["total"]["phase"]))
    lines.extend(counter_table("Suppressed Routes", report["total"]["suppressed"]))

    for group in ["fixed", "damaged", "correct", "wrong", "all"]:
        if group not in report["by_group"]:
            continue
        lines.append(f"## Group: {group}")
        for key in ["route_action", "route_signal", "phase", "suppressed"]:
            counter = report["by_group"][group][key]
            if not counter:
                continue
            total = sum(counter.values())
            top = ", ".join(f"{k}={v} ({pct(v, total)})" for k, v in counter.most_common(8))
            lines.append(f"- `{key}`: {top}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baseline_run_dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    report = collect(
        Path(args.run_dir),
        Path(args.baseline_run_dir) if args.baseline_run_dir else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(report) + "\n", encoding="utf-8")
    if args.output_json:
        compact = {
            "run_dir": report["run_dir"],
            "baseline_run_dir": report["baseline_run_dir"],
            "sample_rows": [
                {
                    **{k: v for k, v in row.items() if not isinstance(v, Counter)},
                    "route_counts": dict(row["route_counts"]),
                    "signal_counts": dict(row["signal_counts"]),
                    "phase_counts": dict(row["phase_counts"]),
                }
                for row in report["sample_rows"]
            ],
            "total": {k: dict(v) for k, v in report["total"].items()},
        }
        Path(args.output_json).write_text(
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

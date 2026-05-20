#!/usr/bin/env python3
import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from lead.evaluator import evaluate_single
from script.exp5_16.analyze_spike_types import (
    classify_spike,
    is_spike,
    pct,
    topk_masses,
)


SPIKE_TYPES = [
    "visual_spike",
    "relation_spike",
    "format_spike",
    "answer_spike",
    "diffuse_low_conf_spike",
    "other",
]


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": st.median(values),
        "p90": values[int(0.9 * (len(values) - 1))],
        "max": values[-1],
    }


def fmt_stats(stats):
    if stats.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={stats['n']} mean={stats['mean']:.2f} "
        f"med={stats['median']:.2f} p90={stats['p90']:.2f} max={stats['max']:.2f}"
    )


def load_results(run_dir):
    rows = {}
    with (run_dir / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ok, extracted = evaluate_single(rec.get("model_answer"), rec.get("answer", ""))
            rows[rec.get("id")] = {
                "correct": ok,
                "extracted": extracted,
                "failed_extraction": extracted is None,
                "output_tokens": rec.get("output_tokens") or 0,
                "answer": rec.get("answer"),
                "subtopic": rec.get("subtopic", "unknown"),
            }
    return rows


def last_k_mean(tokens, field, k):
    vals = [t.get(field) for t in tokens[-k:] if t.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def collect_run(label, run_dir, args):
    results = load_results(run_dir)
    sample_rows = []
    event_spikes = defaultdict(Counter)
    event_samples = Counter()
    event_token_counts = defaultdict(list)
    soft_neighbor_offsets = Counter()
    soft_neighbor_spikes = Counter()

    with (run_dir / "token_entropy_full.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("id")
            meta = results.get(sid, {})
            tokens = rec.get("tokens") or []
            entropies = [t.get("raw_entropy") for t in tokens]
            spike_types = []
            top1_values = []
            margin_values = []
            soft_steps = []
            for idx, token in enumerate(tokens):
                masses, _, top1, margin = topk_masses(token, args.topk_field)
                if top1:
                    top1_values.append(top1)
                margin_values.append(margin)
                if token.get("mode") == "soft":
                    soft_steps.append(idx)
                if is_spike(entropies, idx, args.window, args.alpha, args.min_history, args.min_entropy):
                    cat, _, _, _ = classify_spike(token, args)
                    spike_types.append((idx, cat))

            spike_count = len(spike_types)
            spike_counter = Counter(cat for _, cat in spike_types)
            output_tokens = meta.get("output_tokens", 0)
            max_top1 = max(top1_values) if top1_values else None
            mean_top1 = sum(top1_values) / len(top1_values) if top1_values else None
            tail_top1 = last_k_mean(tokens, "raw_top1_prob", args.tail_k)
            tail_margin = last_k_mean(tokens, "raw_margin", args.tail_k)
            long_output = output_tokens >= args.long_output_threshold
            wrong = meta.get("correct") is False
            high_conf_wrong = wrong and (
                (tail_top1 is not None and tail_top1 >= args.high_conf_tail_threshold)
                or (max_top1 is not None and max_top1 >= args.high_conf_any_threshold)
            )
            sample = {
                "id": sid,
                "correct": meta.get("correct"),
                "wrong": wrong,
                "output_tokens": output_tokens,
                "long_output": long_output,
                "high_conf_wrong": high_conf_wrong,
                "mean_top1": mean_top1,
                "max_top1": max_top1,
                "tail_top1": tail_top1,
                "tail_margin": tail_margin,
                "spike_count": spike_count,
                "spike_counter": spike_counter,
                "soft_steps": len(soft_steps),
            }
            sample_rows.append(sample)

            events = ["all"]
            if wrong:
                events.append("wrong")
            else:
                events.append("correct")
            if high_conf_wrong:
                events.append("high_conf_wrong")
            if long_output:
                events.append("long_output")
            if soft_steps:
                events.append("has_soft_steps")

            for event in events:
                event_samples[event] += 1
                event_token_counts[event].append(output_tokens)
                event_spikes[event].update(spike_counter)

            if soft_steps:
                soft_set = set(soft_steps)
                for idx, cat in spike_types:
                    nearest = min(abs(idx - s) for s in soft_set)
                    if nearest <= args.soft_window:
                        soft_neighbor_spikes[cat] += 1
                        soft_neighbor_offsets[nearest] += 1

    return {
        "label": label,
        "run_dir": str(run_dir),
        "samples": sample_rows,
        "event_samples": event_samples,
        "event_token_counts": event_token_counts,
        "event_spikes": event_spikes,
        "soft_neighbor_spikes": soft_neighbor_spikes,
        "soft_neighbor_offsets": soft_neighbor_offsets,
    }


def spike_table_line(name, sample_n, token_values, counts):
    total_spikes = sum(counts.values())
    cols = [
        name,
        str(sample_n),
        fmt_stats(quantiles(token_values)),
        str(total_spikes),
    ]
    for cat in SPIKE_TYPES:
        cols.append(f"{counts[cat]} ({pct(counts[cat], total_spikes)})")
    return "| " + " | ".join(cols) + " |"


def render_report(runs, args):
    lines = []
    lines.append("# Experiment 2-A Bad Event Attribution")
    lines.append("")
    lines.append(f"- source: `{args.base_dir}`")
    lines.append(f"- spike rule: `H_t > local_mean({args.window}) + {args.alpha} * local_std({args.window})`, min_entropy=`{args.min_entropy}`")
    lines.append(f"- high-confidence wrong: tail-{args.tail_k} mean top1 >= `{args.high_conf_tail_threshold}` or any top1 >= `{args.high_conf_any_threshold}`")
    lines.append(f"- long output threshold: output_tokens >= `{args.long_output_threshold}`")
    lines.append("")

    lines.append("## Sample-Level Summary")
    lines.append("| method | samples | correct | wrong | long_output | high_conf_wrong | mean_len_correct | mean_len_wrong | mean_spikes_correct | mean_spikes_wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        samples = run["samples"]
        correct = [s for s in samples if s["correct"] is True]
        wrong = [s for s in samples if s["wrong"]]
        long_output = [s for s in samples if s["long_output"]]
        high_conf_wrong = [s for s in samples if s["high_conf_wrong"]]
        lines.append(
            f"| {run['label']} | {len(samples)} | {len(correct)} | {len(wrong)} | "
            f"{len(long_output)} | {len(high_conf_wrong)} | "
            f"{(sum(s['output_tokens'] for s in correct) / len(correct)) if correct else 0:.1f} | "
            f"{(sum(s['output_tokens'] for s in wrong) / len(wrong)) if wrong else 0:.1f} | "
            f"{(sum(s['spike_count'] for s in correct) / len(correct)) if correct else 0:.1f} | "
            f"{(sum(s['spike_count'] for s in wrong) / len(wrong)) if wrong else 0:.1f} |"
        )
    lines.append("")

    for run in runs:
        lines.append(f"## {run['label']} Event Attribution")
        lines.append("| event | samples | output_tokens | spikes | visual | relation | format | answer | diffuse_low_conf | other |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for event in ["correct", "wrong", "high_conf_wrong", "long_output", "has_soft_steps", "all"]:
            if run["event_samples"].get(event, 0) == 0:
                continue
            lines.append(spike_table_line(
                event,
                run["event_samples"][event],
                run["event_token_counts"][event],
                run["event_spikes"][event],
            ))
        lines.append("")

        if sum(run["soft_neighbor_spikes"].values()) > 0:
            total = sum(run["soft_neighbor_spikes"].values())
            lines.append(f"## {run['label']} Soft-Neighborhood Spikes")
            lines.append(f"- soft window: `±{args.soft_window}` generated tokens")
            lines.append("| visual | relation | format | answer | diffuse_low_conf | other |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            c = run["soft_neighbor_spikes"]
            lines.append(
                "| "
                + " | ".join(f"{c[cat]} ({pct(c[cat], total)})" for cat in SPIKE_TYPES)
                + " |"
            )
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--min_history", type=int, default=4)
    parser.add_argument("--min_entropy", type=float, default=1.0)
    parser.add_argument("--topk_field", choices=["raw_topk", "filtered_topk"], default="raw_topk")
    parser.add_argument("--visual_tau", type=float, default=0.35)
    parser.add_argument("--relation_tau", type=float, default=0.25)
    parser.add_argument("--format_tau", type=float, default=0.45)
    parser.add_argument("--answer_tau", type=float, default=0.35)
    parser.add_argument("--low_conf_tau", type=float, default=0.20)
    parser.add_argument("--low_margin_tau", type=float, default=0.05)
    parser.add_argument("--tail_k", type=int, default=20)
    parser.add_argument("--high_conf_tail_threshold", type=float, default=0.80)
    parser.add_argument("--high_conf_any_threshold", type=float, default=0.95)
    parser.add_argument("--long_output_threshold", type=int, default=256)
    parser.add_argument("--soft_window", type=int, default=8)
    args = parser.parse_args()

    base = Path(args.base_dir)
    run_specs = [
        ("cot", base / "cot_gpu0"),
        ("lead", base / "lead_gpu1"),
        ("pure_soft", base / "pure_soft_gpu0"),
    ]
    runs = [collect_run(label, run_dir, args) for label, run_dir in run_specs]
    report = render_report(runs, args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

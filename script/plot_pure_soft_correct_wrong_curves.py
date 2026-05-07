#!/usr/bin/env python3
"""Plot correct-vs-wrong confidence and entropy curves for a trace file."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np


def load_extract_fn(evaluator_path: Path):
    spec = importlib.util.spec_from_file_location("lead_evaluator_only", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.extract_mcq_answer


def load_results(results_path: Path, evaluator_path: Path) -> dict[int, dict]:
    extract_mcq_answer = load_extract_fn(evaluator_path)
    records = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pred = extract_mcq_answer(row.get("model_answer"))
            is_correct = pred is not None and pred == row.get("answer")
            records[int(row["id"])] = {
                "gold": row.get("answer"),
                "pred": pred,
                "is_correct": is_correct,
                "output_tokens": int(row.get("output_tokens") or 0),
                "latency_sec": float(row.get("latency_sec") or 0.0),
            }
    return records


def load_specialized_results(results_path: Path) -> dict[int, dict]:
    records = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records[int(row["id"])] = {
                "gold": row.get("specialized_gold"),
                "pred": row.get("specialized_pred"),
                "is_correct": bool(row.get("specialized_is_correct")),
                "output_tokens": int(row.get("output_tokens") or 0),
                "latency_sec": float(row.get("latency_sec") or 0.0),
            }
    return records


def load_traces(trace_path: Path) -> list[dict]:
    rows = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def interpolate_metric(values: list[float], num_bins: int) -> np.ndarray:
    if not values:
        return np.full(num_bins, np.nan, dtype=float)
    if len(values) == 1:
        return np.full(num_bins, values[0], dtype=float)
    src_x = np.linspace(0.0, 1.0, num=len(values))
    dst_x = np.linspace(0.0, 1.0, num=num_bins)
    return np.interp(dst_x, src_x, np.asarray(values, dtype=float))


def group_curves(
    traces: list[dict],
    result_meta: dict[int, dict],
    num_bins: int,
) -> tuple[dict[str, list[np.ndarray]], list[dict]]:
    grouped = {
        "correct_conf": [],
        "wrong_conf": [],
        "correct_entropy": [],
        "wrong_entropy": [],
    }
    sample_rows = []

    for row in traces:
        sample_id = int(row["id"])
        meta = result_meta.get(sample_id)
        if meta is None:
            continue
        tokens = row.get("tokens", [])
        raw_conf = [
            float(token.get("raw_selected_prob", token.get("selected_prob", token.get("confidence", 0.0))))
            for token in tokens
        ]
        raw_entropy = [
            float(token.get("raw_entropy", 0.0))
            for token in tokens
        ]
        conf_curve = interpolate_metric(raw_conf, num_bins)
        entropy_curve = interpolate_metric(raw_entropy, num_bins)
        key_prefix = "correct" if meta["is_correct"] else "wrong"
        grouped[f"{key_prefix}_conf"].append(conf_curve)
        grouped[f"{key_prefix}_entropy"].append(entropy_curve)

        sample_rows.append(
            {
                "id": sample_id,
                "is_correct": meta["is_correct"],
                "pred": meta["pred"],
                "gold": meta["gold"],
                "mean_raw_conf": fmean(raw_conf) if raw_conf else math.nan,
                "last10_raw_conf": fmean(raw_conf[-10:]) if raw_conf else math.nan,
                "last20_raw_conf": fmean(raw_conf[-20:]) if raw_conf else math.nan,
                "mean_raw_entropy": fmean(raw_entropy) if raw_entropy else math.nan,
                "output_tokens": meta["output_tokens"],
                "latency_sec": meta["latency_sec"],
            }
        )

    return grouped, sample_rows


def summarize_group(sample_rows: list[dict], is_correct: bool) -> dict:
    rows = [row for row in sample_rows if row["is_correct"] == is_correct]
    return {
        "count": len(rows),
        "mean_raw_conf": fmean([row["mean_raw_conf"] for row in rows]),
        "last10_raw_conf": fmean([row["last10_raw_conf"] for row in rows]),
        "last20_raw_conf": fmean([row["last20_raw_conf"] for row in rows]),
        "mean_raw_entropy": fmean([row["mean_raw_entropy"] for row in rows]),
        "output_tokens": fmean([row["output_tokens"] for row in rows]),
        "latency_sec": fmean([row["latency_sec"] for row in rows]),
    }


def topk_wrong_rate(sample_rows: list[dict], metric: str, k: int) -> float:
    ranked = sorted(sample_rows, key=lambda row: row[metric], reverse=True)
    subset = ranked[:k]
    if not subset:
        return math.nan
    wrong = sum(not row["is_correct"] for row in subset)
    return wrong / len(subset)


def plot_group_curves(grouped: dict[str, list[np.ndarray]], output_path: Path) -> None:
    x = np.linspace(0.0, 100.0, num=len(grouped["correct_conf"][0]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {
        "correct": "#2563eb",
        "wrong": "#dc2626",
    }
    labels = {
        "correct": "Correct",
        "wrong": "Wrong",
    }

    for prefix, axis, title, y_label in [
        ("conf", axes[0], "Raw confidence over normalized generation progress", "Raw selected prob"),
        ("entropy", axes[1], "Raw entropy over normalized generation progress", "Raw entropy"),
    ]:
        for group in ["correct", "wrong"]:
            curves = np.vstack(grouped[f"{group}_{prefix}"])
            mean_curve = np.nanmean(curves, axis=0)
            std_curve = np.nanstd(curves, axis=0)
            axis.plot(x, mean_curve, color=colors[group], linewidth=2.0, label=labels[group])
            axis.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=colors[group],
                alpha=0.14,
            )
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    axes[1].set_xlabel("Normalized generation progress (%)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument(
        "--results_format",
        choices=["default", "specialized"],
        default="default",
    )
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_bins", type=int, default=200)
    args = parser.parse_args()

    if args.results_format == "specialized":
        result_meta = load_specialized_results(Path(args.results))
    else:
        if not args.evaluator:
            raise SystemExit("--evaluator is required when --results_format=default")
        result_meta = load_results(Path(args.results), Path(args.evaluator))
    traces = load_traces(Path(args.trace))
    grouped, sample_rows = group_curves(traces, result_meta, args.num_bins)
    plot_group_curves(grouped, Path(args.output_png))

    summary = {
        "total_samples": len(sample_rows),
        "correct": summarize_group(sample_rows, True),
        "wrong": summarize_group(sample_rows, False),
        "topk_wrong_rate": {
            "mean_raw_conf_top5": topk_wrong_rate(sample_rows, "mean_raw_conf", 5),
            "last10_raw_conf_top5": topk_wrong_rate(sample_rows, "last10_raw_conf", 5),
            "last20_raw_conf_top5": topk_wrong_rate(sample_rows, "last20_raw_conf", 5),
        },
        "sample_rows": sample_rows,
    }
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

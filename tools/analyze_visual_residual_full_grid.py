#!/usr/bin/env python3
"""Aggregate the full visual-residual grid and apply matched evaluators."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_TOTALS = {
    "mmvp": 300,
    "vstar": 191,
    "realworldqa": 200,
    "visulogic": 1000,
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_dataset(value: str) -> str:
    value = value.lower()
    if "realworld" in value:
        return "realworldqa"
    if "visulogic" in value:
        return "visulogic"
    if "vstar" in value:
        return "vstar"
    if "mmvp" in value:
        return "mmvp"
    raise ValueError(f"Unknown dataset: {value}")


def config_key(row: dict) -> tuple[str, float, int]:
    return (
        str(row["branch"]),
        round(float(row.get("residual_strength", 0.0)), 6),
        int(row.get("residual_duration", 0)),
    )


def evaluate_rows(dataset: str, rows: list[dict], repo: Path) -> tuple[dict, list[dict]]:
    if dataset == "mmvp":
        from script.evaluate_specialized_results import evaluate

        report, enriched = evaluate(rows, rows, "mmvp")
        for row in enriched:
            row["eval_pred"] = row.get("specialized_pred")
            row["eval_correct"] = bool(row.get("specialized_is_correct"))
        return report, enriched

    if dataset == "realworldqa":
        from script.evaluate_realworldqa_mcq import evaluate

        report, enriched = evaluate(rows, rows)
        for row in enriched:
            row["eval_pred"] = row.get("realworldqa_pred")
            row["eval_correct"] = bool(row.get("realworldqa_is_correct"))
        return report, enriched

    from script.evaluate_specialized_results import extract_mcq_letter

    enriched = []
    correct = 0
    failed = 0
    for source in rows:
        row = dict(source)
        pred = extract_mcq_letter(row.get("model_answer", ""))
        gold = str(row.get("answer") or "").strip().upper()
        is_correct = pred is not None and pred == gold
        row["eval_pred"] = pred
        row["eval_correct"] = is_correct
        enriched.append(row)
        correct += int(is_correct)
        failed += int(pred is None)
    return {
        "mode": "corrected_last_answer_mcq",
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "failed_extraction": failed,
    }, enriched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    result_files = sorted(args.experiment_root.glob("task_*/results.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No task results under {args.experiment_root}")

    grouped: dict[str, dict[tuple[str, float, int], dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    duplicates = []
    for path in result_files:
        config_path = path.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dataset = normalize_dataset(str(config.get("dataset", path.parent.name)))
        for row in load_jsonl(path):
            key = config_key(row)
            sample_id = str(row["id"])
            if sample_id in grouped[dataset][key]:
                duplicates.append((dataset, key, sample_id))
            grouped[dataset][key][sample_id] = row
    if duplicates:
        raise RuntimeError(f"Found {len(duplicates)} duplicate results; first={duplicates[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    evaluated: dict[str, dict[tuple[str, float, int], dict[str, dict]]] = defaultdict(dict)
    for dataset in EXPECTED_TOTALS:
        configs = grouped.get(dataset)
        if not configs:
            raise RuntimeError(f"Missing all results for {dataset}")
        for key, by_id in sorted(configs.items()):
            rows = list(by_id.values())
            if len(rows) != EXPECTED_TOTALS[dataset]:
                raise RuntimeError(
                    f"Incomplete {dataset} {key}: {len(rows)}/{EXPECTED_TOTALS[dataset]}"
                )
            if any(row.get("error_type") for row in rows):
                raise RuntimeError(f"Runtime errors in {dataset} {key}")
            report, enriched = evaluate_rows(dataset, rows, args.repo)
            evaluated[dataset][key] = {str(row["id"]): row for row in enriched}
            summary_rows.append({
                "dataset": dataset,
                "branch": key[0],
                "strength": key[1],
                "duration": key[2],
                "total": report["total"],
                "correct": report["correct"],
                "accuracy": report["accuracy"],
                "failed_extraction": report.get("failed_extraction", 0),
                "pair_accuracy": report.get("pair_accuracy"),
                "pair_correct": report.get("pair_correct"),
                "mean_latency_seconds": (
                    sum(float(row.get("latency_seconds", 0.0)) for row in rows)
                    / len(rows)
                    if rows else 0.0
                ),
                "injection_applied": sum(bool(row.get("injection_applied")) for row in rows),
            })

    baseline_key = ("talr", 0.0, 0)
    candidates = sorted({
        key for dataset in evaluated.values() for key in dataset
        if key[0] == "talr_true_residual"
    })
    ranked = []
    for key in candidates:
        dataset_stats = []
        for dataset in EXPECTED_TOTALS:
            baseline = evaluated[dataset][baseline_key]
            treatment = evaluated[dataset][key]
            fixed = damaged = 0
            for sample_id in baseline:
                b = baseline[sample_id]["eval_correct"]
                t = treatment[sample_id]["eval_correct"]
                fixed += int((not b) and t)
                damaged += int(b and (not t))
            net = fixed - damaged
            dataset_stats.append({
                "dataset": dataset,
                "fixed": fixed,
                "damaged": damaged,
                "net_correct": net,
                "delta_accuracy": net / EXPECTED_TOTALS[dataset],
            })
        total_net = sum(item["net_correct"] for item in dataset_stats)
        nonnegative = sum(item["net_correct"] >= 0 for item in dataset_stats)
        worst_pp = min(item["delta_accuracy"] for item in dataset_stats) * 100
        eligible = total_net > 0 and nonnegative >= 3 and worst_pp >= -1.0
        ranked.append({
            "branch": key[0],
            "strength": key[1],
            "duration": key[2],
            "total_net_correct": total_net,
            "nonnegative_datasets": nonnegative,
            "worst_delta_pp": worst_pp,
            "eligible": eligible,
            "datasets": dataset_stats,
        })
    ranked.sort(
        key=lambda row: (
            row["eligible"], row["total_net_correct"],
            row["nonnegative_datasets"], row["worst_delta_pp"],
        ),
        reverse=True,
    )

    artifact = {
        "experiment_root": str(args.experiment_root),
        "selection": "full datasets only; one global configuration",
        "expected_totals": EXPECTED_TOTALS,
        "decision_rule": {
            "total_net_correct": "> 0",
            "nonnegative_datasets": ">= 3/4",
            "worst_single_dataset_drop": ">= -1.0 pp",
        },
        "metrics": summary_rows,
        "ranking": ranked,
        "recommended": ranked[0] if ranked and ranked[0]["eligible"] else None,
    }
    (args.output_dir / "full_grid_summary.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lookup = {
        (row["dataset"], row["branch"], row["strength"], row["duration"]): row
        for row in summary_rows
    }
    lines = [
        "# Visual Residual Full-Grid Summary",
        "",
        "All parameter groups were evaluated on the complete VStar, MMVP, "
        "RealWorldQA fixed200, and VisuLogic datasets. No subset screening was used.",
        "",
        "| Rank | Strength | Duration | MMVP Δ | VStar Δ | RWQA Δ | VisuLogic Δ | Total net | Eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, item in enumerate(ranked, 1):
        deltas = {entry["dataset"]: entry["delta_accuracy"] * 100 for entry in item["datasets"]}
        lines.append(
            f"| {index} | {item['strength']:.3f} | {item['duration']} | "
            f"{deltas['mmvp']:+.2f} | {deltas['vstar']:+.2f} | "
            f"{deltas['realworldqa']:+.2f} | {deltas['visulogic']:+.2f} | "
            f"{item['total_net_correct']:+d} | {'yes' if item['eligible'] else 'no'} |"
        )
    lines.extend(["", "## Absolute Accuracy", ""])
    for dataset in EXPECTED_TOTALS:
        baseline = lookup[(dataset, "talr", 0.0, 0)]
        lines.append(f"- **{dataset} TALR baseline:** {100 * baseline['accuracy']:.2f}%")
    if artifact["recommended"]:
        best = artifact["recommended"]
        lines.extend([
            "",
            "## Pre-registered Selection",
            "",
            f"Selected strength={best['strength']}, duration={best['duration']}; "
            f"total net correct={best['total_net_correct']:+d}.",
        ])
    else:
        lines.extend([
            "", "## Pre-registered Selection", "",
            "No parameter group passed the cross-dataset stability gate.",
        ])
    (args.output_dir / "full_grid_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

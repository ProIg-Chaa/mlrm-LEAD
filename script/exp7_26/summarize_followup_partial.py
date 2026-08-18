#!/usr/bin/env python3
"""Summarize completed follow-up Atlas labels without post-hoc overclaiming."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


STRENGTH = {
    "contracted_soft_l090": 0.90,
    "contracted_soft_l095": 0.95,
    "pure_soft_l100": 1.00,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.labels:
        rows.extend(read_jsonl(path))
    datasets_by_treatment = defaultdict(set)
    for row in rows:
        datasets_by_treatment[row["treatment"]].add(row["dataset"])
    common_datasets = set.intersection(*datasets_by_treatment.values())
    rows = [row for row in rows if row["dataset"] in common_datasets]

    hard_by_dataset = defaultdict(dict)
    totals = defaultdict(Counter)
    per_step = defaultdict(Counter)
    action_map = defaultdict(dict)
    for row in rows:
        dataset = row["dataset"]
        treatment = row["treatment"]
        step = int(row["event_step"])
        original_id = row["original_id"]
        hard_by_dataset[dataset][original_id] = int(bool(row["base_correct"]))
        key = (dataset, treatment)
        totals[key]["events"] += 1
        totals[key]["fixed"] += int(bool(row["fixed"]))
        totals[key]["damaged"] += int(bool(row["damaged"]))
        totals[key]["answer_changed"] += int(bool(row["answer_changed"]))
        totals[key]["failed"] += int(bool(row["treatment_failed_extraction"]))
        step_key = (dataset, treatment, step)
        per_step[step_key]["n"] += 1
        per_step[step_key]["hard_correct"] += int(bool(row["base_correct"]))
        per_step[step_key]["treated_correct"] += int(bool(row["treatment_correct"]))
        per_step[step_key]["fixed"] += int(bool(row["fixed"]))
        per_step[step_key]["damaged"] += int(bool(row["damaged"]))
        per_step[step_key]["failed"] += int(bool(row["treatment_failed_extraction"]))
        action_map[(dataset, original_id, step)][treatment] = row

    sensitivity = defaultdict(Counter)
    for (dataset, _, _), actions in action_map.items():
        if set(actions) != set(STRENGTH):
            continue
        predictions = [actions[name].get("treatment_pred") for name in STRENGTH]
        correctness = [actions[name].get("treatment_correct") for name in STRENGTH]
        sensitivity[dataset]["events"] += 1
        sensitivity[dataset]["prediction_sensitive"] += int(len(set(predictions)) > 1)
        sensitivity[dataset]["correctness_sensitive"] += int(len(set(correctness)) > 1)

    summary = {
        "common_datasets": sorted(common_datasets),
        "rows": len(rows),
        "hard": {
            dataset: {
                "samples": len(samples),
                "correct": sum(samples.values()),
                "accuracy": sum(samples.values()) / len(samples),
            }
            for dataset, samples in sorted(hard_by_dataset.items())
        },
        "totals": {
            f"{dataset}/{treatment}": {
                **dict(counts),
                "net": counts["fixed"] - counts["damaged"],
                "strength": STRENGTH[treatment],
            }
            for (dataset, treatment), counts in sorted(totals.items())
        },
        "per_step": {
            f"{dataset}/{treatment}/{step}": {
                **dict(counts),
                "hard_accuracy": counts["hard_correct"] / counts["n"],
                "treated_accuracy": counts["treated_correct"] / counts["n"],
                "delta": (
                    counts["treated_correct"] - counts["hard_correct"]
                ) / counts["n"],
                "net": counts["fixed"] - counts["damaged"],
            }
            for (dataset, treatment, step), counts in sorted(per_step.items())
        },
        "sensitivity": {
            dataset: {
                **dict(counts),
                "prediction_sensitive_rate": (
                    counts["prediction_sensitive"] / counts["events"]
                ),
                "correctness_sensitive_rate": (
                    counts["correctness_sensitive"] / counts["events"]
                ),
            }
            for dataset, counts in sorted(sensitivity.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partial_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Follow-up Atlas Partial Summary",
        "",
        "Only datasets complete at all three strengths are included.",
        "",
        "## Hard baseline",
        "",
        "| Dataset | Samples | Accuracy |",
        "|---|---:|---:|",
    ]
    for dataset, value in summary["hard"].items():
        lines.append(
            f"| {dataset} | {value['samples']} | {value['accuracy']:.4f} |"
        )
    lines += [
        "",
        "## Event-level totals",
        "",
        "| Dataset | Lambda | Events | Fixed | Damaged | Net | Changed | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in summary["totals"].items():
        dataset, _ = key.split("/", 1)
        lines.append(
            f"| {dataset} | {value['strength']:.2f} | {value['events']} | "
            f"{value['fixed']} | {value['damaged']} | {value['net']:+d} | "
            f"{value['answer_changed']} | {value['failed']} |"
        )
    lines += [
        "",
        "## Per-step policies",
        "",
        "| Dataset | Lambda | Step | Accuracy | Delta | Fixed | Damaged | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in summary["per_step"].items():
        dataset, treatment, step = key.split("/")
        lines.append(
            f"| {dataset} | {STRENGTH[treatment]:.2f} | {step} | "
            f"{value['treated_accuracy']:.4f} | {value['delta']:+.4f} | "
            f"{value['fixed']} | {value['damaged']} | {value['failed']} |"
        )
    lines += [
        "",
        "## Strength sensitivity",
        "",
        "| Dataset | Sample-checkpoints | Prediction sensitive | Correctness sensitive |",
        "|---|---:|---:|---:|",
    ]
    for dataset, value in summary["sensitivity"].items():
        lines.append(
            f"| {dataset} | {value['events']} | "
            f"{value['prediction_sensitive_rate']:.4f} | "
            f"{value['correctness_sensitive_rate']:.4f} |"
        )
    (args.output_dir / "partial_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

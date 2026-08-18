#!/usr/bin/env python3
"""Summarize matched Atlas outcomes across intervention strengths."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRENGTH = {
    "contracted_soft_l075": 0.75,
    "contracted_soft_l090": 0.90,
    "contracted_soft_l095": 0.95,
    "pure_soft_l100": 1.00,
    "pure_soft": 1.00,
}
FIXED_EVENTS = {"fixed_1", "fixed_2", "fixed_4", "fixed_8", "fixed_16", "fixed_32"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extensions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    extension_rows = []
    for path in args.extensions:
        extension_rows.extend(load_jsonl(path))
    extension_event_ids = {row["event_id"] for row in extension_rows}
    base_rows = [
        row
        for row in load_jsonl(args.base)
        if row["event_id"] in extension_event_ids
    ]
    rows = [
        row
        for row in base_rows + extension_rows
        if row.get("event_type") in FIXED_EVENTS
        and row.get("treatment") in STRENGTH
    ]

    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    per_step: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    event_actions: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        treatment = row["treatment"]
        dataset = row["dataset"]
        key = (dataset, treatment)
        totals[key]["events"] += 1
        totals[key]["fixed"] += int(bool(row.get("fixed")))
        totals[key]["damaged"] += int(bool(row.get("damaged")))
        totals[key]["changed"] += int(bool(row.get("answer_changed")))
        totals[key]["prefix_mismatch"] += int(not bool(row.get("prefix_match")))
        totals[key]["failed"] += int(bool(row.get("treatment_failed_extraction")))
        step_key = (dataset, treatment, row["event_type"])
        per_step[step_key]["samples"] += 1
        per_step[step_key]["base_correct"] += int(bool(row.get("base_correct")))
        per_step[step_key]["treatment_correct"] += int(bool(row.get("treatment_correct")))
        per_step[step_key]["fixed"] += int(bool(row.get("fixed")))
        per_step[step_key]["damaged"] += int(bool(row.get("damaged")))
        event_actions[(dataset, row["event_id"])][treatment] = row

    agreement = defaultdict(Counter)
    for (dataset, _), actions in event_actions.items():
        canonical = {}
        for treatment, row in actions.items():
            strength = STRENGTH[treatment]
            canonical[strength] = row
        if len(canonical) < 4:
            continue
        ordered = [canonical[value] for value in (0.75, 0.90, 0.95, 1.00)]
        predictions = [row.get("treatment_pred") for row in ordered]
        correctness = [row.get("treatment_correct") for row in ordered]
        agreement[dataset]["events"] += 1
        agreement[dataset]["all_prediction_agree"] += int(len(set(predictions)) == 1)
        agreement[dataset]["any_prediction_change"] += int(len(set(predictions)) > 1)
        agreement[dataset]["all_correctness_agree"] += int(len(set(correctness)) == 1)
        agreement[dataset]["any_correctness_change"] += int(len(set(correctness)) > 1)

    summary = {
        "scope": {
            "base_rows": len(base_rows),
            "extension_rows": len(extension_rows),
            "analyzed_rows": len(rows),
            "matched_event_ids": len(extension_event_ids),
            "note": "Partial analysis if extension shard 5 is incomplete.",
        },
        "totals": {
            f"{dataset}/{treatment}": {
                **dict(counter),
                "net": counter["fixed"] - counter["damaged"],
                "strength": STRENGTH[treatment],
            }
            for (dataset, treatment), counter in sorted(totals.items())
        },
        "per_step": {
            f"{dataset}/{treatment}/{event_type}": {
                **dict(counter),
                "net": counter["fixed"] - counter["damaged"],
                "base_accuracy": counter["base_correct"] / counter["samples"],
                "treatment_accuracy": counter["treatment_correct"] / counter["samples"],
            }
            for (dataset, treatment, event_type), counter in sorted(per_step.items())
        },
        "strength_agreement": {
            dataset: {
                **dict(counter),
                "prediction_change_rate": counter["any_prediction_change"] / counter["events"],
                "correctness_change_rate": counter["any_correctness_change"] / counter["events"],
            }
            for dataset, counter in sorted(agreement.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all_strength_rows.jsonl", rows)
    (args.output_dir / "strength_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Atlas Intervention Strength Summary",
        "",
        "## Event-level utility",
        "",
        "| Dataset | Lambda | Events | Fixed | Damaged | Net | Answer changed | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in summary["totals"].items():
        dataset, _ = key.split("/", 1)
        lines.append(
            f"| {dataset} | {value['strength']:.2f} | {value['events']} | "
            f"{value['fixed']} | {value['damaged']} | {value['net']:+d} | "
            f"{value['changed']} | {value['failed']} |"
        )
    lines += [
        "",
        "## Best single fixed-step policy",
        "",
        "| Dataset | Lambda | Step | Accuracy | Delta vs hard | Fixed | Damaged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped_steps: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, value in summary["per_step"].items():
        dataset, treatment, event_type = key.split("/", 2)
        grouped_steps[(dataset, treatment)].append((event_type, value))
    for (dataset, treatment), candidates in sorted(grouped_steps.items()):
        event_type, value = max(candidates, key=lambda item: item[1]["treatment_accuracy"])
        lines.append(
            f"| {dataset} | {STRENGTH[treatment]:.2f} | {event_type.removeprefix('fixed_')} | "
            f"{value['treatment_accuracy']:.4f} | "
            f"{value['treatment_accuracy'] - value['base_accuracy']:+.4f} | "
            f"{value['fixed']} | {value['damaged']} |"
        )
    lines += [
        "",
        "## Strength sensitivity",
        "",
        "| Dataset | Matched events | Prediction changes across lambda | Correctness changes across lambda |",
        "|---|---:|---:|---:|",
    ]
    for dataset, value in summary["strength_agreement"].items():
        lines.append(
            f"| {dataset} | {value['events']} | {value['prediction_change_rate']:.4f} | "
            f"{value['correctness_change_rate']:.4f} |"
        )
    (args.output_dir / "strength_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["scope"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

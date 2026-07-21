#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from talr_analysis_common import (
    load_jsonl,
    paired_groups,
    score_row,
    write_json,
)


def summarize_run(run_dir: Path, dataset: str) -> dict:
    rows = load_jsonl(run_dir / "results.jsonl")
    scored = [score_row(row) for row in rows]
    valid = [item for item in scored if not item["runtime_error"]]
    lengths = [
        int(row.get("output_tokens") or 0)
        for row in rows
        if not row.get("error_type")
    ]
    result = {
        "run_dir": str(run_dir),
        "samples": len(rows),
        "accuracy": mean(item["correct"] for item in valid) if valid else None,
        "failed_extraction": sum(item["failed_extraction"] for item in valid),
        "runtime_errors": sum(item["runtime_error"] for item in scored),
        "avg_output_tokens": mean(lengths) if lengths else None,
        "long_ge_256": sum(value >= 256 for value in lengths),
        "maxed_1024": sum(value >= 1024 for value in lengths),
    }
    if dataset == "mmvp":
        ordered = sorted(
            zip(rows, scored),
            key=lambda pair: int(pair[0].get("id", 0)),
        )
        flags = [
            bool(item["correct"]) and not item["runtime_error"]
            for _, item in ordered
        ]
        pairs = [
            flags[index] and flags[index + 1]
            for index in range(0, len(flags) - 1, 2)
        ]
        result["mmvp_pair_accuracy"] = mean(pairs) if pairs else None
    return result


def selected_runs(optimization_root: Path, locked: dict) -> tuple[dict, dict]:
    selected_guard = locked["selected_guard"]
    details = locked["guard_selection"]["details"][selected_guard]["datasets"]
    runs = {
        "r1_onevision_7b_rl": {
            dataset: value["candidate"]["run_dir"]
            for dataset, value in details.items()
        }
    }
    references = {
        "r1_onevision_7b_rl": {
            dataset: value["full_lead"]["run_dir"]
            for dataset, value in details.items()
        }
    }
    validation_path = optimization_root / "locked_validation_runs.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        talr_validation = validation.get("talr", validation)
        for model, datasets in talr_validation.items():
            runs.setdefault(model, {}).update(datasets)
        for model, datasets in validation.get("full_lead", {}).items():
            references.setdefault(model, {}).update(datasets)
    return runs, references


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    locked = json.loads(
        (args.optimization_root / "locked_talr_config.json").read_text(
            encoding="utf-8"
        )
    )
    references = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    selected = references.get("selected", {})
    talr_runs, matched_references = selected_runs(args.optimization_root, locked)

    cells = []
    comparison_manifest = {"comparisons": []}
    for model, datasets in talr_runs.items():
        runs_for_manifest = {}
        for dataset, talr_dir_text in datasets.items():
            reference_key = f"{model}/{dataset}/full_lead"
            reference_dir_text = matched_references.get(model, {}).get(
                dataset
            ) or selected.get(reference_key)
            if not reference_dir_text:
                cells.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "status": "missing_full_lead_reference",
                    }
                )
                continue
            talr_dir = Path(talr_dir_text)
            reference_dir = Path(reference_dir_text)
            talr = summarize_run(talr_dir, dataset)
            lead = summarize_run(reference_dir, dataset)
            left_rows = {
                str(row.get("id")): row
                for row in load_jsonl(reference_dir / "results.jsonl")
            }
            right_rows = {
                str(row.get("id")): row
                for row in load_jsonl(talr_dir / "results.jsonl")
            }
            groups = paired_groups(left_rows, right_rows)
            cell = {
                "model": model,
                "dataset": dataset,
                "status": "complete",
                "full_lead": lead,
                "talr": talr,
                "accuracy_delta": talr["accuracy"] - lead["accuracy"],
                "fixed": len(groups.get("fixed", [])),
                "damaged": len(groups.get("damaged", [])),
                "net_fixed": len(groups.get("fixed", []))
                - len(groups.get("damaged", [])),
            }
            if dataset == "mmvp":
                cell["mmvp_pair_delta"] = (
                    talr["mmvp_pair_accuracy"] - lead["mmvp_pair_accuracy"]
                )
            cells.append(cell)

            runs_for_manifest.setdefault(dataset, {})["full_lead"] = str(
                reference_dir
            )
            runs_for_manifest[dataset]["talr"] = str(talr_dir)
            initial_key = f"{model}/{dataset}/initial_transition"
            if initial_key in selected:
                runs_for_manifest[dataset]["initial_transition"] = selected[
                    initial_key
                ]
        for dataset, runs in runs_for_manifest.items():
            comparison_manifest["comparisons"].append(
                {"model": model, "dataset": dataset, "runs": runs}
            )

    valid = [cell for cell in cells if cell["status"] == "complete"]
    deltas = [cell["accuracy_delta"] for cell in valid]
    total_talr_failed = sum(
        cell["talr"]["failed_extraction"] for cell in valid
    )
    total_lead_failed = sum(
        cell["full_lead"]["failed_extraction"] for cell in valid
    )
    total_talr_long = sum(cell["talr"]["long_ge_256"] for cell in valid)
    total_lead_long = sum(cell["full_lead"]["long_ge_256"] for cell in valid)
    total_talr_maxed = sum(cell["talr"]["maxed_1024"] for cell in valid)
    total_lead_maxed = sum(cell["full_lead"]["maxed_1024"] for cell in valid)
    acceptance = {
        "complete_cells": len(valid),
        "mean_delta_positive": bool(deltas and mean(deltas) > 0),
        "noninferior_cells_ge_6": sum(delta >= -0.005 for delta in deltas) >= 6,
        "no_drop_over_2pp": bool(deltas and min(deltas) >= -0.02),
        "failed_not_worse": total_talr_failed <= total_lead_failed,
        "long_not_worse": total_talr_long <= total_lead_long,
        "maxed_not_worse": total_talr_maxed <= total_lead_maxed,
    }
    acceptance["passed"] = (
        len(valid) == 8 and all(value for key, value in acceptance.items() if key not in {"complete_cells", "passed"})
    )
    summary = {
        "locked_config": locked,
        "cells": cells,
        "mean_accuracy_delta": mean(deltas) if deltas else None,
        "acceptance": acceptance,
    }
    write_json(args.output_dir / "talr_optimization_summary.json", summary)
    write_json(
        args.output_dir / "locked_comparison_manifest.json",
        comparison_manifest,
    )

    lines = [
        "# Locked TALR Optimization Summary",
        "",
        f"Selected: W={locked['refinement_window']}, "
        f"K={locked['refinement_soft_cap']}, guard={locked['selected_guard']}.",
        "",
        "| Model | Dataset | Full LEAD | TALR | Delta | Fixed | Damaged | Failed (L/T) | Long (L/T) | Maxed (L/T) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in valid:
        lead, talr = cell["full_lead"], cell["talr"]
        lines.append(
            f"| {cell['model']} | {cell['dataset']} | "
            f"{100 * lead['accuracy']:.2f}% | {100 * talr['accuracy']:.2f}% | "
            f"{100 * cell['accuracy_delta']:+.2f}pp | {cell['fixed']} | "
            f"{cell['damaged']} | {lead['failed_extraction']}/{talr['failed_extraction']} | "
            f"{lead['long_ge_256']}/{talr['long_ge_256']} | "
            f"{lead['maxed_1024']}/{talr['maxed_1024']} |"
        )
    lines.extend(
        [
            "",
            f"Mean delta: {100 * mean(deltas):+.2f}pp" if deltas else "Mean delta: unavailable",
            "",
            f"Pre-registered acceptance passed: **{acceptance['passed']}**",
            "",
            "Validation results are locked. A failed gate must not trigger another search on these cells.",
        ]
    )
    (args.output_dir / "talr_optimization_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

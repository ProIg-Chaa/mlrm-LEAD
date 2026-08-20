#!/usr/bin/env python3
"""Merge shards and evaluate TALR visual-residual probe results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "script"))
from evaluate_specialized_results import evaluate


BRANCHES = ("talr", "talr_true_residual", "talr_random_residual")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare(target, reference):
    ids = sorted(set(target) & set(reference), key=int)
    fixed = [key for key in ids if target[key]["specialized_is_correct"] and not reference[key]["specialized_is_correct"]]
    damaged = [key for key in ids if reference[key]["specialized_is_correct"] and not target[key]["specialized_is_correct"]]
    changed = [key for key in ids if target[key]["specialized_pred"] != reference[key]["specialized_pred"]]
    return {
        "n": len(ids),
        "fixed": len(fixed),
        "damaged": len(damaged),
        "net": len(fixed) - len(damaged),
        "changed_predictions": len(changed),
        "fixed_ids": fixed,
        "damaged_ids": damaged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    merged_dir = args.root / "merged"
    all_rows = []
    for shard in (0, 1):
        all_rows.extend(read_jsonl(args.root / f"shard_{shard}" / "results.jsonl"))
    unique = {}
    for row in all_rows:
        key = (int(row["id"]), str(row["branch"]))
        if key in unique:
            raise RuntimeError(f"Duplicate result: {key}")
        unique[key] = row
    if len(unique) != 64 * len(BRANCHES):
        raise RuntimeError(f"Expected 192 results, got {len(unique)}")
    dataset = read_jsonl(args.dataset)
    reports = {}
    enriched_maps = {}
    for branch in BRANCHES:
        rows = [row for (_, name), row in unique.items() if name == branch]
        rows.sort(key=lambda row: int(row["id"]))
        report, enriched = evaluate(dataset, rows, "mmvp")
        branch_dir = merged_dir / branch
        write_jsonl(branch_dir / "results.jsonl", rows)
        write_jsonl(branch_dir / "specialized_results.jsonl", enriched)
        (branch_dir / "specialized_eval_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reports[branch] = report
        enriched_maps[branch] = {str(row["id"]): row for row in enriched}

    comparisons = {
        "true_residual_vs_talr": compare(
            enriched_maps["talr_true_residual"], enriched_maps["talr"]
        ),
        "random_residual_vs_talr": compare(
            enriched_maps["talr_random_residual"], enriched_maps["talr"]
        ),
        "true_residual_vs_random": compare(
            enriched_maps["talr_true_residual"], enriched_maps["talr_random_residual"]
        ),
    }
    true_rows = [row for (_, name), row in unique.items() if name == "talr_true_residual"]
    applied = [row for row in true_rows if row.get("injection_applied")]
    summary = {
        "reports": reports,
        "comparisons": comparisons,
        "injection": {
            "applied_samples": len(applied),
            "total_samples": len(true_rows),
            "mean_refinement_step": (
                sum(int(row["refinement_step"]) for row in applied) / len(applied)
                if applied else None
            ),
            "mean_target_soft_hard_norm": (
                sum(float(row["target_soft_hard_norm"]) for row in applied) / len(applied)
                if applied else None
            ),
        },
    }
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# TALR + Visual Residual: MMVP Held-out64",
        "",
        "Atlas samples are excluded; complete MMVP pairs are retained.",
        "",
        "| Method | Sample Acc | Pair Acc | Failed |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "talr": "Frozen TALR",
        "talr_true_residual": "TALR + true visual residual",
        "talr_random_residual": "TALR + random residual",
    }
    for branch in BRANCHES:
        report = reports[branch]
        lines.append(
            f"| {labels[branch]} | {report['accuracy']:.2%} | "
            f"{report['pair_accuracy']:.2%} | {report['failed_extraction']} |"
        )
    lines.extend([
        "",
        "| Comparison | Fixed | Damaged | Net | Changed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, row in comparisons.items():
        lines.append(
            f"| {name} | {row['fixed']} | {row['damaged']} | "
            f"{row['net']:+d} | {row['changed_predictions']} |"
        )
    lines.extend([
        "",
        f"Residual injection applied to {len(applied)}/{len(true_rows)} samples.",
        "",
    ])
    (merged_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

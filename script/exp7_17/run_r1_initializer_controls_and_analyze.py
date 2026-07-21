#!/usr/bin/env python3
"""Run R1 Initial Transition controls and compare against T1.0/T1.25."""

from __future__ import annotations

import json
from pathlib import Path

import run_talr_worst_tuning_queue as queue
from talr_analysis_common import paired_groups


ROOT = queue.ROOT
PHASE = "phase_f_initializer_control"
OUTPUT_JSON = ROOT / "initializer_refiner_control_summary.json"
OUTPUT_MD = ROOT / "initializer_refiner_control_summary.md"


def run_dir(phase: str, dataset: str, name: str) -> Path:
    return ROOT / phase / "r1_rl" / dataset / f"{name}__none"


def reported_metrics(path: Path, dataset: str) -> dict:
    result = queue.metrics(path)
    if dataset == "mmvp":
        report = json.loads(
            (path / "specialized_eval_report.json").read_text(encoding="utf-8")
        )
        result["accuracy"] = report["accuracy"]
        result["pair_accuracy"] = report["pair_accuracy"]
        result["failed"] = report["failed_extraction"]
    return result


def pairwise(reference_dir: Path, method_dir: Path) -> dict:
    reference = {
        str(row.get("id")): row
        for row in queue.load_jsonl(reference_dir / "results.jsonl")
    }
    method = {
        str(row.get("id")): row
        for row in queue.load_jsonl(method_dir / "results.jsonl")
    }
    groups = paired_groups(reference, method)
    fixed = len(groups.get("fixed", []))
    damaged = len(groups.get("damaged", []))
    return {
        "fixed": fixed,
        "damaged": damaged,
        "net": fixed - damaged,
        "both_correct": len(groups.get("both_correct", [])),
        "both_wrong": len(groups.get("both_wrong", [])),
    }


def main() -> int:
    datasets = ("vstar", "mmvp")
    for dataset in datasets:
        queue.run_one(
            "r1_rl",
            dataset,
            queue.FULL_DATASETS[dataset],
            "initial_transition",
            "initial_transition",
            "none",
            PHASE,
        )

    methods = {
        "initializer": (PHASE, "initial_transition"),
        "t100": ("phase_d_full", "r_base_w8k2_t100"),
        "t125": ("phase_d_t125_ab", "r_w8k2_t125_l100"),
    }
    cells = {}
    mean_deltas = {"t100": [], "t125": []}
    for dataset in datasets:
        paths = {
            key: run_dir(phase, dataset, name)
            for key, (phase, name) in methods.items()
        }
        metrics = {
            key: reported_metrics(path, dataset)
            for key, path in paths.items()
        }
        comparisons = {}
        for key in ("t100", "t125"):
            delta = metrics[key]["accuracy"] - metrics["initializer"]["accuracy"]
            mean_deltas[key].append(delta)
            comparisons[key] = {
                "delta_vs_initializer": delta,
                "pairwise": pairwise(paths["initializer"], paths[key]),
            }
        cells[dataset] = {
            "metrics": metrics,
            "comparisons": comparisons,
        }

    average = {
        key: sum(values) / len(values)
        for key, values in mean_deltas.items()
    }
    best = max(average, key=average.get)
    decision = (
        "refiner_contributes_on_key_controls"
        if average[best] > 0
        else "initializer_sufficient"
    )
    payload = {
        "decision": decision,
        "best_refiner": best,
        "mean_delta_vs_initializer": average,
        "datasets": cells,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# R1 Initializer vs Refiner Controls",
        "",
        f"- Decision: `{decision}`",
        f"- Best refiner on these controls: `{best}`",
        f"- T1.0 mean delta vs Initializer: {100 * average['t100']:+.2f} pp",
        f"- T1.25 mean delta vs Initializer: {100 * average['t125']:+.2f} pp",
        "",
        "| Dataset | Initializer | T1.0 | T1.25 | "
        "T1.0 F/D | T1.25 F/D |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, cell in cells.items():
        metrics = cell["metrics"]
        comparisons = cell["comparisons"]
        lines.append(
            f"| {dataset} | {100 * metrics['initializer']['accuracy']:.2f}% | "
            f"{100 * metrics['t100']['accuracy']:.2f}% | "
            f"{100 * metrics['t125']['accuracy']:.2f}% | "
            f"{comparisons['t100']['pairwise']['fixed']}/"
            f"{comparisons['t100']['pairwise']['damaged']} | "
            f"{comparisons['t125']['pairwise']['fixed']}/"
            f"{comparisons['t125']['pairwise']['damaged']} |"
        )
        if dataset == "mmvp":
            lines.append(
                f"| {dataset} pair | "
                f"{100 * metrics['initializer']['pair_accuracy']:.2f}% | "
                f"{100 * metrics['t100']['pair_accuracy']:.2f}% | "
                f"{100 * metrics['t125']['pair_accuracy']:.2f}% | - | - |"
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

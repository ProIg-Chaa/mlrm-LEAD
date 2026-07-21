#!/usr/bin/env python3
"""Create a CPU-only decision packet after TALR full validation completes."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from talr_analysis_common import score_row


ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260718_talr_worst_cell_tuning"
)
LOCKED = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260717_talr_diagnosis_optimization/locked_validation"
)
VISION_COMPACT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260714_vision_r1_compact_matrix/vision_r1_7b"
)

DATASET_ALIASES = {
    "vstar": "vstar",
    "realworldqa": "realworldqa_fixed200",
    "mmvp": "mmvp",
    "visulogic": "visulogic300",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_key(row, index):
    for name in ("id", "sample_id", "question_id", "index"):
        if row.get(name) is not None:
            return f"{name}:{row[name]}"
    return f"position:{index}"


def pair_accuracy(run_dir: Path):
    path = run_dir / "specialized_eval_report.json"
    if not path.exists():
        return None
    return load_json(path).get("pair_accuracy")


def run_metrics(run_dir: Path):
    rows = load_jsonl(run_dir / "results.jsonl")
    scored = [score_row(row) for row in rows]
    traces = load_jsonl(run_dir / "token_entropy.jsonl")
    lengths = [
        int(row["output_tokens"])
        for row in traces
        if row.get("output_tokens") is not None
    ]
    summaries = [row.get("entropy_summary", {}) for row in traces]
    return {
        "run_dir": str(run_dir),
        "samples": len(rows),
        "accuracy": (
            sum(bool(item["correct"]) for item in scored) / len(scored)
            if scored
            else None
        ),
        "failed": sum(item.get("pred") is None for item in scored),
        "runtime_errors": sum(bool(row.get("error_type")) for row in rows),
        "avg_tokens": statistics.mean(lengths) if lengths else None,
        "long": sum(length >= 256 for length in lengths),
        "maxed": sum(length >= 1024 for length in lengths),
        "pair_accuracy": pair_accuracy(run_dir),
        "refinement_candidates": sum(
            item.get("lead_refinement_candidate_count", 0)
            for item in summaries
        ),
        "refinement_active": sum(
            item.get("lead_refinement_active_count", 0)
            for item in summaries
        ),
        "format_active": sum(
            item.get("format_cooldown_active_steps", 0)
            for item in summaries
        ),
        "veto": sum(
            item.get("lead_soft_veto_count", 0)
            for item in summaries
        ),
    }


def paired_delta(reference_dir: Path, current_dir: Path):
    reference_rows = load_jsonl(reference_dir / "results.jsonl")
    current_rows = load_jsonl(current_dir / "results.jsonl")
    reference = {
        row_key(row, index): score_row(row)
        for index, row in enumerate(reference_rows)
    }
    current = {
        row_key(row, index): score_row(row)
        for index, row in enumerate(current_rows)
    }
    shared = sorted(set(reference) & set(current))
    fixed = damaged = both_correct = both_wrong = 0
    for key in shared:
        ref_correct = bool(reference[key]["correct"])
        cur_correct = bool(current[key]["correct"])
        if not ref_correct and cur_correct:
            fixed += 1
        elif ref_correct and not cur_correct:
            damaged += 1
        elif ref_correct:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "paired_samples": len(shared),
        "fixed": fixed,
        "damaged": damaged,
        "net": fixed - damaged,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def compatible(reference_dir: Path, current_dir: Path):
    ref_path = reference_dir / "config.json"
    cur_path = current_dir / "config.json"
    if not ref_path.exists() or not cur_path.exists():
        return {"compatible": False, "reason": "missing config"}
    ref = load_json(ref_path)
    cur = load_json(cur_path)
    fields = [
        "cot_prompt_mode",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "max_new_tokens",
    ]
    mismatches = {
        field: [ref.get(field), cur.get(field)]
        for field in fields
        if ref.get(field) != cur.get(field)
    }
    ref_model = Path(str(ref.get("model_name", ""))).name
    cur_model = Path(str(cur.get("model_name", ""))).name
    if ref_model != cur_model:
        mismatches["model_name"] = [ref_model, cur_model]
    return {"compatible": not mismatches, "mismatches": mismatches}


def full_lead_reference(model, dataset):
    locked_name = DATASET_ALIASES[dataset]
    path = LOCKED / model / locked_name / "full_lead"
    return path if (path / "results.jsonl").exists() else None


def initializer_reference(model, dataset):
    if model != "vision_r1":
        return None
    path = (
        VISION_COMPACT
        / DATASET_ALIASES[dataset]
        / "initial_transition_only"
    )
    return path if (path / "results.jsonl").exists() else None


def current_run_dir(summary, model, dataset):
    config_name = summary["selected_configs"][model][0]
    guard = summary["selected_guards"][model]
    return (
        ROOT
        / "phase_d_full"
        / model
        / dataset
        / f"{config_name}__{guard}"
    )


def classify(lead_deltas, init_deltas):
    if not lead_deltas:
        return "reference_incomplete"
    mean_lead = statistics.mean(lead_deltas)
    minimum = min(lead_deltas)
    noninferior = sum(delta >= -0.005 for delta in lead_deltas)
    strong = (
        mean_lead > 0.005
        and noninferior / len(lead_deltas) >= 0.75
        and minimum >= -0.02
        and (not init_deltas or statistics.mean(init_deltas) > 0)
    )
    if strong:
        return "A_strong_method"
    if mean_lead >= 0 and minimum >= -0.02:
        if init_deltas and statistics.mean(init_deltas) <= 0:
            return "C_initializer_sufficient"
        return "B_competitive_constrained"
    if init_deltas and statistics.mean(init_deltas) <= 0:
        return "C_initializer_sufficient"
    return "D_regression_diagnosis"


def main():
    summary_path = ROOT / "final_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")
    summary = load_json(summary_path)
    cells = []
    lead_deltas = []
    init_deltas = []

    for model in ("r1_rl", "vision_r1"):
        for dataset in ("vstar", "realworldqa", "mmvp", "visulogic"):
            current_dir = current_run_dir(summary, model, dataset)
            current = run_metrics(current_dir)
            cell = {
                "model": model,
                "dataset": dataset,
                "current": current,
                "full_lead": None,
                "initializer": None,
            }
            for label, resolver in (
                ("full_lead", full_lead_reference),
                ("initializer", initializer_reference),
            ):
                reference_dir = resolver(model, dataset)
                if reference_dir is None:
                    continue
                parity = compatible(reference_dir, current_dir)
                reference = run_metrics(reference_dir)
                comparison = None
                delta = None
                if parity["compatible"]:
                    comparison = paired_delta(reference_dir, current_dir)
                    delta = current["accuracy"] - reference["accuracy"]
                    if label == "full_lead":
                        lead_deltas.append(delta)
                    else:
                        init_deltas.append(delta)
                cell[label] = {
                    "metrics": reference,
                    "parity": parity,
                    "delta": delta,
                    "pairwise": comparison,
                }
            cells.append(cell)

    scenario = classify(lead_deltas, init_deltas)
    next_actions = [
        {
            "priority": "P0",
            "action": "Run R1-RL W8K2-T1.25 on full VStar and MMVP",
            "reason": "Screening tie with fewer refinement events",
            "automatic": False,
        }
    ]
    if scenario == "A_strong_method":
        next_actions.append({
            "priority": "P1",
            "action": "Lock the winner, then start component ablation",
            "reason": "Strong full validation gate passed",
            "automatic": False,
        })
    elif scenario == "B_competitive_constrained":
        next_actions.append({
            "priority": "P1",
            "action": "Measure latency, memory, and deleted late-soft events",
            "reason": "Value depends on constrained equivalence",
            "automatic": False,
        })
    elif scenario == "C_initializer_sufficient":
        next_actions.append({
            "priority": "P1",
            "action": "Run initializer-vs-refiner event utility and simplify",
            "reason": "Refiner has no positive average contribution",
            "automatic": False,
        })
    else:
        next_actions.append({
            "priority": "P1",
            "action": "Analyze damaged cells before any new grid search",
            "reason": "Full validation shows regression or missing references",
            "automatic": False,
        })

    packet = {
        "scenario": scenario,
        "selected_configs": summary["selected_configs"],
        "selected_guards": summary["selected_guards"],
        "matched_lead_cells": len(lead_deltas),
        "mean_delta_vs_lead": (
            statistics.mean(lead_deltas) if lead_deltas else None
        ),
        "min_delta_vs_lead": min(lead_deltas) if lead_deltas else None,
        "matched_initializer_cells": len(init_deltas),
        "mean_delta_vs_initializer": (
            statistics.mean(init_deltas) if init_deltas else None
        ),
        "cells": cells,
        "next_actions": next_actions,
    }
    (ROOT / "post_full_decision.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ROOT / "next_experiment_manifest.json").write_text(
        json.dumps({"scenario": scenario, "actions": next_actions},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# TALR Post-Full Decision Packet",
        "",
        f"- Scenario: `{scenario}`",
        f"- Selected configs: `{summary['selected_configs']}`",
        f"- Selected guards: `{summary['selected_guards']}`",
        f"- Matched LEAD cells: {len(lead_deltas)}",
        (
            "- Mean delta vs LEAD: "
            + (f"{statistics.mean(lead_deltas) * 100:+.2f} pp"
               if lead_deltas else "N/A")
        ),
        (
            "- Mean delta vs Initializer: "
            + (f"{statistics.mean(init_deltas) * 100:+.2f} pp"
               if init_deltas else "N/A")
        ),
        "",
        "## Cells",
        "",
        "| Model | Dataset | TALR | vs LEAD | Fixed/Damaged | vs Init |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lead = cell["full_lead"]
        init = cell["initializer"]
        lead_delta = lead["delta"] if lead else None
        init_delta = init["delta"] if init else None
        pair = lead["pairwise"] if lead else None
        lines.append(
            f"| {cell['model']} | {cell['dataset']} | "
            f"{cell['current']['accuracy'] * 100:.2f}% | "
            f"{lead_delta * 100:+.2f} pp"
            if lead_delta is not None
            else
            f"| {cell['model']} | {cell['dataset']} | "
            f"{cell['current']['accuracy'] * 100:.2f}% | N/A"
        )
        suffix = (
            f" | {pair['fixed']}/{pair['damaged']} |"
            if pair else " | N/A |"
        )
        suffix += (
            f" {init_delta * 100:+.2f} pp |"
            if init_delta is not None else " N/A |"
        )
        lines[-1] += suffix
    lines.extend(["", "## Next Actions", ""])
    for action in next_actions:
        lines.append(
            f"- **{action['priority']}** {action['action']}: "
            f"{action['reason']}."
        )
    (ROOT / "post_full_decision.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "scenario": scenario,
        "mean_delta_vs_lead": packet["mean_delta_vs_lead"],
        "outputs": [
            str(ROOT / "post_full_decision.json"),
            str(ROOT / "post_full_decision.md"),
            str(ROOT / "next_experiment_manifest.json"),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

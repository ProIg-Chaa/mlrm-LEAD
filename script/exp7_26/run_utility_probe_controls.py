#!/usr/bin/env python3
"""Run leakage-safe feature and routing controls for the utility probe."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_SETS = {
    "entropy_point": {
        "log1p_raw_entropy",
        "log1p_filtered_entropy",
        "raw_top1_prob",
        "raw_margin",
        "action_lambda",
        "action_is_pure",
    },
    "timing_only": {
        "step_index",
        "log1p_step_index",
        "normalized_position_max1024",
        "prefix_length",
        "action_lambda",
        "action_is_pure",
    },
    "entropy_timing": {
        "log1p_raw_entropy",
        "log1p_filtered_entropy",
        "raw_top1_prob",
        "raw_margin",
        "entropy_delta_1",
        "entropy_mean_4",
        "entropy_std_4",
        "entropy_delta_from_mean_4",
        "entropy_mean_8",
        "entropy_std_8",
        "entropy_delta_from_mean_8",
        "entropy_mean_16",
        "entropy_std_16",
        "entropy_delta_from_mean_16",
        "step_index",
        "log1p_step_index",
        "normalized_position_max1024",
        "prefix_length",
        "action_lambda",
        "action_is_pure",
    },
    "geometry_timing": {
        "hard_embedding_norm",
        "soft_embedding_norm",
        "soft_hard_l2",
        "soft_hard_relative_l2",
        "soft_hard_cosine",
        "step_index",
        "log1p_step_index",
        "normalized_position_max1024",
        "prefix_length",
        "action_lambda",
        "action_is_pure",
    },
    "sample_only": {
        "question_char_count",
        "question_word_count",
        "option_count",
        "action_lambda",
        "action_is_pure",
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def project_rows(rows, feature_names: list[str], selected_names: list[str]):
    indices = [feature_names.index(name) for name in selected_names]
    return [replace(row, x=row.x[indices]) for row in rows]


def random_matched_policy(rows, coverage: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    by_group = {}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)
    correct = fixed = damaged = interventions = 0
    for candidates in by_group.values():
        base_correct = candidates[0].base_correct
        chosen = None
        if rng.random() < coverage:
            chosen = rng.choice(candidates)
        final_correct = chosen.treatment_correct if chosen else base_correct
        correct += final_correct
        fixed += int(not base_correct and final_correct)
        damaged += int(base_correct and not final_correct)
        interventions += int(chosen is not None)
    total = len(by_group)
    return {
        "samples": total,
        "accuracy": correct / total,
        "fixed": fixed,
        "damaged": damaged,
        "net": fixed - damaged,
        "coverage": interventions / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rho", type=float, default=1.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 42])
    args = parser.parse_args()

    sys.path.insert(0, str(args.trainer_dir))
    import train_intervention_utility_probe as probe  # noqa: PLC0415

    raw_rows = probe.read_jsonl(args.atlas)
    rows, feature_names, dataset_stats = probe.prepare_rows(raw_rows)
    feature_sets = dict(FEATURE_SETS)
    feature_sets["full_no_strength"] = set(feature_names) - {
        "action_lambda",
        "action_is_pure",
    }
    feature_sets["full"] = set(feature_names)

    all_groups = {row.group for row in rows}
    results = {}
    for feature_set, requested in feature_sets.items():
        selected = sorted(set(feature_names) & set(requested))
        projected = project_rows(rows, feature_names, selected)
        runs = []
        for seed in args.seeds:
            folds = probe.grouped_folds(projected, args.folds, seed)
            for fold_index, test_groups in enumerate(folds):
                train_groups = all_groups - test_groups
                run = probe.run_split(
                    projected,
                    train_groups,
                    test_groups,
                    "linear",
                    seed + fold_index * 100,
                    args.rho,
                )
                run["fold"] = fold_index
                runs.append(run)
        results[feature_set] = {
            "features": selected,
            "aggregate": probe.aggregate_runs(runs),
            "runs": runs,
        }

    full_mlp_runs = []
    for seed in args.seeds:
        folds = probe.grouped_folds(rows, args.folds, seed)
        for fold_index, test_groups in enumerate(folds):
            full_mlp_runs.append(
                probe.run_split(
                    rows,
                    all_groups - test_groups,
                    test_groups,
                    "mlp",
                    seed + fold_index * 100,
                    args.rho,
                )
            )
    results["full_mlp"] = {
        "features": feature_names,
        "aggregate": probe.aggregate_runs(full_mlp_runs),
        "runs": full_mlp_runs,
    }

    no_strength_names = sorted(feature_sets["full_no_strength"])
    no_strength_rows = project_rows(rows, feature_names, no_strength_names)
    no_strength_mlp_runs = []
    for seed in args.seeds:
        folds = probe.grouped_folds(no_strength_rows, args.folds, seed)
        for fold_index, test_groups in enumerate(folds):
            no_strength_mlp_runs.append(
                probe.run_split(
                    no_strength_rows,
                    all_groups - test_groups,
                    test_groups,
                    "mlp",
                    seed + fold_index * 100,
                    args.rho,
                )
            )
    results["full_no_strength_mlp"] = {
        "features": no_strength_names,
        "aggregate": probe.aggregate_runs(no_strength_mlp_runs),
        "runs": no_strength_mlp_runs,
    }

    random_controls = []
    for run_index, run in enumerate(full_mlp_runs):
        test_seed = args.seeds[run_index // args.folds]
        fold_index = run_index % args.folds
        test_groups = probe.grouped_folds(rows, args.folds, test_seed)[fold_index]
        test_rows = [row for row in rows if row.group in test_groups]
        for repeat in range(20):
            random_controls.append(
                random_matched_policy(
                    test_rows,
                    run["test_policy"]["coverage"],
                    seed=10000 + run_index * 100 + repeat,
                )
            )
    random_accuracy = [row["accuracy"] for row in random_controls]
    random_net = [row["net"] for row in random_controls]

    summary = {
        "dataset": dataset_stats,
        "rho": args.rho,
        "feature_controls": results,
        "matched_rate_random": {
            "runs": len(random_controls),
            "accuracy_mean": float(np.mean(random_accuracy)),
            "accuracy_std": float(np.std(random_accuracy)),
            "net_mean": float(np.mean(random_net)),
            "net_std": float(np.std(random_net)),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "probe_control_results.json", summary)
    lines = [
        "# Utility Probe Control Experiments",
        "",
        "| Control | Features | Accuracy | Coverage | Net fixed-damaged | Fix AUROC | Damage AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        aggregate = result["aggregate"]
        lines.append(
            f"| {name} | {len(result['features'])} | "
            f"{aggregate['accuracy']['mean']:.4f} | "
            f"{aggregate['coverage']['mean']:.4f} | "
            f"{aggregate['net_fixed_minus_damaged']['mean']:.2f} | "
            f"{aggregate['fix_auroc'] if aggregate['fix_auroc'] is not None else 'NA'} | "
            f"{aggregate['damage_auroc'] if aggregate['damage_auroc'] is not None else 'NA'} |"
        )
    lines += [
        "",
        "## Matched-rate random",
        "",
        f"- Accuracy: {summary['matched_rate_random']['accuracy_mean']:.4f} "
        f"+/- {summary['matched_rate_random']['accuracy_std']:.4f}",
        f"- Net fixed-damaged: {summary['matched_rate_random']['net_mean']:.2f} "
        f"+/- {summary['matched_rate_random']['net_std']:.2f}",
    ]
    (args.output_dir / "probe_control_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    main()

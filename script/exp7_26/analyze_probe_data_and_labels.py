#!/usr/bin/env python3
"""Diagnose whether the intervention probe is data- or target-limited."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


CORE_FEATURES = {
    "log1p_raw_entropy",
    "log1p_filtered_entropy",
    "raw_top1_prob",
    "raw_margin",
    "entropy_delta_1",
    "entropy_std_4",
    "entropy_delta_from_mean_4",
    "soft_hard_relative_l2",
    "soft_hard_cosine",
    "log1p_step_index",
    "normalized_position_max1024",
    "recent16_duplicate_ratio",
    "token_is_newline",
    "token_is_answer_marker",
    "action_lambda",
    "action_is_pure",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def project_rows(rows, names: list[str], selected: list[str]):
    indices = [names.index(name) for name in selected]
    return [replace(row, x=row.x[indices]) for row in rows]


def stratified_subsample(rows, groups: set[str], fraction: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    group_dataset: dict[str, str] = {}
    for row in rows:
        if row.group in groups:
            group_dataset[row.group] = row.dataset
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for group, dataset in group_dataset.items():
        by_dataset[dataset].append(group)
    chosen: set[str] = set()
    for dataset_groups in by_dataset.values():
        rng.shuffle(dataset_groups)
        count = max(4, round(len(dataset_groups) * fraction))
        chosen.update(dataset_groups[: min(count, len(dataset_groups))])
    return chosen


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def learning_curves(probe, rows, fractions: list[float], seeds: list[int], folds: int, rho: float):
    all_groups = {row.group for row in rows}
    output: dict[str, Any] = {}
    for fraction in fractions:
        runs = []
        for seed in seeds:
            fold_groups = probe.grouped_folds(rows, folds, seed)
            for fold_index, test_groups in enumerate(fold_groups):
                available = all_groups - test_groups
                train_groups = stratified_subsample(
                    rows, available, fraction, seed + fold_index * 1000
                )
                run = probe.run_split(
                    rows,
                    train_groups,
                    test_groups,
                    "mlp",
                    seed + fold_index * 100,
                    rho,
                )
                runs.append(run)
        output[str(fraction)] = {
            "aggregate": probe.aggregate_runs(runs),
            "fit_samples": aggregate([run["fit_samples"] for run in runs]),
            "calibration_samples": aggregate(
                [run["calibration_samples"] for run in runs]
            ),
            "runs": runs,
        }
    return output


def feature_ablation(probe, rows, feature_names, seeds, folds, rho):
    sets = {
        "core16": sorted(CORE_FEATURES & set(feature_names)),
        "full40": list(feature_names),
    }
    all_groups = {row.group for row in rows}
    result = {}
    for name, selected in sets.items():
        selected_rows = project_rows(rows, feature_names, selected)
        runs = []
        for seed in seeds:
            for fold_index, test_groups in enumerate(
                probe.grouped_folds(selected_rows, folds, seed)
            ):
                runs.append(
                    probe.run_split(
                        selected_rows,
                        all_groups - test_groups,
                        test_groups,
                        "mlp",
                        seed + fold_index * 100,
                        rho,
                    )
                )
        result[name] = {
            "features": selected,
            "aggregate": probe.aggregate_runs(runs),
            "runs": runs,
        }
    return result


def row_key(raw: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(raw["original_id"]),
        str(raw["event_type"]),
        str(raw["treatment"]),
    )


def raw_actionability(raw: dict[str, Any]) -> tuple[int, str]:
    answer_changed = bool(raw.get("answer_changed"))
    first_divergence = raw.get("first_divergence_step") is not None
    mismatch = max(
        float(raw.get("mismatch_ratio_h8") or 0.0),
        float(raw.get("mismatch_ratio_h16") or 0.0),
        float(raw.get("mismatch_ratio_h32") or 0.0),
    )
    if answer_changed:
        return 1, "answer_changed"
    if first_divergence or mismatch > 0:
        return 1, "prefix_only"
    return 0, "no_observed_change"


def binary_grouped_cv(
    probe,
    rows,
    labels: np.ndarray,
    seeds: list[int],
    folds: int,
    kind: str = "mlp",
) -> dict[str, Any]:
    all_groups = {row.group for row in rows}
    aucs: list[float] = []
    aps: list[float] = []
    fold_records = []
    for seed in seeds:
        for fold_index, test_groups in enumerate(
            probe.grouped_folds(rows, folds, seed)
        ):
            train_groups = all_groups - test_groups
            train_indices = [
                index for index, row in enumerate(rows) if row.group in train_groups
            ]
            test_indices = [
                index for index, row in enumerate(rows) if row.group in test_groups
            ]
            x_train = np.stack([rows[index].x for index in train_indices])
            x_test = np.stack([rows[index].x for index in test_indices])
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[std < 1e-6] = 1.0
            model = probe.train_head(
                (x_train - mean) / std,
                labels[train_indices].astype(np.float32),
                kind,
                seed + fold_index * 100,
            )
            with torch.no_grad():
                scores = np.asarray(
                    torch.sigmoid(
                        model(
                            torch.tensor(
                                (x_test - mean) / std, dtype=torch.float32
                            )
                        )
                    ).tolist(),
                    dtype=np.float64,
                )
            y_test = labels[test_indices]
            auc = probe.auc_score(y_test, scores)
            ap = probe.average_precision(y_test, scores)
            if auc is not None:
                aucs.append(auc)
            if ap is not None:
                aps.append(ap)
            fold_records.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "test_rows": len(test_indices),
                    "positive_rate": float(y_test.mean()),
                    "auroc": auc,
                    "auprc": ap,
                }
            )
    return {
        "auroc": aggregate(aucs) if aucs else None,
        "auprc": aggregate(aps) if aps else None,
        "runs": fold_records,
    }


def layered_labels(probe, raw_rows, rows, feature_names, seeds, folds):
    eligible_raw = {}
    reasons = Counter()
    for raw in raw_rows:
        key = row_key(raw)
        actionable, reason = raw_actionability(raw)
        eligible_raw[key] = (actionable, reason)

    actionability = []
    reason_counts = Counter()
    for row in rows:
        value, reason = eligible_raw[(row.group, row.event_type, row.action)]
        actionability.append(value)
        reason_counts[reason] += 1
    actionability_array = np.asarray(actionability, dtype=np.int64)

    core_names = sorted(CORE_FEATURES & set(feature_names))
    core_rows = project_rows(rows, feature_names, core_names)
    actionable_indices = np.flatnonzero(actionability_array)
    actionable_rows = [core_rows[index] for index in actionable_indices]

    fixed_all = np.asarray([row.fixed for row in core_rows], dtype=np.int64)
    damaged_all = np.asarray([row.damaged for row in core_rows], dtype=np.int64)
    fixed_actionable = fixed_all[actionable_indices]
    damaged_actionable = damaged_all[actionable_indices]

    strength_groups: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for row in rows:
        strength_groups[(row.group, row.event_step)].append(row)
    strength_sensitive = 0
    strength_total = 0
    unique_best = Counter()
    for candidates in strength_groups.values():
        if len(candidates) < 2:
            continue
        utilities = {candidate.action: candidate.utility for candidate in candidates}
        if len(set(utilities.values())) > 1:
            strength_sensitive += 1
        strength_total += 1
        best_value = max(utilities.values())
        winners = [action for action, utility in utilities.items() if utility == best_value]
        unique_best[winners[0] if len(winners) == 1 else "tie"] += 1

    return {
        "core_features": core_names,
        "label_counts": {
            "all_rows": len(rows),
            "actionable": int(actionability_array.sum()),
            "non_actionable": int(len(rows) - actionability_array.sum()),
            "actionable_rate": float(actionability_array.mean()),
            "actionability_reasons": dict(reason_counts),
            "actionable_fixed": int(fixed_actionable.sum()),
            "actionable_damaged": int(damaged_actionable.sum()),
            "actionable_neutral": int(
                len(actionable_indices)
                - fixed_actionable.sum()
                - damaged_actionable.sum()
            ),
        },
        "actionability_probe": binary_grouped_cv(
            probe, core_rows, actionability_array, seeds, folds
        ),
        "conditional_fix_probe": binary_grouped_cv(
            probe, actionable_rows, fixed_actionable, seeds, folds
        ),
        "conditional_damage_probe": binary_grouped_cv(
            probe, actionable_rows, damaged_actionable, seeds, folds
        ),
        "strength_heterogeneity": {
            "sample_checkpoints_with_multiple_strengths": strength_total,
            "strength_sensitive": strength_sensitive,
            "strength_sensitive_rate": (
                strength_sensitive / strength_total if strength_total else None
            ),
            "unique_best_or_tie": dict(unique_best),
        },
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Probe Data Sufficiency and Layered-Label Analysis",
        "",
        "## Learning curve",
        "",
        "| Train fraction | Fit samples | Accuracy | Net fixed-damaged | Coverage | Fix AUROC | Damage AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fraction, result in summary["learning_curve"].items():
        agg = result["aggregate"]
        lines.append(
            f"| {fraction} | {result['fit_samples']['mean']:.1f} | "
            f"{agg['accuracy']['mean']:.4f} | "
            f"{agg['net_fixed_minus_damaged']['mean']:.2f} | "
            f"{agg['coverage']['mean']:.4f} | "
            f"{agg['fix_auroc']:.4f} | {agg['damage_auroc']:.4f} |"
        )
    lines += [
        "",
        "## Feature complexity",
        "",
        "| Feature set | Dimensions | Accuracy | Net | Fix AUROC | Damage AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["feature_ablation"].items():
        agg = result["aggregate"]
        lines.append(
            f"| {name} | {len(result['features'])} | "
            f"{agg['accuracy']['mean']:.4f} | "
            f"{agg['net_fixed_minus_damaged']['mean']:.2f} | "
            f"{agg['fix_auroc']:.4f} | {agg['damage_auroc']:.4f} |"
        )
    labels = summary["layered_labels"]
    counts = labels["label_counts"]
    lines += [
        "",
        "## Layered labels",
        "",
        f"- Actionable rows: {counts['actionable']}/{counts['all_rows']} "
        f"({counts['actionable_rate']:.2%})",
        f"- Within actionable: fixed={counts['actionable_fixed']}, "
        f"damaged={counts['actionable_damaged']}, neutral={counts['actionable_neutral']}",
        "",
        "| Target | AUROC | AUPRC |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("Actionability", "actionability_probe"),
        ("Fix conditional on actionable", "conditional_fix_probe"),
        ("Damage conditional on actionable", "conditional_damage_probe"),
    ]:
        result = labels[key]
        lines.append(
            f"| {label} | {result['auroc']['mean']:.4f} | "
            f"{result['auprc']['mean']:.4f} |"
        )
    strength = labels["strength_heterogeneity"]
    lines += [
        "",
        "## Strength heterogeneity",
        "",
        f"- Comparable sample-checkpoints: "
        f"{strength['sample_checkpoints_with_multiple_strengths']}",
        f"- Strength-sensitive checkpoints: {strength['strength_sensitive']} "
        f"({strength['strength_sensitive_rate']:.2%})",
        "",
        "## Interpretation",
        "",
        "- A rising learning curve supports collecting more independent samples.",
        "- Core16 matching Full40 supports simplifying the deployable probe.",
        "- High actionability AUROC but weaker conditional utility AUROC means the current target conflates trajectory change with benefit.",
        "- These are grouped development estimates, not final frozen external validation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 42])
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--rho", type=float, default=1.5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.trainer_dir))
    import train_intervention_utility_probe as probe

    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    raw_rows = probe.read_jsonl(args.atlas)
    rows, feature_names, dataset_stats = probe.prepare_rows(raw_rows)

    summary = {
        "dataset": dataset_stats,
        "learning_curve": learning_curves(
            probe, rows, args.fractions, args.seeds, args.folds, args.rho
        ),
        "feature_ablation": feature_ablation(
            probe, rows, feature_names, args.seeds, args.folds, args.rho
        ),
        "layered_labels": layered_labels(
            probe, raw_rows, rows, feature_names, args.seeds, args.folds
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "probe_data_label_analysis.json", summary)
    (args.output_dir / "probe_data_label_analysis.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

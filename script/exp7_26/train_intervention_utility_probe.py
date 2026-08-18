#!/usr/bin/env python3
"""Train a leakage-safe intervention utility probe from matched Atlas outcomes.

The script intentionally depends only on numpy and torch. Samples, rather than
event rows, are the unit of every split so checkpoints from one question never
cross train/test boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


FIXED_EVENTS = {"fixed_1", "fixed_2", "fixed_4", "fixed_8", "fixed_16", "fixed_32"}
ACTION_LAMBDA = {
    "contracted_soft_l075": 0.75,
    "contracted_soft_l090": 0.90,
    "contracted_soft_l095": 0.95,
    "pure_soft": 1.0,
    "pure_soft_l100": 1.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_training_rows(path: Path, rows: list["ProbeRow"], feature_names: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = {
                "original_id": row.group,
                "dataset": row.dataset,
                "event_type": row.event_type,
                "event_step": row.event_step,
                "action": row.action,
                "fixed": row.fixed,
                "damaged": row.damaged,
                "utility": row.utility,
                "base_correct": row.base_correct,
                "treatment_correct": row.treatment_correct,
                "features": {
                    name: float(value)
                    for name, value in zip(feature_names, row.x.tolist())
                },
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    online = row.get("online_pre_intervention_features") or {}
    sample = row.get("sample_features") or {}
    action = row["treatment"]
    steps_since_newline = online.get("steps_since_newline")
    steps_since_answer = online.get("steps_since_answer_marker")
    raw_entropy = max(0.0, scalar(online.get("raw_entropy")))
    filtered_entropy = max(0.0, scalar(online.get("filtered_entropy")))
    return {
        "log1p_raw_entropy": math.log1p(raw_entropy),
        "log1p_filtered_entropy": math.log1p(filtered_entropy),
        "selected_prob": scalar(online.get("selected_prob")),
        "raw_top1_prob": scalar(online.get("raw_top1_prob")),
        "raw_margin": scalar(online.get("raw_margin")),
        "raw_top2_mass": scalar(online.get("raw_top2_mass")),
        "raw_top5_mass": scalar(online.get("raw_top5_mass")),
        "hard_embedding_norm": scalar(online.get("hard_embedding_norm")),
        "soft_embedding_norm": scalar(online.get("soft_embedding_norm")),
        "soft_hard_l2": scalar(online.get("soft_hard_l2")),
        "soft_hard_relative_l2": scalar(online.get("soft_hard_relative_l2")),
        "soft_hard_cosine": scalar(online.get("soft_hard_cosine")),
        "entropy_delta_1": scalar(online.get("entropy_delta_1")),
        "entropy_mean_4": scalar(online.get("entropy_mean_4")),
        "entropy_std_4": scalar(online.get("entropy_std_4")),
        "entropy_delta_from_mean_4": scalar(online.get("entropy_delta_from_mean_4")),
        "entropy_mean_8": scalar(online.get("entropy_mean_8")),
        "entropy_std_8": scalar(online.get("entropy_std_8")),
        "entropy_delta_from_mean_8": scalar(online.get("entropy_delta_from_mean_8")),
        "entropy_mean_16": scalar(online.get("entropy_mean_16")),
        "entropy_std_16": scalar(online.get("entropy_std_16")),
        "entropy_delta_from_mean_16": scalar(online.get("entropy_delta_from_mean_16")),
        "recent16_duplicate_ratio": scalar(online.get("recent16_duplicate_ratio")),
        "step_index": scalar(online.get("step_index"), scalar(row.get("event_step"))),
        "log1p_step_index": math.log1p(max(0.0, scalar(online.get("step_index"), scalar(row.get("event_step"))))),
        "prefix_length": scalar(online.get("prefix_length")),
        "normalized_position_max1024": scalar(online.get("normalized_position_max1024")),
        "token_is_newline": scalar(online.get("token_is_newline")),
        "token_is_whitespace": scalar(online.get("token_is_whitespace")),
        "token_is_punctuation": scalar(online.get("token_is_punctuation")),
        "token_is_answer_marker": scalar(online.get("token_is_answer_marker")),
        "steps_since_newline": scalar(steps_since_newline),
        "steps_since_newline_missing": float(steps_since_newline is None),
        "steps_since_answer_marker": scalar(steps_since_answer),
        "steps_since_answer_marker_missing": float(steps_since_answer is None),
        "question_char_count": scalar(sample.get("question_char_count")),
        "question_word_count": scalar(sample.get("question_word_count")),
        "option_count": scalar(sample.get("option_count")),
        "action_lambda": ACTION_LAMBDA[action],
        "action_is_pure": float(action.startswith("pure_soft")),
    }


@dataclass
class ProbeRow:
    group: str
    dataset: str
    event_type: str
    event_step: int
    action: str
    x: np.ndarray
    fixed: int
    damaged: int
    utility: int
    base_correct: int
    treatment_correct: int


def prepare_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[ProbeRow], list[str], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    exclusions = Counter()
    for row in raw_rows:
        if row.get("event_type") not in FIXED_EVENTS:
            exclusions["non_deployable_event"] += 1
            continue
        if row.get("treatment") not in ACTION_LAMBDA:
            exclusions["unsupported_action"] += 1
            continue
        if not row.get("prefix_match", False):
            exclusions["prefix_mismatch"] += 1
            continue
        if row.get("base_runtime_error") or row.get("treatment_runtime_error"):
            exclusions["runtime_error"] += 1
            continue
        if row.get("base_failed_extraction") or row.get("treatment_failed_extraction"):
            exclusions["failed_extraction"] += 1
            continue
        accepted.append(row)

    if not accepted:
        raise RuntimeError("No eligible Atlas rows found")

    names = sorted(feature_dict(accepted[0]))
    rows: list[ProbeRow] = []
    for row in accepted:
        features = feature_dict(row)
        rows.append(
            ProbeRow(
                group=str(row["original_id"]),
                dataset=str(row["dataset"]),
                event_type=str(row["event_type"]),
                event_step=int(row["event_step"]),
                action=str(row["treatment"]),
                x=np.asarray([features[name] for name in names], dtype=np.float32),
                fixed=int(bool(row["fixed"])),
                damaged=int(bool(row["damaged"])),
                utility=int(row["utility_acc"]),
                base_correct=int(bool(row["base_correct"])),
                treatment_correct=int(bool(row["treatment_correct"])),
            )
        )

    groups = sorted({row.group for row in rows})
    by_dataset = Counter(row.dataset for row in rows)
    label_counts = Counter(row.utility for row in rows)
    stats = {
        "raw_rows": len(raw_rows),
        "eligible_rows": len(rows),
        "independent_samples": len(groups),
        "feature_count": len(names),
        "rows_by_dataset": dict(sorted(by_dataset.items())),
        "utility_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "fixed_count": sum(row.fixed for row in rows),
        "damaged_count": sum(row.damaged for row in rows),
        "exclusions": dict(exclusions),
        "allowed_events": sorted(FIXED_EVENTS),
        "allowed_actions": sorted(ACTION_LAMBDA),
    }
    return rows, names, stats


def grouped_folds(rows: list[ProbeRow], n_splits: int, seed: int) -> list[set[str]]:
    rng = random.Random(seed)
    dataset_groups: dict[str, list[str]] = defaultdict(list)
    group_dataset: dict[str, str] = {}
    for row in rows:
        group_dataset[row.group] = row.dataset
    for group, dataset in group_dataset.items():
        dataset_groups[dataset].append(group)
    folds = [set() for _ in range(n_splits)]
    for groups in dataset_groups.values():
        rng.shuffle(groups)
        for index, group in enumerate(groups):
            folds[index % n_splits].add(group)
    return folds


def calibration_split(rows: list[ProbeRow], train_groups: set[str], seed: int, fraction: float = 0.2) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row.group in train_groups and row.group not in seen:
            seen.add(row.group)
            by_dataset[row.dataset].append(row.group)
    calibration: set[str] = set()
    for groups in by_dataset.values():
        rng.shuffle(groups)
        count = max(1, round(len(groups) * fraction))
        calibration.update(groups[:count])
    return train_groups - calibration, calibration


class BinaryProbe(nn.Module):
    def __init__(self, width: int, kind: str):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(width, 1)
        elif kind == "mlp":
            hidden = min(48, max(16, width))
            self.net = nn.Sequential(
                nn.Linear(width, hidden),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden, 1),
            )
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class FittedHeads:
    fix: BinaryProbe
    damage: BinaryProbe
    mean: np.ndarray
    std: np.ndarray


def train_head(x: np.ndarray, y: np.ndarray, kind: str, seed: int) -> BinaryProbe:
    torch.manual_seed(seed)
    model = BinaryProbe(x.shape[1], kind)
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    positives = max(1.0, float(tensor_y.sum().item()))
    negatives = max(1.0, float(len(y) - positives))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02 if kind == "linear" else 0.005, weight_decay=1e-3)
    epochs = 350 if kind == "linear" else 500
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(tensor_x), tensor_y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def fit_heads(rows: list[ProbeRow], groups: set[str], kind: str, seed: int) -> FittedHeads:
    selected = [row for row in rows if row.group in groups]
    x = np.stack([row.x for row in selected])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    fix = train_head(z, np.asarray([row.fixed for row in selected], dtype=np.float32), kind, seed)
    damage = train_head(z, np.asarray([row.damaged for row in selected], dtype=np.float32), kind, seed + 1000)
    return FittedHeads(fix=fix, damage=damage, mean=mean, std=std)


def predict_scores(model: FittedHeads, rows: list[ProbeRow], rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([row.x for row in rows])
    z = torch.tensor((x - model.mean) / model.std, dtype=torch.float32)
    with torch.no_grad():
        # list -> ndarray avoids a Torch/NumPy ABI edge case in the remote env.
        p_fix = np.asarray(torch.sigmoid(model.fix(z)).tolist(), dtype=np.float64)
        p_damage = np.asarray(torch.sigmoid(model.damage(z)).tolist(), dtype=np.float64)
    return p_fix, p_damage, p_fix - rho * p_damage


def auc_score(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[y.astype(bool)].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float((precision * sorted_y).sum() / positives)


def event_metrics(rows: list[ProbeRow], p_fix: np.ndarray, p_damage: np.ndarray) -> dict[str, Any]:
    fix = np.asarray([row.fixed for row in rows])
    damage = np.asarray([row.damaged for row in rows])
    return {
        "fix_auroc": auc_score(fix, p_fix),
        "fix_auprc": average_precision(fix, p_fix),
        "fix_brier": float(np.mean((p_fix - fix) ** 2)),
        "damage_auroc": auc_score(damage, p_damage),
        "damage_auprc": average_precision(damage, p_damage),
        "damage_brier": float(np.mean((p_damage - damage) ** 2)),
    }


def simulate_policy(rows: list[ProbeRow], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    indexed: dict[str, list[tuple[ProbeRow, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        indexed[row.group].append((row, float(score)))

    correct = fixed = damaged = interventions = 0
    action_counts: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    per_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for group_rows in indexed.values():
        base_correct = group_rows[0][0].base_correct
        dataset = group_rows[0][0].dataset
        chosen: ProbeRow | None = None
        for step in sorted({row.event_step for row, _ in group_rows}):
            at_step = [(row, score) for row, score in group_rows if row.event_step == step]
            candidate, candidate_score = max(at_step, key=lambda item: item[1])
            if candidate_score > threshold:
                chosen = candidate
                break
        final_correct = chosen.treatment_correct if chosen is not None else base_correct
        correct += final_correct
        fixed += int(not base_correct and final_correct)
        damaged += int(base_correct and not final_correct)
        per_dataset[dataset]["n"] += 1
        per_dataset[dataset]["correct"] += final_correct
        per_dataset[dataset]["fixed"] += int(not base_correct and final_correct)
        per_dataset[dataset]["damaged"] += int(base_correct and not final_correct)
        if chosen is not None:
            interventions += 1
            action_counts[chosen.action] += 1
            step_counts[str(chosen.event_step)] += 1

    total = len(indexed)
    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "fixed": fixed,
        "damaged": damaged,
        "net_fixed_minus_damaged": fixed - damaged,
        "interventions": interventions,
        "coverage": interventions / total if total else None,
        "actions": dict(action_counts),
        "steps": dict(step_counts),
        "by_dataset": {
            dataset: {
                **dict(counts),
                "accuracy": counts["correct"] / counts["n"],
            }
            for dataset, counts in sorted(per_dataset.items())
        },
    }


def choose_threshold(rows: list[ProbeRow], scores: np.ndarray, rho: float) -> tuple[float, dict[str, Any]]:
    sorted_scores = np.sort(np.asarray(scores, dtype=np.float64))
    quantile_indices = np.rint(np.linspace(0, len(sorted_scores) - 1, 81)).astype(np.int64)
    candidates = sorted(
        set(sorted_scores[quantile_indices].tolist() + [float(sorted_scores[-1] + 1e-6)])
    )
    best_threshold = candidates[-1]
    best_result = simulate_policy(rows, scores, best_threshold)
    best_objective = 0.0
    for threshold in candidates:
        result = simulate_policy(rows, scores, threshold)
        objective = result["fixed"] - rho * result["damaged"]
        tie_break = (objective, result["net_fixed_minus_damaged"], -result["interventions"], threshold)
        best_tie_break = (
            best_objective,
            best_result["net_fixed_minus_damaged"],
            -best_result["interventions"],
            best_threshold,
        )
        if tie_break > best_tie_break:
            best_objective = objective
            best_threshold = threshold
            best_result = result
    best_result["selection_objective"] = best_objective
    return best_threshold, best_result


def hard_baseline(rows: list[ProbeRow]) -> dict[str, Any]:
    unique: dict[str, ProbeRow] = {}
    for row in rows:
        unique.setdefault(row.group, row)
    correct = sum(row.base_correct for row in unique.values())
    return {
        "samples": len(unique),
        "correct": correct,
        "accuracy": correct / len(unique),
    }


def run_split(
    rows: list[ProbeRow],
    train_groups: set[str],
    test_groups: set[str],
    kind: str,
    seed: int,
    rho: float,
) -> dict[str, Any]:
    fit_groups, calibration_groups = calibration_split(rows, train_groups, seed)
    model = fit_heads(rows, fit_groups, kind, seed)
    calibration_rows = [row for row in rows if row.group in calibration_groups]
    test_rows = [row for row in rows if row.group in test_groups]
    calibration_predictions = predict_scores(model, calibration_rows, rho)
    threshold, calibration_policy = choose_threshold(calibration_rows, calibration_predictions[2], rho)
    test_predictions = predict_scores(model, test_rows, rho)
    return {
        "seed": seed,
        "model": kind,
        "fit_samples": len(fit_groups),
        "calibration_samples": len(calibration_groups),
        "test_samples": len(test_groups),
        "threshold": threshold,
        "hard_baseline": hard_baseline(test_rows),
        "calibration_policy": calibration_policy,
        "event_metrics": event_metrics(test_rows, test_predictions[0], test_predictions[1]),
        "test_policy": simulate_policy(test_rows, test_predictions[2], threshold),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["accuracy", "coverage", "net_fixed_minus_damaged", "fixed", "damaged"]
    aggregate: dict[str, Any] = {"runs": len(runs)}
    for key in keys:
        values = [float(run["test_policy"][key]) for run in runs]
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    metrics = ["fix_auroc", "fix_auprc", "damage_auroc", "damage_auprc"]
    for key in metrics:
        values = [run["event_metrics"][key] for run in runs if run["event_metrics"][key] is not None]
        aggregate[key] = float(np.mean(values)) if values else None
    return aggregate


def train_final_artifact(
    rows: list[ProbeRow],
    feature_names: list[str],
    kind: str,
    seed: int,
    rho: float,
    output_dir: Path,
) -> dict[str, Any]:
    all_groups = {row.group for row in rows}
    fit_groups, calibration_groups = calibration_split(rows, all_groups, seed)
    calibration_model = fit_heads(rows, fit_groups, kind, seed)
    calibration_rows = [row for row in rows if row.group in calibration_groups]
    predictions = predict_scores(calibration_model, calibration_rows, rho)
    threshold, calibration_policy = choose_threshold(calibration_rows, predictions[2], rho)
    final_model = fit_heads(rows, all_groups, kind, seed)
    artifact_path = output_dir / f"utility_probe_{kind}.pt"
    torch.save(
        {
            "fix_state_dict": final_model.fix.state_dict(),
            "damage_state_dict": final_model.damage.state_dict(),
            "mean": final_model.mean,
            "std": final_model.std,
            "feature_names": feature_names,
            "model_kind": kind,
            "rho": rho,
            "threshold": threshold,
            "actions": ACTION_LAMBDA,
            "events": sorted(FIXED_EVENTS),
            "seed": seed,
        },
        artifact_path,
    )
    return {
        "artifact": str(artifact_path),
        "threshold": threshold,
        "calibration_policy": calibration_policy,
        "training_samples": len(all_groups),
    }


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Intervention Utility Probe V1",
        "",
        "## Data contract",
        "",
        f"- Independent samples: {summary['dataset']['independent_samples']}",
        f"- Eligible event-action rows: {summary['dataset']['eligible_rows']}",
        f"- Features: {summary['dataset']['feature_count']}",
        f"- Fixed labels: {summary['dataset']['fixed_count']}",
        f"- Damaged labels: {summary['dataset']['damaged_count']}",
        "- Split unit: original sample ID; checkpoints from one sample never cross splits.",
        "- Training events: fixed 1/2/4/8/16/32 only.",
        "- Excluded from deployable training: entropy_top1 and random_control.",
        "",
        "## Grouped cross-validation",
        "",
        "| Model | Accuracy | Coverage | Net fixed-damaged | Fix AUROC | Damage AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind, result in summary["grouped_cv"]["aggregate"].items():
        lines.append(
            f"| {kind} | {result['accuracy']['mean']:.4f} | "
            f"{result['coverage']['mean']:.4f} | {result['net_fixed_minus_damaged']['mean']:.2f} | "
            f"{result['fix_auroc'] if result['fix_auroc'] is not None else 'NA'} | "
            f"{result['damage_auroc'] if result['damage_auroc'] is not None else 'NA'} |"
        )
    lines += [
        "",
        "## Leave-one-dataset-out",
        "",
        "| Held-out dataset | Model | Hard acc | Probe acc | Fixed | Damaged | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for held_out, models in summary["leave_one_dataset_out"].items():
        for kind, runs in models.items():
            for run in runs:
                hard = run["hard_baseline"]
                policy = run["test_policy"]
                lines.append(
                    f"| {held_out} | {kind} | {hard['accuracy']:.4f} | {policy['accuracy']:.4f} | "
                    f"{policy['fixed']} | {policy['damaged']} | {policy['coverage']:.4f} |"
                )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- This is a counterfactual utility estimator, not a correctness oracle.",
        "- Thresholds are selected only on grouped calibration samples.",
        "- The default action is hard decoding; intervention requires positive conservative utility.",
        "- Results from the same 256-sample Atlas remain development evidence, not a final held-out claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rho", type=float, default=1.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 42])
    parser.add_argument("--models", nargs="+", choices=["linear", "mlp"], default=["linear", "mlp"])
    args = parser.parse_args()

    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(args.atlas)
    rows, feature_names, dataset_stats = prepare_rows(raw_rows)
    write_json(args.output_dir / "probe_dataset_manifest.json", {
        **dataset_stats,
        "feature_names": feature_names,
        "source": str(args.atlas),
    })
    write_training_rows(
        args.output_dir / "utility_probe_training_rows.jsonl",
        rows,
        feature_names,
    )

    grouped_results: dict[str, list[dict[str, Any]]] = {kind: [] for kind in args.models}
    all_groups = {row.group for row in rows}
    for seed in args.seeds:
        folds = grouped_folds(rows, args.folds, seed)
        for fold_index, test_groups in enumerate(folds):
            train_groups = all_groups - test_groups
            for kind in args.models:
                result = run_split(rows, train_groups, test_groups, kind, seed + fold_index * 100, args.rho)
                result["fold"] = fold_index
                grouped_results[kind].append(result)

    datasets = sorted({row.dataset for row in rows})
    lodo: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset in datasets:
        test_groups = {row.group for row in rows if row.dataset == dataset}
        train_groups = all_groups - test_groups
        lodo[dataset] = {kind: [] for kind in args.models}
        for seed in args.seeds:
            for kind in args.models:
                lodo[dataset][kind].append(run_split(rows, train_groups, test_groups, kind, seed, args.rho))

    artifacts = {
        kind: train_final_artifact(rows, feature_names, kind, 42, args.rho, args.output_dir)
        for kind in args.models
    }
    summary = {
        "probe_version": "v1",
        "rho_damage": args.rho,
        "dataset": dataset_stats,
        "grouped_cv": {
            "runs": grouped_results,
            "aggregate": {kind: aggregate_runs(runs) for kind, runs in grouped_results.items()},
        },
        "leave_one_dataset_out": lodo,
        "final_artifacts": artifacts,
        "scientific_scope": {
            "default_action": "hard",
            "training_unit": "sample_group",
            "future_information_used": False,
            "gold_used_at_inference": False,
            "development_only": True,
        },
    }
    write_json(args.output_dir / "utility_probe_results.json", summary)
    (args.output_dir / "utility_probe_results.md").write_text(report_markdown(summary), encoding="utf-8")
    (args.output_dir / "PROBE_TRAINING_COMPLETE").write_text("ok\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "samples": dataset_stats["independent_samples"],
        "rows": dataset_stats["eligible_rows"],
        "models": args.models,
    }, indent=2))


if __name__ == "__main__":
    main()

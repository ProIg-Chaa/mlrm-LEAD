#!/usr/bin/env python3
"""Train and evaluate a hierarchical latent-intervention utility probe.

Stage A predicts whether an intervention changes the generated trajectory.
Stage U predicts fixed/damaged risk conditional on an actionable intervention.
At each checkpoint, the policy chooses the strength with the highest
actionability-weighted conservative utility, or abstains to hard decoding.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ALLOWED_ACTIONS = {
    "contracted_soft_l090": 0.90,
    "contracted_soft_l095": 0.95,
    "pure_soft_l100": 1.00,
    "pure_soft": 1.00,
}
ALLOWED_EVENTS = {"fixed_1", "fixed_2", "fixed_4", "fixed_8", "fixed_16", "fixed_32"}
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    online = row.get("online_pre_intervention_features") or {}
    sample = row.get("sample_features") or {}
    action = str(row["treatment"])
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
        "log1p_step_index": math.log1p(
            max(0.0, scalar(online.get("step_index"), scalar(row.get("event_step"))))
        ),
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
        "action_lambda": ALLOWED_ACTIONS[action],
        "action_is_pure": float(action.startswith("pure_soft")),
    }


@dataclass
class V2Row:
    group: str
    dataset: str
    event_step: int
    event_type: str
    action: str
    x_action: np.ndarray
    x_utility: np.ndarray
    actionable: int
    answer_changed: int
    fixed: int
    damaged: int
    base_correct: int
    treatment_correct: int


def is_actionable(row: dict[str, Any]) -> int:
    mismatch = max(
        scalar(row.get("mismatch_ratio_h8")),
        scalar(row.get("mismatch_ratio_h16")),
        scalar(row.get("mismatch_ratio_h32")),
    )
    return int(
        bool(row.get("answer_changed"))
        or row.get("first_divergence_step") is not None
        or mismatch > 0
    )


def prepare_rows(
    raw_rows: list[dict[str, Any]],
) -> tuple[list[V2Row], list[str], list[str], dict[str, Any]]:
    accepted = []
    excluded = Counter()
    for row in raw_rows:
        if row.get("event_type") not in ALLOWED_EVENTS:
            excluded["event"] += 1
            continue
        if row.get("treatment") not in ALLOWED_ACTIONS:
            excluded["action"] += 1
            continue
        if not row.get("prefix_match", False):
            excluded["prefix_mismatch"] += 1
            continue
        if row.get("base_runtime_error") or row.get("treatment_runtime_error"):
            excluded["runtime_error"] += 1
            continue
        if row.get("base_failed_extraction") or row.get("treatment_failed_extraction"):
            excluded["failed_extraction"] += 1
            continue
        accepted.append(row)
    if not accepted:
        raise RuntimeError("No eligible rows")

    all_names = sorted(feature_dict(accepted[0]))
    action_names = sorted(CORE_FEATURES & set(all_names))
    utility_names = all_names
    rows = []
    for row in accepted:
        features = feature_dict(row)
        rows.append(
            V2Row(
                group=str(row["original_id"]),
                dataset=str(row["dataset"]),
                event_step=int(row["event_step"]),
                event_type=str(row["event_type"]),
                action=str(row["treatment"]),
                x_action=np.asarray(
                    [features[name] for name in action_names], dtype=np.float32
                ),
                x_utility=np.asarray(
                    [features[name] for name in utility_names], dtype=np.float32
                ),
                actionable=is_actionable(row),
                answer_changed=int(bool(row.get("answer_changed"))),
                fixed=int(bool(row.get("fixed"))),
                damaged=int(bool(row.get("damaged"))),
                base_correct=int(bool(row.get("base_correct"))),
                treatment_correct=int(bool(row.get("treatment_correct"))),
            )
        )
    stats = {
        "raw_rows": len(raw_rows),
        "eligible_rows": len(rows),
        "independent_samples": len({row.group for row in rows}),
        "datasets": dict(Counter(row.dataset for row in rows)),
        "actionable": sum(row.actionable for row in rows),
        "answer_changed": sum(row.answer_changed for row in rows),
        "fixed": sum(row.fixed for row in rows),
        "damaged": sum(row.damaged for row in rows),
        "excluded": dict(excluded),
    }
    return rows, action_names, utility_names, stats


class BinaryProbe(torch.nn.Module):
    def __init__(self, width: int, kind: str):
        super().__init__()
        if kind == "linear":
            self.net = torch.nn.Linear(width, 1)
        else:
            hidden = min(48, max(16, width))
            self.net = torch.nn.Sequential(
                torch.nn.Linear(width, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_binary(x: np.ndarray, y: np.ndarray, kind: str, seed: int) -> BinaryProbe:
    torch.manual_seed(seed)
    model = BinaryProbe(x.shape[1], kind)
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    positives = max(1.0, float(tensor_y.sum()))
    negatives = max(1.0, float(len(tensor_y) - tensor_y.sum()))
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.02 if kind == "linear" else 0.005,
        weight_decay=1e-3,
    )
    epochs = 350 if kind == "linear" else 500
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(tensor_x), tensor_y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@dataclass
class FittedV2:
    action: BinaryProbe
    fix: BinaryProbe
    damage: BinaryProbe
    action_mean: np.ndarray
    action_std: np.ndarray
    utility_mean: np.ndarray
    utility_std: np.ndarray


def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def fit_model(rows: list[V2Row], groups: set[str], kind: str, seed: int) -> FittedV2:
    selected = [row for row in rows if row.group in groups]
    action_x, action_mean, action_std = normalize(
        np.stack([row.x_action for row in selected])
    )
    action = train_binary(
        action_x,
        np.asarray([row.actionable for row in selected], dtype=np.float32),
        kind,
        seed,
    )
    actionable = [row for row in selected if row.actionable]
    if not actionable:
        raise RuntimeError("No actionable training rows")
    utility_x, utility_mean, utility_std = normalize(
        np.stack([row.x_utility for row in actionable])
    )
    fix = train_binary(
        utility_x,
        np.asarray([row.fixed for row in actionable], dtype=np.float32),
        kind,
        seed + 1000,
    )
    damage = train_binary(
        utility_x,
        np.asarray([row.damaged for row in actionable], dtype=np.float32),
        kind,
        seed + 2000,
    )
    return FittedV2(
        action=action,
        fix=fix,
        damage=damage,
        action_mean=action_mean,
        action_std=action_std,
        utility_mean=utility_mean,
        utility_std=utility_std,
    )


def predict(model: FittedV2, rows: list[V2Row], rho: float):
    action_x = torch.tensor(
        (np.stack([row.x_action for row in rows]) - model.action_mean)
        / model.action_std,
        dtype=torch.float32,
    )
    utility_x = torch.tensor(
        (np.stack([row.x_utility for row in rows]) - model.utility_mean)
        / model.utility_std,
        dtype=torch.float32,
    )
    with torch.no_grad():
        q = np.asarray(torch.sigmoid(model.action(action_x)).tolist())
        p_fix = np.asarray(torch.sigmoid(model.fix(utility_x)).tolist())
        p_damage = np.asarray(torch.sigmoid(model.damage(utility_x)).tolist())
    utility = q * (p_fix - rho * p_damage)
    return q, p_fix, p_damage, utility


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
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
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return float(
        (ranks[y.astype(bool)].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float((precision * sorted_y).sum() / positives)


def event_metrics(rows: list[V2Row], predictions) -> dict[str, Any]:
    q, p_fix, p_damage, _ = predictions
    actionable = np.asarray([row.actionable for row in rows])
    actionable_indices = np.flatnonzero(actionable)
    fixed = np.asarray([row.fixed for row in rows])
    damaged = np.asarray([row.damaged for row in rows])
    return {
        "actionability_auroc": auc(actionable, q),
        "actionability_auprc": average_precision(actionable, q),
        "conditional_fix_auroc": auc(fixed[actionable_indices], p_fix[actionable_indices]),
        "conditional_fix_auprc": average_precision(
            fixed[actionable_indices], p_fix[actionable_indices]
        ),
        "conditional_damage_auroc": auc(
            damaged[actionable_indices], p_damage[actionable_indices]
        ),
        "conditional_damage_auprc": average_precision(
            damaged[actionable_indices], p_damage[actionable_indices]
        ),
    }


def grouped_folds(rows: list[V2Row], n_splits: int, seed: int) -> list[set[str]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    group_dataset = {}
    for row in rows:
        group_dataset[row.group] = row.dataset
    for group, dataset in group_dataset.items():
        by_dataset[dataset].append(group)
    folds = [set() for _ in range(n_splits)]
    for groups in by_dataset.values():
        rng.shuffle(groups)
        for index, group in enumerate(groups):
            folds[index % n_splits].add(group)
    return folds


def calibration_split(
    rows: list[V2Row], groups: set[str], seed: int, fraction: float = 0.2
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row.group in groups and row.group not in seen:
            seen.add(row.group)
            by_dataset[row.dataset].append(row.group)
    calibration = set()
    for dataset_groups in by_dataset.values():
        rng.shuffle(dataset_groups)
        count = max(1, round(len(dataset_groups) * fraction))
        calibration.update(dataset_groups[:count])
    return groups - calibration, calibration


def hard_baseline(rows: list[V2Row]) -> dict[str, Any]:
    samples = {}
    for row in rows:
        samples.setdefault(row.group, row.base_correct)
    correct = sum(samples.values())
    return {
        "samples": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples),
    }


def simulate(
    rows: list[V2Row],
    predictions,
    action_threshold: float,
    utility_threshold: float,
) -> dict[str, Any]:
    q, _, _, utility = predictions
    grouped: dict[str, list[tuple[V2Row, float, float]]] = defaultdict(list)
    for row, q_value, u_value in zip(rows, q, utility):
        grouped[row.group].append((row, float(q_value), float(u_value)))
    correct = fixed = damaged = interventions = 0
    actions = Counter()
    steps = Counter()
    by_dataset = defaultdict(Counter)
    for candidates in grouped.values():
        base_correct = candidates[0][0].base_correct
        dataset = candidates[0][0].dataset
        chosen = None
        for step in sorted({row.event_step for row, _, _ in candidates}):
            at_step = [
                item
                for item in candidates
                if item[0].event_step == step and item[1] > action_threshold
            ]
            if not at_step:
                continue
            candidate = max(at_step, key=lambda item: item[2])
            if candidate[2] > utility_threshold:
                chosen = candidate[0]
                break
        final_correct = chosen.treatment_correct if chosen else base_correct
        correct += final_correct
        fixed += int(not base_correct and final_correct)
        damaged += int(base_correct and not final_correct)
        interventions += int(chosen is not None)
        by_dataset[dataset]["n"] += 1
        by_dataset[dataset]["correct"] += final_correct
        by_dataset[dataset]["fixed"] += int(not base_correct and final_correct)
        by_dataset[dataset]["damaged"] += int(base_correct and not final_correct)
        if chosen:
            actions[chosen.action] += 1
            steps[str(chosen.event_step)] += 1
    total = len(grouped)
    return {
        "samples": total,
        "accuracy": correct / total,
        "correct": correct,
        "fixed": fixed,
        "damaged": damaged,
        "net": fixed - damaged,
        "interventions": interventions,
        "coverage": interventions / total,
        "actions": dict(actions),
        "steps": dict(steps),
        "by_dataset": {
            dataset: {
                **dict(counts),
                "accuracy": counts["correct"] / counts["n"],
                "net": counts["fixed"] - counts["damaged"],
            }
            for dataset, counts in sorted(by_dataset.items())
        },
    }


def choose_thresholds(
    rows: list[V2Row], predictions, rho: float
) -> tuple[float, float, dict[str, Any]]:
    q, _, _, utility = predictions
    q_values = np.sort(q)
    u_values = np.sort(utility)
    q_indices = np.rint(np.linspace(0, len(q_values) - 1, 17)).astype(int)
    u_indices = np.rint(np.linspace(0, len(u_values) - 1, 33)).astype(int)
    q_candidates = sorted(set(q_values[q_indices].tolist() + [float(q_values[-1] + 1e-6)]))
    u_candidates = sorted(set(u_values[u_indices].tolist() + [float(u_values[-1] + 1e-6)]))
    best = simulate(rows, predictions, q_candidates[-1], u_candidates[-1])
    best_q = q_candidates[-1]
    best_u = u_candidates[-1]
    best_key = (0.0, best["net"], -best["interventions"], best_q, best_u)
    for q_threshold in q_candidates:
        for utility_threshold in u_candidates:
            result = simulate(rows, predictions, q_threshold, utility_threshold)
            objective = result["fixed"] - rho * result["damaged"]
            key = (
                objective,
                result["net"],
                -result["interventions"],
                q_threshold,
                utility_threshold,
            )
            if key > best_key:
                best_key = key
                best = result
                best_q = q_threshold
                best_u = utility_threshold
    best["selection_objective"] = best_key[0]
    return best_q, best_u, best


def run_split(
    rows: list[V2Row],
    train_groups: set[str],
    test_groups: set[str],
    kind: str,
    seed: int,
    rho: float,
) -> dict[str, Any]:
    fit_groups, calibration_groups = calibration_split(rows, train_groups, seed)
    model = fit_model(rows, fit_groups, kind, seed)
    calibration_rows = [row for row in rows if row.group in calibration_groups]
    test_rows = [row for row in rows if row.group in test_groups]
    calibration_predictions = predict(model, calibration_rows, rho)
    action_threshold, utility_threshold, calibration_policy = choose_thresholds(
        calibration_rows, calibration_predictions, rho
    )
    test_predictions = predict(model, test_rows, rho)
    return {
        "seed": seed,
        "kind": kind,
        "fit_samples": len(fit_groups),
        "calibration_samples": len(calibration_groups),
        "test_samples": len(test_groups),
        "action_threshold": action_threshold,
        "utility_threshold": utility_threshold,
        "hard": hard_baseline(test_rows),
        "calibration_policy": calibration_policy,
        "event_metrics": event_metrics(test_rows, test_predictions),
        "policy": simulate(
            test_rows, test_predictions, action_threshold, utility_threshold
        ),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"runs": len(runs)}
    for key in ("accuracy", "coverage", "fixed", "damaged", "net"):
        values = np.asarray([run["policy"][key] for run in runs], dtype=float)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for key in (
        "actionability_auroc",
        "actionability_auprc",
        "conditional_fix_auroc",
        "conditional_fix_auprc",
        "conditional_damage_auroc",
        "conditional_damage_auprc",
    ):
        values = [
            run["event_metrics"][key]
            for run in runs
            if run["event_metrics"][key] is not None
        ]
        output[key] = float(np.mean(values)) if values else None
    return output


def train_final(
    rows: list[V2Row],
    kind: str,
    seed: int,
    rho: float,
    action_names: list[str],
    utility_names: list[str],
    output: Path,
) -> tuple[FittedV2, float, float, dict[str, Any]]:
    groups = {row.group for row in rows}
    fit_groups, calibration_groups = calibration_split(rows, groups, seed)
    calibration_model = fit_model(rows, fit_groups, kind, seed)
    calibration_rows = [row for row in rows if row.group in calibration_groups]
    thresholds = choose_thresholds(
        calibration_rows, predict(calibration_model, calibration_rows, rho), rho
    )
    final_model = fit_model(rows, groups, kind, seed)
    action_threshold, utility_threshold, policy = thresholds
    torch.save(
        {
            "action_state_dict": final_model.action.state_dict(),
            "fix_state_dict": final_model.fix.state_dict(),
            "damage_state_dict": final_model.damage.state_dict(),
            "action_mean": final_model.action_mean,
            "action_std": final_model.action_std,
            "utility_mean": final_model.utility_mean,
            "utility_std": final_model.utility_std,
            "action_features": action_names,
            "utility_features": utility_names,
            "kind": kind,
            "rho": rho,
            "action_threshold": action_threshold,
            "utility_threshold": utility_threshold,
            "actions": ALLOWED_ACTIONS,
            "events": sorted(ALLOWED_EVENTS),
            "seed": seed,
        },
        output,
    )
    return final_model, action_threshold, utility_threshold, policy


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hierarchical Intervention Utility Probe V2",
        "",
        "## Data",
        "",
        f"- Train independent samples: {summary['train_data']['independent_samples']}",
        f"- Train eligible rows: {summary['train_data']['eligible_rows']}",
        f"- Train actionable/fixed/damaged: {summary['train_data']['actionable']}/"
        f"{summary['train_data']['fixed']}/{summary['train_data']['damaged']}",
        "",
        "## Grouped cross-validation",
        "",
        "| Model | Accuracy | Coverage | Net | Action AUROC | Conditional fix AUROC | Conditional damage AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, value in summary["grouped_cv"]["aggregate"].items():
        lines.append(
            f"| {kind} | {value['accuracy']['mean']:.4f} | "
            f"{value['coverage']['mean']:.4f} | {value['net']['mean']:.2f} | "
            f"{value['actionability_auroc']:.4f} | "
            f"{value['conditional_fix_auroc']:.4f} | "
            f"{value['conditional_damage_auroc']:.4f} |"
        )
    if "external" in summary:
        lines += [
            "",
            "## Frozen external evaluation",
            "",
            f"- External independent samples: {summary['external_data']['independent_samples']}",
            "",
            "| Model | Hard acc | Probe acc | Fixed | Damaged | Net | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for kind, value in summary["external"].items():
            lines.append(
                f"| {kind} | {value['hard']['accuracy']:.4f} | "
                f"{value['policy']['accuracy']:.4f} | {value['policy']['fixed']} | "
                f"{value['policy']['damaged']} | {value['policy']['net']} | "
                f"{value['policy']['coverage']:.4f} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Actionability means observable trajectory divergence, not correctness improvement.",
        "- Thresholds are selected only on grouped calibration samples.",
        "- Fresh external results are evaluated before combined retraining.",
        "- The default action remains hard decoding.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-atlas", type=Path, required=True)
    parser.add_argument("--external-atlas", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 42])
    parser.add_argument("--models", nargs="+", choices=["linear", "mlp"], default=["mlp"])
    parser.add_argument("--rho", type=float, default=1.5)
    args = parser.parse_args()

    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, action_names, utility_names, train_stats = prepare_rows(
        read_jsonl(args.train_atlas)
    )
    groups = {row.group for row in train_rows}
    grouped = {kind: [] for kind in args.models}
    for kind in args.models:
        for seed in args.seeds:
            for fold_index, test_groups in enumerate(
                grouped_folds(train_rows, args.folds, seed)
            ):
                grouped[kind].append(
                    run_split(
                        train_rows,
                        groups - test_groups,
                        test_groups,
                        kind,
                        seed + fold_index * 100,
                        args.rho,
                    )
                )

    summary: dict[str, Any] = {
        "train_data": train_stats,
        "action_features": action_names,
        "utility_features": utility_names,
        "rho": args.rho,
        "grouped_cv": {
            "aggregate": {
                kind: aggregate_runs(runs) for kind, runs in grouped.items()
            },
            "runs": grouped,
        },
    }

    final_models = {}
    for kind in args.models:
        artifact = args.output_dir / f"hierarchical_probe_v2_{kind}.pt"
        final_models[kind] = train_final(
            train_rows,
            kind,
            42,
            args.rho,
            action_names,
            utility_names,
            artifact,
        )

    if args.external_atlas:
        external_rows, external_action_names, external_utility_names, external_stats = (
            prepare_rows(read_jsonl(args.external_atlas))
        )
        if external_action_names != action_names or external_utility_names != utility_names:
            raise RuntimeError("External feature schema mismatch")
        external = {}
        for kind, (
            model,
            action_threshold,
            utility_threshold,
            _,
        ) in final_models.items():
            predictions = predict(model, external_rows, args.rho)
            external[kind] = {
                "hard": hard_baseline(external_rows),
                "event_metrics": event_metrics(external_rows, predictions),
                "policy": simulate(
                    external_rows,
                    predictions,
                    action_threshold,
                    utility_threshold,
                ),
                "action_threshold": action_threshold,
                "utility_threshold": utility_threshold,
            }
        summary["external_data"] = external_stats
        summary["external"] = external

    write_json(args.output_dir / "hierarchical_probe_v2_summary.json", summary)
    (args.output_dir / "hierarchical_probe_v2_summary.md").write_text(
        report(summary), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

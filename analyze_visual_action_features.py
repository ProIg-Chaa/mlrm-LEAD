#!/usr/bin/env python3
"""Offline feature audit for the Visual Action Strength Atlas.

This script deliberately separates pre-action features from post-action response
features. All cross-validation splits are grouped by original sample ID.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_visual_action_strength_atlas import extract_prediction


DATASETS = ("vstar", "mmvp", "realworldqa", "visulogic")
BRANCHES = (
    "true_image",
    "true_mask_residual",
    "true_swap_residual",
    "true_dataset_noise_residual",
    "shuffled_mask_residual",
    "random_residual",
    "reverse_mask_residual",
    "reverse_dataset_noise_residual",
)

ALIGNMENT_KEYS = {
    "true_mask_residual": "true_mask_residual_aligned_hard_cosine",
    "true_swap_residual": "true_swap_residual_aligned_hard_cosine",
    "true_dataset_noise_residual": "true_dataset_noise_residual_aligned_hard_cosine",
    "shuffled_mask_residual": "shuffled_mask_residual_aligned_hard_cosine",
    "random_residual": "random_residual_aligned_hard_cosine",
    "reverse_mask_residual": "reverse_mask_residual_aligned_hard_cosine",
    "reverse_dataset_noise_residual": "reverse_dataset_noise_residual_aligned_hard_cosine",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_letter(row: dict) -> str | None:
    raw = str(row.get("gold") or row.get("answer") or "").upper()
    for letter in "ABCD":
        if letter in raw:
            return letter
    return None


def enrich_result(row: dict) -> dict:
    result = dict(row)
    result["pred"] = extract_prediction(row)
    result["gold_letter"] = gold_letter(row)
    result["correct"] = result["pred"] == result["gold_letter"]
    return result


def clamp_cosine(value):
    if value is None:
        return np.nan
    return float(np.clip(float(value), -1.0, 1.0))


def safe_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def first_divergence(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return -1


def changed_within(left: list[int], right: list[int], horizon: int) -> int:
    return int(left[:horizon] != right[:horizon])


def capture_value(metadata: dict, source: str, key: str):
    return safe_float((metadata.get("capture_statistics") or {}).get(source, {}).get(key))


def action_norm(metadata: dict, branch: str) -> float:
    if branch in {"true_mask_residual", "shuffled_mask_residual", "random_residual", "reverse_mask_residual"}:
        return safe_float(metadata.get("residual_norm_before_matching"))
    if branch in {"true_dataset_noise_residual", "reverse_dataset_noise_residual"}:
        return safe_float(metadata.get("dataset_noise_residual_norm_before_matching"))
    return safe_float(metadata.get("target_delta_norm"))


def build_table(root: Path, datasets: list[str], strengths: list[str]) -> pd.DataFrame:
    records = []
    for dataset in datasets:
        hard_rows = {
            str(row["event_id"]): enrich_result(row)
            for row in read_jsonl(root / "hard" / dataset / "results.jsonl")
        }
        for strength in strengths:
            result_path = root / strength / dataset / "results.jsonl"
            metadata_path = root / strength / dataset / "vector_metadata.jsonl"
            if not result_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Missing complete input: {result_path} or {metadata_path}")
            metadata = {str(row["event_id"]): row for row in read_jsonl(metadata_path)}
            lambda_value = int(strength.split("_")[-1]) / 100.0
            for raw in read_jsonl(result_path):
                event_id = str(raw["event_id"])
                if event_id not in hard_rows or event_id not in metadata:
                    continue
                treatment = enrich_result(raw)
                hard = hard_rows[event_id]
                meta = metadata[event_id]
                branch = str(treatment.get("branch"))
                hard_tokens = list(hard.get("generated_token_ids") or [])
                treatment_tokens = list(treatment.get("generated_token_ids") or [])
                divergence = first_divergence(treatment_tokens, hard_tokens)
                norm = action_norm(meta, branch)
                align_key = ALIGNMENT_KEYS.get(branch)
                action_alignment = clamp_cosine(meta.get(align_key)) if align_key else clamp_cosine(meta.get("true_soft_hard_cosine"))
                event_step = int(treatment.get("event_step") or meta.get("event_step") or 0)
                prefix_len = len(treatment.get("prefix_ids") or [])
                divergence_after_action = divergence - prefix_len if divergence >= prefix_len else -1
                remaining_length = max(1, min(len(hard_tokens), len(treatment_tokens)) - prefix_len)
                record = {
                    "dataset": dataset,
                    "strength": strength,
                    "lambda": lambda_value,
                    "branch": branch,
                    "event_id": event_id,
                    "original_id": str(treatment.get("original_id")),
                    "sample_group": f"{dataset}::{treatment.get('original_id')}",
                    "event_type": str(treatment.get("event_type")),
                    "event_step": event_step,
                    "log_event_step": math.log1p(event_step),
                    "prefix_length": prefix_len,
                    "relative_event_position": event_step / max(1, len(hard_tokens)),
                    "true_entropy": safe_float(meta.get("true_entropy")),
                    "true_selected_probability": safe_float(meta.get("true_selected_probability")),
                    "true_soft_hard_cosine": clamp_cosine(meta.get("true_soft_hard_cosine")),
                    "target_delta_norm": safe_float(meta.get("target_delta_norm")),
                    "mask_residual_norm": safe_float(meta.get("residual_norm_before_matching")),
                    "dataset_noise_residual_norm": safe_float(meta.get("dataset_noise_residual_norm_before_matching")),
                    "action_norm": norm,
                    "action_aligned_hard_cosine": action_alignment,
                    "true_mask_entropy_gap": capture_value(meta, "true", "entropy") - capture_value(meta, "mask", "entropy"),
                    "true_swap_entropy_gap": capture_value(meta, "true", "entropy") - capture_value(meta, "swap", "entropy"),
                    "true_noise_entropy_gap": capture_value(meta, "true", "entropy") - capture_value(meta, "dataset_noise", "entropy"),
                    "true_mask_selected_prob_gap": capture_value(meta, "true", "selected_probability") - capture_value(meta, "mask", "selected_probability"),
                    "true_swap_selected_prob_gap": capture_value(meta, "true", "selected_probability") - capture_value(meta, "swap", "selected_probability"),
                    "true_noise_selected_prob_gap": capture_value(meta, "true", "selected_probability") - capture_value(meta, "dataset_noise", "selected_probability"),
                    "hard_correct": int(bool(hard["correct"])),
                    "treatment_correct": int(bool(treatment["correct"])),
                    "answer_change": int(treatment["pred"] != hard["pred"]),
                    "trajectory_change": int(divergence >= 0),
                    "fixed": int((not hard["correct"]) and treatment["correct"]),
                    "damaged": int(hard["correct"] and (not treatment["correct"])),
                    "first_divergence": divergence,
                    "first_divergence_after_action": divergence_after_action,
                    "first_divergence_normalized": (divergence_after_action / remaining_length) if divergence_after_action >= 0 else 1.0,
                    "changed_by_1": changed_within(treatment_tokens[prefix_len:], hard_tokens[prefix_len:], 1),
                    "changed_by_2": changed_within(treatment_tokens[prefix_len:], hard_tokens[prefix_len:], 2),
                    "changed_by_4": changed_within(treatment_tokens[prefix_len:], hard_tokens[prefix_len:], 4),
                    "changed_by_8": changed_within(treatment_tokens[prefix_len:], hard_tokens[prefix_len:], 8),
                    "output_length_delta": len(treatment_tokens) - len(hard_tokens),
                    "runtime_error": int(bool(treatment.get("error_type"))),
                }
                records.append(record)
    return pd.DataFrame.from_records(records)


BASIC_NUMERIC = [
    "lambda", "log_event_step", "relative_event_position",
    "true_entropy", "true_selected_probability",
]
GEOMETRY_NUMERIC = BASIC_NUMERIC + [
    "true_soft_hard_cosine", "target_delta_norm", "action_norm",
    "action_aligned_hard_cosine",
]
VISUAL_NUMERIC = GEOMETRY_NUMERIC + [
    "mask_residual_norm", "dataset_noise_residual_norm",
    "true_mask_entropy_gap", "true_swap_entropy_gap", "true_noise_entropy_gap",
    "true_mask_selected_prob_gap", "true_swap_selected_prob_gap",
    "true_noise_selected_prob_gap",
]
RESPONSE_NUMERIC = VISUAL_NUMERIC + [
    "changed_by_1", "changed_by_2", "changed_by_4", "changed_by_8",
]
CATEGORICAL = ["dataset", "branch", "event_type", "strength"]


def make_model(numeric: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([
        ("features", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])


def score_predictions(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, probability)) if y_true.sum() else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)) if len(np.unique(y_true)) == 2 else None,
    }


def grouped_cv(frame: pd.DataFrame, target: str, numeric: list[str], folds: int = 5) -> dict:
    frame = frame.copy()
    groups = frame["sample_group"].to_numpy()
    y = frame[target].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < folds:
        return {"n": int(len(frame)), "positive": int(y.sum()), "error": "insufficient_classes_or_groups"}
    probabilities = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=folds)
    for train_index, test_index in splitter.split(frame, y, groups):
        if len(np.unique(y[train_index])) < 2:
            continue
        model = make_model(numeric)
        model.fit(frame.iloc[train_index], y[train_index])
        probabilities[test_index] = model.predict_proba(frame.iloc[test_index])[:, 1]
    valid = np.isfinite(probabilities)
    return score_predictions(y[valid], probabilities[valid])


def leave_one_dataset_out(frame: pd.DataFrame, target: str, numeric: list[str]) -> list[dict]:
    output = []
    for dataset in sorted(frame["dataset"].unique()):
        train = frame[frame["dataset"] != dataset]
        test = frame[frame["dataset"] == dataset]
        y_train = train[target].astype(int).to_numpy()
        y_test = test[target].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            output.append({"held_out_dataset": dataset, "error": "insufficient_classes"})
            continue
        model = make_model(numeric)
        model.fit(train, y_train)
        probability = model.predict_proba(test)[:, 1]
        output.append({"held_out_dataset": dataset, **score_predictions(y_test, probability)})
    return output


def univariate_table(frame: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    records = []
    for target in targets:
        subset = frame
        if target == "fixed": subset = frame[frame["hard_correct"] == 0]
        if target == "damaged": subset = frame[frame["hard_correct"] == 1]
        y = subset[target].astype(int)
        for feature in features:
            values = pd.to_numeric(subset[feature], errors="coerce")
            valid = values.notna()
            if valid.sum() < 20 or y[valid].nunique() < 2:
                continue
            rho, p_value = stats.spearmanr(values[valid], y[valid])
            positive = values[valid & (y == 1)]
            negative = values[valid & (y == 0)]
            try:
                u_stat, u_p = stats.mannwhitneyu(positive, negative, alternative="two-sided")
                rank_biserial = 2.0 * u_stat / (len(positive) * len(negative)) - 1.0
            except ValueError:
                u_p, rank_biserial = np.nan, np.nan
            records.append({
                "target": target,
                "feature": feature,
                "n": int(valid.sum()),
                "positive_n": int(y[valid].sum()),
                "positive_mean": float(positive.mean()),
                "negative_mean": float(negative.mean()),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
                "rank_biserial": float(rank_biserial),
                "mannwhitney_p": float(u_p),
            })
    return pd.DataFrame.from_records(records)


def no_op_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    thresholds = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
    event_frame = frame.drop_duplicates(["dataset", "strength", "event_id"])
    for threshold in thresholds:
        for keys, group in event_frame.groupby(["dataset", "strength"], dropna=False):
            no_op = group["target_delta_norm"].fillna(0.0) <= threshold
            records.append({
                "dataset": keys[0], "strength": keys[1], "threshold": threshold,
                "n_events": len(group), "no_op_events": int(no_op.sum()),
                "no_op_rate": float(no_op.mean()),
                "no_op_entropy_mean": float(group.loc[no_op, "true_entropy"].mean()) if no_op.any() else None,
                "active_entropy_mean": float(group.loc[~no_op, "true_entropy"].mean()) if (~no_op).any() else None,
            })
    return pd.DataFrame.from_records(records)


def label_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in frame.groupby(["dataset", "strength", "branch"], dropna=False):
        records.append({
            "dataset": keys[0], "strength": keys[1], "branch": keys[2], "n": len(group),
            "trajectory_change_rate": group["trajectory_change"].mean(),
            "answer_change_rate": group["answer_change"].mean(),
            "fixed": int(group["fixed"].sum()), "damaged": int(group["damaged"].sum()),
            "net": int(group["fixed"].sum() - group["damaged"].sum()),
        })
    return pd.DataFrame.from_records(records)


def markdown_report(
    frame: pd.DataFrame,
    no_op: pd.DataFrame,
    labels: pd.DataFrame,
    cv_records: list[dict],
    lodo_records: list[dict],
    univariate: pd.DataFrame,
    strengths: list[str],
) -> str:
    lines = [
        "# Visual Action Atlas 离线特征审计",
        "",
        "> 状态：阶段性；仅使用已完整强度  ",
        f"> 强度：{', '.join(strengths)}  ",
        "> 统计约束：交叉验证按原始样本分组，避免同一样本检查点泄漏  ",
        "",
        "## 1. 数据概况",
        "",
        f"- action-event 行数：{len(frame):,}",
        f"- 独立样本数：{frame['sample_group'].nunique():,}",
        f"- 独立事件数：{frame['event_id'].nunique():,}",
        f"- runtime error：{int(frame['runtime_error'].sum())}",
        "",
        "## 2. No-op 审计",
        "",
        "下表使用 `target_delta_norm <= 1e-6` 定义 soft≈hard 的退化事件。",
        "",
        "| 数据集 | 事件数 | no-op | 比例 | no-op entropy | active entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected = no_op[np.isclose(no_op["threshold"], 1e-6)].groupby("dataset", as_index=False).agg({
        "n_events": "sum", "no_op_events": "sum", "no_op_rate": "mean",
        "no_op_entropy_mean": "mean", "active_entropy_mean": "mean",
    })
    for _, row in selected.iterrows():
        lines.append(
            f"| {row.dataset} | {int(row.n_events)} | {int(row.no_op_events)} | {row.no_op_rate:.1%} | "
            f"{row.no_op_entropy_mean:.4f} | {row.active_entropy_mean:.4f} |"
        )

    lines += [
        "",
        "## 3. 分组交叉验证",
        "",
        "四组特征依次为：Basic（位置/分布）、Geometry、Visual contrast、Short response。后者包含 intervention 后前 1/2/4/8 token 的响应，只能用于短 rollout 策略，不能冒充零开销在线特征。",
        "",
        "| 目标 | 样本范围 | 特征组 | N | 正例率 | AUROC | AP | Balanced Acc |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in cv_records:
        lines.append(
            f"| {row['target']} | {row['scope']} | {row['feature_set']} | {row.get('n',0)} | "
            f"{row.get('prevalence',0):.1%} | {row.get('auroc') or 0:.3f} | "
            f"{row.get('average_precision') or 0:.3f} | {row.get('balanced_accuracy') or 0:.3f} |"
        )

    lines += [
        "",
        "## 4. 跨数据集泛化",
        "",
        "以下使用完整 pre-action 特征（不含 short response），每次留出一个数据集。",
        "",
        "| 目标 | 留出数据集 | N | 正例率 | AUROC | AP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in lodo_records:
        if "error" in row: continue
        lines.append(
            f"| {row['target']} | {row['held_out_dataset']} | {row['n']} | {row['prevalence']:.1%} | "
            f"{row['auroc']:.3f} | {row['average_precision']:.3f} |"
        )

    lines += [
        "",
        "## 5. 单特征信号",
        "",
        "按绝对 Spearman 相关排序，仅用于解释，不等价于因果或可部署预测能力。",
        "",
        "| 目标 | 特征 | N | rho | 正例均值 | 负例均值 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    top = (univariate.assign(abs_rho=univariate.spearman_rho.abs())
           .sort_values(["target", "abs_rho"], ascending=[True, False])
           .groupby("target").head(6))
    for _, row in top.iterrows():
        lines.append(
            f"| {row.target} | `{row.feature}` | {int(row.n)} | {row.spearman_rho:.3f} | "
            f"{row.positive_mean:.4f} | {row.negative_mean:.4f} |"
        )

    lines += [
        "",
        "## 6. 解释边界",
        "",
        "- 本报告是离线相关性审计，不证明某个特征具有因果效用。",
        "- fix 与 damage 分别只在 hard-wrong 与 hard-correct 子集中建模。",
        "- 所有 CV 按原始样本分组；事件级行数不能当作独立样本量。",
        "- Short response 特征需要额外短 rollout，后续必须单独报告计算成本。",
        "- Atlas 全部五档完成后，应使用相同脚本重跑最终统计。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--strengths", nargs="+", required=True)
    parser.add_argument("--no-op-threshold", type=float, default=1e-6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_table(args.root, args.datasets, args.strengths)
    frame["is_no_op"] = (frame["target_delta_norm"].fillna(0.0) <= args.no_op_threshold).astype(int)
    frame.to_csv(args.output_dir / "event_feature_table.csv", index=False)
    active = frame[frame["is_no_op"] == 0].copy()

    no_op = no_op_summary(frame)
    labels = label_summary(frame)
    # Keep the main univariate table deployable: only pre-action features.
    # Full-trajectory divergence remains in event_feature_table.csv for post-hoc analysis.
    numeric_features = sorted(set(VISUAL_NUMERIC))
    univariate = univariate_table(active, numeric_features, ["trajectory_change", "answer_change", "fixed", "damaged"])
    no_op.to_csv(args.output_dir / "no_op_threshold_sensitivity.csv", index=False)
    labels.to_csv(args.output_dir / "label_distribution.csv", index=False)
    univariate.to_csv(args.output_dir / "univariate_associations.csv", index=False)

    cv_records = []
    feature_sets = {
        "basic": BASIC_NUMERIC,
        "geometry": GEOMETRY_NUMERIC,
        "visual": VISUAL_NUMERIC,
        "short_response": RESPONSE_NUMERIC,
    }
    targets = {
        "trajectory_change": active,
        "answer_change": active,
        "fixed": active[active["hard_correct"] == 0],
        "damaged": active[active["hard_correct"] == 1],
    }
    for target, subset in targets.items():
        for feature_name, features in feature_sets.items():
            metrics = grouped_cv(subset, target, features)
            cv_records.append({
                "target": target,
                "scope": "active_only",
                "feature_set": feature_name,
                **metrics,
            })
    pd.DataFrame(cv_records).to_csv(args.output_dir / "grouped_cv_metrics.csv", index=False)

    lodo_records = []
    for target, subset in targets.items():
        for record in leave_one_dataset_out(subset, target, VISUAL_NUMERIC):
            lodo_records.append({"target": target, "feature_set": "visual", **record})
    pd.DataFrame(lodo_records).to_csv(args.output_dir / "leave_one_dataset_out.csv", index=False)

    summary = {
        "strengths": args.strengths,
        "datasets": args.datasets,
        "rows": len(frame),
        "independent_samples": int(frame["sample_group"].nunique()),
        "events": int(frame["event_id"].nunique()),
        "no_op_threshold": args.no_op_threshold,
        "no_op_rows": int(frame["is_no_op"].sum()),
        "cv": cv_records,
        "leave_one_dataset_out": lodo_records,
    }
    (args.output_dir / "offline_feature_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "offline_feature_audit.md").write_text(
        markdown_report(frame, no_op, labels, cv_records, lodo_records, univariate, args.strengths),
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(frame),
        "samples": int(frame["sample_group"].nunique()),
        "events": int(frame["event_id"].nunique()),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()

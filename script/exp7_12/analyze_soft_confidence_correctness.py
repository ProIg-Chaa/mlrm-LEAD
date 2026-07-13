#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean


DATASETS = [
    "vstar", "mmvp", "visulogic300", "vmcbench_dev", "mmk12_physics",
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("fixed_damage_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return fmean(values) if values else None


def metric_mean(tokens, field, start=None, stop=None):
    subset = tokens[slice(start, stop)]
    return mean([float(token[field]) for token in subset if token.get(field) is not None])


def confidence_values(tokens):
    values = []
    for token in tokens:
        value = token.get("raw_selected_prob")
        if value is None:
            value = token.get("raw_top1_prob")
        if value is not None:
            values.append(float(value))
    return values


def sample_features(trace, result_row, correct, pred, gold):
    tokens = trace.get("tokens") or []
    confidence = confidence_values(tokens)
    entropy = [float(token.get("raw_entropy")) for token in tokens if token.get("raw_entropy") is not None]
    output_tokens = int(result_row.get("output_tokens") or trace.get("output_tokens") or len(tokens))
    return {
        "id": str(trace.get("id")),
        "correct": bool(correct),
        "pred": pred,
        "gold": gold,
        "failed_extraction": pred is None,
        "runtime_error": bool(result_row.get("error_type")),
        "output_tokens": output_tokens,
        "long_256": output_tokens >= 256,
        "maxed_1024": output_tokens >= 1024,
        "mean_raw_conf": mean(confidence),
        "early32_raw_conf": mean(confidence[:32]),
        "tail20_raw_conf": mean(confidence[-20:]),
        "mean_raw_entropy": mean(entropy),
        "early32_raw_entropy": mean(entropy[:32]),
        "tail20_raw_entropy": mean(entropy[-20:]),
        "entropy_confidence": -mean(entropy) if entropy else None,
        "soft_ratio": (
            sum(token.get("mode") in {"soft", "pure_soft"} for token in tokens) / len(tokens)
            if tokens else None
        ),
    }


def auc(scores, labels):
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels) if score is not None]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def bootstrap_auc(rows, score_key, iterations=200, seed=42):
    eligible = [row for row in rows if row.get(score_key) is not None]
    base = auc([row[score_key] for row in eligible], [row["correct"] for row in eligible])
    if base is None or not eligible:
        return {"auc": base, "ci95": None}
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [eligible[rng.randrange(len(eligible))] for _ in eligible]
        estimate = auc([row[score_key] for row in sample], [row["correct"] for row in sample])
        if estimate is not None:
            estimates.append(estimate)
    estimates.sort()
    if not estimates:
        return {"auc": base, "ci95": None}
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return {"auc": base, "ci95": [low, high]}


def quantile_bins(rows, score_key, bins=10):
    eligible = sorted(
        [row for row in rows if row.get(score_key) is not None],
        key=lambda row: row[score_key],
    )
    result = []
    for bin_index in range(bins):
        start = len(eligible) * bin_index // bins
        stop = len(eligible) * (bin_index + 1) // bins
        subset = eligible[start:stop]
        if not subset:
            continue
        result.append({
            "bin": bin_index + 1,
            "count": len(subset),
            "score_min": subset[0][score_key],
            "score_max": subset[-1][score_key],
            "accuracy": mean([float(row["correct"]) for row in subset]),
            "wrong": sum(not row["correct"] for row in subset),
        })
    return result


def risk_coverage(rows, score_key):
    eligible = sorted(
        [row for row in rows if row.get(score_key) is not None],
        key=lambda row: row[score_key], reverse=True,
    )
    result = []
    for coverage in (0.1, 0.2, 0.5, 1.0):
        count = max(1, math.ceil(len(eligible) * coverage)) if eligible else 0
        subset = eligible[:count]
        result.append({
            "coverage": coverage,
            "count": count,
            "accuracy": mean([float(row["correct"]) for row in subset]),
            "wrong": sum(not row["correct"] for row in subset),
        })
    return result


def length_stratified_auc(rows, score_key):
    ordered = sorted(rows, key=lambda row: row["output_tokens"])
    strata = []
    weighted = []
    for index in range(4):
        start = len(ordered) * index // 4
        stop = len(ordered) * (index + 1) // 4
        subset = [row for row in ordered[start:stop] if row.get(score_key) is not None]
        value = auc([row[score_key] for row in subset], [row["correct"] for row in subset])
        strata.append({"quartile": index + 1, "count": len(subset), "auc": value})
        if value is not None:
            weighted.extend([value] * len(subset))
    return {"weighted_auc": mean(weighted), "strata": strata}


def threshold_group(rows, score_key, threshold):
    subset = [row for row in rows if row.get(score_key) is not None and row[score_key] >= threshold]
    semantic = [row for row in subset if not row["failed_extraction"]]
    return {
        "threshold": threshold,
        "count": len(subset),
        "correct": sum(row["correct"] for row in subset),
        "wrong": sum(not row["correct"] for row in subset),
        "accuracy": mean([float(row["correct"]) for row in subset]),
        "failed_extraction": sum(row["failed_extraction"] for row in subset),
        "semantic_count": len(semantic),
        "semantic_correct": sum(row["correct"] for row in semantic),
        "semantic_wrong": sum(not row["correct"] for row in semantic),
        "semantic_accuracy": mean([float(row["correct"]) for row in semantic]),
    }


def summarize(rows, method):
    strict_rows = [row for row in rows if not row["runtime_error"]]
    semantic_rows = [row for row in strict_rows if not row["failed_extraction"]]
    score_keys = ["entropy_confidence"]
    if method == "pure_soft":
        score_keys += ["mean_raw_conf", "early32_raw_conf", "tail20_raw_conf"]
    score_stats = {}
    for score_key in score_keys:
        eligible = [row for row in strict_rows if row.get(score_key) is not None]
        correct = [row[score_key] for row in eligible if row["correct"]]
        wrong = [row[score_key] for row in eligible if not row["correct"]]
        deciles = quantile_bins(strict_rows, score_key)
        semantic_deciles = quantile_bins(semantic_rows, score_key)
        overall = mean([float(row["correct"]) for row in eligible])
        top_accuracy = deciles[-1]["accuracy"] if deciles else None
        semantic_overall = mean([float(row["correct"]) for row in semantic_rows])
        semantic_top_accuracy = semantic_deciles[-1]["accuracy"] if semantic_deciles else None
        semantic_bootstrap = bootstrap_auc(semantic_rows, score_key, seed=142)
        score_stats[score_key] = {
            **bootstrap_auc(strict_rows, score_key),
            "coverage": len(eligible) / len(strict_rows) if strict_rows else 0.0,
            "correct_mean": mean(correct),
            "wrong_mean": mean(wrong),
            "wrong_minus_correct": (
                mean(wrong) - mean(correct) if mean(wrong) is not None and mean(correct) is not None else None
            ),
            "deciles": deciles,
            "semantic_deciles": semantic_deciles,
            "top_decile_accuracy": top_accuracy,
            "overall_accuracy_on_eligible": overall,
            "top_decile_delta": top_accuracy - overall if top_accuracy is not None and overall is not None else None,
            "semantic_top_decile_accuracy": semantic_top_accuracy,
            "semantic_overall_accuracy": semantic_overall,
            "semantic_top_decile_delta": (
                semantic_top_accuracy - semantic_overall
                if semantic_top_accuracy is not None and semantic_overall is not None else None
            ),
            "risk_coverage": risk_coverage(strict_rows, score_key),
            "length_stratified": length_stratified_auc(strict_rows, score_key),
            "semantic_only_auc": semantic_bootstrap["auc"],
            "semantic_only_auc_ci95": semantic_bootstrap["ci95"],
        }
    return {
        "total": len(rows),
        "accuracy": mean([float(row["correct"]) for row in strict_rows]),
        "correct": sum(row["correct"] for row in strict_rows),
        "wrong": sum(not row["correct"] for row in strict_rows),
        "failed_extraction": sum(row["failed_extraction"] for row in strict_rows),
        "runtime_error": sum(row["runtime_error"] for row in rows),
        "mean_output_tokens_correct": mean([row["output_tokens"] for row in strict_rows if row["correct"]]),
        "mean_output_tokens_wrong": mean([row["output_tokens"] for row in strict_rows if not row["correct"]]),
        "score_stats": score_stats,
        "high_confidence_groups": (
            {
                "mean_raw_conf_ge_090": threshold_group(strict_rows, "mean_raw_conf", 0.90),
                "tail20_raw_conf_ge_095": threshold_group(strict_rows, "tail20_raw_conf", 0.95),
            }
            if method == "pure_soft" else {}
        ),
    }


def run_dir(root, dataset, method):
    return root / "output/experiments/20260706_format_stability_full_baselines/format_stability_full_baselines/r1_onevision_7b" / dataset / "pure_soft_gpu0"


def analyze_run(root, evaluator, dataset, method):
    directory = run_dir(root, dataset, method)
    results = evaluator.row_map(directory, dataset)
    trace_path = directory / "token_entropy_full.jsonl"
    rows = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            sid = str(trace.get("id"))
            result_row = results.get(sid)
            if result_row is None:
                continue
            pred, correct, _ = evaluator.adjudicate(result_row, dataset)
            gold = evaluator.normalize_gold(result_row.get("answer"), result_row.get("options"))
            feature = sample_features(trace, result_row, correct, pred, gold)
            feature.update({"dataset": dataset, "method": method})
            rows.append(feature)
    return directory, rows, summarize(rows, method)


def render_markdown(stats):
    lines = [
        "# Soft Reasoning Confidence vs Correctness",
        "",
        "主问题：token-level 高置信度能否预测最终答案正确。Pure-soft 使用 `raw_selected_prob`，并以 `-mean(raw_entropy)` 作一致性检查。不使用经过 top-k/top-p 过滤后常接近 1 的 `selected_prob`。",
        "",
        "| dataset | acc | failed | strict AUC | semantic AUC | top10 strict delta | top10 semantic delta | wrong-conf - correct-conf | length-controlled AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        method = item["method"]
        summary = item["summary"]
        key = "mean_raw_conf"
        score = summary["score_stats"][key]
        ci = score.get("ci95")
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        lines.append(
            f"| {item['dataset']} | {100*summary['accuracy']:.2f}% | {summary['failed_extraction']} | "
            f"{score['auc']:.3f} | {score['semantic_only_auc']:.3f} | "
            f"{100*score['top_decile_delta']:+.2f}pp | {100*score['semantic_top_decile_delta']:+.2f}pp | "
            f"{score['wrong_minus_correct']:+.4f} | "
            f"{score['length_stratified']['weighted_auc']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- AUC=0.5 表示置信度不能区分最终正确/错误；AUC<0.5 表示错误样本反而更自信。",
        "- top-decile delta<=0 表示只保留最高置信 10% 样本并不会提高准确率。",
        "- length-controlled AUC 按输出长度四分位计算，避免长输出单独造成表面相关。",
        "- failed extraction 同时计入 strict accuracy，并另外给出排除抽取失败后的 semantic-only AUC。",
        "",
    ]
    return "\n".join(lines)


def plot(stats, output_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    soft = [item for item in stats if item["method"] == "pure_soft"]
    labels = [item["dataset"] for item in soft]
    auc_values = [item["summary"]["score_stats"]["mean_raw_conf"]["auc"] for item in soft]
    deltas = [100 * item["summary"]["score_stats"]["mean_raw_conf"]["top_decile_delta"] for item in soft]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].bar(x, auc_values, color=["#b91c1c" if value < 0.5 else "#2563eb" for value in auc_values])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Confidence AUROC")
    axes[0].set_title("Does mean raw confidence predict final correctness?")
    axes[1].bar(x, deltas, color=["#b91c1c" if value <= 0 else "#15803d" for value in deltas])
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Top-decile accuracy delta (pp)")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "soft_confidence_predictiveness.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output_dir = args.root.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_module(root / "script/exp7_11/analyze_fixed_damaged_mechanisms.py")
    all_rows, stats = [], []
    for dataset in DATASETS:
        method = "pure_soft"
        print(f"[ANALYZE] {dataset}/{method}", flush=True)
        directory, rows, summary = analyze_run(root, evaluator, dataset, method)
        all_rows.extend(rows)
        stats.append({"dataset": dataset, "method": method, "run_dir": str(directory), "summary": summary})
    with (output_dir / "sample_confidence_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    high_confidence_wrong = []
    for dataset in DATASETS:
        candidates = [
            row for row in all_rows
            if row["dataset"] == dataset and not row["correct"]
            and not row["failed_extraction"] and row.get("mean_raw_conf") is not None
        ]
        candidates.sort(key=lambda row: row["mean_raw_conf"], reverse=True)
        high_confidence_wrong.extend(candidates[:20])
    with (output_dir / "high_confidence_semantic_wrong_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in high_confidence_wrong:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "confidence_correctness_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_markdown(stats)
    (output_dir / "confidence_correctness_summary.md").write_text(report, encoding="utf-8")
    plot(stats, output_dir)
    print(f"[DONE] {output_dir}")


if __name__ == "__main__":
    main()

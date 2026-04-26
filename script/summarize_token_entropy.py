#!/usr/bin/env python3
"""Summarize token-level entropy traces saved by main.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from collections import Counter


def weighted_mean(pairs):
    total_weight = sum(weight for _, weight in pairs)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def summarize(path: Path) -> dict:
    records = []
    token_counts = []
    raw_entropies = []
    filtered_entropies = []
    selected_probs = []
    soft_tokens = 0
    normal_tokens = 0
    reasoning_tokens = 0
    think_opened = 0
    think_closed = 0
    compact_all_raw_means = []
    compact_reasoning_raw_means = []
    compact_reasoning_raw_p90 = []
    compact_reasoning_raw_high_gt_1 = []
    compact_reasoning_raw_high_gt_2 = []
    compact_all_filtered_means = []
    compact_reasoning_filtered_means = []
    relation_tokens = 0
    reasoning_relation_tokens = 0
    compact_relation_raw_means = []
    compact_non_relation_raw_means = []
    compact_reasoning_relation_raw_means = []
    compact_reasoning_non_relation_raw_means = []
    compact_relation_raw_p90 = []
    compact_relation_raw_high_gt_1 = []
    compact_reasoning_relation_raw_p90 = []
    compact_reasoning_relation_raw_high_gt_1 = []
    relation_marker_counts = Counter()
    relation_category_counts = Counter()
    relation_category_means = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            summary = record.get("entropy_summary")
            if summary is not None:
                token_count = int(summary.get("token_count") or 0)
                reasoning_count = int(summary.get("reasoning_token_count") or 0)
                token_counts.append(token_count)
                reasoning_tokens += reasoning_count
                soft_tokens += int(summary.get("soft_token_count") or 0)
                normal_tokens += token_count - int(summary.get("soft_token_count") or 0)
                think_opened += int(bool(summary.get("think_opened")))
                think_closed += int(bool(summary.get("think_closed")))

                all_raw = summary.get("all_raw_entropy", {})
                reasoning_raw = summary.get("reasoning_raw_entropy", {})
                all_filtered = summary.get("all_filtered_entropy", {})
                reasoning_filtered = summary.get("reasoning_filtered_entropy", {})
                if all_raw.get("mean") is not None:
                    compact_all_raw_means.append((float(all_raw["mean"]), token_count))
                if reasoning_raw.get("mean") is not None:
                    compact_reasoning_raw_means.append((
                        float(reasoning_raw["mean"]),
                        reasoning_count,
                    ))
                if reasoning_raw.get("p90") is not None:
                    compact_reasoning_raw_p90.append(float(reasoning_raw["p90"]))
                if reasoning_raw.get("high_gt_1_ratio") is not None:
                    compact_reasoning_raw_high_gt_1.append(float(reasoning_raw["high_gt_1_ratio"]))
                if reasoning_raw.get("high_gt_2_ratio") is not None:
                    compact_reasoning_raw_high_gt_2.append(float(reasoning_raw["high_gt_2_ratio"]))
                if all_filtered.get("mean") is not None:
                    compact_all_filtered_means.append((float(all_filtered["mean"]), token_count))
                if reasoning_filtered.get("mean") is not None:
                    compact_reasoning_filtered_means.append((
                        float(reasoning_filtered["mean"]),
                        reasoning_count,
                    ))

                relation_count = int(summary.get("relation_token_count") or 0)
                reasoning_relation_count = int(
                    summary.get("reasoning_relation_token_count") or 0
                )
                relation_tokens += relation_count
                reasoning_relation_tokens += reasoning_relation_count
                relation_raw = summary.get("relation_raw_entropy", {})
                non_relation_raw = summary.get("non_relation_raw_entropy", {})
                reasoning_relation_raw = summary.get(
                    "reasoning_relation_raw_entropy", {}
                )
                reasoning_non_relation_raw = summary.get(
                    "reasoning_non_relation_raw_entropy", {}
                )
                if relation_raw.get("mean") is not None:
                    compact_relation_raw_means.append((
                        float(relation_raw["mean"]),
                        relation_count,
                    ))
                if non_relation_raw.get("mean") is not None:
                    compact_non_relation_raw_means.append((
                        float(non_relation_raw["mean"]),
                        int(non_relation_raw.get("count") or 0),
                    ))
                if reasoning_relation_raw.get("mean") is not None:
                    compact_reasoning_relation_raw_means.append((
                        float(reasoning_relation_raw["mean"]),
                        reasoning_relation_count,
                    ))
                if reasoning_non_relation_raw.get("mean") is not None:
                    compact_reasoning_non_relation_raw_means.append((
                        float(reasoning_non_relation_raw["mean"]),
                        int(reasoning_non_relation_raw.get("count") or 0),
                    ))
                if relation_raw.get("p90") is not None:
                    compact_relation_raw_p90.append(float(relation_raw["p90"]))
                if relation_raw.get("high_gt_1_ratio") is not None:
                    compact_relation_raw_high_gt_1.append(
                        float(relation_raw["high_gt_1_ratio"])
                    )
                if reasoning_relation_raw.get("p90") is not None:
                    compact_reasoning_relation_raw_p90.append(
                        float(reasoning_relation_raw["p90"])
                    )
                if reasoning_relation_raw.get("high_gt_1_ratio") is not None:
                    compact_reasoning_relation_raw_high_gt_1.append(
                        float(reasoning_relation_raw["high_gt_1_ratio"])
                    )
                relation_marker_counts.update(
                    summary.get("relation_marker_counts") or {}
                )
                for category, stats in (
                    summary.get("relation_category_stats") or {}
                ).items():
                    count = int(stats.get("count") or 0)
                    relation_category_counts[category] += count
                    if stats.get("mean") is not None:
                        relation_category_means.setdefault(category, []).append((
                            float(stats["mean"]),
                            count,
                        ))
                continue

            tokens = record.get("tokens", [])
            token_counts.append(len(tokens))
            for token in tokens:
                if token.get("raw_entropy") is not None:
                    raw_entropies.append(float(token["raw_entropy"]))
                if token.get("filtered_entropy") is not None:
                    filtered_entropies.append(float(token["filtered_entropy"]))
                if token.get("selected_prob") is not None:
                    selected_probs.append(float(token["selected_prob"]))
                if token.get("mode") == "soft":
                    soft_tokens += 1
                elif token.get("mode") == "normal":
                    normal_tokens += 1
                if token.get("is_reasoning_token"):
                    reasoning_tokens += 1

    total_tokens = soft_tokens + normal_tokens
    compact = bool(compact_all_raw_means or compact_reasoning_raw_means)
    return {
        "path": str(path),
        "records": len(records),
        "total_tokens": sum(token_counts),
        "avg_tokens_per_sample": mean(token_counts) if token_counts else None,
        "median_tokens_per_sample": median(token_counts) if token_counts else None,
        "avg_raw_entropy": (
            weighted_mean(compact_all_raw_means)
            if compact
            else mean(raw_entropies) if raw_entropies else None
        ),
        "median_raw_entropy": median(raw_entropies) if raw_entropies else None,
        "avg_filtered_entropy": (
            weighted_mean(compact_all_filtered_means)
            if compact
            else mean(filtered_entropies) if filtered_entropies else None
        ),
        "median_filtered_entropy": median(filtered_entropies) if filtered_entropies else None,
        "avg_selected_prob": None if compact else mean(selected_probs) if selected_probs else None,
        "soft_tokens": soft_tokens,
        "normal_tokens": normal_tokens,
        "soft_ratio": soft_tokens / total_tokens if total_tokens else None,
        "think_opened_records": think_opened if compact else None,
        "think_closed_records": think_closed if compact else None,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_ratio": reasoning_tokens / total_tokens if total_tokens else None,
        "avg_reasoning_raw_entropy": (
            weighted_mean(compact_reasoning_raw_means)
            if compact
            else None
        ),
        "avg_reasoning_filtered_entropy": (
            weighted_mean(compact_reasoning_filtered_means)
            if compact
            else None
        ),
        "avg_sample_reasoning_raw_p90": (
            mean(compact_reasoning_raw_p90)
            if compact_reasoning_raw_p90
            else None
        ),
        "avg_sample_reasoning_raw_high_gt_1_ratio": (
            mean(compact_reasoning_raw_high_gt_1)
            if compact_reasoning_raw_high_gt_1
            else None
        ),
        "avg_sample_reasoning_raw_high_gt_2_ratio": (
            mean(compact_reasoning_raw_high_gt_2)
            if compact_reasoning_raw_high_gt_2
            else None
        ),
        "relation_tokens": relation_tokens if compact else None,
        "relation_ratio": (
            relation_tokens / sum(token_counts)
            if compact and sum(token_counts)
            else None
        ),
        "reasoning_relation_tokens": (
            reasoning_relation_tokens if compact else None
        ),
        "reasoning_relation_ratio": (
            reasoning_relation_tokens / reasoning_tokens
            if compact and reasoning_tokens
            else None
        ),
        "avg_relation_raw_entropy": (
            weighted_mean(compact_relation_raw_means)
            if compact_relation_raw_means
            else None
        ),
        "avg_non_relation_raw_entropy": (
            weighted_mean(compact_non_relation_raw_means)
            if compact_non_relation_raw_means
            else None
        ),
        "avg_reasoning_relation_raw_entropy": (
            weighted_mean(compact_reasoning_relation_raw_means)
            if compact_reasoning_relation_raw_means
            else None
        ),
        "avg_reasoning_non_relation_raw_entropy": (
            weighted_mean(compact_reasoning_non_relation_raw_means)
            if compact_reasoning_non_relation_raw_means
            else None
        ),
        "avg_sample_relation_raw_p90": (
            mean(compact_relation_raw_p90)
            if compact_relation_raw_p90
            else None
        ),
        "avg_sample_relation_raw_high_gt_1_ratio": (
            mean(compact_relation_raw_high_gt_1)
            if compact_relation_raw_high_gt_1
            else None
        ),
        "avg_sample_reasoning_relation_raw_p90": (
            mean(compact_reasoning_relation_raw_p90)
            if compact_reasoning_relation_raw_p90
            else None
        ),
        "avg_sample_reasoning_relation_raw_high_gt_1_ratio": (
            mean(compact_reasoning_relation_raw_high_gt_1)
            if compact_reasoning_relation_raw_high_gt_1
            else None
        ),
        "relation_marker_counts": (
            dict(relation_marker_counts.most_common(30))
            if compact and relation_marker_counts
            else None
        ),
        "relation_category_counts": (
            dict(relation_category_counts)
            if compact and relation_category_counts
            else None
        ),
        "relation_category_avg_raw_entropy": (
            {
                category: weighted_mean(values)
                for category, values in relation_category_means.items()
            }
            if compact and relation_category_means
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for raw_path in args.paths:
        data = summarize(Path(raw_path))
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

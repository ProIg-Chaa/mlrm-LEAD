#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "exp7_17"))
from talr_analysis_common import (  # noqa: E402
    explicit_answers,
    load_jsonl,
    repeat_ratio,
    score_row,
    trace_by_id,
    write_json,
    write_jsonl,
)


TREATMENTS = ("contracted_soft_l095", "pure_soft_l100")


def by_id(path: Path) -> dict[str, dict]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def token_ids(trace: dict | None) -> list[int]:
    return [
        int(token["token_id"])
        for token in (trace or {}).get("tokens", [])
        if token.get("token_id") is not None
    ]


def first_divergence(left: list[int], right: list[int], start: int) -> int | None:
    for index in range(start, min(len(left), len(right))):
        if left[index] != right[index]:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def mismatch_ratio(
    left: list[int], right: list[int], start: int, horizon: int
) -> float:
    stop = min(max(len(left), len(right)), start + horizon)
    if stop <= start:
        return 0.0
    mismatches = 0
    for index in range(start, stop):
        lval = left[index] if index < len(left) else None
        rval = right[index] if index < len(right) else None
        mismatches += int(lval != rval)
    return mismatches / (stop - start)


def rolling_features(tokens: list[dict], step: int) -> dict:
    current = tokens[step]
    entropy = [
        float(token.get("raw_entropy") or 0.0)
        for token in tokens[: step + 1]
    ]
    result = {
        key: current.get(key)
        for key in (
            "raw_entropy",
            "filtered_entropy",
            "selected_prob",
            "raw_top1_prob",
            "raw_margin",
            "raw_top2_mass",
            "raw_top5_mass",
            "hard_embedding_norm",
            "soft_embedding_norm",
            "soft_hard_l2",
            "soft_hard_relative_l2",
            "soft_hard_cosine",
            "token_id",
            "token_text",
            "token_is_newline",
            "token_is_whitespace",
            "token_is_punctuation",
            "token_is_answer_marker",
        )
    }
    result["entropy_delta_1"] = (
        entropy[-1] - entropy[-2] if len(entropy) >= 2 else 0.0
    )
    for window in (4, 8, 16):
        values = entropy[-window:]
        result[f"entropy_mean_{window}"] = statistics.fmean(values)
        result[f"entropy_std_{window}"] = (
            statistics.pstdev(values) if len(values) > 1 else 0.0
        )
        result[f"entropy_delta_from_mean_{window}"] = (
            entropy[-1] - result[f"entropy_mean_{window}"]
        )
    recent_ids = [
        int(token.get("token_id"))
        for token in tokens[max(0, step - 15) : step + 1]
        if token.get("token_id") is not None
    ]
    result["recent16_duplicate_ratio"] = (
        1.0 - len(set(recent_ids)) / len(recent_ids)
        if recent_ids
        else 0.0
    )
    result["step_index"] = step
    result["prefix_length"] = step + 1
    result["normalized_position_max1024"] = step / 1024.0
    result["previous_token_id"] = (
        tokens[step - 1].get("token_id") if step > 0 else None
    )
    result["previous_token_text"] = (
        tokens[step - 1].get("token_text") if step > 0 else None
    )
    result["recent4_token_ids"] = [
        token.get("token_id") for token in tokens[max(0, step - 3) : step + 1]
    ]
    result["recent4_token_text"] = [
        token.get("token_text") for token in tokens[max(0, step - 3) : step + 1]
    ]
    prior_newlines = [
        index for index, token in enumerate(tokens[: step + 1])
        if token.get("token_is_newline")
    ]
    prior_answer_markers = [
        index for index, token in enumerate(tokens[: step + 1])
        if token.get("token_is_answer_marker")
    ]
    result["steps_since_newline"] = (
        step - prior_newlines[-1] if prior_newlines else None
    )
    result["steps_since_answer_marker"] = (
        step - prior_answer_markers[-1] if prior_answer_markers else None
    )
    return result


def sample_features(row: dict) -> dict:
    question = str(row.get("question") or "")
    options = str(row.get("options") or "")
    return {
        "question_char_count": len(question),
        "question_word_count": len(question.split()),
        "option_count": len(set(re.findall(r"\(([A-Ea-e])\)", options))),
        "subtopic": row.get("subtopic") or row.get("subject"),
        "benchmark": row.get("benchmark"),
    }


def analysis_only_context(tokens: list[dict], step: int) -> dict:
    entropies = [float(token.get("raw_entropy") or 0.0) for token in tokens]
    current = entropies[step]
    return {
        "baseline_trace_length": len(tokens),
        "normalized_position_in_realized_baseline": step / max(1, len(tokens) - 1),
        "global_entropy_percentile": (
            sum(value <= current for value in entropies) / len(entropies)
            if entropies else None
        ),
    }


def output_features(row: dict) -> dict:
    text = str(row.get("model_answer") or "")
    answers = explicit_answers(text)
    output_tokens = int(row.get("output_tokens") or 0)
    return {
        "predicted_answer_sequence": answers,
        "answer_reversal": len(set(answers)) > 1,
        "output_tokens": output_tokens,
        "long_ge_256": output_tokens >= 256,
        "maxed_1024": output_tokens >= 1024,
        "repeat_ratio_3gram": repeat_ratio(text, n=3),
        "latency_sec": row.get("latency_sec"),
        "cuda_peak_allocated_mb": row.get("cuda_peak_allocated_mb"),
        "cuda_peak_reserved_mb": row.get("cuda_peak_reserved_mb"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    args = parser.parse_args()

    event_manifest = load_jsonl(args.shard_dir / "event_manifest.jsonl")
    events = {row["event_id"]: row for row in event_manifest}
    baseline_rows = by_id(args.shard_dir / "hard_baseline" / "results.jsonl")
    baseline_traces = trace_by_id(args.shard_dir / "hard_baseline")
    labels = []
    totals = defaultdict(Counter)

    for treatment in TREATMENTS:
        treatment_dir = args.shard_dir / treatment
        treatment_rows = by_id(treatment_dir / "results.jsonl")
        treatment_traces = trace_by_id(treatment_dir)
        for event_id, event in events.items():
            original_id = str(event["original_id"])
            if original_id not in baseline_rows or event_id not in treatment_rows:
                continue
            base_row = baseline_rows[original_id]
            treated_row = treatment_rows[event_id]
            base_trace = baseline_traces.get(original_id, {})
            treated_trace = treatment_traces.get(event_id, {})
            base_tokens = base_trace.get("tokens") or []
            treated_tokens = treated_trace.get("tokens") or []
            step = int(event["event_step"])
            if step >= len(base_tokens):
                continue
            base_score = score_row(base_row)
            treated_score = score_row(treated_row)
            base_ids = token_ids(base_trace)
            treated_ids = token_ids(treated_trace)
            prefix_match = base_ids[: step + 1] == treated_ids[: step + 1]
            divergence = first_divergence(base_ids, treated_ids, step + 1)
            utility_acc = (
                None
                if base_score["correct"] is None or treated_score["correct"] is None
                else int(treated_score["correct"]) - int(base_score["correct"])
            )
            row = {
                **event,
                "treatment": treatment,
                "prefix_match": prefix_match,
                "base_pred": base_score["pred"],
                "treatment_pred": treated_score["pred"],
                "gold": base_score["gold"],
                "base_correct": base_score["correct"],
                "treatment_correct": treated_score["correct"],
                "utility_acc": utility_acc,
                "fixed": utility_acc == 1,
                "damaged": utility_acc == -1,
                "answer_changed": base_score["pred"] != treated_score["pred"],
                "base_failed_extraction": base_score["failed_extraction"],
                "treatment_failed_extraction": treated_score["failed_extraction"],
                "base_runtime_error": base_score["runtime_error"],
                "treatment_runtime_error": treated_score["runtime_error"],
                "first_divergence_step": divergence,
                "divergence_delay": (
                    None if divergence is None else divergence - step
                ),
                "mismatch_ratio_h8": mismatch_ratio(
                    base_ids, treated_ids, step + 1, 8
                ),
                "mismatch_ratio_h16": mismatch_ratio(
                    base_ids, treated_ids, step + 1, 16
                ),
                "mismatch_ratio_h32": mismatch_ratio(
                    base_ids, treated_ids, step + 1, 32
                ),
                "base_output": output_features(base_row),
                "treatment_output": output_features(treated_row),
                "sample_features": sample_features(base_row),
                "online_pre_intervention_features": rolling_features(
                    base_tokens, step
                ),
                "analysis_only_context": analysis_only_context(
                    base_tokens, step
                ),
            }
            labels.append(row)
            key = f"{event['dataset']}/{treatment}"
            totals[key]["events"] += 1
            totals[key]["prefix_match"] += int(prefix_match)
            totals[key]["fixed"] += int(utility_acc == 1)
            totals[key]["damaged"] += int(utility_acc == -1)
            totals[key]["answer_changed"] += int(row["answer_changed"])
            totals[key]["failed"] += int(
                treated_score["failed_extraction"]
            )

    write_jsonl(args.shard_dir / "event_labels.jsonl", labels)
    summary = {key: dict(value) for key, value in sorted(totals.items())}
    write_json(args.shard_dir / "event_label_summary.json", summary)
    lines = [
        "# Intervention Atlas V0B Shard Summary",
        "",
        "| Dataset/treatment | Events | Prefix match | Fixed | Damaged | Net | Answer changed | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(summary.items()):
        lines.append(
            f"| {key} | {value['events']} | {value['prefix_match']} | "
            f"{value['fixed']} | {value['damaged']} | "
            f"{value['fixed'] - value['damaged']:+d} | "
            f"{value['answer_changed']} | {value['failed']} |"
        )
    (args.shard_dir / "event_label_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


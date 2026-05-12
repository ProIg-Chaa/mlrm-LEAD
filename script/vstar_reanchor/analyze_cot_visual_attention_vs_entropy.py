#!/usr/bin/env python3
"""Analyze whether high-entropy COT tokens correlate with weak visual attention."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import fmean


def load_extract_fn(evaluator_path: Path):
    spec = importlib.util.spec_from_file_location("lead_evaluator_only", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.extract_mcq_answer


def load_results(results_path: Path, evaluator_path: Path) -> dict[int, bool]:
    extract_mcq_answer = load_extract_fn(evaluator_path)
    records = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pred = extract_mcq_answer(row.get("model_answer"))
            records[int(row["id"])] = pred is not None and pred == row.get("answer")
    return records


def load_specialized_results(results_path: Path) -> dict[int, bool]:
    records = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records[int(row["id"])] = bool(row.get("specialized_is_correct"))
    return records


def load_trace_rows(trace_path: Path) -> list[dict]:
    rows = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def token_is_wordlike(token: dict) -> bool:
    normalized = (token.get("normalized_token_text") or "").strip()
    return any(ch.isalnum() for ch in normalized)


def summarize_tokens(tokens: list[dict]) -> dict:
    if not tokens:
        return {
            "count": 0,
            "mean_raw_entropy": None,
            "mean_selected_prob": None,
            "mean_visual_attn_mass": None,
            "mean_visual_attn_top1": None,
            "mean_visual_attn_top4_sum": None,
            "mean_visual_attn_entropy": None,
        }
    return {
        "count": len(tokens),
        "mean_raw_entropy": fmean(float(t["raw_entropy"]) for t in tokens),
        "mean_selected_prob": fmean(float(t.get("selected_prob", 0.0)) for t in tokens),
        "mean_visual_attn_mass": fmean(float(t["visual_attn_mass"]) for t in tokens),
        "mean_visual_attn_top1": fmean(float(t["visual_attn_top1"]) for t in tokens),
        "mean_visual_attn_top4_sum": fmean(float(t["visual_attn_top4_sum"]) for t in tokens),
        "mean_visual_attn_entropy": fmean(
            float(t["visual_attn_entropy"]) for t in tokens if t.get("visual_attn_entropy") is not None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--results_format",
        choices=["default", "specialized"],
        default="default",
    )
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--sample_quantile", type=float, default=0.9)
    parser.add_argument("--reasoning_only", action="store_true")
    parser.add_argument("--exclude_nonword", action="store_true")
    args = parser.parse_args()

    if args.results_format == "specialized":
        result_meta = load_specialized_results(Path(args.results))
    else:
        if not args.evaluator:
            raise SystemExit("--evaluator is required when --results_format=default")
        result_meta = load_results(Path(args.results), Path(args.evaluator))

    trace_rows = load_trace_rows(Path(args.trace))
    split_tokens = {
        "overall": {"all": [], "high_abs": [], "low_abs": [], "high_rel": [], "low_rel": []},
        "correct": {"all": [], "high_abs": [], "low_abs": [], "high_rel": [], "low_rel": []},
        "wrong": {"all": [], "high_abs": [], "low_abs": [], "high_rel": [], "low_rel": []},
    }
    sample_rows = []

    for row in trace_rows:
        sample_id = int(row["id"])
        is_correct = result_meta.get(sample_id)
        if is_correct is None:
            continue
        split_name = "correct" if is_correct else "wrong"
        tokens = []
        for token in row.get("tokens", []):
            if not token.get("visual_attn_available"):
                continue
            if args.reasoning_only and not token.get("is_reasoning_token"):
                continue
            if args.exclude_nonword and not token_is_wordlike(token):
                continue
            if token.get("raw_entropy") is None:
                continue
            tokens.append(token)
        if not tokens:
            continue

        entropies = sorted(float(token["raw_entropy"]) for token in tokens)
        q_index = round((len(entropies) - 1) * args.sample_quantile)
        rel_threshold = entropies[q_index]
        high_abs_tokens = [t for t in tokens if float(t["raw_entropy"]) >= args.entropy_threshold]
        low_abs_tokens = [t for t in tokens if float(t["raw_entropy"]) < args.entropy_threshold]
        high_rel_tokens = [t for t in tokens if float(t["raw_entropy"]) >= rel_threshold]
        low_rel_tokens = [t for t in tokens if float(t["raw_entropy"]) < rel_threshold]

        for split in ("overall", split_name):
            split_tokens[split]["all"].extend(tokens)
            split_tokens[split]["high_abs"].extend(high_abs_tokens)
            split_tokens[split]["low_abs"].extend(low_abs_tokens)
            split_tokens[split]["high_rel"].extend(high_rel_tokens)
            split_tokens[split]["low_rel"].extend(low_rel_tokens)

        sample_rows.append(
            {
                "id": sample_id,
                "is_correct": is_correct,
                "token_count": len(tokens),
                "sample_relative_entropy_threshold": rel_threshold,
                "high_abs_count": len(high_abs_tokens),
                "high_rel_count": len(high_rel_tokens),
                "mean_visual_attn_mass_all": fmean(float(t["visual_attn_mass"]) for t in tokens),
                "mean_visual_attn_mass_high_abs": (
                    fmean(float(t["visual_attn_mass"]) for t in high_abs_tokens)
                    if high_abs_tokens
                    else None
                ),
                "mean_visual_attn_mass_high_rel": (
                    fmean(float(t["visual_attn_mass"]) for t in high_rel_tokens)
                    if high_rel_tokens
                    else None
                ),
            }
        )

    summary = {
        "trace": args.trace,
        "results": args.results,
        "entropy_threshold": args.entropy_threshold,
        "sample_quantile": args.sample_quantile,
        "reasoning_only": args.reasoning_only,
        "exclude_nonword": args.exclude_nonword,
        "splits": {},
        "sample_rows": sample_rows,
    }
    for split_name, buckets in split_tokens.items():
        split_summary = {}
        for bucket_name, bucket_tokens in buckets.items():
            split_summary[bucket_name] = summarize_tokens(bucket_tokens)
        all_mass = split_summary["all"]["mean_visual_attn_mass"]
        high_abs_mass = split_summary["high_abs"]["mean_visual_attn_mass"]
        high_rel_mass = split_summary["high_rel"]["mean_visual_attn_mass"]
        split_summary["delta"] = {
            "high_abs_minus_all_visual_mass": (
                high_abs_mass - all_mass
                if high_abs_mass is not None and all_mass is not None
                else None
            ),
            "high_rel_minus_all_visual_mass": (
                high_rel_mass - all_mass
                if high_rel_mass is not None and all_mass is not None
                else None
            ),
        }
        summary["splits"][split_name] = split_summary

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

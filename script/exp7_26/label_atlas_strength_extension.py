#!/usr/bin/env python3
"""Create Atlas labels for additional intervention strengths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--runs-subdir", default="newgpu3")
    parser.add_argument("--run-dir-name", required=True)
    parser.add_argument("--treatment-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exp_dir = args.repo / "script" / "exp7_23"
    sys.path.insert(0, str(exp_dir))
    import summarize_intervention_atlas_v0b as atlas  # noqa: PLC0415
    from talr_analysis_common import (  # type: ignore  # noqa: PLC0415
        load_jsonl,
        score_row,
        trace_by_id,
        write_jsonl,
    )

    labels = []
    shards = []
    for shard_dir in sorted((args.atlas_root / args.runs_subdir).glob("shard_*")):
        run_dir = shard_dir / args.run_dir_name
        required = [
            run_dir / "results.jsonl",
            run_dir / "token_entropy_full.jsonl",
            shard_dir / "hard_baseline" / "results.jsonl",
            shard_dir / "event_manifest.jsonl",
        ]
        if not all(path.exists() for path in required):
            shards.append({"shard": shard_dir.name, "status": "incomplete"})
            continue

        events = {
            row["event_id"]: row
            for row in load_jsonl(shard_dir / "event_manifest.jsonl")
        }
        baseline_rows = atlas.by_id(shard_dir / "hard_baseline" / "results.jsonl")
        baseline_traces = trace_by_id(shard_dir / "hard_baseline")
        treatment_rows = atlas.by_id(run_dir / "results.jsonl")
        treatment_traces = trace_by_id(run_dir)
        before = len(labels)

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
            base_ids = atlas.token_ids(base_trace)
            treated_ids = atlas.token_ids(treated_trace)
            prefix_match = base_ids[: step + 1] == treated_ids[: step + 1]
            divergence = atlas.first_divergence(base_ids, treated_ids, step + 1)
            utility_acc = (
                None
                if base_score["correct"] is None or treated_score["correct"] is None
                else int(treated_score["correct"]) - int(base_score["correct"])
            )
            labels.append(
                {
                    **event,
                    "treatment": args.treatment_name,
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
                    "divergence_delay": None if divergence is None else divergence - step,
                    "mismatch_ratio_h8": atlas.mismatch_ratio(base_ids, treated_ids, step + 1, 8),
                    "mismatch_ratio_h16": atlas.mismatch_ratio(base_ids, treated_ids, step + 1, 16),
                    "mismatch_ratio_h32": atlas.mismatch_ratio(base_ids, treated_ids, step + 1, 32),
                    "base_output": atlas.output_features(base_row),
                    "treatment_output": atlas.output_features(treated_row),
                    "sample_features": atlas.sample_features(base_row),
                    "online_pre_intervention_features": atlas.rolling_features(base_tokens, step),
                    "analysis_only_context": atlas.analysis_only_context(base_tokens, step),
                }
            )
        shards.append(
            {"shard": shard_dir.name, "status": "complete", "labels": len(labels) - before}
        )

    labels.sort(
        key=lambda row: (
            row["dataset"],
            row["original_id"],
            row["event_step"],
            row["treatment"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, labels)
    manifest = {
        "treatment": args.treatment_name,
        "run_dir_name": args.run_dir_name,
        "labels": len(labels),
        "shards": shards,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from talr_analysis_common import (
    load_jsonl,
    score_row,
    trace_by_id,
    write_json,
    write_jsonl,
)


def token_ids(trace: dict) -> list[int]:
    return [
        int(token["token_id"])
        for token in trace.get("tokens", [])
        if "token_id" in token
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        (args.replay_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    rows_out = []
    totals = {}
    for item in manifest:
        event_steps = json.loads(
            Path(item["event_steps"]).read_text(encoding="utf-8")
        )
        reference_rows = {
            str(row.get("id")): row
            for row in load_jsonl(Path(item["reference_dir"]) / "results.jsonl")
        }
        replay_rows = {
            str(row.get("id")): row
            for row in load_jsonl(Path(item["run_dir"]) / "results.jsonl")
        }
        reference_traces = trace_by_id(Path(item["reference_dir"]))
        replay_traces = trace_by_id(Path(item["run_dir"]))
        key = f"{item['dataset']}/{item['branch']}"
        totals[key] = {
            "samples": 0,
            "prefix_match": 0,
            "answer_changed": 0,
            "fixed": 0,
            "damaged": 0,
        }
        for sample_id, step in event_steps.items():
            if sample_id not in reference_rows or sample_id not in replay_rows:
                continue
            reference_score = score_row(reference_rows[sample_id])
            replay_score = score_row(replay_rows[sample_id])
            reference_ids = token_ids(reference_traces.get(sample_id, {}))
            replay_ids = token_ids(replay_traces.get(sample_id, {}))
            prefix_match = reference_ids[: step + 1] == replay_ids[: step + 1]
            answer_changed = reference_score["pred"] != replay_score["pred"]
            fixed = not reference_score["correct"] and replay_score["correct"]
            damaged = reference_score["correct"] and not replay_score["correct"]
            row = {
                "dataset": item["dataset"],
                "branch": item["branch"],
                "id": sample_id,
                "event_step": step,
                "prefix_match": prefix_match,
                "reference_pred": reference_score["pred"],
                "replay_pred": replay_score["pred"],
                "answer_changed": answer_changed,
                "fixed": fixed,
                "damaged": damaged,
            }
            rows_out.append(row)
            total = totals[key]
            total["samples"] += 1
            total["prefix_match"] += int(prefix_match)
            total["answer_changed"] += int(answer_changed)
            total["fixed"] += int(fixed)
            total["damaged"] += int(damaged)
    write_jsonl(args.replay_dir / "talr_event_replay.jsonl", rows_out)
    write_json(args.replay_dir / "talr_event_replay_summary.json", totals)
    lines = [
        "# TALR Single-Event Replay",
        "",
        "Only rows with `prefix_match` may support a causal event-level interpretation.",
        "",
        "| Dataset/branch | N | Prefix match | Answer changed | Fixed | Damaged | Net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(totals.items()):
        lines.append(
            f"| {key} | {value['samples']} | {value['prefix_match']} | "
            f"{value['answer_changed']} | {value['fixed']} | "
            f"{value['damaged']} | {value['fixed'] - value['damaged']:+d} |"
        )
    (args.replay_dir / "talr_event_replay_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

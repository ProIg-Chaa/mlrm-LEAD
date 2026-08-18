#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    shard_status = []
    for root in args.roots:
        for shard_dir in sorted(root.glob("shard_*")):
            labels = load_jsonl(shard_dir / "event_labels.jsonl")
            rows.extend(labels)
            shard_status.append(
                {
                    "path": str(shard_dir),
                    "complete": (shard_dir / "SHARD_COMPLETE").exists(),
                    "labels": len(labels),
                }
            )
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["original_id"],
            row["event_step"],
            row["treatment"],
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "intervention_atlas_v0b.jsonl", rows)

    totals = defaultdict(Counter)
    sample_outcomes: dict[tuple[str, str], dict] = {}
    event_pairs = defaultdict(dict)
    for row in rows:
        key = f"{row['dataset']}/{row['treatment']}"
        totals[key]["events"] += 1
        totals[key]["prefix_match"] += int(row["prefix_match"])
        totals[key]["fixed"] += int(row["fixed"])
        totals[key]["damaged"] += int(row["damaged"])
        totals[key]["answer_changed"] += int(row["answer_changed"])
        sample_key = (row["dataset"], row["original_id"])
        state = sample_outcomes.setdefault(
            sample_key,
            {
                "base_correct": bool(row["base_correct"]),
                "contracted_correct": False,
                "pure_correct": False,
            },
        )
        if row["treatment"] == "contracted_soft_l095":
            state["contracted_correct"] |= bool(row["treatment_correct"])
        elif row["treatment"] == "pure_soft_l100":
            state["pure_correct"] |= bool(row["treatment_correct"])
        event_pairs[row["event_id"]][row["treatment"]] = row

    pairwise = []
    for event_id, pair in sorted(event_pairs.items()):
        if set(pair) != {"contracted_soft_l095", "pure_soft_l100"}:
            continue
        contracted = pair["contracted_soft_l095"]
        pure = pair["pure_soft_l100"]
        pairwise.append(
            {
                "event_id": event_id,
                "dataset": contracted["dataset"],
                "original_id": contracted["original_id"],
                "event_step": contracted["event_step"],
                "event_type": contracted["event_type"],
                "contracted_correct": contracted["treatment_correct"],
                "pure_correct": pure["treatment_correct"],
                "contracted_better": (
                    bool(contracted["treatment_correct"])
                    and not bool(pure["treatment_correct"])
                ),
                "pure_better": (
                    bool(pure["treatment_correct"])
                    and not bool(contracted["treatment_correct"])
                ),
                "prediction_agreement": (
                    contracted["treatment_pred"] == pure["treatment_pred"]
                ),
            }
        )
    write_jsonl(args.output_dir / "contracted_vs_pure.jsonl", pairwise)

    oracle = defaultdict(Counter)
    for (dataset, _), state in sample_outcomes.items():
        oracle[dataset]["samples"] += 1
        oracle[dataset]["base_correct"] += int(state["base_correct"])
        oracle[dataset]["contracted_oracle_correct"] += int(
            state["base_correct"] or state["contracted_correct"]
        )
        oracle[dataset]["pure_oracle_correct"] += int(
            state["base_correct"] or state["pure_correct"]
        )
        oracle[dataset]["joint_oracle_correct"] += int(
            state["base_correct"]
            or state["contracted_correct"]
            or state["pure_correct"]
        )

    summary = {
        "shards": shard_status,
        "event_totals": {
            key: dict(value) for key, value in sorted(totals.items())
        },
        "oracle": {
            key: dict(value) for key, value in sorted(oracle.items())
        },
        "pairwise_events": len(pairwise),
    }
    (args.output_dir / "atlas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

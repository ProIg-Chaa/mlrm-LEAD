#!/usr/bin/env python3
"""Analyze outcome-agnostic image-source transplant branches."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from analyze_soft_state_transplant import gold_choice, prediction, read_jsonl


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--gaussian", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    by_event: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    event_meta = {}
    for row in manifest:
        event_id = str(row["event_id"])
        event_meta[event_id] = row
        by_event[event_id]["hard"] = {
            "pred": row.get("expected_hard_pred"),
            "correct": bool(row.get("expected_hard_correct")),
        }
        by_event[event_id]["true_image_l095"] = {
            "pred": row.get("expected_true_l095_pred"),
            "correct": bool(row.get("expected_true_l095_correct")),
        }

    inputs = [("core", args.core)]
    for path in args.gaussian:
        inputs.append((path.parent.name.replace("_1024", ""), path))
    for source_name, path in inputs:
        for row in read_jsonl(path):
            branch = str(row["branch"])
            if branch == "generic_noise":
                branch = source_name
            pred = prediction(row)
            gold = gold_choice(row)
            by_event[str(row["event_id"])][branch] = {
                "pred": pred,
                "correct": pred is not None and pred == gold,
            }

    branches = sorted({branch for items in by_event.values() for branch in items})

    def metrics(event_ids: list[str], branch: str) -> dict[str, Any]:
        values = [by_event[event_id][branch] for event_id in event_ids if branch in by_event[event_id]]
        return {
            "n": len(values),
            "correct": sum(item["correct"] for item in values),
            "accuracy": mean([float(item["correct"]) for item in values]),
            "failed": sum(item["pred"] is None for item in values),
        }

    def paired(event_ids: list[str], reference: str, branch: str) -> dict[str, Any]:
        pairs = [
            (by_event[event_id][reference], by_event[event_id][branch])
            for event_id in event_ids
            if reference in by_event[event_id] and branch in by_event[event_id]
        ]
        fixed = sum(not left["correct"] and right["correct"] for left, right in pairs)
        damaged = sum(left["correct"] and not right["correct"] for left, right in pairs)
        changed = sum(left["pred"] != right["pred"] for left, right in pairs)
        return {
            "n": len(pairs),
            "fixed": fixed,
            "damaged": damaged,
            "net": fixed - damaged,
            "prediction_changed": changed,
            "change_rate": changed / len(pairs) if pairs else None,
        }

    complete_ids = [
        event_id
        for event_id, items in by_event.items()
        if all(branch in items for branch in branches)
    ]
    result: dict[str, Any] = {
        "manifest_events": len(manifest),
        "complete_events_for_available_branches": len(complete_ids),
        "branches": branches,
        "overall": {},
        "by_dataset": {},
        "by_event_type": {},
        "true_effect_conditionals": {},
    }
    all_ids = list(by_event)
    for branch in branches:
        result["overall"][branch] = {
            **metrics(all_ids, branch),
            "vs_hard": paired(all_ids, "hard", branch) if branch != "hard" else None,
            "vs_true": paired(all_ids, "true_image_l095", branch)
            if branch != "true_image_l095"
            else None,
        }

    for field, output_key in (("dataset", "by_dataset"), ("event_type", "by_event_type")):
        values = sorted({str(row[field]) for row in manifest})
        for value in values:
            ids = [event_id for event_id, row in event_meta.items() if str(row[field]) == value]
            result[output_key][value] = {
                branch: {
                    **metrics(ids, branch),
                    "vs_hard": paired(ids, "hard", branch) if branch != "hard" else None,
                }
                for branch in branches
            }

    true_fixed = [
        event_id
        for event_id, items in by_event.items()
        if not items["hard"]["correct"] and items["true_image_l095"]["correct"]
    ]
    true_damaged = [
        event_id
        for event_id, items in by_event.items()
        if items["hard"]["correct"] and not items["true_image_l095"]["correct"]
    ]
    for branch in branches:
        if branch in {"hard", "true_image_l095"}:
            continue
        fixed_retained = sum(
            by_event[event_id].get(branch, {}).get("correct", False)
            for event_id in true_fixed
        )
        damage_repeated = sum(
            not by_event[event_id].get(branch, {}).get("correct", True)
            for event_id in true_damaged
        )
        result["true_effect_conditionals"][branch] = {
            "true_fixed_total": len(true_fixed),
            "true_fixed_retained": fixed_retained,
            "true_fixed_retention_rate": fixed_retained / len(true_fixed) if true_fixed else None,
            "true_damaged_total": len(true_damaged),
            "true_damage_repeated": damage_repeated,
            "true_damage_repeat_rate": damage_repeated / len(true_damaged) if true_damaged else None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

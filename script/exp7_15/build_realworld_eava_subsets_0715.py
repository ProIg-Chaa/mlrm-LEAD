#!/usr/bin/env python3
"""Build deterministic RealWorldQA subsets for the early visual-anchor study."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.evaluate_realworldqa_mcq import evaluate, load_jsonl  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_length(row: dict) -> int:
    value = row.get("output_tokens")
    if isinstance(value, int):
        return value
    return len((row.get("model_answer") or "").split())


def matched_correct_controls(
    hard_rows: list[dict], cot_enriched: list[dict], count: int
) -> list[int]:
    candidates = {
        int(row["id"]): row for row in cot_enriched if row["realworldqa_is_correct"]
    }
    targets = sorted(
        hard_rows,
        key=lambda row: (
            row.get("answer", ""),
            output_length(row["cot_result"]),
            int(row["id"]),
        ),
    )
    if not targets or not candidates:
        return []
    if count < len(targets):
        indices = [
            min(len(targets) - 1, math.floor((idx + 0.5) * len(targets) / count))
            for idx in range(count)
        ]
        targets = [targets[idx] for idx in indices]

    selected: list[int] = []
    used: set[int] = set()
    for target in targets:
        gold = target.get("answer", "").strip().upper()
        target_len = output_length(target["cot_result"])
        pool = [
            row
            for sample_id, row in candidates.items()
            if sample_id not in used and row.get("realworldqa_gold") == gold
        ]
        if not pool:
            pool = [row for sample_id, row in candidates.items() if sample_id not in used]
        if not pool:
            break
        chosen = min(
            pool,
            key=lambda row: (
                abs(output_length(row) - target_len),
                int(row["id"]),
            ),
        )
        sample_id = int(chosen["id"])
        selected.append(sample_id)
        used.add(sample_id)
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cot-results", type=Path, required=True)
    parser.add_argument("--initial-results", type=Path, required=True)
    parser.add_argument("--talr-results", type=Path, required=True)
    parser.add_argument("--hard-output", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--control-count", type=int, default=20)
    args = parser.parse_args()

    dataset_rows = load_jsonl(args.dataset)
    dataset_by_id = {int(row["id"]): row for row in dataset_rows}
    result_paths = {
        "cot": args.cot_results,
        "initial_transition": args.initial_results,
        "talr": args.talr_results,
    }
    enriched_by_method: dict[str, dict[int, dict]] = {}
    reports = {}
    for method, path in result_paths.items():
        report, enriched = evaluate(dataset_rows, load_jsonl(path))
        reports[method] = report
        enriched_by_method[method] = {int(row["id"]): row for row in enriched}

    common_ids = set(dataset_by_id)
    for rows in enriched_by_method.values():
        common_ids &= set(rows)
    hard_ids = sorted(
        sample_id
        for sample_id in common_ids
        if all(
            not enriched_by_method[method][sample_id]["realworldqa_is_correct"]
            for method in enriched_by_method
        )
    )
    hard_rows = []
    for sample_id in hard_ids:
        item = dict(dataset_by_id[sample_id])
        item["cot_result"] = enriched_by_method["cot"][sample_id]
        hard_rows.append(item)

    control_ids = matched_correct_controls(
        hard_rows, list(enriched_by_method["cot"].values()), args.control_count
    )
    hard_dataset_rows = [dataset_by_id[sample_id] for sample_id in hard_ids]
    control_dataset_rows = [dataset_by_id[sample_id] for sample_id in control_ids]
    write_jsonl(args.hard_output, hard_dataset_rows)
    write_jsonl(args.control_output, control_dataset_rows)

    manifest = {
        "dataset": str(args.dataset),
        "result_sources": {key: str(value) for key, value in result_paths.items()},
        "specialized_reports": reports,
        "hard_wrong_definition": "COT, initial_transition_only, and TALR all incorrect",
        "hard_wrong_count": len(hard_ids),
        "hard_wrong_ids": hard_ids,
        "hard_wrong_gold_distribution": dict(
            sorted(Counter(dataset_by_id[idx].get("answer", "") for idx in hard_ids).items())
        ),
        "control_definition": "COT-correct, greedily matched by gold answer and COT output length",
        "control_count": len(control_ids),
        "control_ids": control_ids,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if len(hard_ids) != 54:
        raise SystemExit(f"Expected 54 hard-wrong samples, found {len(hard_ids)}")
    if len(control_ids) != args.control_count:
        raise SystemExit(
            f"Expected {args.control_count} controls, found {len(control_ids)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

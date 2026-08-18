#!/usr/bin/env python3
"""Prepare non-overlapping follow-up samples for Atlas validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


SPECS = {
    "visulogic": ("visulogic.jsonl", 64, 128),
    "mmk12_math": ("mmk12_math.jsonl", 32, 64),
    "mmk12_physics": ("mmk12_physics.jsonl", 32, 64),
    "vmcbench": ("vmcbench_dev.jsonl", 32, 64),
    "pope_adversarial": ("pope_adversarial.jsonl", 32, 64),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def exclusion_ids(paths: list[Path]) -> dict[str, set[str]]:
    excluded: dict[str, set[str]] = {}
    for path in paths:
        for row in load_jsonl(path):
            dataset = str(row.get("_atlas_dataset") or "")
            original_id = str(
                row.get("_atlas_original_id")
                or str(row.get("id")).split("::", 1)[-1]
            )
            excluded.setdefault(dataset, set()).add(original_id)
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "script" / "exp7_23"))
    import prepare_intervention_atlas_v0b as base  # noqa: PLC0415

    excluded = exclusion_ids(args.exclude)
    selected: dict[str, list[dict]] = {}
    for dataset, (filename, new_count, candidate_count) in SPECS.items():
        source = base.load_jsonl(args.data_root / filename)
        if dataset == "visulogic":
            source = source[:300]
        candidates = base.stratified_sample(
            source, candidate_count, args.seed, dataset
        )
        fresh = [
            row
            for row in candidates
            if str(row.get("id")) not in excluded.get(dataset, set())
        ][:new_count]
        if len(fresh) != new_count:
            # Continue through the full deterministic order if the initial
            # candidate window overlaps more than expected.
            candidates = base.stratified_sample(
                source, len(source), args.seed, dataset
            )
            fresh = [
                row
                for row in candidates
                if str(row.get("id")) not in excluded.get(dataset, set())
            ][:new_count]
        if len(fresh) != new_count:
            raise RuntimeError(
                f"Only {len(fresh)} fresh samples for {dataset}, need {new_count}"
            )
        enriched = []
        for row in fresh:
            item = dict(row)
            original_id = str(row.get("id"))
            item["id"] = f"{dataset}::{original_id}"
            item["image"] = base.resolve_image(
                row.get("image"), args.data_root, None
            )
            item["_atlas_dataset"] = dataset
            item["_atlas_original_id"] = original_id
            enriched.append(item)
        selected[dataset] = enriched

    shards = {
        0: selected["visulogic"],
        1: selected["mmk12_math"],
        2: selected["mmk12_physics"],
        3: [
            row
            for pair in zip(
                selected["vmcbench"], selected["pope_adversarial"]
            )
            for row in pair
        ],
    }
    all_rows = [row for dataset in SPECS for row in selected[dataset]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "selected_all.jsonl", all_rows)

    manifest = {
        "seed": args.seed,
        "exclusion_files": [str(path) for path in args.exclude],
        "datasets": {
            dataset: {
                "source": str(args.data_root / SPECS[dataset][0]),
                "selected": len(rows),
                "excluded_before_selection": len(excluded.get(dataset, set())),
            }
            for dataset, rows in selected.items()
        },
        "shards": [],
    }
    all_ids = [row["id"] for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Duplicate IDs in follow-up selection")
    for index, rows in shards.items():
        path = args.output_dir / "shards" / f"shard_{index}.jsonl"
        write_jsonl(path, rows)
        manifest["shards"].append(
            {
                "index": index,
                "path": str(path),
                "samples": len(rows),
                "dataset_counts": dict(
                    Counter(row["_atlas_dataset"] for row in rows)
                ),
            }
        )
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

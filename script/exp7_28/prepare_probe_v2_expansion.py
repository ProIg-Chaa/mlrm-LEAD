#!/usr/bin/env python3
"""Prepare a strictly non-overlapping external set for Probe V2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


DATASETS = {
    "vstar": ("vstar.jsonl", None),
    "mmvp": ("mmvp.jsonl", None),
    "realworldqa": ("realworldqa_fixed_mcq_random200_seed42.jsonl", None),
    "visulogic": ("visulogic.jsonl", 300),
    "mmk12_math": ("mmk12_math.jsonl", None),
    "mmk12_physics": ("mmk12_physics.jsonl", None),
    "vmcbench": ("vmcbench_dev.jsonl", None),
    "pope_adversarial": ("pope_adversarial.jsonl", None),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_exclusions(paths: list[Path]) -> dict[str, set[str]]:
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
    parser.add_argument("--per-dataset", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "script" / "exp7_23"))
    import prepare_intervention_atlas_v0b as base  # noqa: PLC0415

    excluded = collect_exclusions(args.exclude)
    selected: dict[str, list[dict]] = {}
    for dataset, (filename, limit) in DATASETS.items():
        source = load_jsonl(args.data_root / filename)
        if limit is not None:
            source = source[:limit]
        ordered = base.stratified_sample(source, len(source), args.seed, dataset)
        fresh = [
            row
            for row in ordered
            if str(row.get("id")) not in excluded.get(dataset, set())
        ][: args.per_dataset]
        if len(fresh) != args.per_dataset:
            raise RuntimeError(
                f"{dataset}: only {len(fresh)} fresh samples; "
                f"requested {args.per_dataset}"
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

    interleaved = []
    for offset in range(args.per_dataset):
        for dataset in DATASETS:
            interleaved.append(selected[dataset][offset])
    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for index, row in enumerate(interleaved):
        shards[index % args.num_shards].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for dataset in DATASETS for row in selected[dataset]]
    all_ids = [row["id"] for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Duplicate IDs in expansion selection")
    write_jsonl(args.output_dir / "selected_all.jsonl", all_rows)

    manifest = {
        "purpose": "frozen_probe_v2_external_expansion",
        "seed": args.seed,
        "per_dataset": args.per_dataset,
        "num_shards": args.num_shards,
        "total_samples": len(all_rows),
        "exclusion_files": [str(path.resolve()) for path in args.exclude],
        "datasets": {
            dataset: {
                "source": str((args.data_root / filename).resolve()),
                "selected": len(selected[dataset]),
                "excluded_before_selection": len(excluded.get(dataset, set())),
            }
            for dataset, (filename, _) in DATASETS.items()
        },
        "shards": [],
    }
    for index, rows in enumerate(shards):
        path = args.output_dir / "shards" / f"shard_{index}.jsonl"
        write_jsonl(path, rows)
        manifest["shards"].append(
            {
                "index": index,
                "path": str(path.resolve()),
                "samples": len(rows),
                "dataset_counts": dict(
                    sorted(Counter(row["_atlas_dataset"] for row in rows).items())
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

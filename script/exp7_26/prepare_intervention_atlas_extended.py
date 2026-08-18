#!/usr/bin/env python3
"""Prepare a stratified multi-dataset selection for the extended Atlas."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASETS = {
    "vmcbench": "vmcbench_dev.jsonl",
    "pope_adversarial": "pope_adversarial.jsonl",
    "mmk12_math": "mmk12_math.jsonl",
    "mmk12_physics": "mmk12_physics.jsonl",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Optional NAME=FILENAME override; may be repeated.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "script" / "exp7_23"))
    import prepare_intervention_atlas_v0b as base  # noqa: PLC0415

    datasets = dict(DEFAULT_DATASETS)
    if args.dataset:
        datasets = {}
        for spec in args.dataset:
            name, filename = spec.split("=", 1)
            datasets[name] = filename

    selected_by_dataset: dict[str, list[dict]] = {}
    image_failures = []
    for dataset, filename in datasets.items():
        source = base.load_jsonl(args.data_root / filename)
        selected = base.stratified_sample(
            source, args.per_dataset, args.seed, dataset
        )
        enriched = []
        for row in selected:
            item = dict(row)
            original_id = str(row.get("id"))
            item["id"] = f"{dataset}::{original_id}"
            try:
                item["image"] = base.resolve_image(
                    row.get("image"), args.data_root, args.image_root
                )
            except (FileNotFoundError, ValueError) as exc:
                image_failures.append(
                    {"dataset": dataset, "id": original_id, "error": str(exc)}
                )
                continue
            item["_atlas_dataset"] = dataset
            item["_atlas_original_id"] = original_id
            enriched.append(item)
        selected_by_dataset[dataset] = enriched

    if image_failures:
        raise RuntimeError(
            f"Image resolution failed for {len(image_failures)} rows: "
            f"{image_failures[:3]}"
        )
    for dataset, rows in selected_by_dataset.items():
        if len(rows) != min(args.per_dataset, len(base.load_jsonl(args.data_root / datasets[dataset]))):
            raise RuntimeError(f"Unexpected selected count for {dataset}: {len(rows)}")

    interleaved = []
    max_count = max(len(rows) for rows in selected_by_dataset.values())
    for index in range(max_count):
        for dataset in datasets:
            rows = selected_by_dataset[dataset]
            if index < len(rows):
                interleaved.append(rows[index])

    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for index, row in enumerate(interleaved):
        shards[index % args.num_shards].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "selected_all.jsonl",
        [row for dataset in datasets for row in selected_by_dataset[dataset]],
    )
    manifest = {
        "seed": args.seed,
        "per_dataset": args.per_dataset,
        "num_shards": args.num_shards,
        "datasets": {
            dataset: {
                "source": str(args.data_root / filename),
                "selected": len(selected_by_dataset[dataset]),
            }
            for dataset, filename in datasets.items()
        },
        "shards": [],
    }
    for shard_index, rows in enumerate(shards):
        shard_path = args.output_dir / "shards" / f"shard_{shard_index}.jsonl"
        write_jsonl(shard_path, rows)
        counts = defaultdict(int)
        for row in rows:
            counts[row["_atlas_dataset"]] += 1
        manifest["shards"].append(
            {
                "index": shard_index,
                "path": str(shard_path),
                "samples": len(rows),
                "dataset_counts": dict(counts),
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DATASETS = {
    "vstar": "vstar.jsonl",
    "mmvp": "mmvp.jsonl",
    "realworldqa": "realworldqa_fixed_mcq_random200_seed42.jsonl",
    "visulogic": "visulogic.jsonl",
}
LIMITS = {"visulogic": 300}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image(value, data_root: Path, image_root: Path) -> str | list[str]:
    def resolve_one(raw: str) -> str:
        path = Path(raw)
        candidates = [path] if path.is_absolute() else [
            data_root / path,
            data_root.parent / path,
        ]
        normalized = raw.replace("\\", "/")
        marker = "datasets/mlrm-LEAD/"
        suffix = normalized.split(marker, 1)[-1] if marker in normalized else raw
        candidates.append(image_root / suffix)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(f"Missing image: {raw}")

    if isinstance(value, list):
        return [resolve_one(str(item)) for item in value]
    return resolve_one(str(value))


def atlas_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in load_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--full-reuse", type=Path, required=True)
    parser.add_argument("--partial-reuse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=12)
    args = parser.parse_args()

    fully_reused = atlas_ids(args.full_reuse)
    partially_reused = {
        sample_id
        for sample_id in atlas_ids(args.partial_reuse)
        if sample_id.startswith("visulogic::")
    }
    selected_by_dataset: dict[str, list[dict]] = {}
    dataset_stats = {}

    for dataset, filename in DATASETS.items():
        source = load_jsonl(args.data_root / filename)
        source = source[: LIMITS.get(dataset, len(source))]
        pending = []
        full_count = 0
        partial_count = 0
        for row in source:
            original_id = str(row["id"])
            atlas_id = f"{dataset}::{original_id}"
            if atlas_id in fully_reused:
                full_count += 1
                continue
            if atlas_id in partially_reused:
                partial_count += 1
                continue
            item = dict(row)
            item["id"] = atlas_id
            item["image"] = resolve_image(
                row.get("image"), args.data_root, args.image_root
            )
            item["_atlas_dataset"] = dataset
            item["_atlas_original_id"] = original_id
            pending.append(item)
        selected_by_dataset[dataset] = pending
        dataset_stats[dataset] = {
            "source_samples": len(source),
            "fully_reused": full_count,
            "partially_reused": partial_count,
            "pending_full_matrix": len(pending),
        }

    interleaved = []
    max_count = max(len(rows) for rows in selected_by_dataset.values())
    for index in range(max_count):
        for dataset in DATASETS:
            rows = selected_by_dataset[dataset]
            if index < len(rows):
                interleaved.append(rows[index])

    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for index, row in enumerate(interleaved):
        shards[index % args.num_shards].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "pending_all.jsonl", interleaved)
    manifest = {
        "matrix": {
            "fixed_steps": [1, 2, 4, 8, 16, 32],
            "adaptive_events": ["entropy_top1", "random_control"],
            "soft_actions": ["contracted_soft_l095", "pure_soft_l100"],
        },
        "full_reuse_file": str(args.full_reuse),
        "partial_reuse_file": str(args.partial_reuse),
        "datasets": dataset_stats,
        "pending_samples": len(interleaved),
        "num_shards": args.num_shards,
        "shards": [],
    }
    for index, rows in enumerate(shards):
        path = args.output_dir / "shards" / f"shard_{index}.jsonl"
        write_jsonl(path, rows)
        counts = defaultdict(int)
        for row in rows:
            counts[row["_atlas_dataset"]] += 1
        manifest["shards"].append(
            {
                "index": index,
                "path": str(path),
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

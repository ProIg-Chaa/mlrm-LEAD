#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path


DATASETS = {
    "vstar": "vstar.jsonl",
    "mmvp": "mmvp.jsonl",
    "realworldqa": "realworldqa_fixed_mcq_random200_seed42.jsonl",
    "visulogic": "visulogic.jsonl",
}
DATASET_LIMITS = {"visulogic": 300}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image(value, data_root: Path, image_root: Path | None):
    def resolve_one(raw: str) -> str:
        path = Path(raw)
        candidates = [path] if path.is_absolute() else [
            data_root / path,
            data_root.parent / path,
        ]
        if image_root is not None:
            marker = "datasets/mlrm-LEAD/"
            normalized = raw.replace("\\", "/")
            suffix = normalized.split(marker, 1)[-1] if marker in normalized else raw
            candidates.append(image_root / suffix)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(f"Missing image: {raw}")

    if isinstance(value, list):
        return [resolve_one(str(item)) for item in value]
    if value is None:
        raise ValueError("Sample has no image field")
    return resolve_one(str(value))


def stratified_sample(
    rows: list[dict], count: int, seed: int, dataset: str
) -> list[dict]:
    rng = random.Random(f"{seed}:{dataset}")
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("subtopic") or row.get("subject") or "unknown"),
            str(row.get("answer") or "unknown").strip().upper(),
        )
        buckets[key].append(row)
    queues: list[deque[dict]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        rng.shuffle(bucket)
        queues.append(deque(bucket))
    selected: list[dict] = []
    while queues and len(selected) < min(count, len(rows)):
        next_round = []
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected_by_dataset: dict[str, list[dict]] = {}
    for dataset, filename in DATASETS.items():
        source = load_jsonl(args.data_root / filename)
        source = source[: DATASET_LIMITS.get(dataset, len(source))]
        selected = stratified_sample(
            source, args.per_dataset, args.seed, dataset
        )
        enriched = []
        for row in selected:
            item = dict(row)
            original_id = str(row.get("id"))
            item["id"] = f"{dataset}::{original_id}"
            item["image"] = resolve_image(
                row.get("image"), args.data_root, args.image_root
            )
            item["_atlas_dataset"] = dataset
            item["_atlas_original_id"] = original_id
            enriched.append(item)
        selected_by_dataset[dataset] = enriched

    interleaved: list[dict] = []
    for index in range(args.per_dataset):
        for dataset in DATASETS:
            rows = selected_by_dataset[dataset]
            if index < len(rows):
                interleaved.append(rows[index])

    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for index, row in enumerate(interleaved):
        shards[index % args.num_shards].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "selected_all.jsonl",
        [row for dataset in DATASETS for row in selected_by_dataset[dataset]],
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
            for dataset, filename in DATASETS.items()
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



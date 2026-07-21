#!/usr/bin/env python3
"""Create deterministic, label-balanced POPE subsets for quick evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


DATA_DIR = Path("/root/gushuo/proj/mlrm-LEAD/data")
VARIANTS = ("adversarial", "popular", "random")
SEED = 42
PER_LABEL = 250


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = {
        "sampling": "answer-label-stratified random sample; restored source order",
        "seed": SEED,
        "per_label": PER_LABEL,
        "variants": {},
    }

    for offset, variant in enumerate(VARIANTS):
        active = DATA_DIR / f"pope_{variant}.jsonl"
        full = DATA_DIR / f"pope_{variant}_full3000.jsonl"
        subset = DATA_DIR / f"pope_{variant}_balanced500_seed42.jsonl"

        if not full.exists():
            shutil.copy2(active, full)
        rows = read_jsonl(full)
        if len(rows) != 3000:
            raise RuntimeError(f"{full} has {len(rows)} rows, expected 3000")

        by_label: dict[str, list[int]] = {"A": [], "B": []}
        for index, row in enumerate(rows):
            answer = str(row.get("answer", "")).strip().upper()
            if answer not in by_label:
                raise RuntimeError(f"unexpected answer {answer!r} in {full}")
            by_label[answer].append(index)

        rng = random.Random(SEED + offset)
        selected = set()
        for label in ("A", "B"):
            selected.update(rng.sample(by_label[label], PER_LABEL))
        sampled = [row for index, row in enumerate(rows) if index in selected]

        write_jsonl_atomic(subset, sampled)
        write_jsonl_atomic(active, sampled)

        counts = Counter(str(row["answer"]).upper() for row in sampled)
        missing_images = sum(
            1 for row in sampled if not Path(str(row.get("image", ""))).is_file()
        )
        manifest["variants"][variant] = {
            "full_path": str(full),
            "active_path": str(active),
            "subset_path": str(subset),
            "rows": len(sampled),
            "answer_counts": dict(sorted(counts.items())),
            "unique_images": len({row.get("image_source") for row in sampled}),
            "missing_images": missing_images,
            "sha256": sha256(subset),
        }

    manifest_path = DATA_DIR / "pope_balanced500_seed42_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

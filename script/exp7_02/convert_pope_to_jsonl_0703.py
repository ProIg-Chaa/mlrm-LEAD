#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


SPLITS = {
    "pope_adversarial": "Full/adversarial-00000-of-00001.parquet",
    "pope_popular": "Full/popular-00000-of-00001.parquet",
    "pope_random": "Full/random-00000-of-00001.parquet",
}


def normalize_answer(answer: str) -> str:
    value = (answer or "").strip().lower()
    if value == "yes":
        return "A"
    if value == "no":
        return "B"
    raise ValueError(f"unexpected POPE answer: {answer!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        default="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE",
    )
    parser.add_argument(
        "--project_root",
        default="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD",
    )
    parser.add_argument(
        "--image_root",
        default="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/pope",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    project_root = Path(args.project_root)
    image_root = Path(args.image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    (project_root / "data").mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, rel in SPLITS.items():
        parquet_path = source_root / rel
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        out_jsonl = project_root / "data" / f"{name}.jsonl"
        split_image_dir = image_root / name
        split_image_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                image = row.get("image") or {}
                image_bytes = image.get("bytes")
                if not image_bytes:
                    raise ValueError(f"missing image bytes in {name} row {row.get('id')}")
                image_path = split_image_dir / f"{int(row['id']):06d}.jpg"
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)

                item = {
                    "id": f"{name}_{row['id']}",
                    "question": row["question"],
                    "options": "(A) yes\n(B) no",
                    "answer": normalize_answer(row["answer"]),
                    "answer_text": row["answer"].strip().lower(),
                    "image": str(image_path),
                    "subtopic": row.get("category", name.replace("pope_", "")),
                    "benchmark": "POPE",
                    "pope_category": row.get("category"),
                    "image_source": row.get("image_source"),
                    "question_id": row.get("question_id"),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

        summary[name] = {
            "rows": written,
            "jsonl": str(out_jsonl),
            "image_dir": str(split_image_dir),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

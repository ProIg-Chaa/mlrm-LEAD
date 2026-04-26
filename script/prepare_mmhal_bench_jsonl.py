#!/usr/bin/env python3
"""Convert downloaded MMHal-Bench files to this project's JSONL format."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE_ROOT = Path(
    "/share/home/wangzixu/liudinghao/gushuo/datasets/sources/Shengcao1006__MMHal-Bench"
)
OUTPUT_PATH = Path(
    "/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/data/mmhal_bench.jsonl"
)


def main() -> int:
    template_path = SOURCE_ROOT / "response_template.json"
    image_dir = SOURCE_ROOT / "images"
    rows = json.loads(template_path.read_text(encoding="utf-8"))

    converted = []
    missing_images = []
    for idx, row in enumerate(rows):
        image_name = Path(row["image_src"]).name
        image_path = image_dir / image_name
        if not image_path.is_file():
            missing_images.append(str(image_path))

        converted.append(
            {
                "id": idx,
                "image": str(image_path),
                "question": row.get("question", ""),
                "answer": row.get("gt_answer", ""),
                "gt_answer": row.get("gt_answer", ""),
                "image_content": row.get("image_content", []),
                "image_id": row.get("image_id", ""),
                "image_src": row.get("image_src", ""),
                "question_type": row.get("question_type", ""),
                "question_topic": row.get("question_topic", ""),
                "subtopic": row.get("question_type", "unknown"),
                "difficulty": 1,
                "language": "english",
                "benchmark": "MMHal-Bench",
            }
        )

    if missing_images:
        raise FileNotFoundError(
            "Missing MMHal-Bench images:\n" + "\n".join(missing_images[:20])
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(converted)} samples to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

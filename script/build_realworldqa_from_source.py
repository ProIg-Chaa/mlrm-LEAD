#!/usr/bin/env python3
"""Build RealWorldQA jsonl from the local HuggingFace parquet source."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


INSTRUCTION_RE = re.compile(
    r"\n?Please answer directly with (?:only the letter of the correct option and nothing else|a single word or number)\.?",
    re.I,
)
OPTION_RE = re.compile(
    r"(?ms)(?:^|\n)\s*([A-D])\.\s*(.*?)(?=(?:\n\s*[A-D]\.\s)|\Z)"
)


def split_question_options(text: str) -> tuple[str, str]:
    text = INSTRUCTION_RE.sub("", text or "").strip()
    matches = list(OPTION_RE.finditer(text))
    if not matches:
        return text, ""
    question = text[: matches[0].start()].strip()
    options = []
    for match in matches:
        options.append(f"{match.group(1).upper()}. {match.group(2).strip()}")
    return question, "\n".join(options)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--mcq_output_jsonl", required=True)
    parser.add_argument("--mcq_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    image_dir = Path(args.image_dir)
    frames = [
        pd.read_parquet(path, columns=["question", "answer"])
        for path in sorted((source_dir / "data").glob("*.parquet"))
    ]
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for idx, row in df.iterrows():
        question, options = split_question_options(str(row["question"]))
        image = image_dir / f"{idx:06d}.webp"
        rows.append(
            {
                "id": int(idx),
                "image": str(image),
                "question": question,
                "options": options,
                "answer": str(row["answer"]).strip(),
            }
        )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mcq_rows = [
        row
        for row in rows
        if row["options"] and re.fullmatch(r"[A-Da-d]", row["answer"] or "")
    ]
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(mcq_rows, min(args.mcq_n, len(mcq_rows))), key=lambda x: x["id"])
    mcq_output_jsonl = Path(args.mcq_output_jsonl)
    mcq_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with mcq_output_jsonl.open("w", encoding="utf-8") as f:
        for row in selected:
            row = dict(row)
            row["answer"] = row["answer"].upper()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing = sum(not Path(row["image"]).exists() for row in rows)
    answer_dist: dict[str, int] = {}
    for row in selected:
        answer = row["answer"].upper()
        answer_dist[answer] = answer_dist.get(answer, 0) + 1
    print(
        json.dumps(
            {
                "total": len(rows),
                "mcq_total": len(mcq_rows),
                "mcq_selected": len(selected),
                "missing_images": missing,
                "mcq_answer_dist": dict(sorted(answer_dist.items())),
                "output_jsonl": str(output_jsonl),
                "mcq_output_jsonl": str(mcq_output_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

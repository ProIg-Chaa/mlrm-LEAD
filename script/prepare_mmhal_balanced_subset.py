#!/usr/bin/env python3
"""Prepare a balanced MMHal-Bench subset with a fixed number per type."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/mmhal_bench.jsonl",
        help="Source MMHal JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="data/mmhal_bench_balanced_2pertype.jsonl",
        help="Output subset JSONL path.",
    )
    parser.add_argument(
        "--per_type",
        type=int,
        default=2,
        help="Number of samples to keep for each question_type.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("question_type", "unknown")].append(row)

    selected = []
    for question_type in sorted(buckets):
        bucket = sorted(buckets[question_type], key=lambda row: row.get("id", 0))
        selected.extend(bucket[: args.per_type])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        question_type: len(
            [row for row in selected if row.get("question_type") == question_type]
        )
        for question_type in sorted(buckets)
    }
    print(f"Wrote {len(selected)} samples to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

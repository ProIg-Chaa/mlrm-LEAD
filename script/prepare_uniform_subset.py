#!/usr/bin/env python3
"""Prepare a uniformly spaced subset from a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_uniform(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= len(rows):
        return rows

    last = len(rows) - 1
    indices = {
        round(i * last / (limit - 1))
        for i in range(limit)
    } if limit > 1 else {0}
    ordered = sorted(indices)

    if len(ordered) != limit:
        selected = []
        used = set()
        cursor = 0.0
        step = len(rows) / limit
        while len(selected) < limit:
            idx = min(int(round(cursor)), len(rows) - 1)
            while idx in used and idx + 1 < len(rows):
                idx += 1
            if idx in used:
                idx = max(i for i in range(len(rows)) if i not in used)
            used.add(idx)
            selected.append(rows[idx])
            cursor += step
        return selected

    return [rows[idx] for idx in ordered]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", required=True, type=int)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    selected = select_uniform(rows, args.limit)
    save_jsonl(selected, Path(args.output))
    print(f"Wrote {len(selected)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict[int, str]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[int(row["id"])] = row.get("model_answer") or ""
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = read(Path(args.expected))
    actual = read(Path(args.actual))
    ids = sorted(set(expected) & set(actual))
    if args.limit is not None:
        ids = ids[: args.limit]
    mismatched = [sample_id for sample_id in ids if expected[sample_id] != actual[sample_id]]
    report = {"total": len(ids), "matched": len(ids) - len(mismatched), "mismatched_ids": mismatched}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())

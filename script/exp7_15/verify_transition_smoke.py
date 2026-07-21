#!/usr/bin/env python3
"""Verify that LEAD's force-normal wrapper is token-identical to greedy COT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_trace(path: Path) -> dict[int, list[int]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["id"])] = [int(token["token_id"]) for token in row.get("tokens", [])]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", required=True)
    parser.add_argument("--force-normal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cot = load_trace(Path(args.cot))
    normal = load_trace(Path(args.force_normal))
    ids = sorted(set(cot) & set(normal))
    mismatches = [sample_id for sample_id in ids if cot[sample_id] != normal[sample_id]]
    report = {"total": len(ids), "matched": len(ids) - len(mismatches), "mismatched_ids": mismatches}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise SystemExit(f"force-normal differs from COT for {len(mismatches)}/{len(ids)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

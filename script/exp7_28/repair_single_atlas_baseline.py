#!/usr/bin/env python3
"""Select failed IDs and merge successful single-process repair rows."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


FILES = ("results.jsonl", "token_entropy.jsonl", "token_entropy_full.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--failed-baseline", type=Path, required=True)
    parser.add_argument("--repair-baseline", type=Path)
    parser.add_argument("--selected-output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    failed_results = read_jsonl(args.failed_baseline / "results.jsonl")
    failed_ids = {
        str(row["id"]) for row in failed_results if row.get("error_type")
    }
    if args.prepare:
        if not failed_ids:
            raise RuntimeError("No failed IDs to repair")
        selected = [
            row
            for row in read_jsonl(args.selection)
            if str(row["id"]) in failed_ids
        ]
        if {str(row["id"]) for row in selected} != failed_ids:
            raise RuntimeError("Failed IDs missing from source selection")
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.selected_output, selected)
        print(json.dumps({"failed_ids": sorted(failed_ids)}, indent=2))

    if args.merge:
        if args.repair_baseline is None:
            raise RuntimeError("--repair-baseline is required for --merge")
        repair_results = read_jsonl(args.repair_baseline / "results.jsonl")
        if any(row.get("error_type") for row in repair_results):
            raise RuntimeError("Repair output still contains runtime errors")
        repair_ids = {str(row["id"]) for row in repair_results}
        if repair_ids != failed_ids:
            raise RuntimeError(
                f"Repair ID mismatch: expected={failed_ids}, actual={repair_ids}"
            )
        backup = args.failed_baseline.with_name(
            args.failed_baseline.name + f".pre_repair.{int(time.time())}"
        )
        shutil.copytree(args.failed_baseline, backup)
        for filename in FILES:
            original = read_jsonl(args.failed_baseline / filename)
            replacements = {
                str(row["id"]): row
                for row in read_jsonl(args.repair_baseline / filename)
            }
            merged = [
                replacements.get(str(row["id"]), row) for row in original
            ]
            if len(merged) != len(original):
                raise RuntimeError(f"Row count changed for {filename}")
            write_jsonl(args.failed_baseline / filename, merged)
        manifest = {
            "failed_ids": sorted(failed_ids),
            "repair_dir": str(args.repair_baseline.resolve()),
            "backup_dir": str(backup.resolve()),
            "files": list(FILES),
        }
        (args.failed_baseline / "repair_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare a VStar wrong-only subset from a completed result file."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_extract_fn(evaluator_path: Path):
    spec = importlib.util.spec_from_file_location("lead_evaluator_only", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.extract_mcq_answer


def load_dataset(dataset_path: Path) -> list[dict]:
    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include_error_samples",
        action="store_true",
        help="Include samples with error_type/model_answer=None in the wrong subset",
    )
    args = parser.parse_args()

    dataset_rows = load_dataset(Path(args.dataset))
    dataset_by_id = {int(row["id"]): row for row in dataset_rows}
    extract_mcq_answer = load_extract_fn(Path(args.evaluator))

    wrong_rows = []
    skipped_missing = 0
    total = 0
    wrong = 0
    error_rows = 0

    with Path(args.results).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            sample_id = int(row["id"])
            source = dataset_by_id.get(sample_id)
            if source is None:
                skipped_missing += 1
                continue

            if row.get("error_type") or row.get("model_answer") is None:
                error_rows += 1
                if not args.include_error_samples:
                    continue
                wrong_rows.append(source)
                wrong += 1
                continue

            pred = extract_mcq_answer(row.get("model_answer"))
            is_correct = pred is not None and pred == row.get("answer")
            if not is_correct:
                wrong_rows.append(source)
                wrong += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in wrong_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "total_results": total,
                "wrong_subset_size": wrong,
                "error_rows": error_rows,
                "include_error_samples": args.include_error_samples,
                "skipped_missing_dataset_ids": skipped_missing,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

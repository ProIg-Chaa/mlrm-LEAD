#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "script/exp7_17"))
    from talr_analysis_common import load_jsonl, score_row, write_json, write_jsonl

    rows = load_jsonl(args.run_dir / "results.jsonl")
    evaluated = [
        {"id": str(row.get("id")), **score_row(row)}
        for row in rows
    ]
    report = {
        "accuracy": sum(item["correct"] is True for item in evaluated) / len(evaluated),
        "correct": sum(item["correct"] is True for item in evaluated),
        "total": len(evaluated),
        "failed_extraction": sum(item["failed_extraction"] for item in evaluated),
        "runtime_errors": sum(item["runtime_error"] for item in evaluated),
        "evaluator": "talr_analysis_common.corrected_last_answer",
    }
    write_json(args.run_dir / "corrected_eval_report.json", report)
    write_jsonl(args.run_dir / "corrected_eval_rows.jsonl", evaluated)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

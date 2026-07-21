#!/usr/bin/env python3
"""Fail fast when a bridge smoke does not actually use the intended handoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-source", required=True, choices=["soft", "hard"])
    parser.add_argument("--expected-anchor", required=True, choices=["end_thinking", "generated_token"])
    args = parser.parse_args()
    run = Path(args.run_dir)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for key, expected in {
        "lead_force_initial_transition_step1": True,
        "lead_transition_source": args.expected_source,
        "lead_transition_anchor": args.expected_anchor,
    }.items():
        if config.get(key) != expected:
            raise SystemExit(f"config mismatch: {key}={config.get(key)!r}, expected {expected!r}")
    rows = read_jsonl(run / "results.jsonl")
    traces = {str(row["id"]): row.get("tokens", []) for row in read_jsonl(run / "token_entropy_full.jsonl")}
    if len(rows) != 2 or len(traces) != 2:
        raise SystemExit(f"expected two result/trace rows, got {len(rows)}/{len(traces)}")
    for row in rows:
        if row.get("error_type"):
            raise SystemExit(f"runtime error for {row.get('id')}: {row['error_type']}")
        step1 = [token for token in traces[str(row["id"])] if token.get("step") == 1]
        if len(step1) != 1:
            raise SystemExit(f"missing/duplicate step-1 trace for {row['id']}")
        token = step1[0]
        checks = {
            "to_normal": True,
            "forced_transition_step1": True,
            "lead_transition_source": args.expected_source,
            "lead_transition_anchor": args.expected_anchor,
        }
        for key, expected in checks.items():
            if token.get(key) != expected:
                raise SystemExit(f"trace mismatch {row['id']} {key}={token.get(key)!r}, expected {expected!r}")
    print(json.dumps({"status": "ok", "run": str(run), "source": args.expected_source, "anchor": args.expected_anchor}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

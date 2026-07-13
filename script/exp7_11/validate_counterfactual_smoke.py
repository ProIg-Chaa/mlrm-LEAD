#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.replay_dir / "counterfactual_branches.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = []
    for row in rows:
        probe = row.get("forced_answer_probe") or {}
        if not row.get("prefix_match"):
            failures.append({"id": row.get("id"), "reason": "prefix_mismatch"})
        if row.get("branch") == "actual" and not row.get("trajectory_match"):
            failures.append({"id": row.get("id"), "reason": "instrumentation_changed_trajectory"})
        if not probe.get("available"):
            failures.append({"id": row.get("id"), "reason": "probe_unavailable", "detail": probe})
        if not row.get("event_geometry") or row["event_geometry"].get("hard_emb_norm") is None:
            failures.append({"id": row.get("id"), "reason": "geometry_missing"})
    report = {"passed": bool(rows) and not failures, "rows": len(rows), "failures": failures}
    (args.replay_dir / "smoke_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())

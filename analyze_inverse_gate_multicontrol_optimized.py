#!/usr/bin/env python3
"""Apply the locked inverse gate to full TALR versus residual predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from inverse_gate_policy import derive_features


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("shard_*/optimized_gate_probe_results.jsonl"))
    rows = [row for path in paths for row in read_jsonl(path)]
    if not rows:
        raise RuntimeError("No optimized gate rows")
    failures = [row for row in rows if row.get("error_type")]
    if failures:
        raise RuntimeError("Cannot analyze %d failed rows" % len(failures))
    keys = [(str(row["dataset"]), str(row["id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate dataset/sample IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched = []
    for row in rows:
        features = derive_features(row["probes"]) if row["probe_eligible"] else {"reject_action": False}
        reject = bool(features["reject_action"])
        chosen = "baseline" if reject else "action"
        enriched.append(dict(row, gate_features=features, gate_reject=reject, chosen_branch=chosen,
                             chosen_pred=row["baseline_pred"] if reject else row["action_pred"],
                             chosen_correct=bool(row["baseline_correct"] if reject else row["action_correct"])))

    reports: Dict[str, Any] = {}
    grouped = defaultdict(list)
    for row in enriched:
        grouped[str(row["dataset"])].append(row)
    for dataset, items in sorted(grouped.items()):
        total = len(items)
        eligible = [row for row in items if row["probe_eligible"]]
        baseline_correct = sum(bool(row["baseline_correct"]) for row in items)
        action_correct = sum(bool(row["action_correct"]) for row in items)
        gated_correct = sum(bool(row["chosen_correct"]) for row in items)
        fixed = [row for row in items if (not row["baseline_correct"]) and row["action_correct"]]
        damaged = [row for row in items if row["baseline_correct"] and (not row["action_correct"])]
        report = {
            "total": total, "probed": len(eligible), "rejected": sum(row["gate_reject"] for row in items),
            "coverage": sum(row["gate_reject"] for row in items) / total,
            "talr_correct": baseline_correct, "talr_accuracy": baseline_correct / total,
            "ungated_correct": action_correct, "ungated_accuracy": action_correct / total,
            "gated_correct": gated_correct, "gated_accuracy": gated_correct / total,
            "net_correct_vs_ungated": gated_correct - action_correct,
            "net_correct_vs_talr": gated_correct - baseline_correct,
            "fixed": len(fixed), "damaged": len(damaged),
            "fixed_retention": (sum(not row["gate_reject"] for row in fixed) / len(fixed)) if fixed else None,
            "damage_rejection": (sum(row["gate_reject"] for row in damaged) / len(damaged)) if damaged else None,
        }
        if dataset == "mmvp":
            pair_ids = sorted({int(row["pair_id"]) for row in items if row.get("pair_id") is not None})
            pair_correct = sum(all(row["chosen_correct"] for row in items if int(row["pair_id"]) == pair_id) for pair_id in pair_ids)
            report.update({"pair_correct": pair_correct, "pair_total": len(pair_ids), "pair_accuracy": pair_correct / len(pair_ids)})
        reports[dataset] = report

    artifact = {
        "action": "TALR true visual residual strength=1.0 duration=4",
        "gate": "exact locked three-swap median plus mask h4/h8 inverse veto",
        "optimization": "20 unique probes instead of 36 repeated probes; policy unchanged",
        "datasets": reports,
    }
    with (args.output_dir / "gated_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "optimized_inverse_gate_summary.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Optimized Exact Inverse Gate", "", "Candidate: residual strength 1.0 for 4 tokens. Gate semantics are identical to the original three-swap+mask policy.", "", "| Dataset | TALR | Ungated | Gated | Rejected | Fixed kept | Damage rejected |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, report in reports.items():
        kept = "-" if report["fixed_retention"] is None else "%.1f%%" % (100 * report["fixed_retention"])
        rejected = "-" if report["damage_rejection"] is None else "%.1f%%" % (100 * report["damage_rejection"])
        lines.append("| %s | %.2f%% | %.2f%% | %.2f%% | %d/%d | %s | %s |" % (dataset, 100*report["talr_accuracy"], 100*report["ungated_accuracy"], 100*report["gated_accuracy"], report["rejected"], report["total"], kept, rejected))
    (args.output_dir / "optimized_inverse_gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

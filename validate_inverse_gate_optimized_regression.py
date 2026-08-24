#!/usr/bin/env python3
"""Prove the locked policy implementation reproduces historical original gate outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from inverse_gate_policy import legacy_feature_decision


RELATIVE = {
    "mmvp": "mmvp/full_merged/merged",
    "vstar": "vstar/full_merged/merged",
    "realworldqa": "realworldqa/full/merged",
    "visulogic": "visulogic/full/merged",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmap(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(str(row["dataset"]), str(row["id"])): row for row in read_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-core", type=Path, required=True)
    parser.add_argument("--primary-extension", type=Path, required=True)
    parser.add_argument("--alt1-core", type=Path, required=True)
    parser.add_argument("--alt2-core", type=Path, required=True)
    parser.add_argument("--alt1-extension", type=Path, required=True)
    parser.add_argument("--alt2-extension", type=Path, required=True)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--core-summary", type=Path, required=True)
    parser.add_argument("--extension-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core = [fmap(path) for path in (args.primary_core, args.alt1_core, args.alt2_core)]
    extension = [fmap(path) for path in (args.primary_extension, args.alt1_extension, args.alt2_extension)]
    expected_summaries = {
        **{name: json.loads(args.core_summary.read_text(encoding="utf-8"))["datasets"][name]["multi_swap_mask_consensus"] for name in ("mmvp", "vstar")},
        **{name: json.loads(args.extension_summary.read_text(encoding="utf-8"))["datasets"][name]["multi_swap_mask_consensus"] for name in ("realworldqa", "visulogic")},
    }
    report: Dict[str, Any] = {"policy": "three-swap median plus mask consensus", "datasets": {}}
    all_pass = True
    for dataset, relative in RELATIVE.items():
        features = core if dataset in {"mmvp", "vstar"} else extension
        common = set(features[0]) & set(features[1]) & set(features[2])
        root = args.branch_root / relative
        baseline = {str(row["id"]): row for row in read_jsonl(root / "talr/specialized_results.jsonl")}
        action = {str(row["id"]): row for row in read_jsonl(root / "talr_true_residual/specialized_results.jsonl")}
        rejected = gated_correct = 0
        selected_predictions = []
        for sample_id in sorted(set(baseline) & set(action), key=int):
            key = (dataset, sample_id)
            reject = legacy_feature_decision(features[0][key], features[1][key], features[2][key])["reject_action"] if key in common else False
            rejected += int(reject)
            chosen = baseline[sample_id] if reject else action[sample_id]
            gated_correct += int(bool(chosen["specialized_is_correct"]))
            selected_predictions.append((sample_id, chosen.get("specialized_pred"), bool(chosen["specialized_is_correct"])))
        expected = expected_summaries[dataset]
        checks = {
            "total": len(selected_predictions) == int(expected["total"]),
            "probed": sum((dataset, sid) in common for sid in baseline) == int(expected["probed"]),
            "rejected": rejected == int(expected["rejected"]),
            "gated_correct": gated_correct == round(float(expected["gated_accuracy"]) * int(expected["total"])),
        }
        passed = all(checks.values())
        all_pass = all_pass and passed
        report["datasets"][dataset] = {"checks": checks, "passed": passed, "rejected": rejected, "gated_correct": gated_correct, "prediction_digest_rows": len(selected_predictions)}
    report["exact_historical_regression_passed"] = all_pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_pass:
        raise RuntimeError("Optimized policy failed historical regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a full manifest for the locked gate over a chosen residual action."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXPECTED = {"mmvp": 300, "vstar": 191, "realworldqa": 200, "visulogic": 1000}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_map(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(str(row["dataset"]), str(row["id"])): row for row in read_jsonl(path)}


def merged_feature_map(paths: List[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    output: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in paths:
        for key, row in feature_map(path).items():
            if key in output:
                raise RuntimeError("Duplicate feature key %r across %s" % (key, paths))
            output[key] = row
    return output


def gold_choice(row: Dict[str, Any]) -> str:
    value = row.get("specialized_gold", row.get("realworldqa_gold", row.get("answer", "")))
    match = re.search(r"[A-Ea-e]", str(value))
    if not match:
        raise ValueError("Cannot normalize gold choice: %r" % (value,))
    return match.group(0).upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--primary", type=Path, action="append", required=True)
    parser.add_argument("--alt1", type=Path, action="append", required=True)
    parser.add_argument("--alt2", type=Path, action="append", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    analyzer = load_module("full_grid_eval", args.repo / "tools/analyze_visual_residual_full_grid.py")
    controls = [merged_feature_map(paths) for paths in (args.primary, args.alt1, args.alt2)]
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for config_path in sorted(args.grid_root.glob("task_*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        by_dataset[str(config["dataset_name"])] = read_jsonl(config_path.parent / "results.jsonl")

    output_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"strength": args.strength, "duration": args.duration, "datasets": {}, "gate": "locked three-swap median plus mask consensus"}
    for dataset, expected in EXPECTED.items():
        rows = by_dataset[dataset]
        baseline_raw = [row for row in rows if row["branch"] == "talr"]
        action_raw = [row for row in rows if row["branch"] == "talr_true_residual" and abs(float(row["residual_strength"]) - args.strength) < 1e-9 and int(row["residual_duration"]) == args.duration]
        if len(baseline_raw) != expected or len(action_raw) != expected:
            raise RuntimeError("Incomplete %s: baseline=%d action=%d expected=%d" % (dataset, len(baseline_raw), len(action_raw), expected))
        _, baseline_eval = analyzer.evaluate_rows(dataset, baseline_raw, args.repo)
        _, action_eval = analyzer.evaluate_rows(dataset, action_raw, args.repo)
        baseline = {str(row["id"]): row for row in baseline_eval}
        action = {str(row["id"]): row for row in action_eval}
        counts = {"total": expected, "eligible": 0, "no_injection": 0, "short": 0, "prefix_mismatch": 0, "missing_controls": 0, "control_collisions": 0}
        for sample_id in sorted(set(baseline) & set(action), key=int):
            base, treated = baseline[sample_id], action[sample_id]
            event_step = int(treated.get("refinement_step", -1))
            base_tokens = [int(value) for value in base["generated_token_ids"]]
            action_tokens = [int(value) for value in treated["generated_token_ids"]]
            key = (dataset, sample_id)
            reason = None
            if event_step < 0 or not bool(treated.get("injection_applied", False)):
                reason = "no_injection"
            elif len(base_tokens) <= event_step + 8 or len(action_tokens) <= event_step + 8:
                reason = "short"
            elif base_tokens[:event_step + 1] != action_tokens[:event_step + 1]:
                reason = "prefix_mismatch"
            elif not all(key in control for control in controls):
                reason = "missing_controls"
            if reason:
                counts[reason] += 1
            eligible = reason is None
            counts["eligible"] += int(eligible)
            base_correct = bool(base["eval_correct"])
            action_correct = bool(treated["eval_correct"])
            group = "fixed" if (not base_correct and action_correct) else "damaged" if (base_correct and not action_correct) else "unchanged_correct" if base_correct else "unchanged_wrong"
            swaps = []
            if eligible:
                swaps = [{"image": str(control[key]["swapped_image"]), "id": str(control[key].get("swapped_id", ""))} for control in controls]
                paths = [entry["image"] for entry in swaps]
                if str(base["image"]) in paths:
                    raise RuntimeError("Control equals true image for %s:%s" % key)
                counts["control_collisions"] += int(len(set(paths)) != 3)
            output_rows.append({
                "dataset": dataset, "id": sample_id, "pair_id": base.get("pair_id"),
                "image": base["image"], "question": base["question"], "options": base.get("options", ""), "answer": base.get("answer"),
                "gold_choice": gold_choice(base), "event_step": event_step,
                "prefix_ids": base_tokens[:event_step + 1] if event_step >= 0 else [],
                "baseline_generated_token_ids": base_tokens, "action_generated_token_ids": action_tokens,
                "baseline_pred": base.get("eval_pred"), "action_pred": treated.get("eval_pred"),
                "baseline_correct": base_correct, "action_correct": action_correct, "group": group,
                "probe_eligible": eligible, "exclusion_reason": reason, "swaps": swaps,
                "action_strength": args.strength, "action_duration": args.duration,
            })
        summary["datasets"][dataset] = counts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["rows"] = len(output_rows)
    summary["eligible"] = sum(item["eligible"] for item in summary["datasets"].values())
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

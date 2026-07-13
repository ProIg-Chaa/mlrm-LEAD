#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("fixed_damage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(module, run_dir: Path, baseline_dir: Path):
    rows = load_jsonl(run_dir / "results.jsonl")
    baseline = {str(row.get("id")): row for row in load_jsonl(baseline_dir / "results.jsonl")}
    fixed = damaged = failed = long = maxed = 0
    for row in rows:
        sid = str(row.get("id"))
        pred = module.extract_prediction(row.get("model_answer"), row.get("options"))
        gold = module.normalize_gold(row.get("answer"), row.get("options"))
        base = baseline.get(sid, {})
        base_pred = module.extract_prediction(base.get("model_answer"), base.get("options"))
        base_gold = module.normalize_gold(base.get("answer"), base.get("options"))
        correct, base_correct = bool(pred and pred == gold), bool(base_pred and base_pred == base_gold)
        fixed += int(not base_correct and correct)
        damaged += int(base_correct and not correct)
        failed += int(pred is None)
        tokens = int(row.get("output_tokens") or 0)
        long += int(tokens >= 256)
        maxed += int(tokens >= 1024)
    return {"total": len(rows), "fixed": fixed, "damaged": damaged, "failed": failed, "long_256": long, "maxed_1024": maxed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    args = parser.parse_args()
    module = load_module(args.root / "script/exp7_11/analyze_fixed_damaged_mechanisms.py")
    baseline = args.root / "output/experiments/20260705_integrated_cot_lead_baselines/integrated_repo_cot_lead_baselines/r1_onevision_7b/vstar/cot_orign_greedy_gpu0"
    min0 = metrics(module, args.base_dir / "vstar/quota05_guard_min0", baseline)
    min2 = metrics(module, args.base_dir / "vstar/transition_preserving_quota05_guard_min2", baseline)
    passed = (
        min2["fixed"] > min2["damaged"]
        and min2["failed"] <= min0["failed"]
        and min2["long_256"] <= min0["long_256"]
        and min2["maxed_1024"] <= min0["maxed_1024"]
    )
    report = {"passed": passed, "criterion": "fixed>damaged and failed/long/maxed no worse than min0", "min0": min0, "min2": min2}
    (args.base_dir / "vstar_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())

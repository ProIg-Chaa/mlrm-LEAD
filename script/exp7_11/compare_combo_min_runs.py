#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path("output/experiments/20260711_fixed_damaged_mechanism_analysis/transition_preserving_combo")
for dataset in ("vstar", "mmvp", "visulogic300", "realworldqa_fixed200"):
    left = ROOT / dataset / "quota05_guard_min0" / "results.jsonl"
    right = ROOT / dataset / "transition_preserving_quota05_guard_min2" / "results.jsonl"
    if not left.exists() or not right.exists():
        continue
    a = {str(row.get("id")): row for row in map(json.loads, left.read_text(encoding="utf-8").splitlines())}
    b = {str(row.get("id")): row for row in map(json.loads, right.read_text(encoding="utf-8").splitlines())}
    ids = sorted(set(a) & set(b))
    print(json.dumps({
        "dataset": dataset,
        "paired": len(ids),
        "model_answer_changed": sum(a[sid].get("model_answer") != b[sid].get("model_answer") for sid in ids),
        "output_tokens_changed": sum(a[sid].get("output_tokens") != b[sid].get("output_tokens") for sid in ids),
    }))

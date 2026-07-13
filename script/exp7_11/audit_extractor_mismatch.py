#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

from lead.evaluator import extract_mcq_answer

root = Path.cwd()
spec = importlib.util.spec_from_file_location("analysis", root / "script/exp7_11/analyze_fixed_damaged_mechanisms.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
path = root / "output/experiments/20260705_integrated_cot_lead_baselines/integrated_repo_cot_lead_baselines/r1_onevision_7b/vstar/cot_orign_greedy_gpu0/results.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
mismatches = []
for row in rows:
    old = extract_mcq_answer(row.get("model_answer"))
    new = analysis.extract_prediction(row.get("model_answer"), row.get("options"))
    if old != new:
        mismatches.append({"id": row["id"], "gold": row["answer"], "old": old, "new": new, "tail": row["model_answer"][-500:]})
print(json.dumps({"count": len(mismatches), "head": mismatches[:30]}, ensure_ascii=False, indent=2))

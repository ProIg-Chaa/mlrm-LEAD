#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

RUN_DIR="$(
  {
    ls -td output/experiments/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/vstar_lead_paper_params_* 2>/dev/null
    ls -td output/experiments/vstar_lead_paper_params_* 2>/dev/null
  } | head -n 1
)"

if [[ -z "${RUN_DIR}" ]]; then
  echo "No VStar LEAD paper-params experiment directory found." >&2
  exit 1
fi

echo "Run dir: $RUN_DIR"
echo

if [[ -f "$RUN_DIR/eval_report.json" ]]; then
  echo "Evaluation:"
  python3 - <<PY
import json
from pathlib import Path
path = Path("$RUN_DIR/eval_report.json")
data = json.loads(path.read_text())
print(json.dumps({
    "accuracy": data.get("accuracy"),
    "correct": data.get("correct"),
    "total": data.get("total"),
    "failed_extraction": data.get("failed_extraction"),
    "by_subtopic": data.get("by_subtopic"),
}, ensure_ascii=False, indent=2))
PY
  echo
else
  echo "Evaluation report not found yet: $RUN_DIR/eval_report.json"
  echo
fi

if [[ -f "$RUN_DIR/token_entropy.jsonl" ]]; then
  echo "Token entropy summary:"
  python3 script/summarize_token_entropy.py "$RUN_DIR/token_entropy.jsonl"
else
  echo "Token entropy file not found yet: $RUN_DIR/token_entropy.jsonl" >&2
  exit 1
fi

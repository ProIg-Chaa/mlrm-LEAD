#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

RUN_DIR="$(
  {
    ls -td output/experiments/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/mmhal_lead_paper_params_* 2>/dev/null
    ls -td output/experiments/mmhal_lead_paper_params_* 2>/dev/null
  } | head -n 1
)"

if [[ -z "${RUN_DIR}" ]]; then
  echo "No MMHal-Bench LEAD paper-params experiment directory found." >&2
  exit 1
fi

echo "Run dir: $RUN_DIR"
echo

if [[ -f "$RUN_DIR/results.jsonl" ]]; then
  echo "Result count:"
  wc -l "$RUN_DIR/results.jsonl"
  echo

  python3 - <<PY
import json
from pathlib import Path

source = Path("/share/home/wangzixu/liudinghao/gushuo/datasets/sources/Shengcao1006__MMHal-Bench/response_template.json")
results = Path("$RUN_DIR/results.jsonl")
output = Path("$RUN_DIR/mmhal_response.json")

template = json.loads(source.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]

for item, row in zip(template, rows):
    item["model_answer"] = row.get("model_answer") or ""

output.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote MMHal official response file: {output}")
print("Use this file with MMHal-Bench/eval_gpt4.py for 0-6 hallucination scoring.")
PY
  echo
else
  echo "Results file not found yet: $RUN_DIR/results.jsonl"
  echo
fi

if [[ -f "$RUN_DIR/token_entropy.jsonl" ]]; then
  echo "Token entropy summary:"
  python3 script/summarize_token_entropy.py "$RUN_DIR/token_entropy.jsonl"
else
  echo "Token entropy file not found yet: $RUN_DIR/token_entropy.jsonl" >&2
  exit 1
fi

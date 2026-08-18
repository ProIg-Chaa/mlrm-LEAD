#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD-transplant
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260801_soft_state_transplant_expanded
MANIFEST="$BASE/outcome_agnostic_1024_events.jsonl"
OUT="$BASE/ttr_dose_response"

mkdir -p "$OUT"

printf '[%s] lane A waiting for high-strength true/swap queue\n' "$(date -Is)"
while [[ ! -e "$BASE/STRENGTH_EXTENSION_COMPLETE" ]]; do
  sleep 60
done

cd "$ROOT"
"$PY" -m py_compile run_soft_state_transplant.py

for lambda in 0.20 0.40 0.60 0.70; do
  tag=${lambda/./}
  run="$OUT/lambda${tag}_true_swap_1024"
  if [[ -e "$run/RUN_COMPLETE" ]]; then
    printf '[%s] lane A skip complete lambda=%s\n' "$(date -Is)" "$lambda"
    continue
  fi
  printf '[%s] lane A start lambda=%s true/swap\n' "$(date -Is)" "$lambda"
  "$PY" run_soft_state_transplant.py \
    --manifest "$MANIFEST" \
    --model "$MODEL" \
    --output-dir "$run" \
    --branches true_image,swapped_image \
    --mix-lambda "$lambda" \
    --trace-topk 0
done

touch "$OUT/LANE_A_COMPLETE"
printf '[%s] lane A complete\n' "$(date -Is)"

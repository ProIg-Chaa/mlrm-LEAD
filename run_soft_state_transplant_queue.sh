#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD-transplant
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
ATLAS=/root/autodl-tmp/gushuo/outputs/experiments/20260730_intervention_atlas_full
OUT=/root/autodl-tmp/gushuo/outputs/experiments/20260731_soft_state_transplant
MANIFEST="$OUT/soft_state_transplant_manifest.jsonl"

mkdir -p "$OUT"
printf '[%s] queue started; waiting for Atlas shards\n' "$(date -Is)"
while true; do
  complete=$(find "$ATLAS" -name SHARD_COMPLETE 2>/dev/null | wc -l)
  active=$(pgrep -af 'main.py.*20260730_intervention_atlas_full' | grep -v grep | wc -l || true)
  printf '[%s] atlas complete=%s/12 active=%s\n' "$(date -Is)" "$complete" "$active"
  if [[ "$complete" -ge 12 && "$active" -eq 0 ]]; then
    break
  fi
  sleep 120
done

cd "$ROOT"
"$PYTHON" -m py_compile \
  lead/generation_utils.py \
  lead/inference.py \
  prepare_soft_state_transplant.py \
  run_soft_state_transplant.py

if [[ ! -s "$MANIFEST" ]]; then
  "$PYTHON" prepare_soft_state_transplant.py \
    --enriched-atlas "$ATLAS/multimodal_analysis_interim_20260731/multimodal_event_labels.jsonl" \
    --roots \
      /root/autodl-tmp/gushuo/outputs/experiments/20260724_intervention_atlas_v0b/newgpu3 \
      "$ATLAS/runs" \
      "$ATLAS/reused_partial" \
    --output "$MANIFEST" \
    --per-class 10 \
    --seed 42
fi

printf '[%s] starting two-event reproduction smoke\n' "$(date -Is)"
"$PYTHON" run_soft_state_transplant.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --output-dir "$OUT/smoke2" \
  --limit 2 \
  --branches hard,true_image \
  --require-reproduction \
  --trace-topk 20

printf '[%s] smoke passed; starting full 40-event factorial\n' "$(date -Is)"
"$PYTHON" run_soft_state_transplant.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --output-dir "$OUT/full40" \
  --branches hard,true_image,swapped_image,masked_image,random \
  --require-reproduction \
  --trace-topk 20

printf '[%s] soft-state transplant queue complete\n' "$(date -Is)"
touch "$OUT/QUEUE_COMPLETE"

#!/usr/bin/env bash
set -euo pipefail

PARENT_PID="${1:?parent runner PID required}"
ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_followup/runs/shard_0
REPO=/root/gushuo/proj/mlrm-LEAD-atlas-v0b
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
OUT="$ROOT/pure_soft_l100"

resume_parent() {
  kill -CONT "$PARENT_PID" 2>/dev/null || true
}
trap resume_parent EXIT

cd "$REPO"
"$PY" main.py \
  --model_name /dev/shm/wangzixu_models/R1-Onevision-7B-RL \
  --dataset "$ROOT/event_dataset.jsonl" \
  --output_dir "$OUT" \
  --method cot_greedy \
  --cot_prompt_mode orign \
  --no-do_sample \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42 \
  --max_new_tokens 1024 \
  --device cuda \
  --save_token_entropy \
  --save_full_token_entropy \
  --trace_topk 0 \
  --trace_route_override_manifest "$ROOT/event_override_manifest.json" \
  --trace_route_override_kind raw_soft \
  --trace_route_override_mix_lambda 1.0

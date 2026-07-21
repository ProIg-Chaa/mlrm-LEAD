#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
OUT=/root/autodl-tmp/gushuo/outputs/experiments/20260718_dynamic_transition_holdout32
DATA="$OUT/vstar_heldout_ids32_63.jsonl"

mkdir -p "$OUT"

# Let the already-running W8 cap search finish before loading two more models.
while pgrep -f "script/exp7_17/run_w8_cap_smoke.sh" >/dev/null; do
    sleep 30
done

common=(
    --model_name "$MODEL"
    --dataset "$DATA"
    --method lead
    --alpha 0.4
    --max_switch_count 5
    --window_size 128
    --cot_prompt_mode orign
    --no-do_sample
    --temperature 0.6
    --top_p 0.95
    --top_k 20
    --seed 42
    --max_new_tokens 1024
    --device cuda
    --save_token_entropy
    --save_full_token_entropy
    --trace_topk 0
    --lead_initial_transition_only
)

cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 "$PY" main.py \
    "${common[@]}" \
    --output_dir "$OUT/boundary_step2" \
    --lead_transition_dynamic_entropy_window 2 \
    --lead_transition_dynamic_entropy_ratio 0.5 \
    --lead_transition_dynamic_min_history 2 \
    --lead_transition_dynamic_max_step 4 \
    >"$OUT/boundary_step2.log" 2>&1 &
pid_boundary=$!

CUDA_VISIBLE_DEVICES=0 "$PY" main.py \
    "${common[@]}" \
    --output_dir "$OUT/semantic_adaptive_tau080" \
    --lead_transition_semantic_adaptive \
    --lead_transition_semantic_entropy_threshold 0.80 \
    --lead_transition_semantic_max_extra_steps 1 \
    >"$OUT/semantic_adaptive_tau080.log" 2>&1 &
pid_adaptive=$!

wait "$pid_boundary" "$pid_adaptive"

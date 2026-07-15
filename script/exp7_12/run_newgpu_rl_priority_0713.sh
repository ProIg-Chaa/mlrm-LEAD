#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260712_uniform_multimodel_full_matrix/uniform_multimodel_full_matrix/r1_onevision_7b_rl

cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py

COMMON=(
  --model_name "$MODEL" --cot_prompt_mode orign
  --temperature 0.6 --top_p 0.95 --top_k 20
  --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample
  --save_token_entropy --trace_topk 0
)

complete() {
  local run_dir=$1 expected=$2
  [[ -f "$run_dir/config.json" && -f "$run_dir/results.jsonl" &&
     -f "$run_dir/eval_report.json" && -f "$run_dir/token_entropy.jsonl" &&
     "$(wc -l < "$run_dir/results.jsonl")" -eq "$expected" ]]
}

run_one() {
  local dataset_key=$1 dataset=$2 expected=$3 method_key=$4
  shift 4
  local run_dir="$BASE/$dataset_key/$method_key"
  if complete "$run_dir" "$expected"; then
    echo "[SKIP] $dataset_key/$method_key"
    return
  fi
  mkdir -p "$run_dir"
  echo "[START] $(date '+%F %T') $dataset_key/$method_key"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py "${COMMON[@]}" --dataset "$dataset" \
    --output_dir "$run_dir" "$@"
  complete "$run_dir" "$expected"
  echo "[DONE] $(date '+%F %T') $dataset_key/$method_key"
}

run_initial_transition() {
  run_one "$1" "$2" "$3" initial_transition_only \
    --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 \
    --lead_initial_transition_only
}

run_talr() {
  run_one "$1" "$2" "$3" transition_preserving_quota05_guard_min2 \
    --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 \
    --lead_soft_quota_ratio 0.05 --lead_format_cooldown \
    --format_cooldown_steps 2 --format_cooldown_min_step 2 \
    --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen \
    --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 \
    --lead_veto_recent_repeat_tau 0.35
}

POPE="$ROOT/data/pope_adversarial.jsonl"
PHYSICS="$ROOT/data/mmk12_physics.jsonl"

run_initial_transition pope_adversarial "$POPE" 3000
run_talr pope_adversarial "$POPE" 3000
run_initial_transition mmk12_physics "$PHYSICS" 500
run_talr mmk12_physics "$PHYSICS" 500

echo "[ALL DONE] $(date '+%F %T')"

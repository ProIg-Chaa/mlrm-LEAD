#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:?Usage: $0 pope_talr|mmk_methods}
ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260712_uniform_multimodel_full_matrix/uniform_multimodel_full_matrix/r1_onevision_7b_rl

COMMON=(
  --model_name "$MODEL" --cot_prompt_mode orign
  --temperature 0.6 --top_p 0.95 --top_k 20
  --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample
  --save_token_entropy --trace_topk 0
)
LEAD=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128)
TALR=(
  "${LEAD[@]}" --lead_soft_quota_ratio 0.05 --lead_format_cooldown
  --format_cooldown_steps 2 --format_cooldown_min_step 2
  --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen
  --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64
  --lead_veto_recent_repeat_tau 0.35
)

cd "$ROOT"

run_one() {
  local dataset=$1 run_dir=$2 expected=$3
  shift 3
  if [[ -f "$run_dir/results.jsonl" && -f "$run_dir/eval_report.json" &&
        "$(wc -l < "$run_dir/results.jsonl")" -eq "$expected" ]]; then
    echo "[SKIP] $run_dir"
    return
  fi
  mkdir -p "$run_dir"
  echo "[START] $(date '+%F %T') $run_dir"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py "${COMMON[@]}" --dataset "$dataset" \
    --output_dir "$run_dir" "$@"
  [[ "$(wc -l < "$run_dir/results.jsonl")" -eq "$expected" ]]
  echo "[DONE] $(date '+%F %T') $run_dir"
}

case "$SCOPE" in
  pope_talr)
    run_one "$ROOT/data/pope_adversarial.jsonl" \
      "$BASE/pope_adversarial/transition_preserving_quota05_guard_min2" 3000 \
      "${TALR[@]}"
    ;;
  mmk_methods)
    run_one "$ROOT/data/mmk12_physics.jsonl" \
      "$BASE/mmk12_physics/initial_transition_only" 500 \
      "${LEAD[@]}" --lead_initial_transition_only
    run_one "$ROOT/data/mmk12_physics.jsonl" \
      "$BASE/mmk12_physics/transition_preserving_quota05_guard_min2" 500 \
      "${TALR[@]}"
    ;;
  *) echo "Unknown scope: $SCOPE" >&2; exit 2 ;;
esac

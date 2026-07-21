#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/Vision-R1-7B
OUT=/root/gushuo/outputs/experiments/20260717_legacy_talr_vision_core
GPU=0

COMMON=(
  --model_name "$MODEL" --cot_prompt_mode orign
  --temperature 0.6 --top_p 0.95 --top_k 20
  --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample
  --save_token_entropy --trace_topk 0
)

# Exact legacy TALR policy: full/legacy LEAD routing under a 5% soft quota,
# with format cooldown and the late diffuse/repeat veto. It intentionally does
# not enable --lead_initial_transition_with_refinement.
LEGACY_TALR=(
  --method lead --alpha 0.4 --max_switch_count 5 --window_size 128
  --lead_soft_quota_ratio 0.05
  --lead_format_cooldown --format_cooldown_steps 2 --format_cooldown_min_step 2
  --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen
  --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64
  --lead_veto_recent_repeat_tau 0.35
)

cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py
mkdir -p "$OUT/logs"

run_one() {
  local key=$1 dataset=$2 expected=$3
  shift 3
  local run_dir="$OUT/$key/legacy_talr_quota05_guard_min2"
  if [[ -f "$run_dir/results.jsonl" && "$(wc -l < "$run_dir/results.jsonl")" -eq "$expected" && -f "$run_dir/eval_report.json" ]]; then
    echo "[SKIP] $key"
    return
  fi
  mkdir -p "$run_dir"
  echo "[START] $(date '+%F %T') $key"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nice -n 10 "$PYTHON" main.py --dataset "$dataset" --output_dir "$run_dir" \
    "${COMMON[@]}" "${LEGACY_TALR[@]}" "$@"
  [[ -f "$run_dir/results.jsonl" && "$(wc -l < "$run_dir/results.jsonl")" -eq "$expected" ]] || {
    echo "[INCOMPLETE] $key" >&2
    exit 4
  }
  echo "[DONE] $(date '+%F %T') $key"
}

run_long_lane() {
  run_one visulogic300 "$ROOT/data/visulogic.jsonl" 300 --limit 300
}

run_short_lane() {
  run_one vstar "$ROOT/data/vstar.jsonl" 191
  run_one mmvp "$ROOT/data/mmvp.jsonl" 300
  run_one realworldqa_fixed200 "$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl" 200
}

run_long_lane >"$OUT/logs/visulogic.log" 2>&1 &
LONG_PID=$!
run_short_lane >"$OUT/logs/core_short.log" 2>&1 &
SHORT_PID=$!

echo "[LAUNCHED] long_pid=$LONG_PID short_pid=$SHORT_PID gpu=$GPU"
wait "$LONG_PID"
wait "$SHORT_PID"
echo "[ALL-DONE] $(date '+%F %T')"

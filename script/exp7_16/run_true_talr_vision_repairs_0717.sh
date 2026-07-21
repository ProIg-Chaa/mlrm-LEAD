#!/usr/bin/env bash
set -euo pipefail

# Repair only the two missing/invalid Vision-R1 True TALR cells.  They can use
# two model processes on a single A800; each process remains below 25 GB.
ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/Vision-R1-7B
BASE=/root/gushuo/outputs/experiments/20260716_talr_dual_line/true_talr_core_runs/vision_r1_7b
LOG=/root/gushuo/outputs/experiments/20260716_talr_dual_line/repair_logs

COMMON=(--cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --trace_topk 0)
TALR=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_with_refinement --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 --format_cooldown_min_step 2 --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 --lead_veto_recent_repeat_tau 0.35)

cd "$ROOT"
mkdir -p "$LOG"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py

complete() {
  local run=$1 expected=$2
  [[ -f "$run/results.jsonl" && -f "$run/eval_report.json" ]] || return 1
  [[ "$(wc -l < "$run/results.jsonl")" -eq "$expected" ]] || return 1
  ! grep -q 'invalid index of a 0-dim tensor\|"error_type": "' "$run/results.jsonl"
}

run_realworldqa() {
  local out="$BASE/realworldqa_fixed200/talr_early_quota05_guard_min2"
  if ! complete "$out" 200; then
    if [[ -e "$out" ]]; then
      mv "$out" "${out}.tracebug_invalid.$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" main.py --model_name "$MODEL" \
      --dataset "$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl" --output_dir "$out" \
      "${COMMON[@]}" "${TALR[@]}"
    "$PYTHON" script/evaluate_realworldqa_mcq.py \
      --dataset "$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl" \
      --results "$out/results.jsonl" --output_json "$out/realworldqa_mcq_eval.json" \
      --output_results_jsonl "$out/realworldqa_mcq_rows.jsonl"
  fi
  complete "$out" 200
}

run_visulogic() {
  local out="$BASE/visulogic300/talr_early_quota05_guard_min2"
  if ! complete "$out" 300; then
    if [[ -e "$out" ]]; then
      mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" main.py --model_name "$MODEL" --dataset "$ROOT/data/visulogic.jsonl" --limit 300 --output_dir "$out" \
      "${COMMON[@]}" "${TALR[@]}"
  fi
  complete "$out" 300
}

run_realworldqa > "$LOG/vision_realworldqa.log" 2>&1 &
pid_rw=$!
run_visulogic > "$LOG/vision_visulogic.log" 2>&1 &
pid_visu=$!

status=0
wait "$pid_rw" || status=1
wait "$pid_visu" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "[FAILED] True TALR Vision repair $(date '+%F %T')" >&2
  exit "$status"
fi
echo "[ALL DONE] True TALR Vision repair $(date '+%F %T')"

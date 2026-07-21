#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
VISION=/dev/shm/wangzixu_models/Vision-R1-7B
VISION_ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260714_vision_r1_compact_matrix/vision_r1_7b
RL_ROOT=/root/gushuo/migrated_results/rl_compact_matrix_migration_20260713/reusable_results/r1_onevision_7b_rl
OUT=/root/gushuo/outputs/experiments/20260716_talr_dual_line/main_summary
COMMON=(--cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --trace_topk 0)
TALR=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 --format_cooldown_min_step 2 --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 --lead_veto_recent_repeat_tau 0.35)

cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py \
  script/exp7_16/summarize_talr_core_main.py

TARGET="$VISION_ROOT/visulogic300/transition_preserving_quota05_guard_min2"
if [[ ! -f "$TARGET/results.jsonl" || "$(wc -l < "$TARGET/results.jsonl")" -ne 300 || ! -f "$TARGET/eval_report.json" ]]; then
  if [[ -e "$TARGET" ]]; then
    mv "$TARGET" "${TARGET}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$TARGET"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py --model_name "$VISION" --dataset "$ROOT/data/visulogic.jsonl" \
    --limit 300 --output_dir "$TARGET" "${COMMON[@]}" "${TALR[@]}"
fi

evaluate_specialized() {
  local model_root=$1
  for method in cot_orign_greedy lead initial_transition_only transition_preserving_quota05_guard_min2 talr; do
    local mmvp="$model_root/mmvp/$method"
    if [[ -f "$mmvp/results.jsonl" ]]; then
      "$PYTHON" script/evaluate_specialized_results.py --dataset "$ROOT/data/mmvp.jsonl" \
        --results "$mmvp/results.jsonl" --output_json "$mmvp/specialized_eval_report.json" \
        --output_results_jsonl "$mmvp/specialized_eval_rows.jsonl"
    fi
    local rw="$model_root/realworldqa_fixed200/$method"
    if [[ -f "$rw/results.jsonl" ]]; then
      "$PYTHON" script/evaluate_realworldqa_mcq.py \
        --dataset "$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl" \
        --results "$rw/results.jsonl" --output_json "$rw/realworldqa_mcq_eval.json" \
        --output_results_jsonl "$rw/realworldqa_mcq_rows.jsonl"
    fi
  done
}

evaluate_specialized "$VISION_ROOT"
evaluate_specialized "$RL_ROOT"
mkdir -p "$OUT"
"$PYTHON" script/exp7_16/summarize_talr_core_main.py \
  --rl-root "$RL_ROOT" --vision-root "$VISION_ROOT" --output-dir "$OUT"
echo "[ALL DONE] $(date '+%F %T') TALR core main"

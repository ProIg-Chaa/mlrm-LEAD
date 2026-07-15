#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/Vision-R1-7B
BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260714_vision_r1_compact_matrix/vision_r1_7b

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
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py

run_one() {
  local dataset_key=$1 dataset=$2 expected=$3 method_key=$4
  shift 4
  local out="$BASE/$dataset_key/$method_key"
  if [[ -f "$out/results.jsonl" && -f "$out/eval_report.json" &&
        "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]; then
    echo "[SKIP] $dataset_key/$method_key"
    return
  fi
  mkdir -p "$out"
  echo "[START] $(date '+%F %T') $dataset_key/$method_key"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py "${COMMON[@]}" --dataset "$dataset" \
    --output_dir "$out" "$@"
  [[ "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]
  echo "[DONE] $(date '+%F %T') $dataset_key/$method_key"
}

for spec in \
  "mmk12_physics:$ROOT/data/mmk12_physics.jsonl:500" \
  "pope_adversarial:$ROOT/data/pope_adversarial.jsonl:3000"; do
  IFS=: read -r key dataset expected <<<"$spec"
  run_one "$key" "$dataset" "$expected" cot_orign_greedy --method cot_greedy
  run_one "$key" "$dataset" "$expected" lead "${LEAD[@]}"
  run_one "$key" "$dataset" "$expected" initial_transition_only \
    "${LEAD[@]}" --lead_initial_transition_only
  run_one "$key" "$dataset" "$expected" \
    transition_preserving_quota05_guard_min2 "${TALR[@]}"
done

echo "[ALL DONE] $(date '+%F %T')"

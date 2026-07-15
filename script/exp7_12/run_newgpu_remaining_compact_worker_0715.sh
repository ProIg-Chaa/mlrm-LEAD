#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:?Usage: $0 worker_a|worker_b}
ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
VISION_MODEL=/dev/shm/wangzixu_models/Vision-R1-7B
RL_MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
VISION_BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260714_vision_r1_compact_matrix/vision_r1_7b
RL_BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260712_uniform_multimodel_full_matrix/uniform_multimodel_full_matrix/r1_onevision_7b_rl

COMMON=(
  --cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20
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
  local model=$1 base=$2 dataset_key=$3 dataset=$4 expected=$5 method_key=$6 limit=$7
  shift 7
  local out="$base/$dataset_key/$method_key"
  if [[ -f "$out/results.jsonl" && -f "$out/eval_report.json" &&
        "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]; then
    echo "[SKIP] $dataset_key/$method_key"
    return
  fi
  if [[ -d "$out" ]]; then
    case "$out" in "$base"/*) rm -rf -- "$out" ;; *) exit 3 ;; esac
  fi
  mkdir -p "$out"
  local limit_args=()
  [[ "$limit" != none ]] && limit_args=(--limit "$limit")
  echo "[START] $(date '+%F %T') $dataset_key/$method_key model=$(basename "$model")"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py "${COMMON[@]}" --model_name "$model" \
    --dataset "$dataset" --output_dir "$out" "${limit_args[@]}" "$@"
  [[ -f "$out/eval_report.json" &&
     "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]
  echo "[DONE] $(date '+%F %T') $dataset_key/$method_key"
}

run_four_vision() {
  local key=$1 file=$2 expected=$3 limit=$4
  run_one "$VISION_MODEL" "$VISION_BASE" "$key" "$file" "$expected" cot_orign_greedy "$limit" --method cot_greedy
  run_one "$VISION_MODEL" "$VISION_BASE" "$key" "$file" "$expected" lead "$limit" "${LEAD[@]}"
  run_one "$VISION_MODEL" "$VISION_BASE" "$key" "$file" "$expected" initial_transition_only "$limit" \
    "${LEAD[@]}" --lead_initial_transition_only
  run_one "$VISION_MODEL" "$VISION_BASE" "$key" "$file" "$expected" transition_preserving_quota05_guard_min2 "$limit" \
    "${TALR[@]}"
}

case "$SCOPE" in
  worker_a)
    run_one "$RL_MODEL" "$RL_BASE" vmcbench_dev "$ROOT/data/vmcbench_dev.jsonl" 1000 initial_transition_only none \
      "${LEAD[@]}" --lead_initial_transition_only
    run_one "$RL_MODEL" "$RL_BASE" vmcbench_dev "$ROOT/data/vmcbench_dev.jsonl" 1000 transition_preserving_quota05_guard_min2 none \
      "${TALR[@]}"
    run_four_vision vstar "$ROOT/data/vstar.jsonl" 191 none
    run_four_vision realworldqa_fixed200 "$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl" 200 none
    run_four_vision visulogic300 "$ROOT/data/visulogic.jsonl" 300 300
    ;;
  worker_b)
    run_four_vision mmvp "$ROOT/data/mmvp.jsonl" 300 none
    run_four_vision vmcbench_dev "$ROOT/data/vmcbench_dev.jsonl" 1000 none
    ;;
  *) echo "Unknown scope: $SCOPE" >&2; exit 2 ;;
esac

echo "[ALL DONE] $(date '+%F %T') $SCOPE"

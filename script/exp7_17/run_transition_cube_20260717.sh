#!/usr/bin/env bash
set -euo pipefail

# Controlled 2x2x2 decomposition. The EOT half uses a forced step-1 handoff
# so every cell has the same intervention time; it is not labelled exact-original.
ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
OUT=/root/gushuo/outputs/experiments/20260717_transition_cube
DIRECT=/root/gushuo/outputs/experiments/20260717_step0_initializer_2x2
FULL=/root/gushuo/outputs/experiments/20260716_token_anchored_transition

cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py \
  script/exp7_17/summarize_transition_cube.py
mkdir -p "$OUT/logs"

COMMON=(--method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 \
  --lead_initial_transition_only --lead_force_initial_transition_step1 \
  --lead_transition_source hard --lead_transition_anchor end_thinking --lead_transition_beta0 0.7 \
  --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample \
  --save_token_entropy --save_full_token_entropy --trace_topk 0)

complete() {
  [[ -f "$1/results.jsonl" && -f "$1/eval_report.json" && "$(wc -l < "$1/results.jsonl")" -eq "$2" ]] \
    && ! grep -q '"error_type": "' "$1/results.jsonl"
}

run_one() {
  local dataset_key=$1 dataset=$2 expected=$3 condition=$4
  shift 4
  local out="$OUT/$dataset_key/$condition"
  local limit=()
  [[ "$dataset_key" == smoke ]] && limit=(--limit 2)
  if ! complete "$out" "$expected"; then
    [[ -e "$out" ]] && mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" main.py --model_name "$MODEL" --dataset "$dataset" --output_dir "$out" \
      "${COMMON[@]}" "${limit[@]}" "$@"
  fi
  complete "$out" "$expected"
  if [[ "$dataset_key" == mmvp ]]; then
    "$PYTHON" script/evaluate_specialized_results.py --dataset "$ROOT/data/mmvp.jsonl" \
      --results "$out/results.jsonl" --output_json "$out/specialized_eval_report.json" \
      --output_results_jsonl "$out/specialized_eval_rows.jsonl"
  fi
}

lane() {
  local condition=$1
  shift
  run_one smoke "$ROOT/data/vstar.jsonl" 2 "$condition" "$@"
  run_one vstar "$ROOT/data/vstar.jsonl" 191 "$condition" "$@"
  run_one mmvp "$ROOT/data/mmvp.jsonl" 300 "$condition" "$@"
}

# A=hard/soft at step 0; B=newline off/on. C=EOT bridge, beta=.7.
lane eot_hard_no_newline --lead_initial_transition_hard_boundary_only --lead_disable_step0_linebreak_mix \
  > "$OUT/logs/eot_hard_no_newline.log" 2>&1 & pid_a=$!
lane eot_hard_with_newline --lead_initial_transition_hard_boundary_only \
  > "$OUT/logs/eot_hard_with_newline.log" 2>&1 & pid_b=$!
wait "$pid_a"
lane eot_soft_no_newline --lead_disable_step0_linebreak_mix \
  > "$OUT/logs/eot_soft_no_newline.log" 2>&1
wait "$pid_b"
lane eot_soft_with_newline \
  > "$OUT/logs/eot_soft_with_newline.log" 2>&1

"$PYTHON" script/exp7_17/summarize_transition_cube.py --direct-root "$DIRECT" --full-root "$FULL" --eot-root "$OUT"
echo "[ALL DONE] $(date '+%F %T') transition cube"

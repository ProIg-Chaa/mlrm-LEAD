#!/usr/bin/env bash
set -euo pipefail

# This worker deliberately waits for the current True TALR repair queue.  The
# four bridge controls then share every generation condition except the step-1
# source/anchor definition.
ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
SOURCE=/root/gushuo/outputs/experiments/20260716_transition_causal_newgpu/transition_causal_decomposition
OUT=/root/gushuo/outputs/experiments/20260716_token_anchored_transition
TRUE_TALR_WORKER=run_true_talr_repair_0716.sh

COMMON=(--cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 0)
EARLY=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_force_initial_transition_step1)

cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py \
  script/exp7_16/summarize_token_anchored_transition.py \
  script/exp7_16/verify_token_anchor_smoke.py

while pgrep -f "[${TRUE_TALR_WORKER:0:1}]${TRUE_TALR_WORKER:1}" >/dev/null; do
  echo "[$(date '+%F %T')] Waiting for True TALR repair queue to finish."
  sleep 60
done

mkdir -p "$OUT"

complete_run() {
  local out=$1 expected=$2
  [[ -f "$out/results.jsonl" && -f "$out/eval_report.json" ]] || return 1
  [[ "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]] || return 1
  ! grep -q '"error_type"[[:space:]]*:[[:space:]]*"[^"[:space:]][^"]*"' "$out/results.jsonl"
}

run_one() {
  local dataset_key=$1 dataset=$2 expected=$3 name=$4; shift 4
  local out="$OUT/$dataset_key/$name"
  if complete_run "$out" "$expected"; then
    echo "[SKIP] $dataset_key/$name"
    return
  fi
  if [[ -e "$out" ]]; then
    mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py --model_name "$MODEL" --dataset "$dataset" --output_dir "$out" \
    "${COMMON[@]}" "$@"
  complete_run "$out" "$expected"
}

smoke_one() {
  local name=$1 source=$2 anchor=$3; shift 3
  run_one smoke "$ROOT/data/vstar.jsonl" 2 "$name" --limit 2 "${EARLY[@]}" "$@"
  "$PYTHON" script/exp7_16/verify_token_anchor_smoke.py \
    --run-dir "$OUT/smoke/$name" --expected-source "$source" --expected-anchor "$anchor"
}

# $1/$2 after EARLY are source and anchor.  These smokes assert that step 1,
# rather than an entropy event, performed the handoff.
smoke_one original_eot_bridge_step1 soft end_thinking \
  --lead_transition_source soft --lead_transition_anchor end_thinking
smoke_one hard_eot_bridge_step1 hard end_thinking \
  --lead_transition_source hard --lead_transition_anchor end_thinking
smoke_one direct_token_step1 hard generated_token \
  --lead_transition_source hard --lead_transition_anchor generated_token
smoke_one token_anchored_transition_step1 soft generated_token \
  --lead_transition_source soft --lead_transition_anchor generated_token

for spec in \
  'original_eot_bridge_step1 soft end_thinking' \
  'hard_eot_bridge_step1 hard end_thinking' \
  'direct_token_step1 hard generated_token' \
  'token_anchored_transition_step1 soft generated_token'; do
  read -r name source anchor <<< "$spec"
  run_one vstar "$ROOT/data/vstar.jsonl" 191 "$name" "${EARLY[@]}" \
    --lead_transition_source "$source" --lead_transition_anchor "$anchor"
  run_one mmvp "$ROOT/data/mmvp.jsonl" 300 "$name" "${EARLY[@]}" \
    --lead_transition_source "$source" --lead_transition_anchor "$anchor"
  "$PYTHON" script/evaluate_specialized_results.py \
    --dataset "$ROOT/data/mmvp.jsonl" \
    --results "$OUT/mmvp/$name/results.jsonl" \
    --output_json "$OUT/mmvp/$name/specialized_eval_report.json" \
    --output_results_jsonl "$OUT/mmvp/$name/specialized_eval_rows.jsonl"
done

"$PYTHON" script/exp7_16/summarize_token_anchored_transition.py \
  --source-root "$SOURCE" --experiment-root "$OUT"

echo "[ALL DONE] $(date '+%F %T') token-anchored transition"

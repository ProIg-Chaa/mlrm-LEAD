#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
STAGE=/root/gushuo/anchor_beta_stage_0717
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
OUT=/root/gushuo/outputs/experiments/20260717_eot_beta_curve
BASE=/root/gushuo/outputs/experiments/20260716_token_anchored_transition
REPAIR=run_true_talr_vision_repairs_0717.sh

while pgrep -f "[${REPAIR:0:1}]${REPAIR:1}" >/dev/null; do
  echo "[$(date '+%F %T')] waiting for Vision-R1 True TALR repairs"
  sleep 60
done

cp "$STAGE/main.py" "$ROOT/main.py"
cp "$STAGE/lead/inference.py" "$ROOT/lead/inference.py"
cp "$STAGE/lead/generation_utils.py" "$ROOT/lead/generation_utils.py"
cp "$STAGE/script/exp7_17/summarize_eot_beta_curve.py" "$ROOT/script/exp7_17/summarize_eot_beta_curve.py"
cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py script/exp7_17/summarize_eot_beta_curve.py
mkdir -p "$OUT/logs"

COMMON=(--method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_force_initial_transition_step1 --lead_transition_source hard --lead_transition_anchor end_thinking --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 0)

complete() {
  local out=$1 expected=$2
  [[ -f "$out/results.jsonl" && -f "$out/eval_report.json" ]] || return 1
  [[ "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]] || return 1
  ! grep -q '"error_type": "' "$out/results.jsonl"
}

run_one() {
  local dataset_key=$1 dataset=$2 expected=$3 beta=$4
  local label="beta$(printf '%.2f' "$beta" | tr '.' 'p')"
  local out="$OUT/$dataset_key/$label"
  if complete "$out" "$expected"; then return; fi
  if [[ -e "$out" ]]; then mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"; fi
  mkdir -p "$out"
  local limit_args=()
  if [[ "$dataset_key" == "smoke" ]]; then
    limit_args=(--limit 2)
  fi
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py --model_name "$MODEL" --dataset "$dataset" --output_dir "$out" \
    "${COMMON[@]}" "${limit_args[@]}" --lead_transition_beta0 "$beta"
  complete "$out" "$expected"
  if [[ "$dataset_key" == "mmvp" ]]; then
    "$PYTHON" script/evaluate_specialized_results.py --dataset "$ROOT/data/mmvp.jsonl" --results "$out/results.jsonl" --output_json "$out/specialized_eval_report.json" --output_results_jsonl "$out/specialized_eval_rows.jsonl"
  fi
}

lane() {
  for beta in "$@"; do
    run_one smoke "$ROOT/data/vstar.jsonl" 2 "$beta"
    run_one vstar "$ROOT/data/vstar.jsonl" 191 "$beta"
    run_one mmvp "$ROOT/data/mmvp.jsonl" 300 "$beta"
  done
}

# Two resident model processes: lane A handles two endpoints, lane B the middle point.
lane 0.40 0.85 > "$OUT/logs/lane_a.log" 2>&1 &
pid_a=$!
lane 0.55 > "$OUT/logs/lane_b.log" 2>&1 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"
"$PYTHON" script/exp7_17/summarize_eot_beta_curve.py --root "$OUT" --baseline-root "$BASE"
echo "[ALL DONE] $(date '+%F %T') EOT beta curve"

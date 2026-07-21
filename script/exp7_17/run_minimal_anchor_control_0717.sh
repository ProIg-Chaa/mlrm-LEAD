#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
STAGE=/root/gushuo/minimal_anchor_stage_0717
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
OUT=/root/gushuo/outputs/experiments/20260717_minimal_anchor_control
BASE=/root/gushuo/outputs/experiments/20260716_token_anchored_transition
BETA=run_eot_beta_curve_0717.sh
BETA_LOG=/root/gushuo/outputs/experiments/20260717_eot_beta_curve/worker.log

while pgrep -f "[${BETA:0:1}]${BETA:1}" >/dev/null; do
  if [[ -f "$BETA_LOG" ]] && grep -q '\[ALL DONE\]' "$BETA_LOG"; then
    break
  fi
  echo "[$(date '+%F %T')] waiting for EOT beta curve"
  sleep 60
done

cp "$STAGE/main.py" "$ROOT/main.py"
cp "$STAGE/lead/inference.py" "$ROOT/lead/inference.py"
cp "$STAGE/lead/generation_utils.py" "$ROOT/lead/generation_utils.py"
cp "$STAGE/script/exp7_17/summarize_minimal_anchor.py" "$ROOT/script/exp7_17/summarize_minimal_anchor.py"
cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py script/exp7_17/summarize_minimal_anchor.py
mkdir -p "$OUT/logs"
COMMON=(--method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_force_initial_transition_step1 --lead_transition_source soft --lead_transition_beta0 0.7 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 0)

complete() { [[ -f "$1/results.jsonl" && -f "$1/eval_report.json" && "$(wc -l < "$1/results.jsonl")" -eq "$2" ]] && ! grep -q '"error_type": "' "$1/results.jsonl"; }
run_one() {
  local key=$1 data=$2 expected=$3 anchor=$4
  local out="$OUT/$key/$anchor"
  local limit=()
  [[ "$key" == smoke ]] && limit=(--limit 2)
  if ! complete "$out" "$expected"; then
    [[ -e "$out" ]] && mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON" main.py --model_name "$MODEL" --dataset "$data" --output_dir "$out" "${COMMON[@]}" "${limit[@]}" --lead_transition_anchor "$anchor"
  fi
  complete "$out" "$expected"
  [[ "$key" != mmvp ]] || "$PYTHON" script/evaluate_specialized_results.py --dataset "$ROOT/data/mmvp.jsonl" --results "$out/results.jsonl" --output_json "$out/specialized_eval_report.json" --output_results_jsonl "$out/specialized_eval_rows.jsonl"
}
lane() { local anchor=$1; run_one smoke "$ROOT/data/vstar.jsonl" 2 "$anchor"; run_one vstar "$ROOT/data/vstar.jsonl" 191 "$anchor"; run_one mmvp "$ROOT/data/mmvp.jsonl" 300 "$anchor"; }
lane start_thinking > "$OUT/logs/start_thinking.log" 2>&1 & pid_a=$!
lane newline > "$OUT/logs/newline.log" 2>&1 & pid_b=$!
wait "$pid_a"; wait "$pid_b"
"$PYTHON" script/exp7_17/summarize_minimal_anchor.py --root "$OUT" --baseline-root "$BASE"
echo "[ALL DONE] $(date '+%F %T') minimal anchor control"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
OUT=/root/gushuo/outputs/experiments/20260717_step0_initializer_2x2
FULL=/root/gushuo/outputs/experiments/20260716_token_anchored_transition
COT=/root/gushuo/outputs/experiments/20260716_transition_causal_newgpu/transition_causal_decomposition
cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py script/exp7_17/summarize_step0_initializer_2x2.py
mkdir -p "$OUT/logs"
COMMON=(--method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_force_initial_transition_step1 --lead_transition_source hard --lead_transition_anchor end_thinking --lead_transition_beta0 1.0 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 0)

complete() { [[ -f "$1/results.jsonl" && -f "$1/eval_report.json" && "$(wc -l < "$1/results.jsonl")" -eq "$2" ]] && ! grep -q '"error_type": "' "$1/results.jsonl"; }
run_one() {
  local key=$1 data=$2 expected=$3 name=$4; shift 4
  local out="$OUT/$key/$name"
  local limit=()
  [[ "$key" == smoke ]] && limit=(--limit 2)
  if ! complete "$out" "$expected"; then
    [[ -e "$out" ]] && mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON" main.py --model_name "$MODEL" --dataset "$data" --output_dir "$out" "${COMMON[@]}" "${limit[@]}" "$@"
  fi
  complete "$out" "$expected"
  [[ "$key" != mmvp ]] || "$PYTHON" script/evaluate_specialized_results.py --dataset "$ROOT/data/mmvp.jsonl" --results "$out/results.jsonl" --output_json "$out/specialized_eval_report.json" --output_results_jsonl "$out/specialized_eval_rows.jsonl"
}
lane() {
  local name=$1; shift
  run_one smoke "$ROOT/data/vstar.jsonl" 2 "$name" "$@"
  if [[ "$name" == hard_no_newline ]]; then
    "$PYTHON" script/exp7_16/verify_result_text_equivalence.py \
      --expected "$COT/vstar/cot_orign_greedy/results.jsonl" \
      --actual "$OUT/smoke/$name/results.jsonl" \
      --limit 2 \
      --output "$OUT/smoke/$name/cot_equivalence.json"
  fi
  run_one vstar "$ROOT/data/vstar.jsonl" 191 "$name" "$@"
  run_one mmvp "$ROOT/data/mmvp.jsonl" 300 "$name" "$@"
}
lane hard_no_newline --lead_initial_transition_hard_boundary_only --lead_disable_step0_linebreak_mix > "$OUT/logs/hard_no_newline.log" 2>&1 & pid_a=$!
lane hard_with_newline --lead_initial_transition_hard_boundary_only > "$OUT/logs/hard_with_newline.log" 2>&1 & pid_b=$!
wait "$pid_a"
lane soft_no_newline --lead_disable_step0_linebreak_mix > "$OUT/logs/soft_no_newline.log" 2>&1
wait "$pid_b"
"$PYTHON" script/exp7_17/summarize_step0_initializer_2x2.py --root "$OUT" --full-run-root "$FULL" --cot-root "$COT"
echo "[ALL DONE] $(date '+%F %T') step0 initializer 2x2"

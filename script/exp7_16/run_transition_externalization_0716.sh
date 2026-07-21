#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
SOURCE=/root/gushuo/outputs/experiments/20260716_transition_causal_newgpu/transition_causal_decomposition
OUT=/root/gushuo/outputs/experiments/20260716_talr_dual_line/transition_externalization
COMMON=(--cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --trace_topk 0)
LEAD=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only)

cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py \
  script/exp7_16/run_transition_same_prefix_replay.py \
  script/exp7_16/summarize_transition_externalization.py
mkdir -p "$OUT"

run_one() {
  local dataset_key=$1 dataset=$2 expected=$3 name=$4; shift 4
  local out="$OUT/$dataset_key/$name"
  if [[ -f "$out/results.jsonl" && -f "$out/eval_report.json" && "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]; then
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
  [[ "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]
}

# Default and backward-compatibility smoke tests must reproduce prior text exactly.
run_one smoke_default "$ROOT/data/vstar.jsonl" 2 initial_transition --limit 2 "${LEAD[@]}"
"$PYTHON" script/exp7_16/verify_result_text_equivalence.py \
  --expected "$SOURCE/vstar/initial_transition_only/results.jsonl" \
  --actual "$OUT/smoke_default/initial_transition/results.jsonl" --limit 2 \
  --output "$OUT/smoke_default/default_equivalence.json"

run_one smoke_prefix2 "$ROOT/data/vstar.jsonl" 2 cache_rebuild_prefix2 --limit 2 \
  "${LEAD[@]}" --lead_initial_transition_cache_rebuild_prefix_len 2
"$PYTHON" script/exp7_16/verify_result_text_equivalence.py \
  --expected "$SOURCE/vstar/initial_transition_cache_rebuild/results.jsonl" \
  --actual "$OUT/smoke_prefix2/cache_rebuild_prefix2/results.jsonl" --limit 2 \
  --output "$OUT/smoke_prefix2/prefix2_equivalence.json"

run_one smoke_controls "$ROOT/data/vstar.jsonl" 2 cache_rebuild_prefix1 --limit 2 \
  "${LEAD[@]}" --lead_initial_transition_cache_rebuild_prefix_len 1
run_one smoke_controls "$ROOT/data/vstar.jsonl" 2 hard_boundary_only --limit 2 \
  "${LEAD[@]}" --lead_initial_transition_hard_boundary_only

run_one vstar "$ROOT/data/vstar.jsonl" 191 cache_rebuild_prefix1 \
  "${LEAD[@]}" --lead_initial_transition_cache_rebuild_prefix_len 1
run_one vstar "$ROOT/data/vstar.jsonl" 191 hard_boundary_only \
  "${LEAD[@]}" --lead_initial_transition_hard_boundary_only

# Prefix-1 MMVP is cheap and provides the preregistered external validation.
run_one mmvp "$ROOT/data/mmvp.jsonl" 300 cache_rebuild_prefix1 \
  "${LEAD[@]}" --lead_initial_transition_cache_rebuild_prefix_len 1
"$PYTHON" script/evaluate_specialized_results.py \
  --dataset "$ROOT/data/mmvp.jsonl" \
  --results "$OUT/mmvp/cache_rebuild_prefix1/results.jsonl" \
  --output_json "$OUT/mmvp/cache_rebuild_prefix1/specialized_eval_report.json" \
  --output_results_jsonl "$OUT/mmvp/cache_rebuild_prefix1/specialized_eval_rows.jsonl"

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" script/exp7_16/run_transition_same_prefix_replay.py \
  --root "$ROOT" --model "$MODEL" \
  --cot-results "$SOURCE/vstar/cot_orign_greedy/results.jsonl" \
  --cot-trace "$SOURCE/vstar/cot_orign_greedy/token_entropy_full.jsonl" \
  --transition-results "$SOURCE/vstar/initial_transition_only/results.jsonl" \
  --output-dir "$OUT/vstar/same_prefix_replay" --prefix-lengths 1,2,4 \
  --device cuda --cot-prompt-mode orign --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 1024

"$PYTHON" script/exp7_16/summarize_transition_externalization.py \
  --source-root "$SOURCE" --experiment-root "$OUT" \
  --rl-main-root /root/gushuo/migrated_results/rl_compact_matrix_migration_20260713/reusable_results/r1_onevision_7b_rl \
  --timing-summary "$OUT/evidence/timing_summary.json"

echo "[ALL DONE] $(date '+%F %T') transition externalization"

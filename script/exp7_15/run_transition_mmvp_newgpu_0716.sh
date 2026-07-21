#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python"
MODEL="/dev/shm/wangzixu_models/R1-Onevision-7B-RL"
OUT_ROOT="/root/gushuo/outputs/experiments/20260716_transition_causal_newgpu/transition_causal_decomposition"
DATASET="${ROOT}/data/mmvp.jsonl"
COMMON=(--cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20)

run_one() {
  local name="$1"; shift
  local out="${OUT_ROOT}/mmvp/${name}"
  if [[ -f "${out}/results.jsonl" ]] && [[ "$(wc -l < "${out}/results.jsonl")" -eq 300 ]] && [[ -f "${out}/specialized_eval_report.json" ]]; then
    echo "[skip] ${name}"
    return
  fi
  if [[ -e "${out}" ]]; then
    echo "Refusing to overwrite incomplete output: ${out}" >&2
    return 1
  fi
  mkdir -p "${out}"
  "${PYTHON_BIN}" "${ROOT}/main.py" --model_name "${MODEL}" --dataset "${DATASET}" --output_dir "${out}" "${COMMON[@]}" "$@"
  "${PYTHON_BIN}" "${ROOT}/script/evaluate_specialized_results.py" --dataset "${DATASET}" --results "${out}/results.jsonl" --output_json "${out}/specialized_eval_report.json" --output_results_jsonl "${out}/specialized_eval_rows.jsonl"
}

run_one cot_orign_greedy --method cot_greedy
run_one initial_soft_only --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_soft_only
run_one initial_transition_only --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only
run_one initial_transition_cache_rebuild --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_initial_transition_cache_rebuild_after_step 1

#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python"
MODEL="/dev/shm/wangzixu_models/R1-Onevision-7B-RL"
OUT_ROOT="/root/gushuo/outputs/experiments/20260716_transition_causal_newgpu/transition_causal_decomposition"
COMMON=(--cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20)
VSTAR="${ROOT}/data/vstar.jsonl"
MMVP="${ROOT}/data/mmvp.jsonl"

run_one() {
  local dataset_key="$1" dataset="$2" expected="$3" run_name="$4"; shift 4
  local out="${OUT_ROOT}/${dataset_key}/${run_name}"
  if [[ -f "${out}/results.jsonl" ]] && [[ "$(wc -l < "${out}/results.jsonl")" -eq "${expected}" ]] && [[ -f "${out}/eval_report.json" ]]; then
    echo "[skip] ${dataset_key}/${run_name}"
    return
  fi
  if [[ -e "${out}" ]]; then
    echo "Refusing to overwrite incomplete output: ${out}" >&2
    return 1
  fi
  mkdir -p "${out}"
  "${PYTHON_BIN}" "${ROOT}/main.py" --model_name "${MODEL}" --dataset "${dataset}" --output_dir "${out}" "${COMMON[@]}" "$@"
}

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

run_one vstar_smoke20 "${VSTAR}" 20 cot_orign_greedy --limit 20 --method cot_greedy
run_one vstar_smoke20 "${VSTAR}" 20 lead_force_normal --limit 20 --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_force_normal
"${PYTHON_BIN}" script/exp7_15/verify_transition_smoke.py --cot "${OUT_ROOT}/vstar_smoke20/cot_orign_greedy/token_entropy_full.jsonl" --force-normal "${OUT_ROOT}/vstar_smoke20/lead_force_normal/token_entropy_full.jsonl" --output "${OUT_ROOT}/vstar_smoke20/force_normal_equivalence.json"

run_one vstar "${VSTAR}" 191 cot_orign_greedy --method cot_greedy
run_one vstar "${VSTAR}" 191 lead_force_normal --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_force_normal
run_one vstar "${VSTAR}" 191 initial_soft_only --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_soft_only
run_one vstar "${VSTAR}" 191 initial_transition_only --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only
run_one vstar "${VSTAR}" 191 initial_transition_no_to_normal --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_disable_to_normal_transition
run_one vstar "${VSTAR}" 191 initial_transition_no_linebreak --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_disable_step0_linebreak_mix
run_one vstar "${VSTAR}" 191 initial_transition_cache_rebuild --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_initial_transition_cache_rebuild_after_step 1

for run in cot_orign_greedy initial_soft_only initial_transition_only initial_transition_cache_rebuild; do
  case "${run}" in
    cot_orign_greedy) args=(--method cot_greedy) ;;
    initial_soft_only) args=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_soft_only) ;;
    initial_transition_only) args=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only) ;;
    initial_transition_cache_rebuild) args=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_initial_transition_cache_rebuild_after_step 1) ;;
  esac
  run_one mmvp "${MMVP}" 300 "${run}" "${args[@]}"
  "${PYTHON_BIN}" script/evaluate_specialized_results.py --dataset "${MMVP}" --results "${OUT_ROOT}/mmvp/${run}/results.jsonl" --output_json "${OUT_ROOT}/mmvp/${run}/specialized_eval_report.json" --output_results_jsonl "${OUT_ROOT}/mmvp/${run}/specialized_eval_rows.jsonl"
done

"${PYTHON_BIN}" script/exp7_15/run_transition_same_token_replay.py --root "${ROOT}" --model "${MODEL}" --cot-results "${OUT_ROOT}/vstar/cot_orign_greedy/results.jsonl" --cot-trace "${OUT_ROOT}/vstar/cot_orign_greedy/token_entropy_full.jsonl" --transition-results "${OUT_ROOT}/vstar/initial_transition_only/results.jsonl" --output-dir "${OUT_ROOT}/vstar/same_token_replay" --device cuda --cot-prompt-mode orign --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 1024
"${PYTHON_BIN}" script/exp7_15/summarize_transition_causal.py --root "${OUT_ROOT}"

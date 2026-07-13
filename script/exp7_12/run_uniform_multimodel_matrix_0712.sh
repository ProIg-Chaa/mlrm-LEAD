#!/usr/bin/env bash
# Full uniform matrix: baselines and retained methods over every available model
# and every locally integrated benchmark. Safe to interrupt and resume.
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
PYTHON_BIN=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
GPU="${GPU:-0}"
STAMP=20260712_uniform_multimodel_full_matrix
BASE_DIR="${ROOT}/output/experiments/${STAMP}/uniform_multimodel_full_matrix/${MODEL_KEY}"
SOURCE_MODEL="/share/home/wangzixu/liudinghao/gushuo/models/${MODEL_NAME}"
RAM_MODEL="/dev/shm/wangzixu_models/${MODEL_NAME}"

cd "${ROOT}"
mkdir -p "${BASE_DIR}/logs" "$(dirname "${RAM_MODEL}")"
"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py

for required in config.json model.safetensors.index.json tokenizer.json preprocessor_config.json; do
  [[ -f "${SOURCE_MODEL}/${required}" ]] || { echo "[MISSING] ${SOURCE_MODEL}/${required}"; exit 2; }
done

if [[ ! -f "${RAM_MODEL}/model.safetensors.index.json" ]]; then
  echo "[MODEL] $(date '+%F %T') copy ${MODEL_NAME} to ${RAM_MODEL}"
  mkdir -p "${RAM_MODEL}"
  rsync -a --info=progress2 "${SOURCE_MODEL}/" "${RAM_MODEL}/"
fi

# Some downloaded Qwen-family checkpoints use an older processor class alias.
if grep -q 'Qwen2_5_VLImageProcessor' "${RAM_MODEL}/preprocessor_config.json"; then
  sed -i 's/Qwen2_5_VLImageProcessor/Qwen2VLImageProcessor/' "${RAM_MODEL}/preprocessor_config.json"
fi

DATASETS=(
  "vstar:data/vstar.jsonl:"
  "realworldqa_fixed200:data/realworldqa_fixed_mcq_random200_seed42.jsonl:"
  "mmvp:data/mmvp.jsonl:"
  "visulogic300:data/visulogic.jsonl:300"
  "mmhal:data/mmhal_bench.jsonl:"
  "mathvista200:data/math_vista.jsonl:200"
  "vmcbench_dev:data/vmcbench_dev.jsonl:"
  "mmk12_math:data/mmk12_math.jsonl:"
  "mmk12_physics:data/mmk12_physics.jsonl:"
  "mmk12_chemistry:data/mmk12_chemistry.jsonl:"
  "mmk12_biology:data/mmk12_biology.jsonl:"
  "pope_random:data/pope_random.jsonl:"
  "pope_popular:data/pope_popular.jsonl:"
  "pope_adversarial:data/pope_adversarial.jsonl:"
)

CORE_DATASETS=(
  "vstar:data/vstar.jsonl:"
  "realworldqa_fixed200:data/realworldqa_fixed_mcq_random200_seed42.jsonl:"
  "mmvp:data/mmvp.jsonl:"
  "visulogic300:data/visulogic.jsonl:300"
)

TIMING_DATASETS=(
  "vstar:data/vstar.jsonl:"
  "mmvp:data/mmvp.jsonl:"
)

COMMON=(
  --model_name "${RAM_MODEL}"
  --cot_prompt_mode orign
  --temperature 0.6 --top_p 0.95 --top_k 20
  --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample
  --save_token_entropy --trace_topk 0
)

expected_rows() {
  local dataset_path="$1" limit="$2"
  local all
  all="$(wc -l < "${dataset_path}")"
  if [[ -n "${limit}" && "${limit}" -lt "${all}" ]]; then
    echo "${limit}"
  else
    echo "${all}"
  fi
}

is_complete() {
  local run_dir="$1" expected="$2"
  [[ -f "${run_dir}/eval_report.json" && -f "${run_dir}/results.jsonl" ]] || return 1
  [[ "$(wc -l < "${run_dir}/results.jsonl")" -eq "${expected}" ]]
}

run_one() {
  local method_key="$1" dataset_key="$2" dataset_rel="$3" limit="$4"
  shift 4
  local dataset_path="${ROOT}/${dataset_rel}"
  local run_dir="${BASE_DIR}/${dataset_key}/${method_key}"
  local expected
  expected="$(expected_rows "${dataset_path}" "${limit}")"
  if is_complete "${run_dir}" "${expected}"; then
    echo "[SKIP] ${MODEL_KEY}/${dataset_key}/${method_key} (${expected} rows)"
    return
  fi
  local limit_args=()
  [[ -n "${limit}" ]] && limit_args=(--limit "${limit}")
  mkdir -p "${run_dir}"
  echo "[START] $(date '+%F %T') model=${MODEL_KEY} dataset=${dataset_key} method=${method_key} gpu=${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nice -n 10 "${PYTHON_BIN}" main.py \
      "${COMMON[@]}" --dataset "${dataset_path}" --output_dir "${run_dir}" \
      "${limit_args[@]}" "$@"
  [[ "$(wc -l < "${run_dir}/results.jsonl")" -eq "${expected}" ]] || {
    echo "[INCOMPLETE] ${run_dir}: expected ${expected} rows" >&2
    exit 4
  }
  [[ -f "${run_dir}/eval_report.json" ]] || {
    echo "[MISSING-EVAL] ${run_dir}" >&2
    exit 5
  }
  echo "[DONE] $(date '+%F %T') model=${MODEL_KEY} dataset=${dataset_key} method=${method_key}"
}

run_method_over_all_datasets() {
  local method_key="$1"
  shift
  for spec in "${DATASETS[@]}"; do
    IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"
    run_one "${method_key}" "${dataset_key}" "${dataset_rel}" "${limit}" "$@"
  done
}

run_method_over_core_datasets() {
  local method_key="$1"
  shift
  for spec in "${CORE_DATASETS[@]}"; do
    IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"
    run_one "${method_key}" "${dataset_key}" "${dataset_rel}" "${limit}" "$@"
  done
}

run_method_over_timing_datasets() {
  local method_key="$1"
  shift
  for spec in "${TIMING_DATASETS[@]}"; do
    IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"
    run_one "${method_key}" "${dataset_key}" "${dataset_rel}" "${limit}" "$@"
  done
}

# Tier A: paper-facing full benchmark matrix. Baselines come first, followed by
# the proposed engineering method and the strongest pure-soft stabilizer.
run_method_over_all_datasets cot_orign_greedy --method cot_greedy
run_method_over_all_datasets lead --method lead --alpha 0.4 --max_switch_count 5 --window_size 128
run_method_over_all_datasets transition_preserving_quota05_guard_min2 --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 --format_cooldown_min_step 2 --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 --lead_veto_recent_repeat_tau 0.35
run_method_over_all_datasets pure_soft_guard --method pure_soft --pure_soft_format_cooldown --format_cooldown_steps 2 --pure_soft_collapse_on_diffuse --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 64 --collapse_recent_repeat_tau 0.35

# Tier B: insight-essential controls on the four core benchmarks.
run_method_over_core_datasets pure_soft --method pure_soft
run_method_over_core_datasets initial_soft_only --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_soft_only
run_method_over_core_datasets initial_transition_only --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only
run_method_over_core_datasets initial_transition_no_to_normal --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_disable_to_normal_transition
run_method_over_core_datasets initial_transition_no_anchor --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_disable_simple_visual_anchor

# Tier C: compact timing controls for the early-commitment claim.
run_method_over_timing_datasets transition_step2 --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_initial_transition_delay_steps 2
run_method_over_timing_datasets transition_step16 --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only --lead_initial_transition_delay_steps 16

echo "[ALL-DONE] $(date '+%F %T') model=${MODEL_KEY}"

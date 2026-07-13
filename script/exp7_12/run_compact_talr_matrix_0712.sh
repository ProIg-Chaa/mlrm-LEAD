#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
PYTHON_BIN=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
GROUP="${GROUP:?GROUP is required: baseline, methods, or openvl}"
GPU="${GPU:-0}"
BASE_DIR="${ROOT}/output/experiments/20260712_uniform_multimodel_full_matrix/uniform_multimodel_full_matrix/${MODEL_KEY}"
SOURCE_MODEL="/share/home/wangzixu/liudinghao/gushuo/models/${MODEL_NAME}"
RAM_MODEL="/dev/shm/wangzixu_models/${MODEL_NAME}"

MAIN_DATASETS=(
  "vstar:data/vstar.jsonl:"
  "realworldqa_fixed200:data/realworldqa_fixed_mcq_random200_seed42.jsonl:"
  "mmvp:data/mmvp.jsonl:"
  "visulogic300:data/visulogic.jsonl:300"
  "vmcbench_dev:data/vmcbench_dev.jsonl:"
  "pope_adversarial:data/pope_adversarial.jsonl:"
  "mmk12_physics:data/mmk12_physics.jsonl:"
)

OPENVL_DATASETS=(
  "vstar:data/vstar.jsonl:"
  "mmvp:data/mmvp.jsonl:"
)

cd "${ROOT}"
mkdir -p "${BASE_DIR}/logs" "$(dirname "${RAM_MODEL}")"
"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py

for required in config.json model.safetensors.index.json tokenizer.json preprocessor_config.json; do
  [[ -f "${SOURCE_MODEL}/${required}" ]] || { echo "[MISSING] ${SOURCE_MODEL}/${required}"; exit 2; }
done
if [[ ! -f "${RAM_MODEL}/model.safetensors.index.json" ]]; then
  echo "[MODEL] $(date '+%F %T') copy ${MODEL_NAME} to RAM"
  mkdir -p "${RAM_MODEL}"
  rsync -a --info=progress2 "${SOURCE_MODEL}/" "${RAM_MODEL}/"
fi
if grep -q 'Qwen2_5_VLImageProcessor' "${RAM_MODEL}/preprocessor_config.json"; then
  sed -i 's/Qwen2_5_VLImageProcessor/Qwen2VLImageProcessor/' "${RAM_MODEL}/preprocessor_config.json"
fi

COMMON=(
  --model_name "${RAM_MODEL}"
  --cot_prompt_mode orign
  --temperature 0.6 --top_p 0.95 --top_k 20
  --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample
  --save_token_entropy --trace_topk 0
)

expected_rows() {
  local path="$1" limit="$2" count
  count="$(wc -l < "${path}")"
  if [[ -n "${limit}" && "${limit}" -lt "${count}" ]]; then echo "${limit}"; else echo "${count}"; fi
}

is_complete() {
  local run_dir="$1" expected="$2"
  [[ -f "${run_dir}/config.json" && -f "${run_dir}/results.jsonl" && -f "${run_dir}/eval_report.json" && -f "${run_dir}/token_entropy.jsonl" ]] || return 1
  [[ "$(wc -l < "${run_dir}/results.jsonl")" -eq "${expected}" ]]
}

run_one() {
  local dataset_key="$1" dataset_rel="$2" limit="$3" method_key="$4"
  shift 4
  local dataset_path="${ROOT}/${dataset_rel}"
  local run_dir="${BASE_DIR}/${dataset_key}/${method_key}"
  local expected limit_args=()
  expected="$(expected_rows "${dataset_path}" "${limit}")"
  if is_complete "${run_dir}" "${expected}"; then
    echo "[SKIP] ${MODEL_KEY}/${dataset_key}/${method_key} (${expected} rows)"
    return
  fi
  [[ -n "${limit}" ]] && limit_args=(--limit "${limit}")
  mkdir -p "${run_dir}"
  echo "[START] $(date '+%F %T') model=${MODEL_KEY} group=${GROUP} dataset=${dataset_key} method=${method_key} gpu=${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nice -n 10 "${PYTHON_BIN}" main.py "${COMMON[@]}" \
      --dataset "${dataset_path}" --output_dir "${run_dir}" "${limit_args[@]}" "$@"
  is_complete "${run_dir}" "${expected}" || { echo "[INCOMPLETE] ${run_dir}" >&2; exit 4; }
  echo "[DONE] $(date '+%F %T') model=${MODEL_KEY} dataset=${dataset_key} method=${method_key}"
}

run_over() {
  local list_name="$1" method_key="$2"
  shift 2
  local specs=()
  if [[ "${list_name}" == main ]]; then specs=("${MAIN_DATASETS[@]}"); else specs=("${OPENVL_DATASETS[@]}"); fi
  for spec in "${specs[@]}"; do
    IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"
    run_one "${dataset_key}" "${dataset_rel}" "${limit}" "${method_key}" "$@"
  done
}

run_baseline() {
  local list_name="$1"
  run_over "${list_name}" cot_orign_greedy --method cot_greedy
  run_over "${list_name}" lead --method lead --alpha 0.4 --max_switch_count 5 --window_size 128
}

run_methods() {
  local list_name="$1"
  run_over "${list_name}" initial_transition_only --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only
  run_over "${list_name}" transition_preserving_quota05_guard_min2 \
    --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 \
    --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 \
    --format_cooldown_min_step 2 --lead_soft_veto_on_diffuse \
    --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 \
    --lead_veto_recent_repeat_window 64 --lead_veto_recent_repeat_tau 0.35
}

case "${GROUP}" in
  baseline) run_baseline main ;;
  methods) run_methods main ;;
  openvl) run_baseline openvl; run_methods openvl ;;
  *) echo "Unknown GROUP=${GROUP}" >&2; exit 2 ;;
esac

echo "[ALL-DONE] $(date '+%F %T') model=${MODEL_KEY} group=${GROUP}"

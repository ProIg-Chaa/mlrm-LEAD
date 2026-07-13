#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
MODEL_NAME="${MODEL_NAME:-R1-Onevision-7B}"
MODEL_KEY="${MODEL_KEY:-r1_onevision_7b_correct}"
SOURCE_MODEL="/share/home/wangzixu/liudinghao/gushuo/models/${MODEL_NAME}"
RAM_MODEL="/dev/shm/wangzixu_models/${MODEL_NAME}"
PYTHON_BIN=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
BASE_DIR="${ROOT}/output/experiments/20260711_correct_model_our_methods_priority/our_methods_priority"
GPU="${GPU:-0}"

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
if (( free_mib < 30000 )); then
  echo "[ABORT] GPU${GPU} has only ${free_mib} MiB free; need at least 30000 MiB"
  exit 3
fi

cd "${ROOT}"
mkdir -p "${BASE_DIR}/logs" "$(dirname "${RAM_MODEL}")"
"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py

if [[ ! -f "${RAM_MODEL}/model.safetensors.index.json" ]]; then
  echo "[MODEL] $(date '+%F %T') copying ${MODEL_NAME} to RAM"
  mkdir -p "${RAM_MODEL}"
  rsync -a --info=progress2 "${SOURCE_MODEL}/" "${RAM_MODEL}/"
fi
if grep -q 'Qwen2_5_VLImageProcessor' "${RAM_MODEL}/preprocessor_config.json"; then
  sed -i 's/Qwen2_5_VLImageProcessor/Qwen2VLImageProcessor/' \
    "${RAM_MODEL}/preprocessor_config.json"
fi

COMMON=(
  --model_name "${RAM_MODEL}"
  --max_new_tokens 1024
  --temperature 0.6
  --top_p 0.95
  --top_k 20
  --seed 42
  --device cuda
  --no-do_sample
  --save_token_entropy
  --save_full_token_entropy
  --trace_topk 0
  --cot_prompt_mode orign
)

DATASETS=(
  "vstar:data/vstar.jsonl:"
  "realworldqa_fixed200:data/realworldqa_fixed_mcq_random200_seed42.jsonl:"
  "mmvp:data/mmvp.jsonl:"
  "visulogic300:data/visulogic.jsonl:300"
)

run_one() {
  local dataset_key="$1" dataset_rel="$2" limit="$3" method_key="$4"
  shift 4
  local run_dir="${BASE_DIR}/${MODEL_KEY}/${dataset_key}/${method_key}_gpu${GPU}"
  if [[ -f "${run_dir}/eval_report.json" ]]; then
    echo "[SKIP] ${dataset_key}/${method_key}"
    return
  fi
  local limit_args=()
  if [[ -n "${limit}" ]]; then limit_args=(--limit "${limit}"); fi
  mkdir -p "${run_dir}"
  echo "[START] $(date '+%F %T') ${dataset_key}/${method_key}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nice -n 10 "${PYTHON_BIN}" main.py \
      "${COMMON[@]}" \
      --dataset "${ROOT}/${dataset_rel}" \
      --output_dir "${run_dir}" \
      "${limit_args[@]}" \
      "$@"
  echo "[DONE] $(date '+%F %T') ${dataset_key}/${method_key}"
}

for spec in "${DATASETS[@]}"; do
  IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"

  run_one "${dataset_key}" "${dataset_rel}" "${limit}" pure_soft_format2 \
    --method pure_soft --pure_soft_format_cooldown --format_cooldown_steps 2
  run_one "${dataset_key}" "${dataset_rel}" "${limit}" pure_soft_guard \
    --method pure_soft --pure_soft_format_cooldown --format_cooldown_steps 2 \
    --pure_soft_collapse_on_diffuse --collapse_require_repeat_degen \
    --collapse_repeat_ngram 3 --collapse_recent_repeat_window 64 \
    --collapse_recent_repeat_tau 0.35
  run_one "${dataset_key}" "${dataset_rel}" "${limit}" quota05_format2 \
    --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 \
    --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2
  run_one "${dataset_key}" "${dataset_rel}" "${limit}" quota05_guard \
    --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 \
    --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 \
    --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen \
    --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 \
    --lead_veto_recent_repeat_tau 0.35
  run_one "${dataset_key}" "${dataset_rel}" "${limit}" lead_format2 \
    --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 \
    --lead_format_cooldown --format_cooldown_steps 2
  run_one "${dataset_key}" "${dataset_rel}" "${limit}" lead_guard \
    --method lead --alpha 0.4 --max_switch_count 5 --window_size 128 \
    --lead_format_cooldown --format_cooldown_steps 2 \
    --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen \
    --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 \
    --lead_veto_recent_repeat_tau 0.35
done

echo "[ALL_DONE] $(date '+%F %T')"

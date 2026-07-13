#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
SOURCE_MODEL=/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B
RAM_MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B
PYTHON_BIN=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
BASE_DIR="${ROOT}/output/experiments/20260710_correct_model_cot_lead_audit/correct_model_cot_lead_audit"
GPU="${GPU:-0}"

cd "${ROOT}"
mkdir -p "${BASE_DIR}/logs" "$(dirname "${RAM_MODEL}")"

"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py

if [[ ! -f "${RAM_MODEL}/model.safetensors.index.json" ]]; then
  echo "[MODEL] $(date '+%F %T') copying model to RAM"
  rsync -a --info=progress2 "${SOURCE_MODEL}/" "${RAM_MODEL}/"
fi

# This transformers build recognizes the equivalent legacy Qwen2-VL image
# processor class name used by the existing R1-Onevision checkpoint.
if grep -q 'Qwen2_5_VLImageProcessor' "${RAM_MODEL}/preprocessor_config.json"; then
  echo "[MODEL] applying RAM-only image processor compatibility alias"
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
)

run_one() {
  local dataset_key="$1"
  local dataset_path="$2"
  local method_key="$3"
  shift 3
  local run_dir="${BASE_DIR}/r1_onevision_7b_correct/${dataset_key}/${method_key}_gpu${GPU}"

  if [[ -f "${run_dir}/eval_report.json" ]]; then
    echo "[SKIP] ${dataset_key}/${method_key} already complete"
    return
  fi

  mkdir -p "${run_dir}"
  echo "[START] $(date '+%F %T') ${dataset_key}/${method_key}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" main.py \
      "${COMMON[@]}" \
      --dataset "${ROOT}/${dataset_path}" \
      --output_dir "${run_dir}" \
      "$@"
  echo "[DONE] $(date '+%F %T') ${dataset_key}/${method_key}"
}

for spec in \
  "vstar:data/vstar.jsonl" \
  "mmvp:data/mmvp.jsonl" \
  "mmk12_physics:data/mmk12_physics.jsonl" \
  "pope_random:data/pope_random.jsonl"
do
  IFS=: read -r dataset_key dataset_path <<<"${spec}"
  run_one "${dataset_key}" "${dataset_path}" cot_orign_greedy \
    --method cot_greedy --cot_prompt_mode orign
  run_one "${dataset_key}" "${dataset_path}" lead \
    --method lead --cot_prompt_mode orign \
    --alpha 0.4 --max_switch_count 5 --window_size 128
done

echo "[ALL_DONE] $(date '+%F %T')"

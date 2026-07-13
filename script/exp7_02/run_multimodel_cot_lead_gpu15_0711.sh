#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
MODEL_ROOT=/share/home/wangzixu/liudinghao/gushuo/models
RAM_ROOT=/dev/shm/wangzixu_models
PYTHON_BIN=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
BASE_DIR="${ROOT}/output/experiments/20260711_multimodel_cot_lead_baselines/integrated_repo_cot_lead_baselines"

MODELS=(
  "Vision-R1-7B:vision_r1_7b"
  "VL-Cogito-7B:vl_cogito_7b"
  "OpenVLThinker-7B:openvlthinker_7b"
)

DATASETS=(
  "vstar:data/vstar.jsonl:"
  "realworldqa_fixed200:data/realworldqa_fixed_mcq_random200_seed42.jsonl:"
  "mmvp:data/mmvp.jsonl:"
  "visulogic300:data/visulogic.jsonl:300"
  "vmcbench_dev:data/vmcbench_dev.jsonl:"
  "mmk12_math:data/mmk12_math.jsonl:"
  "mmk12_physics:data/mmk12_physics.jsonl:"
  "mmk12_chemistry:data/mmk12_chemistry.jsonl:"
  "mmk12_biology:data/mmk12_biology.jsonl:"
  "pope_random:data/pope_random.jsonl:"
  "pope_popular:data/pope_popular.jsonl:"
  "pope_adversarial:data/pope_adversarial.jsonl:"
)

COMMON=(
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

cd "${ROOT}"
mkdir -p "${BASE_DIR}/logs" "${RAM_ROOT}"
"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py

ensure_ram_model() {
  local model_name="$1"
  local source="${MODEL_ROOT}/${model_name}"
  local target="${RAM_ROOT}/${model_name}"
  if [[ ! -f "${target}/model.safetensors.index.json" ]]; then
    echo "[MODEL] $(date '+%F %T') copying ${model_name} to RAM"
    mkdir -p "${target}"
    rsync -a --info=progress2 "${source}/" "${target}/"
  fi
}

run_one() {
  local model_name="$1"
  local model_key="$2"
  local dataset_key="$3"
  local dataset_rel="$4"
  local limit="$5"
  local method_key="$6"
  shift 6

  local run_dir="${BASE_DIR}/${model_key}/${dataset_key}/${method_key}_gpu0"
  if [[ -f "${run_dir}/eval_report.json" ]]; then
    echo "[SKIP] ${model_key}/${dataset_key}/${method_key}"
    return
  fi

  local limit_args=()
  if [[ -n "${limit}" ]]; then
    limit_args=(--limit "${limit}")
  fi

  mkdir -p "${run_dir}"
  echo "[START] $(date '+%F %T') ${model_key}/${dataset_key}/${method_key}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" main.py \
      --model_name "${RAM_ROOT}/${model_name}" \
      --dataset "${ROOT}/${dataset_rel}" \
      --output_dir "${run_dir}" \
      "${COMMON[@]}" \
      "${limit_args[@]}" \
      "$@"
  echo "[DONE] $(date '+%F %T') ${model_key}/${dataset_key}/${method_key}"
}

for model_spec in "${MODELS[@]}"; do
  IFS=: read -r model_name model_key <<<"${model_spec}"
  ensure_ram_model "${model_name}"
done

for dataset_spec in "${DATASETS[@]}"; do
  IFS=: read -r dataset_key dataset_rel limit <<<"${dataset_spec}"
  for model_spec in "${MODELS[@]}"; do
    IFS=: read -r model_name model_key <<<"${model_spec}"
    run_one "${model_name}" "${model_key}" "${dataset_key}" "${dataset_rel}" "${limit}" \
      cot_orign_greedy --method cot_greedy --cot_prompt_mode orign
    run_one "${model_name}" "${model_key}" "${dataset_key}" "${dataset_rel}" "${limit}" \
      lead --method lead --cot_prompt_mode orign \
      --alpha 0.4 --max_switch_count 5 --window_size 128
  done
done

echo "[ALL_DONE] $(date '+%F %T')"

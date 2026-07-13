#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python}"
MODEL_NAME="${MODEL_NAME:-R1-Onevision-7B-RL}"
MODEL_KEY="${MODEL_KEY:-r1_onevision_7b}"
SOURCE_MODEL="${SOURCE_MODEL:-/share/home/wangzixu/liudinghao/gushuo/models/${MODEL_NAME}}"
RAM_MODEL="${RAM_MODEL:-/dev/shm/wangzixu_models/${MODEL_NAME}}"
STAMP="${STAMP:-20260705_integrated_cot_lead_baselines}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/integrated_repo_cot_lead_baselines}"
GPU="${GPU:-0}"
GROUP="${1:-cot}"

cd "${ROOT}"

COMMON_ARGS=(
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

COT_ARGS=(--method cot_greedy --cot_prompt_mode orign)
LEAD_ARGS=(--method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128)

DATASETS=(
  "mmhal:data/mmhal_bench.jsonl:"
  "vstar:data/vstar.jsonl:"
  "realworldqa_fixed200:data/realworldqa_fixed_mcq_random200_seed42.jsonl:"
  "mmvp:data/mmvp.jsonl:"
  "visulogic300:data/visulogic.jsonl:300"
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

ensure_ram_model() {
  if [[ ! -f "${RAM_MODEL}/config.json" || ! -f "${RAM_MODEL}/model.safetensors.index.json" ]]; then
    echo "[MODEL] Copying ${SOURCE_MODEL} -> ${RAM_MODEL}"
    mkdir -p "$(dirname "${RAM_MODEL}")"
    rsync -a --info=progress2 "${SOURCE_MODEL}/" "${RAM_MODEL}/"
  else
    echo "[MODEL] Using RAM model at ${RAM_MODEL}"
  fi
}

run_one() {
  local dataset_key="$1"
  local dataset_rel="$2"
  local limit="$3"
  local method_name="$4"
  shift 4
  local method_args=("$@")
  local run_dir="${BASE_DIR}/${MODEL_KEY}/${dataset_key}/${method_name}_gpu${GPU}"
  local dataset_path="${ROOT}/${dataset_rel}"

  mkdir -p "${run_dir}"
  echo "[START] $(date '+%F %T') model=${MODEL_KEY} dataset=${dataset_key} method=${method_name} gpu=${GPU}"

  local limit_args=()
  if [[ -n "${limit}" ]]; then
    limit_args=(--limit "${limit}")
  fi

  export CUDA_VISIBLE_DEVICES="${GPU}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "${PYTHON_BIN}" main.py \
    --model_name "${RAM_MODEL}" \
    --dataset "${dataset_path}" \
    --output_dir "${run_dir}" \
    "${COMMON_ARGS[@]}" \
    "${limit_args[@]}" \
    "${method_args[@]}"

  echo "[DONE]  $(date '+%F %T') model=${MODEL_KEY} dataset=${dataset_key} method=${method_name} gpu=${GPU}"
}

"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py script/exp7_02/summarize_integrated_cot_lead_baselines_0705.py
ensure_ram_model

case "${GROUP}" in
  cot)
    for spec in "${DATASETS[@]}"; do
      IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"
      run_one "${dataset_key}" "${dataset_rel}" "${limit}" cot_orign_greedy "${COT_ARGS[@]}"
    done
    ;;
  lead)
    for spec in "${DATASETS[@]}"; do
      IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"
      run_one "${dataset_key}" "${dataset_rel}" "${limit}" lead "${LEAD_ARGS[@]}"
    done
    ;;
  *)
    echo "usage: $0 {cot|lead}" >&2
    exit 2
    ;;
esac


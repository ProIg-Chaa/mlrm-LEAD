#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
MODEL_ROOT=/share/home/wangzixu/liudinghao/gushuo/models
RAM_ROOT=/dev/shm/wangzixu_models
PYTHON_BIN=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
BASE_DIR="${ROOT}/output/experiments/20260711_paper_aligned_vstar_priority/paper_aligned_vstar"
GPU="${GPU:-0}"

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
if (( free_mib < 30000 )); then
  echo "[ABORT] GPU${GPU} has only ${free_mib} MiB free; paper-aligned run requires at least 30000 MiB"
  exit 3
fi

# OpenVLThinker is first because its local COT differs from Table 2 by 12.53 pp.
MODELS=(
  "OpenVLThinker-7B:openvlthinker_7b"
  "Vision-R1-7B:vision_r1_7b"
  "VL-Cogito-7B:vl_cogito_7b"
)

cd "${ROOT}"
mkdir -p "${BASE_DIR}/logs" "${RAM_ROOT}"
"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py

ensure_ram_model() {
  local model_name="$1"
  local source="${MODEL_ROOT}/${model_name}"
  local target="${RAM_ROOT}/${model_name}"
  for required in config.json model.safetensors.index.json tokenizer.json preprocessor_config.json; do
    [[ -f "${source}/${required}" ]] || { echo "[MISSING] ${source}/${required}"; exit 2; }
  done
  if [[ ! -f "${target}/model.safetensors.index.json" ]]; then
    echo "[MODEL] $(date '+%F %T') copying ${model_name} to RAM"
    mkdir -p "${target}"
    rsync -a --info=progress2 "${source}/" "${target}/"
  fi
}

run_one() {
  local model_name="$1" model_key="$2" scope="$3" method_key="$4"
  shift 4
  local run_dir="${BASE_DIR}/${model_key}/vstar/${scope}/${method_key}_gpu${GPU}"
  local limit_args=()
  [[ "${scope}" == smoke20 ]] && limit_args=(--limit 20)
  if [[ -f "${run_dir}/eval_report.json" ]]; then
    echo "[SKIP] ${model_key}/${scope}/${method_key}"
    return
  fi
  mkdir -p "${run_dir}"
  echo "[START] $(date '+%F %T') ${model_key}/${scope}/${method_key}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" main.py \
      --model_name "${RAM_ROOT}/${model_name}" \
      --dataset "${ROOT}/data/vstar.jsonl" \
      --output_dir "${run_dir}" \
      --cot_prompt_mode orign \
      --temperature 0.6 --top_p 0.95 --top_k 20 \
      --seed 42 --max_new_tokens 25600 --device cuda \
      --do_sample --save_token_entropy --trace_topk 0 \
      "${limit_args[@]}" "$@"
  echo "[DONE] $(date '+%F %T') ${model_key}/${scope}/${method_key}"
}

for spec in "${MODELS[@]}"; do
  IFS=: read -r model_name model_key <<<"${spec}"
  ensure_ram_model "${model_name}"

  # Paper-known primary protocol: sampled discrete COT and sampled LEAD use the
  # same decoder settings. Paper lambda=0.4 maps to released-code alpha=0.6.
  for scope in smoke20 full191; do
    run_one "${model_name}" "${model_key}" "${scope}" cot_paper_sampled \
      --method cot
    run_one "${model_name}" "${model_key}" "${scope}" lead_paper_sampled_a06_w128 \
      --method lead --alpha 0.6 --max_switch_count 5 --window_size 128
  done

  # OpenVLThinker-only controls isolate the old greedy/1024 protocol and the
  # released code's default window=256 from the paper-text protocol.
  if [[ "${model_key}" == openvlthinker_7b ]]; then
    run_one "${model_name}" "${model_key}" full191 cot_greedy_25600 \
      --method cot_greedy
    run_one "${model_name}" "${model_key}" full191 lead_released_default_w256 \
      --method lead --alpha 0.6 --max_switch_count 5 --window_size 256
  fi
done

"${PYTHON_BIN}" "${ROOT}/script/exp7_02/summarize_paper_aligned_vstar_0711.py" \
  --base-dir "${BASE_DIR}"
echo "[ALL_DONE] $(date '+%F %T')"

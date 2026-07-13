#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python}"
MODEL_NAME="${MODEL_NAME:-R1-Onevision-7B-RL}"
MODEL_KEY="${MODEL_KEY:-r1_onevision_7b}"
SOURCE_MODEL="${SOURCE_MODEL:-/share/home/wangzixu/liudinghao/gushuo/models/${MODEL_NAME}}"
RAM_MODEL="${RAM_MODEL:-/dev/shm/wangzixu_models/${MODEL_NAME}}"
STAMP="${STAMP:-20260706_format_stability_full_baselines}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/format_stability_full_baselines}"
GPU="${GPU:-0}"
GROUP="${1:-main_a}"

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

DIAG_DATASETS=(
  "vstar:data/vstar.jsonl:"
  "mmvp:data/mmvp.jsonl:"
  "visulogic300:data/visulogic.jsonl:300"
  "mmk12_physics:data/mmk12_physics.jsonl:"
  "vmcbench_dev:data/vmcbench_dev.jsonl:"
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

check_datasets() {
  local missing=0
  for spec in "${DATASETS[@]}" "${DIAG_DATASETS[@]}"; do
    IFS=: read -r _ dataset_rel _limit <<<"${spec}"
    if [[ ! -s "${ROOT}/${dataset_rel}" ]]; then
      echo "[MISSING] ${ROOT}/${dataset_rel}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    exit 3
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

  if [[ -s "${run_dir}/results.jsonl" && -s "${run_dir}/eval_report.json" && -s "${run_dir}/token_entropy_full.jsonl" ]]; then
    echo "[SKIP] $(date '+%F %T') ${MODEL_KEY} ${dataset_key} ${method_name}: existing complete-looking run"
    return 0
  fi

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

  if [[ "${dataset_key}" == "mmvp" ]]; then
    "${PYTHON_BIN}" script/evaluate_specialized_results.py \
      --dataset "${dataset_path}" \
      --results "${run_dir}/results.jsonl" \
      --mode mmvp \
      --output_json "${run_dir}/specialized_eval_report.json" \
      --output_results_jsonl "${run_dir}/specialized_eval_results.jsonl"
  elif [[ "${dataset_key}" == "realworldqa_fixed200" ]]; then
    "${PYTHON_BIN}" script/evaluate_realworldqa_mcq.py \
      --dataset "${dataset_path}" \
      --results "${run_dir}/results.jsonl" \
      --output_json "${run_dir}/realworldqa_mcq_eval.json" \
      --output_results_jsonl "${run_dir}/realworldqa_mcq_results.jsonl"
  fi

  echo "[DONE]  $(date '+%F %T') model=${MODEL_KEY} dataset=${dataset_key} method=${method_name} gpu=${GPU}"
}

run_method_over() {
  local dataset_array_name="$1"
  local method_name="$2"
  shift 2
  local method_args=("$@")
  local specs=()
  case "${dataset_array_name}" in
    main) specs=("${DATASETS[@]}") ;;
    diag) specs=("${DIAG_DATASETS[@]}") ;;
    *) echo "unknown dataset group: ${dataset_array_name}" >&2; exit 2 ;;
  esac
  for spec in "${specs[@]}"; do
    IFS=: read -r dataset_key dataset_rel limit <<<"${spec}"
    run_one "${dataset_key}" "${dataset_rel}" "${limit}" "${method_name}" "${method_args[@]}"
  done
}

"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py
check_datasets
ensure_ram_model

case "${GROUP}" in
  main_a)
    run_method_over main pure_soft_format2 \
      --method pure_soft --cot_prompt_mode orign \
      --pure_soft_format_cooldown --format_cooldown_steps 2
    run_method_over main pure_soft_guard \
      --method pure_soft --cot_prompt_mode orign \
      --pure_soft_format_cooldown --format_cooldown_steps 2 \
      --pure_soft_collapse_on_diffuse --collapse_require_repeat_degen \
      --collapse_repeat_ngram 3 --collapse_recent_repeat_window 64 --collapse_recent_repeat_tau 0.35
    run_method_over main lead_format2 \
      --method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 \
      --lead_format_cooldown --format_cooldown_steps 2
    ;;
  main_b)
    run_method_over main quota05 \
      --method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 \
      --lead_soft_quota_ratio 0.05
    run_method_over main quota05_format2 \
      --method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 \
      --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2
    run_method_over main quota05_guard \
      --method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 \
      --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 \
      --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen \
      --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 --lead_veto_recent_repeat_tau 0.35
    run_method_over main lead_guard \
      --method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128 \
      --lead_format_cooldown --format_cooldown_steps 2 \
      --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen \
      --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 --lead_veto_recent_repeat_tau 0.35
    ;;
  diag_a)
    run_method_over diag pure_soft \
      --method pure_soft --cot_prompt_mode orign
    run_method_over diag pure_soft_diffuse_collapse \
      --method pure_soft --cot_prompt_mode orign \
      --pure_soft_collapse_on_diffuse
    run_method_over diag answer_zone_discrete \
      --method pure_soft --cot_prompt_mode orign \
      --pure_soft_answer_zone_discrete
    ;;
  diag_b)
    run_method_over diag format_cooldown4 \
      --method pure_soft --cot_prompt_mode orign \
      --pure_soft_format_cooldown --format_cooldown_steps 4
    run_method_over diag highrisk_only_cooldown2 \
      --method pure_soft --cot_prompt_mode orign \
      --pure_soft_format_cooldown --format_cooldown_steps 2 --format_cooldown_highrisk_only
    ;;
  *)
    echo "usage: $0 {main_a|main_b|diag_a|diag_b}" >&2
    exit 2
    ;;
esac

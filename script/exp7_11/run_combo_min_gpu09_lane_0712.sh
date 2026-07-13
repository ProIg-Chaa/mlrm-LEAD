#!/usr/bin/env bash
set -euo pipefail

LANE=${1:?usage: run_combo_min_gpu09_lane_0712.sh <0|1>}
ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
PYTHON=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
BASE=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/transition_preserving_combo
STATE=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/gpu09_combo_state

cd "${ROOT}"
mkdir -p "${STATE}"
${PYTHON} -m py_compile main.py lead/inference.py lead/generation_utils.py

free_mib="$(${PYTHON} -c 'import torch; print(torch.cuda.mem_get_info(0)[0] // 1024 // 1024)')"
if (( free_mib < 30000 )); then
  echo "[ABORT] lane ${LANE}: only ${free_mib} MiB free"
  exit 3
fi

common=(
  --model_name "${MODEL}" --max_new_tokens 1024
  --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42
  --device cuda --no-do_sample --cot_prompt_mode orign
  --save_token_entropy --save_full_token_entropy --trace_topk 0
  --method lead --alpha 0.4 --max_switch_count 5 --window_size 128
  --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2
  --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen
  --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64
  --lead_veto_recent_repeat_tau 0.35 --lead_veto_min_step 64
)

run_one() {
  local dataset=$1 data=$2 name=$3 min_step=$4 limit=${5:-}
  local out=${BASE}/${dataset}/${name}
  if [[ -f ${out}/eval_report.json ]]; then
    echo "[SKIP] ${dataset}/${name}"
    return
  fi
  local limit_args=()
  if [[ -n ${limit} ]]; then limit_args=(--limit "${limit}"); fi
  echo "[START] $(date '+%F %T') lane=${LANE} ${dataset}/${name}"
  ${PYTHON} main.py --dataset "${ROOT}/${data}" --output_dir "${out}" \
    "${common[@]}" --format_cooldown_min_step "${min_step}" "${limit_args[@]}"
  echo "[DONE] $(date '+%F %T') lane=${LANE} ${dataset}/${name}"
}

if [[ ${LANE} == 0 ]]; then
  echo "[WAIT] gpu11 completes VisuLogic min0"
  while [[ ! -f ${STATE}/gpu11_released ]]; do sleep 10; done
  run_one visulogic300 data/visulogic.jsonl transition_preserving_quota05_guard_min2 2 300
elif [[ ${LANE} == 1 ]]; then
  run_one realworldqa_fixed200 data/realworldqa_fixed_mcq_random200_seed42.jsonl quota05_guard_min0 0
  run_one realworldqa_fixed200 data/realworldqa_fixed_mcq_random200_seed42.jsonl transition_preserving_quota05_guard_min2 2
else
  echo "Unknown lane ${LANE}" >&2
  exit 2
fi

touch "${STATE}/lane${LANE}.done"

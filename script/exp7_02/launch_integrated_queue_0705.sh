#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
JOBID="${JOBID:-26484}"
NODE="${NODE:-gpu09}"
GROUP="${1:?usage: $0 cot-or-lead gpu [model_name] [model_key]}"
GPU="${2:?usage: $0 cot-or-lead gpu [model_name] [model_key]}"
MODEL_NAME="${3:-R1-Onevision-7B-RL}"
MODEL_KEY="${4:-r1_onevision_7b}"
STAMP="${STAMP:-20260705_integrated_cot_lead_baselines}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/integrated_repo_cot_lead_baselines}"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

export MODEL_NAME
export MODEL_KEY
export SOURCE_MODEL="${SOURCE_MODEL:-/share/home/wangzixu/liudinghao/gushuo/models/${MODEL_NAME}}"
export RAM_MODEL="${RAM_MODEL:-/dev/shm/wangzixu_models/${MODEL_NAME}}"
export GPU
export STAMP
export BASE_DIR

log="${LOG_DIR}/${MODEL_KEY}_${GROUP}_gpu${GPU}.log"
echo "[LAUNCH] $(date '+%F %T') node=${NODE} job=${JOBID} model=${MODEL_KEY} group=${GROUP} gpu=${GPU}" | tee -a "${log}"
srun --jobid="${JOBID}" --overlap -w "${NODE}" bash -lc "cd '${ROOT}' && GPU='${GPU}' MODEL_NAME='${MODEL_NAME}' MODEL_KEY='${MODEL_KEY}' SOURCE_MODEL='${SOURCE_MODEL}' RAM_MODEL='${RAM_MODEL}' STAMP='${STAMP}' BASE_DIR='${BASE_DIR}' bash script/exp7_02/run_integrated_cot_lead_baselines_0705.sh '${GROUP}'" 2>&1 | tee -a "${log}"
status=${PIPESTATUS[0]}
echo "[EXIT] $(date '+%F %T') model=${MODEL_KEY} group=${GROUP} gpu=${GPU} status=${status}" | tee -a "${log}"
exit "${status}"

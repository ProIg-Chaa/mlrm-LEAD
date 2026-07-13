#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-26691}"
RUN_SCRIPT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/exp7_02/run_correct_model_minimal_0710.sh
LOG=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/correct_model_cot_lead_audit_0710.launch.log

while true; do
  state="$(squeue -h -j "${JOB_ID}" -o '%T')"
  if [[ -z "${state}" ]]; then
    echo "[WATCH] $(date '+%F %T') job ${JOB_ID} disappeared" | tee -a "${LOG}"
    exit 1
  fi
  if [[ "${state}" == "RUNNING" ]]; then
    node="$(squeue -h -j "${JOB_ID}" -o '%N')"
    echo "[WATCH] $(date '+%F %T') job ${JOB_ID} running on ${node}" | tee -a "${LOG}"
    srun --jobid="${JOB_ID}" --overlap -w "${node}" bash "${RUN_SCRIPT}" 2>&1 | tee -a "${LOG}"
    exit "${PIPESTATUS[0]}"
  fi
  echo "[WATCH] $(date '+%F %T') job ${JOB_ID} state=${state}"
  sleep 30
done

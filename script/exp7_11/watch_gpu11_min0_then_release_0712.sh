#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
REPORT=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/transition_preserving_combo/visulogic300/quota05_guard_min0/eval_report.json
STATE=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/gpu09_combo_state
JOB_ID=26729

mkdir -p "${STATE}"
while [[ ! -f ${REPORT} ]]; do
  echo "[WAIT] $(date '+%F %T') gpu11 VisuLogic min0"
  sleep 10
done

echo "[RELEASE] $(date '+%F %T') min0 complete; cancel ${JOB_ID} before the next run"
scancel "${JOB_ID}" 2>/dev/null || true
while squeue -h -j "${JOB_ID}" | grep -q .; do sleep 2; done
touch "${STATE}/gpu11_released"

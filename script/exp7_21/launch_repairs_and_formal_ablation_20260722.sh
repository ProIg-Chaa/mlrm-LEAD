#!/usr/bin/env bash
set -euo pipefail

REPO=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260722_talr_formal_ablation
mkdir -p "${ROOT}"

while pgrep -f '^/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python (main.py|script/exp7_21/run_locked_l095_openvl_sequential_20260721.py)' >/dev/null; do
  date '+%F %T | waiting for locked OpenVL queue' >> "${ROOT}/watcher.log"
  sleep 60
done

cd "${REPO}"
"${PYTHON}" script/exp7_21/run_pre_ablation_repairs_20260722.py >> "${ROOT}/repair.log" 2>&1
exec "${PYTHON}" script/exp7_21/run_formal_ablation_20260722_final.py >> "${ROOT}/launcher.log" 2>&1

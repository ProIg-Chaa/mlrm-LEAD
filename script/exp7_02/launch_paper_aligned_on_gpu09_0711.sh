#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
LOG="${ROOT}/output/experiments/20260711_paper_aligned_vstar_priority/gpu09_shared.log"
SESSION=paper_align_0711

mkdir -p "$(dirname "${LOG}")"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "srun --jobid=26623 --overlap --ntasks=1 bash -lc 'GPU=0 bash ${ROOT}/script/exp7_02/run_paper_aligned_vstar_0711.sh 2>&1 | tee ${LOG}'"
echo "[LAUNCHED] ${SESSION}"

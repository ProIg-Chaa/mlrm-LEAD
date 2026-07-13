#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
LOG="${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/gpu_queue.log"
SESSION=fixed_damage_gpu_queue

mkdir -p "$(dirname "${LOG}")"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "srun --jobid=26711 --overlap --ntasks=1 bash -lc '${ROOT}/script/exp7_11/run_analysis_gpu_queue_after_vision_0711.sh 2>&1 | tee ${LOG}'"
echo "[LAUNCHED] ${SESSION}"

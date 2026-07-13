#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
OUT="${ROOT}/output/experiments/20260712_uniform_multimodel_full_matrix"
SESSION=compact_r1_methods_0712
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "srun --jobid=26711 --overlap --gres=gpu:1 --ntasks=1 env GPU=0 MODEL_NAME=R1-Onevision-7B MODEL_KEY=r1_onevision_7b GROUP=methods bash ${ROOT}/script/exp7_12/run_compact_talr_matrix_0712.sh 2>&1 | tee ${OUT}/compact_r1_methods_gpu15.log"
echo "[LAUNCHED] ${SESSION}"

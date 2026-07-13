#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/exp7_02/run_our_methods_shared_gpu26_0711.sh
JOB_ID="${JOB_ID:-26711}"

exec srun --jobid="${JOB_ID}" --overlap --gres=gpu:1 --ntasks=1 --nodes=1 \
  env GPU=0 MODEL_NAME=Vision-R1-7B MODEL_KEY=vision_r1_7b \
  bash "${SCRIPT}"

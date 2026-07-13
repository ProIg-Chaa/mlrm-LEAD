#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/exp7_02/run_multimodel_cot_lead_gpu15_0711.sh
JOB_ID="${JOB_ID:-26708}"
exec srun --jobid="${JOB_ID}" --overlap --gres=gpu:1 --ntasks=1 --nodes=1 bash "${SCRIPT}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/exp7_02/run_correct_model_minimal_0710.sh
exec srun --jobid=26623 --overlap -w gpu09 env GPU=1 bash "${SCRIPT}"

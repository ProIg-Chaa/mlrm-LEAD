#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
OUT=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/mmvp_lowercase_probe_smoke
PYTHON=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash "${OUT}/run_counterfactual_replay.sh"
"${PYTHON}" script/exp7_11/validate_counterfactual_smoke.py --replay-dir "${OUT}"

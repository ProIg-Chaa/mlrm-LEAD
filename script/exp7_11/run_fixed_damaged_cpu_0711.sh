#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
OUT="${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/fixed_damaged_mechanism_analysis"
LOG="${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/analysis.log"
PYTHON=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")"
exec "${PYTHON}" -u script/exp7_11/analyze_fixed_damaged_mechanisms.py \
  --root "${ROOT}" --output-dir "${OUT}" 2>&1 | tee "${LOG}"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
PYTHON=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
SMOKE="${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/counterfactual_replay_smoke"
ACTUAL_SMOKE="${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/instrumentation_actual_visual_smoke"
FULL="${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/counterfactual_replay"

while pgrep -f 'run_our_methods_shared_gpu26_0711.sh' >/dev/null 2>&1; do
  echo "[WAIT] $(date '+%F %T') Vision-R1 method queue still active"
  sleep 60
done

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
if (( free_mib < 30000 )); then
  echo "[ABORT] only ${free_mib} MiB free after Vision queue"
  exit 3
fi

cd "${ROOT}"
exec 9>"${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/.gpu_analysis.lock"
if ! flock -n 9; then
  echo "[SKIP] another GPU analysis worker owns the shared output lock"
  exit 0
fi
bash "${SMOKE}/run_counterfactual_replay.sh"
"${PYTHON}" script/exp7_11/validate_counterfactual_smoke.py --replay-dir "${SMOKE}"
bash "${ACTUAL_SMOKE}/run_counterfactual_replay.sh"
"${PYTHON}" script/exp7_11/validate_counterfactual_smoke.py --replay-dir "${ACTUAL_SMOKE}"

bash "${FULL}/run_counterfactual_replay.sh"
bash script/exp7_11/run_transition_preserving_combo_0711.sh
bash script/exp7_11/run_fixed_damaged_cpu_0711.sh
"${PYTHON}" script/exp7_11/extract_representative_cases.py >/dev/null
"${PYTHON}" script/exp7_11/finalize_semantic_audit.py
"${PYTHON}" script/exp7_11/build_fixed_damaged_report.py

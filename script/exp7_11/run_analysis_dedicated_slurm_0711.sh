#!/usr/bin/env bash
#SBATCH --job-name=fixed_damage_analysis
#SBATCH --partition=ubuntu
#SBATCH --account=maxgpu4
#SBATCH --qos=maxgpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --output=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260711_fixed_damaged_mechanism_analysis/slurm_analysis_%j.log

set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
PYTHON=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
MODEL=/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL
RAM_MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
SMOKE=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/counterfactual_replay_smoke
ACTUAL_SMOKE=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/instrumentation_actual_visual_smoke
FULL=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/counterfactual_replay

cd "${ROOT}"
exec 9>"${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/.gpu_analysis.lock"
if ! flock -n 9; then
  echo "[SKIP] another GPU analysis worker owns the shared output lock"
  exit 0
fi
free_mib="$(${PYTHON} -c 'import torch; print(torch.cuda.mem_get_info(0)[0] // 1024 // 1024)')"
if (( free_mib < 30000 )); then
  echo "[ABORT] assigned CUDA device has only ${free_mib} MiB free"
  exit 3
fi

mkdir -p "${RAM_MODEL}"
rsync -a --ignore-existing "${MODEL}/" "${RAM_MODEL}/"
${PYTHON} -m py_compile main.py lead/inference.py lead/generation_utils.py script/exp7_11/*.py

bash "${SMOKE}/run_counterfactual_replay.sh"
${PYTHON} script/exp7_11/validate_counterfactual_smoke.py --replay-dir "${SMOKE}"
bash "${ACTUAL_SMOKE}/run_counterfactual_replay.sh"
${PYTHON} script/exp7_11/validate_counterfactual_smoke.py --replay-dir "${ACTUAL_SMOKE}"
bash "${FULL}/run_counterfactual_replay.sh"
bash script/exp7_11/run_transition_preserving_combo_0711.sh
bash script/exp7_11/run_fixed_damaged_cpu_0711.sh
${PYTHON} script/exp7_11/extract_representative_cases.py >/dev/null
${PYTHON} script/exp7_11/finalize_semantic_audit.py
${PYTHON} script/exp7_11/build_fixed_damaged_report.py

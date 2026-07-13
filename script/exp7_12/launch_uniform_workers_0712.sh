#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
OUT="${ROOT}/output/experiments/20260712_uniform_multimodel_full_matrix"
RUNNER="${ROOT}/script/exp7_12/run_uniform_multimodel_matrix_0712.sh"
SBATCH_SCRIPT="${ROOT}/script/exp7_12/submit_uniform_multimodel_worker_0712.sbatch"

mkdir -p "${OUT}"

launch_shared() {
  local session="$1" job_id="$2" gpu="$3" model_name="$4" model_key="$5" log_name="$6"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[EXISTS] ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "srun --jobid=${job_id} --overlap --gres=gpu:1 --ntasks=1 env GPU=${gpu} MODEL_NAME=${model_name} MODEL_KEY=${model_key} bash ${RUNNER} 2>&1 | tee ${OUT}/${log_name}"
  echo "[LAUNCHED] ${session}"
}

launch_shared uniform_r1_0712 26623 0 R1-Onevision-7B r1_onevision_7b worker_r1_gpu09_0.log
launch_shared uniform_vision_0712 26623 1 Vision-R1-7B vision_r1_7b worker_vision_gpu09_1.log
launch_shared uniform_cogito_0712 26711 0 VL-Cogito-7B vl_cogito_7b worker_cogito_gpu15_0.log

if [[ ! -f "${OUT}/openvlthinker_sbatch_jobid.txt" ]]; then
  job_id="$(sbatch --export=ALL,MODEL_NAME=OpenVLThinker-7B,MODEL_KEY=openvlthinker_7b "${SBATCH_SCRIPT}" | awk '{print $NF}')"
  echo "${job_id}" > "${OUT}/openvlthinker_sbatch_jobid.txt"
  echo "[SUBMITTED] OpenVLThinker job=${job_id}"
else
  echo "[EXISTS] OpenVLThinker job=$(cat "${OUT}/openvlthinker_sbatch_jobid.txt")"
fi

tmux ls 2>/dev/null | grep 'uniform_.*_0712' || true

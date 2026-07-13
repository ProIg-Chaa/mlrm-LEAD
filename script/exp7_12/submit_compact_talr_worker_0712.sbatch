#!/usr/bin/env bash
#SBATCH --job-name=talr_compact
#SBATCH --partition=ubuntu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260712_uniform_multimodel_full_matrix/compact-slurm-%j.out

set -euo pipefail
ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
# The cluster occasionally grants GRES on nodes where the driver exposes no
# device.  Fail before creating a run directory; the dispatcher can resubmit
# with that node excluded.
nvidia-smi -L >/dev/null
/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
# Slurm exposes the allocated device as logical cuda:0 inside the job.  Some
# cluster nodes leave CUDA_VISIBLE_DEVICES empty, so never propagate it here.
exec env GPU=0 MODEL_NAME="${MODEL_NAME:?}" MODEL_KEY="${MODEL_KEY:?}" GROUP="${GROUP:?}" \
  bash "${ROOT}/script/exp7_12/run_compact_talr_matrix_0712.sh"

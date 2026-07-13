#!/usr/bin/env bash
# Submit with: sbatch --export=ALL,MODEL_NAME=...,MODEL_KEY=... this_file
# Slurm sets CUDA_VISIBLE_DEVICES for a single-GPU allocation; map it to the
# physical index expected by the shared runner.
#SBATCH --job-name=mlrm_uniform
#SBATCH --partition=ubuntu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260712_uniform_multimodel_full_matrix/slurm-%j.out

set -euo pipefail
ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
GPU="${CUDA_VISIBLE_DEVICES%%,*}"
exec env GPU="${GPU}" MODEL_NAME="${MODEL_NAME:?}" MODEL_KEY="${MODEL_KEY:?}" \
  bash "${ROOT}/script/exp7_12/run_uniform_multimodel_matrix_0712.sh"

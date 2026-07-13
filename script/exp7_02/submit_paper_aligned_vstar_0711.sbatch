#!/usr/bin/env bash
#SBATCH --job-name=lead_paper_align
#SBATCH --partition=ubuntu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu04
#SBATCH --output=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260711_paper_aligned_vstar_priority/slurm-%j.out

set -euo pipefail
exec bash /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/exp7_02/run_paper_aligned_vstar_0711.sh

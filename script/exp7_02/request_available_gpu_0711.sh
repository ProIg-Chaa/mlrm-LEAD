#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-ubuntu}"
NODE="${NODE-gpu15}"
GPUS="${GPUS:-1}"
JOB_NAME="${JOB_NAME:-mlrm_correct_priority}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
MEMORY="${MEMORY:-64G}"

echo "[REQUEST] $(date '+%F %T') partition=${PARTITION} node=${NODE:-any} gpus=${GPUS}"

args=(
  srun
  -p "${PARTITION}"
  --job-name="${JOB_NAME}"
  --nodes=1
  --gres="gpu:${GPUS}"
  --mem="${MEMORY}"
)
if [[ -n "${NODE}" ]]; then
  args+=(-w "${NODE}")
fi
if [[ -n "${EXCLUDE_NODES}" ]]; then
  args+=(--exclude="${EXCLUDE_NODES}")
fi
args+=(--pty bash)

exec "${args[@]}"

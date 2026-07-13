#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${JOB_ID:-26623}"
PIDS=(491760 449922)
EXPECTED=(pure_soft_diffuse_collapse highrisk_only_cooldown2)

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  expected="${EXPECTED[$i]}"
  cmd="$(srun --jobid="${JOB_ID}" --overlap --ntasks=1 ps -ww -p "${pid}" -o args= 2>/dev/null || true)"
  if [[ "${cmd}" != *"${expected}"* ]]; then
    echo "[SKIP] PID ${pid} is absent or no longer matches ${expected}"
    continue
  fi
  srun --jobid="${JOB_ID}" --overlap --ntasks=1 kill -CONT "${pid}"
  echo "[RESUMED] PID ${pid}: ${expected}"
done

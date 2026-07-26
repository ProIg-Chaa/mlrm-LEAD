#!/usr/bin/env bash
set -euo pipefail

repo=/root/gushuo/proj/mlrm-LEAD
python=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
root=/root/autodl-tmp/gushuo/outputs/experiments/20260721_locked_l095_all_models

while pgrep -f '^/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python main.py ' >/dev/null; do
  date '+%F %T waiting for single-process VStar repair' >> "$root/openvl_waiter.log"
  sleep 60
done

cd "$repo"
"$python" -m py_compile script/exp7_21/run_locked_l095_openvl_sequential_20260721.py
exec "$python" script/exp7_21/run_locked_l095_openvl_sequential_20260721.py \
  >> "$root/openvl_sequential_launcher.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail

repo=/root/gushuo/proj/mlrm-LEAD
python=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
root=/root/autodl-tmp/gushuo/outputs/experiments/20260721_locked_l095_all_models

mkdir -p "$root"
while pgrep -f '[p]ython main.py' >/dev/null; do
  date '+%F %T waiting for current main.py workers' >> "$root/watcher.log"
  sleep 60
done

cd "$repo"
"$python" -m py_compile \
  main.py lead/inference.py lead/generation_utils.py \
  script/exp7_21/run_locked_l095_all_models_20260721.py

exec "$python" script/exp7_21/run_locked_l095_all_models_20260721.py \
  >> "$root/launcher.log" 2>&1

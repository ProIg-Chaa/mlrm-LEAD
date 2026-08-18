#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260728_probe_v2_external_expansion
REPO=/root/gushuo/proj/mlrm-LEAD-atlas-v0b
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
RUNNER="$REPO/script/exp7_26/run_intervention_atlas_extended_shard.py"

mkdir -p "$ROOT"
"$PY" "$REPO/script/exp7_28/prepare_probe_v2_expansion.py" \
  --repo "$REPO" \
  --data-root "$REPO/data" \
  --output-dir "$ROOT/selection" \
  --per-dataset 64 \
  --num-shards 4 \
  --seed 20260728 \
  --exclude /root/autodl-tmp/gushuo/outputs/experiments/20260724_intervention_atlas_v0b/selection/selected_all.jsonl \
  --exclude /root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_extended/selection/selected_all.jsonl \
  --exclude /root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_followup/selection/selected_all.jsonl \
  >"$ROOT/prepare.log"

python_pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$RUNNER" \
    --repo "$REPO" \
    --python "$PY" \
    --model "$MODEL" \
    --shard "$ROOT/selection/shards/shard_${shard}.jsonl" \
    --output-dir "$ROOT/runs/shard_${shard}" \
    --shard-index "$shard" \
    >"$ROOT/shard_${shard}.log" 2>&1 &
  python_pids+=("$!")
done
printf '%s\n' "${python_pids[@]}" >"$ROOT/shard_pids.txt"

nohup bash "$REPO/script/exp7_28/run_probe_v2_expansion_after_atlas.sh" \
  >"$ROOT/watcher.log" 2>&1 &
echo "$!" >"$ROOT/watcher.pid"

printf 'root=%s\nshards=%s\nwatcher=%s\n' \
  "$ROOT" "${python_pids[*]}" "$(cat "$ROOT/watcher.pid")"

wait "${python_pids[@]}"
date >"$ROOT/GENERATION_WORKERS_COMPLETE"

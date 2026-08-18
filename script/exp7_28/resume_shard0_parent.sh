#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260728_probe_v2_external_expansion
PARENT_PID=776127
EXPECTED=768

for run in contracted_soft_l090 contracted_soft_l095 pure_soft_l100; do
  result="$ROOT/runs/shard_0/$run/results.jsonl"
  while [[ ! -f "$result" ]] || [[ $(wc -l <"$result") -ne $EXPECTED ]]; do
    if ! kill -0 "$PARENT_PID" 2>/dev/null; then
      echo "parent disappeared: $PARENT_PID" >&2
      exit 1
    fi
    sleep 120
  done
done

kill -CONT "$PARENT_PID"
date >"$ROOT/SHARD0_PARENT_RESUMED"

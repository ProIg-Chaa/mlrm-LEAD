#!/usr/bin/env bash
set -euo pipefail

SOURCE=/root/autodl-tmp/gushuo/models/Vision-R1-7B
RAM=/dev/shm/wangzixu_models/Vision-R1-7B
LOG=/root/autodl-tmp/gushuo/outputs/logs/vision_r1_watcher_0714.log
EXPECTED_INDEX_SHA=34f0c8eed3b28b55a2c48941e6499304b95a417c2eed93761cd5f87686903113

echo "[WATCH] $(date '+%F %T') waiting for model" >> "$LOG"
while true; do
  download_pid=$(cat /root/gushuo/vision_r1_download.pid 2>/dev/null || true)
  if [[ -n "$download_pid" ]] && kill -0 "$download_pid" 2>/dev/null; then
    sleep 60
    continue
  fi
  shard_count=$(find "$SOURCE" -maxdepth 1 -name 'model-*-of-*.safetensors' -type f | wc -l)
  if [[ "$shard_count" -eq 4 && -f "$SOURCE/model.safetensors.index.json" &&
        -f "$SOURCE/config.json" && -f "$SOURCE/tokenizer.json" &&
        -f "$SOURCE/preprocessor_config.json" ]]; then
    actual_sha=$(sha256sum "$SOURCE/model.safetensors.index.json" | awk '{print $1}')
    [[ "$actual_sha" == "$EXPECTED_INDEX_SHA" ]] || {
      echo "[ERROR] index SHA mismatch: $actual_sha" >> "$LOG"
      exit 2
    }
    break
  fi
  echo "[WAIT] $(date '+%F %T') downloader stopped but model incomplete" >> "$LOG"
  sleep 60
done

echo "[MODEL OK] $(date '+%F %T') copying to RAM" >> "$LOG"
mkdir -p "$RAM"
rsync -a "$SOURCE/" "$RAM/"

while [[ $(pgrep -fc 'python.*main.py') -gt 2 ]]; do
  sleep 60
done

echo "[LAUNCH] $(date '+%F %T') Vision-R1 queue" >> "$LOG"
tmux kill-session -t vision_r1_matrix_0714 2>/dev/null || true
tmux new-session -d -s vision_r1_matrix_0714 \
  "bash /root/gushuo/proj/mlrm-LEAD/script/exp7_12/run_newgpu_vision_priority_0714.sh >> /root/autodl-tmp/gushuo/outputs/logs/vision_r1_matrix_0714.log 2>&1"

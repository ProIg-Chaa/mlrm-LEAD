#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
MODELS_ROOT="${MODELS_ROOT:-/share/home/wangzixu/liudinghao/gushuo/models}"
STAMP="${STAMP:-20260705_integrated_cot_lead_baselines}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/integrated_repo_cot_lead_baselines}"
LOG_DIR="${BASE_DIR}/logs"
SLEEP_SECONDS="${SLEEP_SECONDS:-600}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

complete_model() {
  local model_dir="$1"
  /share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python - "$model_dir" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
required = ["config.json", "model.safetensors.index.json", "tokenizer_config.json", "preprocessor_config.json"]
ok = all((p / name).exists() for name in required)
shards = [x for x in p.glob("*.safetensors") if x.stat().st_size > 100_000_000]
sys.exit(0 if ok and len(shards) >= 4 else 1)
PY
}

active_baseline_queue() {
  tmux ls 2>/dev/null | grep -q 'integrated_baseline_.*_0705'
}

launch_model() {
  local model_name="$1"
  local model_key="$2"
  local marker="${BASE_DIR}/.${model_key}.launched"
  if [[ -f "${marker}" ]]; then
    return 0
  fi
  if active_baseline_queue; then
    echo "[WAIT] $(date '+%F %T') another integrated baseline queue is still active"
    return 0
  fi
  if ! complete_model "${MODELS_ROOT}/${model_name}"; then
    echo "[WAIT] $(date '+%F %T') ${model_key} is not complete yet"
    return 0
  fi

  echo "[LAUNCH_MODEL] $(date '+%F %T') ${model_key}"
  tmux new-session -d -s "integrated_baseline_${model_key}_cot_0705" "cd '${ROOT}'; bash script/exp7_02/launch_integrated_queue_0705.sh cot 0 '${model_name}' '${model_key}'"
  tmux new-session -d -s "integrated_baseline_${model_key}_lead_0705" "cd '${ROOT}'; bash script/exp7_02/launch_integrated_queue_0705.sh lead 1 '${model_name}' '${model_key}'"
  date '+%F %T' > "${marker}"
}

echo "[WATCH] $(date '+%F %T') integrated model watcher started"
while true; do
  launch_model "Vision-R1-7B" "vision_r1_7b"
  launch_model "VL-Cogito-7B" "vl_cogito_7b"
  launch_model "OpenVLThinker-7B" "openvlthinker_7b"

  if [[ -f "${BASE_DIR}/.vision_r1_7b.launched" && -f "${BASE_DIR}/.vl_cogito_7b.launched" && -f "${BASE_DIR}/.openvlthinker_7b.launched" ]]; then
    echo "[WATCH] $(date '+%F %T') all downloaded models have been queued"
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done

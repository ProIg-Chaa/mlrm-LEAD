#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
MODELS_ROOT="${MODELS_ROOT:-/share/home/wangzixu/liudinghao/gushuo/models}"
STAMP="${STAMP:-20260706_format_stability_full_baselines}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/format_stability_full_baselines}"
LOG_DIR="${BASE_DIR}/logs"
SLEEP_SECONDS="${SLEEP_SECONDS:-600}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"
unset TMUX

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

active_blocking_queue() {
  tmux ls 2>/dev/null \
    | grep -Ev '^format_stability_watcher_0706:' \
    | grep -Eq '(integrated_baseline_.*_0705|format_stability_.*_0706)'
}

active_format_main_for_model() {
  local key="$1"
  tmux ls 2>/dev/null | grep -Eq "format_stability_${key}_main_(a|b)_0706"
}

launch_main() {
  local model_name="$1"
  local model_key="$2"
  local marker="${BASE_DIR}/.${model_key}.main.launched"
  if [[ -f "${marker}" ]]; then
    return 0
  fi
  if active_blocking_queue; then
    echo "[WAIT] $(date '+%F %T') baseline/format queue is active; not launching ${model_key} main"
    return 0
  fi
  if ! complete_model "${MODELS_ROOT}/${model_name}"; then
    echo "[WAIT] $(date '+%F %T') ${model_key} is not complete yet"
    return 0
  fi
  echo "[LAUNCH_MAIN] $(date '+%F %T') ${model_key}"
  tmux new-session -d -s "format_stability_${model_key}_main_a_0706" "cd '${ROOT}'; bash script/exp7_02/launch_format_stability_queue_0706.sh main_a 0 '${model_name}' '${model_key}'"
  tmux new-session -d -s "format_stability_${model_key}_main_b_0706" "cd '${ROOT}'; bash script/exp7_02/launch_format_stability_queue_0706.sh main_b 1 '${model_name}' '${model_key}'"
  date '+%F %T' > "${marker}"
}

launch_r1_diag() {
  local marker="${BASE_DIR}/.r1_onevision_7b.diag.launched"
  if [[ -f "${marker}" ]]; then
    return 0
  fi
  if [[ ! -f "${BASE_DIR}/.r1_onevision_7b.main.launched" ]]; then
    return 0
  fi
  if active_blocking_queue; then
    echo "[WAIT] $(date '+%F %T') waiting for R1 main queues before diagnostics"
    return 0
  fi
  echo "[LAUNCH_DIAG] $(date '+%F %T') r1_onevision_7b"
  tmux new-session -d -s "format_stability_r1_onevision_7b_diag_a_0706" "cd '${ROOT}'; bash script/exp7_02/launch_format_stability_queue_0706.sh diag_a 0 R1-Onevision-7B-RL r1_onevision_7b"
  tmux new-session -d -s "format_stability_r1_onevision_7b_diag_b_0706" "cd '${ROOT}'; bash script/exp7_02/launch_format_stability_queue_0706.sh diag_b 1 R1-Onevision-7B-RL r1_onevision_7b"
  date '+%F %T' > "${marker}"
}

echo "[WATCH] $(date '+%F %T') format stability watcher started"
while true; do
  launch_main "R1-Onevision-7B-RL" "r1_onevision_7b"
  launch_r1_diag
  launch_main "Vision-R1-7B" "vision_r1_7b"
  launch_main "VL-Cogito-7B" "vl_cogito_7b"
  launch_main "OpenVLThinker-7B" "openvlthinker_7b"
  sleep "${SLEEP_SECONDS}"
done

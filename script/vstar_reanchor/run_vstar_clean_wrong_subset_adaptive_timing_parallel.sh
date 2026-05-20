#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/vstar_wrong_subset_from_cot_clean.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT}/output/experiments/${STAMP}/vstar_clean_wrong_subset_adaptive_timing_parallel"

mkdir -p "${RUN_ROOT}"

write_run_command() {
  local run_dir="$1"
  local gpu="$2"
  local trigger_mode="$3"
  local entropy_delta="$4"
  local visual_drop="$5"
  local min_step="$6"
  local max_step="$7"

  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \
  --model_name "${MODEL}" \
  --dataset "${DATASET}" \
  --output_dir "${run_dir}" \
  --method cot_visual_reanchor \
  --max_new_tokens 1024 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42 \
  --device cuda \
  --no-do_sample \
  --save_token_entropy \
  --save_full_token_entropy \
  --save_visual_attn_summary \
  --visual_attn_summary_last_k 4 \
  --reanchor_entropy_threshold 1.0 \
  --reanchor_visual_attn_threshold 0.12 \
  --reanchor_lambda 0.15 \
  --reanchor_top_m 4 \
  --reanchor_attn_last_k 4 \
  --reanchor_max_trigger_count 1 \
  --reanchor_cooldown 32 \
  --reanchor_min_step ${min_step} \
  --reanchor_max_step ${max_step} \
  --reanchor_anchor_mode dynamic \
  --reanchor_trigger_mode "${trigger_mode}" \
  --reanchor_rolling_window 8 \
  --reanchor_min_history 3 \
  --reanchor_entropy_delta_threshold ${entropy_delta} \
  --reanchor_visual_drop_threshold ${visual_drop}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run_command "${RUN_ROOT}/entropy_delta_step0_30_gpu0" 0 "entropy_delta" 0.5 0.03 0 30
write_run_command "${RUN_ROOT}/entropy_delta_visual_drop_step0_50_gpu0" 0 "entropy_delta_visual_drop" 0.5 0.03 0 50
write_run_command "${RUN_ROOT}/visual_drop_step0_30_gpu1" 1 "visual_drop" 0.5 0.03 0 30
write_run_command "${RUN_ROOT}/entropy_delta_step0_50_gpu1" 1 "entropy_delta" 0.5 0.03 0 50

cat > "${RUN_ROOT}/gpu0_worker.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "\$(date '+%F %T') START entropy_delta_step0_30_gpu0 GPU=0" | tee -a "${RUN_ROOT}/launcher.log"
bash "${RUN_ROOT}/entropy_delta_step0_30_gpu0/run_command.sh" > "${RUN_ROOT}/entropy_delta_step0_30_gpu0/nohup.log" 2>&1
echo "\$(date '+%F %T') DONE  entropy_delta_step0_30_gpu0 GPU=0" | tee -a "${RUN_ROOT}/launcher.log"
echo "\$(date '+%F %T') START entropy_delta_visual_drop_step0_50_gpu0 GPU=0" | tee -a "${RUN_ROOT}/launcher.log"
bash "${RUN_ROOT}/entropy_delta_visual_drop_step0_50_gpu0/run_command.sh" > "${RUN_ROOT}/entropy_delta_visual_drop_step0_50_gpu0/nohup.log" 2>&1
echo "\$(date '+%F %T') DONE  entropy_delta_visual_drop_step0_50_gpu0 GPU=0" | tee -a "${RUN_ROOT}/launcher.log"
EOF
chmod +x "${RUN_ROOT}/gpu0_worker.sh"

cat > "${RUN_ROOT}/gpu1_worker.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "\$(date '+%F %T') START visual_drop_step0_30_gpu1 GPU=1" | tee -a "${RUN_ROOT}/launcher.log"
bash "${RUN_ROOT}/visual_drop_step0_30_gpu1/run_command.sh" > "${RUN_ROOT}/visual_drop_step0_30_gpu1/nohup.log" 2>&1
echo "\$(date '+%F %T') DONE  visual_drop_step0_30_gpu1 GPU=1" | tee -a "${RUN_ROOT}/launcher.log"
echo "\$(date '+%F %T') START entropy_delta_step0_50_gpu1 GPU=1" | tee -a "${RUN_ROOT}/launcher.log"
bash "${RUN_ROOT}/entropy_delta_step0_50_gpu1/run_command.sh" > "${RUN_ROOT}/entropy_delta_step0_50_gpu1/nohup.log" 2>&1
echo "\$(date '+%F %T') DONE  entropy_delta_step0_50_gpu1 GPU=1" | tee -a "${RUN_ROOT}/launcher.log"
EOF
chmod +x "${RUN_ROOT}/gpu1_worker.sh"

setsid bash "${RUN_ROOT}/gpu0_worker.sh" > "${RUN_ROOT}/gpu0_worker.log" 2>&1 < /dev/null &
echo $! > "${RUN_ROOT}/gpu0_worker.pid"

setsid bash "${RUN_ROOT}/gpu1_worker.sh" > "${RUN_ROOT}/gpu1_worker.log" 2>&1 < /dev/null &
echo $! > "${RUN_ROOT}/gpu1_worker.pid"

echo "Started adaptive timing workers:"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  GPU0_WORKER_PID=$(cat "${RUN_ROOT}/gpu0_worker.pid")"
echo "  GPU1_WORKER_PID=$(cat "${RUN_ROOT}/gpu1_worker.pid")"
echo
echo "Runs:"
echo "  GPU0 sequential: entropy_delta_step0_30_gpu0 -> entropy_delta_visual_drop_step0_50_gpu0"
echo "  GPU1 sequential: visual_drop_step0_30_gpu1 -> entropy_delta_step0_50_gpu1"

#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"

RUN_DATE="$(date +%Y%m%d)"
RUN_NAME="lead_mmvp_vstar_parallel_$(date +%H%M%S)"
RUN_ROOT="output/experiments/${RUN_DATE}/${RUN_NAME}"

MMVP_DIR="${RUN_ROOT}/mmvp_full_gpu0"
VSTAR_DIR="${RUN_ROOT}/vstar_full_gpu1"
mkdir -p "$MMVP_DIR" "$VSTAR_DIR"

MMVP_CMD="${MMVP_DIR}/run_command.sh"
VSTAR_CMD="${VSTAR_DIR}/run_command.sh"

printf '%s\n' '#!/usr/bin/env bash' > "$MMVP_CMD"
printf '%s\n' 'set -euo pipefail' >> "$MMVP_CMD"
printf '%s\n' 'cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD' >> "$MMVP_CMD"
printf '%s\n' 'export CUDA_VISIBLE_DEVICES=0' >> "$MMVP_CMD"
printf '%s\n' 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> "$MMVP_CMD"
printf '%s\n' "exec ${PYTHON_BIN} main.py --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL --dataset data/mmvp.jsonl --output_dir ${MMVP_DIR} --method lead --max_new_tokens 1024 --alpha 0.4 --max_switch_count 5 --window_size 128 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy" >> "$MMVP_CMD"

printf '%s\n' '#!/usr/bin/env bash' > "$VSTAR_CMD"
printf '%s\n' 'set -euo pipefail' >> "$VSTAR_CMD"
printf '%s\n' 'cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD' >> "$VSTAR_CMD"
printf '%s\n' 'export CUDA_VISIBLE_DEVICES=1' >> "$VSTAR_CMD"
printf '%s\n' 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> "$VSTAR_CMD"
printf '%s\n' "exec ${PYTHON_BIN} main.py --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL --dataset data/vstar.jsonl --output_dir ${VSTAR_DIR} --method lead --max_new_tokens 1024 --alpha 0.4 --max_switch_count 5 --window_size 128 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy" >> "$VSTAR_CMD"

chmod +x "$MMVP_CMD" "$VSTAR_CMD"

setsid "$MMVP_CMD" > "${MMVP_DIR}/nohup.log" 2>&1 < /dev/null &
MMVP_PID=$!
echo "$MMVP_PID" > "${MMVP_DIR}/pid.txt"

setsid "$VSTAR_CMD" > "${VSTAR_DIR}/nohup.log" 2>&1 < /dev/null &
VSTAR_PID=$!
echo "$VSTAR_PID" > "${VSTAR_DIR}/pid.txt"

echo "Started parallel LEAD experiments"
echo "Run root: $RUN_ROOT"
echo "MMVP GPU: 0"
echo "MMVP PID: $MMVP_PID"
echo "MMVP dir: $MMVP_DIR"
echo "VStar GPU: 1"
echo "VStar PID: $VSTAR_PID"
echo "VStar dir: $VSTAR_DIR"
echo
echo "Watch logs:"
echo "  tail -f ${MMVP_DIR}/nohup.log"
echo "  tail -f ${VSTAR_DIR}/nohup.log"

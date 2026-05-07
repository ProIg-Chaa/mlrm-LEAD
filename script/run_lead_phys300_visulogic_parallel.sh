#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"

"$PYTHON_BIN" script/prepare_uniform_subset.py \
  --input data/physunibench.jsonl \
  --output data/physunibench_uniform300.jsonl \
  --limit 300

RUN_DATE="$(date +%Y%m%d)"
RUN_NAME="lead_phys300_visulogic_parallel_$(date +%H%M%S)"
RUN_ROOT="output/experiments/${RUN_DATE}/${RUN_NAME}"

PHYS_DIR="${RUN_ROOT}/physunibench_uniform300_gpu0"
VISU_DIR="${RUN_ROOT}/visulogic_full_gpu1"
mkdir -p "$PHYS_DIR" "$VISU_DIR"

PHYS_CMD="${PHYS_DIR}/run_command.sh"
VISU_CMD="${VISU_DIR}/run_command.sh"

printf '%s\n' '#!/usr/bin/env bash' > "$PHYS_CMD"
printf '%s\n' 'set -euo pipefail' >> "$PHYS_CMD"
printf '%s\n' 'cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD' >> "$PHYS_CMD"
printf '%s\n' 'export CUDA_VISIBLE_DEVICES=0' >> "$PHYS_CMD"
printf '%s\n' 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> "$PHYS_CMD"
printf '%s\n' "exec ${PYTHON_BIN} main.py --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL --dataset data/physunibench_uniform300.jsonl --output_dir ${PHYS_DIR} --method lead --max_new_tokens 1024 --alpha 0.4 --max_switch_count 5 --window_size 128 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy" >> "$PHYS_CMD"

printf '%s\n' '#!/usr/bin/env bash' > "$VISU_CMD"
printf '%s\n' 'set -euo pipefail' >> "$VISU_CMD"
printf '%s\n' 'cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD' >> "$VISU_CMD"
printf '%s\n' 'export CUDA_VISIBLE_DEVICES=1' >> "$VISU_CMD"
printf '%s\n' 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> "$VISU_CMD"
printf '%s\n' "exec ${PYTHON_BIN} main.py --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL --dataset data/visulogic.jsonl --output_dir ${VISU_DIR} --method lead --max_new_tokens 1024 --alpha 0.4 --max_switch_count 5 --window_size 128 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy" >> "$VISU_CMD"

chmod +x "$PHYS_CMD" "$VISU_CMD"

setsid "$PHYS_CMD" > "${PHYS_DIR}/nohup.log" 2>&1 < /dev/null &
PHYS_PID=$!
echo "$PHYS_PID" > "${PHYS_DIR}/pid.txt"

setsid "$VISU_CMD" > "${VISU_DIR}/nohup.log" 2>&1 < /dev/null &
VISU_PID=$!
echo "$VISU_PID" > "${VISU_DIR}/pid.txt"

echo "Started parallel LEAD experiments"
echo "Run root: $RUN_ROOT"
echo "PhysUniBench subset: data/physunibench_uniform300.jsonl"
echo "PhysUniBench GPU: 0"
echo "PhysUniBench PID: $PHYS_PID"
echo "PhysUniBench dir: $PHYS_DIR"
echo "VisuLogic GPU: 1"
echo "VisuLogic PID: $VISU_PID"
echo "VisuLogic dir: $VISU_DIR"
echo
echo "Watch logs:"
echo "  tail -f ${PHYS_DIR}/nohup.log"
echo "  tail -f ${VISU_DIR}/nohup.log"

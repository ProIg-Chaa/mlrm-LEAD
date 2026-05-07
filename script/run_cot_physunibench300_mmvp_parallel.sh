#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

python3 script/prepare_uniform_subset.py \
  --input data/physunibench.jsonl \
  --output data/physunibench_uniform300.jsonl \
  --limit 300

RUN_DATE="$(date +%Y%m%d)"
RUN_NAME="cot_phys300_mmvp_parallel_$(date +%H%M%S)"
RUN_ROOT="output/experiments/${RUN_DATE}/${RUN_NAME}"

PHYS_DIR="${RUN_ROOT}/physunibench_uniform300_gpu0"
MMVP_DIR="${RUN_ROOT}/mmvp_full_gpu1"
mkdir -p "$PHYS_DIR" "$MMVP_DIR"

PHYS_CMD="${PHYS_DIR}/run_command.sh"
MMVP_CMD="${MMVP_DIR}/run_command.sh"

printf '%s\n' '#!/usr/bin/env bash' > "$PHYS_CMD"
printf '%s\n' 'set -euo pipefail' >> "$PHYS_CMD"
printf '%s\n' 'cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD' >> "$PHYS_CMD"
printf '%s\n' 'export CUDA_VISIBLE_DEVICES=0' >> "$PHYS_CMD"
printf '%s\n' 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> "$PHYS_CMD"
printf '%s\n' "exec micromamba run -n mlrm-lead python main.py --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL --dataset data/physunibench_uniform300.jsonl --output_dir ${PHYS_DIR} --method cot --max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy" >> "$PHYS_CMD"

printf '%s\n' '#!/usr/bin/env bash' > "$MMVP_CMD"
printf '%s\n' 'set -euo pipefail' >> "$MMVP_CMD"
printf '%s\n' 'cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD' >> "$MMVP_CMD"
printf '%s\n' 'export CUDA_VISIBLE_DEVICES=1' >> "$MMVP_CMD"
printf '%s\n' 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> "$MMVP_CMD"
printf '%s\n' "exec micromamba run -n mlrm-lead python main.py --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL --dataset data/mmvp.jsonl --output_dir ${MMVP_DIR} --method cot --max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy" >> "$MMVP_CMD"

chmod +x "$PHYS_CMD" "$MMVP_CMD"

setsid "$PHYS_CMD" > "${PHYS_DIR}/nohup.log" 2>&1 < /dev/null &
PHYS_PID=$!
echo "$PHYS_PID" > "${PHYS_DIR}/pid.txt"

setsid "$MMVP_CMD" > "${MMVP_DIR}/nohup.log" 2>&1 < /dev/null &
MMVP_PID=$!
echo "$MMVP_PID" > "${MMVP_DIR}/pid.txt"

echo "Started parallel cot experiments"
echo "Run root: $RUN_ROOT"
echo "PhysUniBench subset: data/physunibench_uniform300.jsonl"
echo "PhysUniBench GPU: 0"
echo "PhysUniBench PID: $PHYS_PID"
echo "PhysUniBench dir: $PHYS_DIR"
echo "MMVP GPU: 1"
echo "MMVP PID: $MMVP_PID"
echo "MMVP dir: $MMVP_DIR"
echo
echo "Watch logs:"
echo "  tail -f ${PHYS_DIR}/nohup.log"
echo "  tail -f ${MMVP_DIR}/nohup.log"

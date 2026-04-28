#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

python3 script/prepare_mmhal_bench_jsonl.py
python3 script/prepare_mmhal_balanced_subset.py \
  --input data/mmhal_bench.jsonl \
  --output data/mmhal_bench_balanced_2pertype.jsonl \
  --per_type 2

RUN_DATE="$(date +%Y%m%d)"
RUN_NAME="mmhal_balanced_entropy_curves_$(date +%H%M%S)"
RUN_ROOT="output/experiments/${RUN_DATE}/${RUN_NAME}"

COT_DIR="${RUN_ROOT}/cot"
LEAD_DIR="${RUN_ROOT}/lead"
mkdir -p "$COT_DIR" "$LEAD_DIR"

nohup env CUDA_VISIBLE_DEVICES=1 \
micromamba run -n mlrm-lead python main.py \
  --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL \
  --dataset data/mmhal_bench_balanced_2pertype.jsonl \
  --output_dir "$COT_DIR" \
  --method cot \
  --max_new_tokens 1024 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42 \
  --device cuda \
  --no-do_sample \
  --save_token_entropy \
  --save_full_token_entropy \
  > "$COT_DIR/nohup.log" 2>&1 &

COT_PID=$!

nohup env CUDA_VISIBLE_DEVICES=1 \
micromamba run -n mlrm-lead python main.py \
  --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL \
  --dataset data/mmhal_bench_balanced_2pertype.jsonl \
  --output_dir "$LEAD_DIR" \
  --method lead \
  --max_new_tokens 1024 \
  --alpha 0.4 \
  --max_switch_count 5 \
  --window_size 128 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42 \
  --device cuda \
  --no-do_sample \
  --save_token_entropy \
  --save_full_token_entropy \
  > "$LEAD_DIR/nohup.log" 2>&1 &

LEAD_PID=$!

echo "Started balanced MMHal entropy-curve experiment"
echo "Run root: $RUN_ROOT"
echo "Subset dataset: data/mmhal_bench_balanced_2pertype.jsonl"
echo "COT PID: $COT_PID"
echo "LEAD PID: $LEAD_PID"
echo "Execution mode: parallel on GPU 1"
echo
echo "Logs:"
echo "  tail -f $COT_DIR/nohup.log"
echo "  tail -f $LEAD_DIR/nohup.log"
echo
echo "After both runs finish, generate plots:"
echo "  python3 script/plot_token_entropy_curve.py $COT_DIR/token_entropy_full.jsonl --metric raw_entropy"
echo "  python3 script/plot_token_entropy_curve.py $LEAD_DIR/token_entropy_full.jsonl --metric raw_entropy"
echo
echo "Filtered-entropy plots if needed:"
echo "  python3 script/plot_token_entropy_curve.py $COT_DIR/token_entropy_full.jsonl --metric filtered_entropy"
echo "  python3 script/plot_token_entropy_curve.py $LEAD_DIR/token_entropy_full.jsonl --metric filtered_entropy"

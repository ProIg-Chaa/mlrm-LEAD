#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

python3 script/prepare_mmhal_bench_jsonl.py

RUN_DATE="$(date +%Y%m%d)"
RUN_NAME="mmhal_lead_paper_params_$(date +%H%M%S)"
RUN_DIR="output/experiments/${RUN_DATE}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

nohup env CUDA_VISIBLE_DEVICES=1 \
micromamba run -n mlrm-lead python main.py \
  --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL \
  --dataset data/mmhal_bench.jsonl \
  --output_dir "$RUN_DIR" \
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
  > "$RUN_DIR/nohup.log" 2>&1 &

PID=$!

echo "Started MMHal-Bench LEAD experiment"
echo "PID: $PID"
echo "Run dir: $RUN_DIR"
echo "Log: $RUN_DIR/nohup.log"
echo
echo "Watch progress:"
echo "  tail -f $RUN_DIR/nohup.log"
echo
echo "Summarize after completion:"
echo "  bash script/summarize_latest_mmhal_lead_paper_params.sh"

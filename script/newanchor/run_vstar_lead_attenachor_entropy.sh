#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

RUN_DATE="$(date +%Y%m%d)"
RUN_NAME="vstar_lead_attenachor_entropy_$(date +%H%M%S)"
RUN_DIR="output/experiments/${RUN_DATE}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

# Single-GPU run for the new anchor method. Keep token entropy summaries and
# full per-token traces so reasoning-token entropy changes can be analyzed later.
nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  micromamba run -n mlrm-lead python main.py \
    --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL \
    --dataset data/vstar.jsonl \
    --output_dir "$RUN_DIR" \
    --method lead_attenachor \
    --max_new_tokens 1024 \
    --alpha 0.4 \
    --max_switch_count 5 \
    --window_size 128 \
    --visual_anchor_top_m 32 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --seed 42 \
    --device cuda \
    --no-do_sample \
    --save_token_entropy \
    --save_full_token_entropy \
    > "$RUN_DIR/nohup.log" 2>&1 &

PID=$!

echo "Started VStar LEAD new-anchor entropy experiment"
echo "PID: $PID"
echo "Run dir: $RUN_DIR"
echo "Log: $RUN_DIR/nohup.log"
echo
echo "Watch progress:"
echo "  tail -f $RUN_DIR/nohup.log"
echo
echo "Key outputs after completion:"
echo "  $RUN_DIR/results.jsonl"
echo "  $RUN_DIR/token_entropy.jsonl"
echo "  $RUN_DIR/token_entropy_full.jsonl"
echo
echo "Compact entropy summary:"
echo "  python3 script/summarize_token_entropy.py $RUN_DIR/token_entropy.jsonl"

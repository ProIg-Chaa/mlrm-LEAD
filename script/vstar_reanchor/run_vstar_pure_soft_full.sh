#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"

RUN_DATE="$(date +%Y%m%d)"
RUN_NAME="vstar_pure_soft_full_$(date +%H%M%S)"
RUN_DIR="output/experiments/${RUN_DATE}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

nohup env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" main.py \
  --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL \
  --dataset data/vstar.jsonl \
  --output_dir "$RUN_DIR" \
  --method pure_soft \
  --max_new_tokens 1024 \
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
echo "$PID" > "$RUN_DIR/pid.txt"

cat > "$RUN_DIR/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec $PYTHON_BIN main.py \\
  --model_name /share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL \\
  --dataset data/vstar.jsonl \\
  --output_dir "$RUN_DIR" \\
  --method pure_soft \\
  --max_new_tokens 1024 \\
  --temperature 0.6 \\
  --top_p 0.95 \\
  --top_k 20 \\
  --seed 42 \\
  --device cuda \\
  --no-do_sample \\
  --save_token_entropy \\
  --save_full_token_entropy
EOF
chmod +x "$RUN_DIR/run_command.sh"

echo "Started VStar pure_soft full experiment"
echo "PID: $PID"
echo "Run dir: $RUN_DIR"
echo "Log: $RUN_DIR/nohup.log"
echo
echo "Watch progress:"
echo "  tail -f $RUN_DIR/nohup.log"

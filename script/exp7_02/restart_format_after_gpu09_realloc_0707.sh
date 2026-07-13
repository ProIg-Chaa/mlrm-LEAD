#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
cd "${ROOT}"

chmod +x script/exp7_02/launch_format_stability_queue_0706.sh
bash -n script/exp7_02/launch_format_stability_queue_0706.sh

for s in \
  format_stability_r1_onevision_7b_main_a_0706 \
  format_stability_r1_onevision_7b_main_b_0706 \
  format_stability_r1_onevision_7b_diag_a_0706 \
  format_stability_r1_onevision_7b_diag_b_0706 \
  format_stability_vision_r1_7b_main_a_0706 \
  format_stability_vision_r1_7b_main_b_0706 \
  format_stability_vl_cogito_7b_main_a_0706 \
  format_stability_vl_cogito_7b_main_b_0706 \
  format_stability_openvlthinker_7b_main_a_0706 \
  format_stability_openvlthinker_7b_main_b_0706; do
  tmux kill-session -t "${s}" 2>/dev/null || true
done

tmux new-session -d -s format_stability_r1_onevision_7b_main_a_0706 \
  "cd '${ROOT}'; JOBID=26623 NODE=gpu09 bash script/exp7_02/launch_format_stability_queue_0706.sh main_a 0 R1-Onevision-7B-RL r1_onevision_7b"
tmux new-session -d -s format_stability_r1_onevision_7b_main_b_0706 \
  "cd '${ROOT}'; JOBID=26623 NODE=gpu09 bash script/exp7_02/launch_format_stability_queue_0706.sh main_b 1 R1-Onevision-7B-RL r1_onevision_7b"

tmux ls | grep -E 'format_stability'

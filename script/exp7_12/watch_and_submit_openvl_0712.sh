#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
EXP=${ROOT}/output/experiments/20260712_uniform_multimodel_full_matrix
PY=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python

while true; do
  cd "${ROOT}"
  "${PY}" script/exp7_12/summarize_compact_talr_matrix_0712.py --root "${EXP}"
  pending="$(${PY} - "${EXP}/compact_talr_summary/compact_manifest.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
main=[i for i in x if i['model'] in {'r1_onevision_7b','vision_r1_7b'}]
print(sum(i['status'] != 'complete' for i in main))
PY
)"
  echo "[WATCH] $(date '+%F %T') compact main pending=${pending}"
  [[ "${pending}" == 0 ]] && break
  sleep 600
done

job_id="$(sbatch --parsable --export=ALL,MODEL_NAME=OpenVLThinker-7B,MODEL_KEY=openvlthinker_7b,GROUP=openvl \
  script/exp7_12/submit_compact_talr_worker_0712.sbatch)"
echo "[SUBMITTED] $(date '+%F %T') OpenVL compact validation job=${job_id}"

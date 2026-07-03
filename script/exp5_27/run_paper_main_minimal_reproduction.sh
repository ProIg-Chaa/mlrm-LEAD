#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python}"
MODEL="${MODEL:-/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL}"
MODEL_KEY="${MODEL_KEY:-r1_onevision_7b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/paper_main_minimal_reproduction}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-20000}"

mkdir -p "${BASE_DIR}/datasets"
cd "${ROOT}"

"${PYTHON_BIN}" -m py_compile \
  main.py \
  lead/inference.py \
  lead/generation_utils.py \
  script/exp5_27/summarize_paper_main_minimal_reproduction.py

if [[ ! -f data/mmhal_bench.jsonl ]]; then
  "${PYTHON_BIN}" script/prepare_mmhal_bench_jsonl.py
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path("${ROOT}")
src = root / "data/math_vista.jsonl"
out = Path("${BASE_DIR}") / "datasets/math_vista_first200.jsonl"
meta = Path("${BASE_DIR}") / "datasets/math_vista_first200_meta.json"
rows = []
with src.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 200:
            break
        if line.strip():
            rows.append(json.loads(line))
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
meta.write_text(
    json.dumps(
        {
            "source": str(src),
            "selection": "first_200_rows",
            "count": len(rows),
            "purpose": "paper_main_minimal_reproduction_mathvista200",
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Wrote {out} with {len(rows)} rows")
PY

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 0"
LEAD_ARGS="--method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128"
COT_ARGS="--method cot_greedy --cot_prompt_mode orign"

dataset_path() {
  case "$1" in
    vstar) echo "${ROOT}/data/vstar.jsonl" ;;
    realworldqa_fixed200) echo "${ROOT}/data/realworldqa_fixed_mcq_random200_seed42.jsonl" ;;
    mmvp) echo "${ROOT}/data/mmvp.jsonl" ;;
    mmhal) echo "${ROOT}/data/mmhal_bench.jsonl" ;;
    mathvista200) echo "${BASE_DIR}/datasets/math_vista_first200.jsonl" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | awk 'NR==1 {print $1}')"
    if [[ "${used}" -lt "${GPU_MEM_LIMIT_MB}" ]]; then
      break
    fi
    echo "GPU${gpu} busy: ${used} MiB used; waiting..."
    sleep 60
  done
}

write_run() {
  local dataset="$1"
  local run_name="$2"
  local gpu="$3"
  local method_args="$4"
  local run_dir="${BASE_DIR}/${MODEL_KEY}/${dataset}/${run_name}_gpu${gpu}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "$(dataset_path "${dataset}")" \\
  --output_dir "${run_dir}" \\
  ${COMMON_ARGS} \\
  ${method_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

rm -f "${BASE_DIR}/queue_gpu${GPU0}.body" "${BASE_DIR}/queue_gpu${GPU1}.body"

for dataset in realworldqa_fixed200 mmvp vstar mmhal mathvista200; do
  write_run "${dataset}" cot_orign_greedy "${GPU0}" "${COT_ARGS}"
  write_run "${dataset}" lead "${GPU1}" "${LEAD_ARGS}"
  {
    echo "wait_for_gpu \"${GPU0}\""
    echo "bash \"${BASE_DIR}/${MODEL_KEY}/${dataset}/cot_orign_greedy_gpu${GPU0}/run_command.sh\""
  } >> "${BASE_DIR}/queue_gpu${GPU0}.body"
  {
    echo "wait_for_gpu \"${GPU1}\""
    echo "bash \"${BASE_DIR}/${MODEL_KEY}/${dataset}/lead_gpu${GPU1}/run_command.sh\""
  } >> "${BASE_DIR}/queue_gpu${GPU1}.body"
done

for gpu in "${GPU0}" "${GPU1}"; do
  cat > "${BASE_DIR}/queue_gpu${gpu}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(declare -f wait_for_gpu)
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB}"
$(cat "${BASE_DIR}/queue_gpu${gpu}.body")
EOF
  chmod +x "${BASE_DIR}/queue_gpu${gpu}.sh"
done

cat > "${BASE_DIR}/summarize_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" script/exp5_27/summarize_paper_main_minimal_reproduction.py \\
  --base_dir "${BASE_DIR}" \\
  --root "${ROOT}"
EOF
chmod +x "${BASE_DIR}/summarize_after_done.sh"

if [[ "${START}" == "1" ]]; then
  setsid bash "${BASE_DIR}/queue_gpu${GPU0}.sh" > "${BASE_DIR}/queue_gpu${GPU0}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU0}.pid"
  setsid bash "${BASE_DIR}/queue_gpu${GPU1}.sh" > "${BASE_DIR}/queue_gpu${GPU1}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU1}.pid"
else
  echo "START=0, queues were generated but not launched."
fi

echo "BASE_DIR=${BASE_DIR}"
for gpu in "${GPU0}" "${GPU1}"; do
  if [[ -f "${BASE_DIR}/queue_gpu${gpu}.pid" ]]; then
    echo "GPU${gpu} queue PID=$(cat "${BASE_DIR}/queue_gpu${gpu}.pid")"
  fi
  echo "GPU${gpu} queue log: ${BASE_DIR}/queue_gpu${gpu}.log"
done
echo "Summarize after done: bash ${BASE_DIR}/summarize_after_done.sh"

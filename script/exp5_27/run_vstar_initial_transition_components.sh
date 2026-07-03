#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/vstar.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/vstar_initial_transition_components"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
LEAD_ARGS="--cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128"

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | awk 'NR==1 {print $1}')"
    if [[ "${used}" -lt 4096 ]]; then
      break
    fi
    echo "GPU${gpu} busy: ${used} MiB used; waiting..."
    sleep 60
  done
}

write_run() {
  local run_name="$1"
  local gpu="$2"
  local extra_args="$3"
  local run_dir="${BASE_DIR}/${run_name}_gpu${gpu}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "${DATASET}" \\
  --output_dir "${run_dir}" \\
  --method lead \\
  ${COMMON_ARGS} \\
  ${LEAD_ARGS} \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

# Component controls for the early LEAD transition finding.
#
# initial_transition_only:
#   keep step0 soft and the first soft->normal transition mix, then no later soft.
# initial_soft_only:
#   keep only step0 soft; skip the soft->normal transition mix.
# *_no_linebreak:
#   disable step0 soft_emb = 0.9 * soft_emb + 0.1 * newline_emb.
# *_no_to_normal:
#   disable normal_emb = beta * soft_emb + (1 - beta) * </think>_emb on to_normal.
# no_anchor:
#   disable the simple <think> -> <|image_pad|> anchor replacement.
write_run "initial_transition_only" "${GPU0}" "--lead_initial_transition_only"
write_run "initial_transition_no_linebreak" "${GPU0}" "--lead_initial_transition_only --lead_disable_step0_linebreak_mix"
write_run "initial_transition_no_to_normal" "${GPU0}" "--lead_initial_transition_only --lead_disable_to_normal_transition"

write_run "initial_soft_only" "${GPU1}" "--lead_initial_soft_only"
write_run "initial_transition_no_linebreak_no_to_normal" "${GPU1}" "--lead_initial_transition_only --lead_disable_step0_linebreak_mix --lead_disable_to_normal_transition"
write_run "initial_transition_no_anchor" "${GPU1}" "--lead_initial_transition_only --lead_disable_simple_visual_anchor"

cat > "${BASE_DIR}/queue_gpu${GPU0}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(declare -f wait_for_gpu)
wait_for_gpu "${GPU0}"
bash "${BASE_DIR}/initial_transition_only_gpu${GPU0}/run_command.sh"
wait_for_gpu "${GPU0}"
bash "${BASE_DIR}/initial_transition_no_linebreak_gpu${GPU0}/run_command.sh"
wait_for_gpu "${GPU0}"
bash "${BASE_DIR}/initial_transition_no_to_normal_gpu${GPU0}/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu${GPU1}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(declare -f wait_for_gpu)
wait_for_gpu "${GPU1}"
bash "${BASE_DIR}/initial_soft_only_gpu${GPU1}/run_command.sh"
wait_for_gpu "${GPU1}"
bash "${BASE_DIR}/initial_transition_no_linebreak_no_to_normal_gpu${GPU1}/run_command.sh"
wait_for_gpu "${GPU1}"
bash "${BASE_DIR}/initial_transition_no_anchor_gpu${GPU1}/run_command.sh"
EOF
chmod +x "${BASE_DIR}/queue_gpu${GPU0}.sh" "${BASE_DIR}/queue_gpu${GPU1}.sh"

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
import re
from pathlib import Path

base = Path("${BASE_DIR}")
runs = [
    ("initial_transition_only", "${GPU0}"),
    ("initial_transition_no_linebreak", "${GPU0}"),
    ("initial_transition_no_to_normal", "${GPU0}"),
    ("initial_soft_only", "${GPU1}"),
    ("initial_transition_no_linebreak_no_to_normal", "${GPU1}"),
    ("initial_transition_no_anchor", "${GPU1}"),
]

def load_rows(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").open("r", encoding="utf-8")
        if line.strip()
    ]

patterns = [
    r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
    r"[Aa]nswer\s*[:\s]+\(?\s*([A-Da-d])\)?",
    r"\\boxed\{\s*([A-Da-d])\s*\}",
    r"\*\*Answer:\*\*\s*([A-Da-d])",
    r"\*\*([A-Da-d])\*\*",
]

def extract(text):
    if not text:
        return None
    tail = text[-1200:]
    for pattern in patterns:
        match = re.search(pattern, tail)
        if match:
            return match.group(1).upper()
    region = tail.split("</think>")[-1]
    letters = re.findall(r"\b([A-D])\b", region[-300:])
    return letters[-1].upper() if letters else None

summary = {}
correct_sets = {}
for run, gpu in runs:
    run_dir = base / f"{run}_gpu{gpu}"
    if not (run_dir / "results.jsonl").exists():
        print(f"{run}: MISSING")
        continue
    report = json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = load_rows(run_dir)
    lens = [int(row.get("output_tokens") or 0) for row in rows]
    correct_ids = set()
    failed = 0
    for idx, row in enumerate(rows):
        pred = extract(row.get("model_answer"))
        failed += int(pred is None)
        if pred == str(row.get("answer", "")).strip().upper()[:1]:
            correct_ids.add(row.get("id", idx))
    correct_sets[run] = correct_ids
    summary[run] = {
        "accuracy": report.get("accuracy"),
        "correct": report.get("correct"),
        "total": report.get("total"),
        "avg_len": sum(lens) / len(lens) if lens else 0.0,
        "long_ge_256": sum(x >= 256 for x in lens),
        "maxed": sum(x >= 1024 for x in lens),
        "local_correct": len(correct_ids),
        "local_failed_extraction": failed,
        "lead_initial_soft_only": cfg.get("lead_initial_soft_only"),
        "lead_initial_transition_only": cfg.get("lead_initial_transition_only"),
        "lead_disable_simple_visual_anchor": cfg.get("lead_disable_simple_visual_anchor"),
        "lead_disable_step0_linebreak_mix": cfg.get("lead_disable_step0_linebreak_mix"),
        "lead_disable_to_normal_transition": cfg.get("lead_disable_to_normal_transition"),
    }

print("# VStar Initial Transition Component Controls")
for run, _ in runs:
    if run not in summary:
        continue
    item = summary[run]
    acc = item["accuracy"]
    print(
        f"{run}: {item['correct']}/{item['total']} = {acc * 100:.2f}% | "
        f"local={item['local_correct']} failed={item['local_failed_extraction']} | "
        f"avg_len={item['avg_len']:.1f} long>=256={item['long_ge_256']} maxed={item['maxed']} | "
        f"initial_soft={item['lead_initial_soft_only']} transition_only={item['lead_initial_transition_only']} "
        f"no_anchor={item['lead_disable_simple_visual_anchor']} "
        f"no_linebreak={item['lead_disable_step0_linebreak_mix']} "
        f"no_to_normal={item['lead_disable_to_normal_transition']}"
    )

print("\\n# Pairwise Deltas Against initial_transition_only")
ref = "initial_transition_only"
if ref in correct_sets:
    for run, _ in runs:
        if run == ref or run not in correct_sets:
            continue
        fixed = sorted(correct_sets[run] - correct_sets[ref])
        damaged = sorted(correct_sets[ref] - correct_sets[run])
        print(
            f"{ref} -> {run}: fixed={len(fixed)} damaged={len(damaged)} "
            f"net={len(fixed) - len(damaged)}"
        )
        print(f"  fixed_ids={fixed[:50]}")
        print(f"  damaged_ids={damaged[:50]}")

(base / "component_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

cat > "${BASE_DIR}/route_summary_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
for spec in \\
  "initial_transition_only:${GPU0}" \\
  "initial_transition_no_linebreak:${GPU0}" \\
  "initial_transition_no_to_normal:${GPU0}" \\
  "initial_soft_only:${GPU1}" \\
  "initial_transition_no_linebreak_no_to_normal:${GPU1}" \\
  "initial_transition_no_anchor:${GPU1}"; do
  run="\${spec%%:*}"
  gpu="\${spec##*:}"
  run_dir="${BASE_DIR}/\${run}_gpu\${gpu}"
  if [[ -f "\${run_dir}/token_entropy_full.jsonl" ]]; then
    "${PYTHON_BIN}" script/exp5_16/analyze_route_summary.py \\
      --run_dir "\${run_dir}" \\
      --output "\${run_dir}/route_summary.md" \\
      --output_json "\${run_dir}/route_summary.json"
  fi
done
EOF
chmod +x "${BASE_DIR}/route_summary_after_done.sh"

if [[ "${START}" == "1" ]]; then
  setsid bash "${BASE_DIR}/queue_gpu${GPU0}.sh" > "${BASE_DIR}/queue_gpu${GPU0}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU0}.pid"
  setsid bash "${BASE_DIR}/queue_gpu${GPU1}.sh" > "${BASE_DIR}/queue_gpu${GPU1}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU1}.pid"
else
  echo "START=0, queues were generated but not launched."
fi

echo "BASE_DIR=${BASE_DIR}"
if [[ -f "${BASE_DIR}/queue_gpu${GPU0}.pid" ]]; then
  echo "GPU${GPU0} queue PID=$(cat "${BASE_DIR}/queue_gpu${GPU0}.pid")"
fi
if [[ -f "${BASE_DIR}/queue_gpu${GPU1}.pid" ]]; then
  echo "GPU${GPU1} queue PID=$(cat "${BASE_DIR}/queue_gpu${GPU1}.pid")"
fi
echo "After runs complete:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
echo "  bash ${BASE_DIR}/route_summary_after_done.sh"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/vstar.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/vstar_lead_cot_sanity_matrix"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
LEAD_PAPER_ARGS="--alpha 0.4 --max_switch_count 5 --window_size 128"

write_run() {
  local run_name="$1"
  local gpu="$2"
  local method="$3"
  local extra_args="$4"
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
  --method "${method}" \\
  ${COMMON_ARGS} \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

# Clean sanity matrix:
# - all runs are greedy: --no-do_sample
# - COT differs only by prompt mode: orign vs step
# - LEAD variants use paper-like params: alpha=0.4, window_size=128, max_switch_count=5
# - lead_force_normal keeps the LEAD generation path but disables all soft embedding use
write_run "cot_orign_greedy" "${GPU0}" "cot" "--cot_prompt_mode orign"
write_run "cot_step_greedy" "${GPU1}" "cot" "--cot_prompt_mode step"
write_run "lead_force_normal" "${GPU0}" "lead" "--cot_prompt_mode orign ${LEAD_PAPER_ARGS} --lead_force_normal"
write_run "lead_no_anchor" "${GPU1}" "lead" "--cot_prompt_mode orign ${LEAD_PAPER_ARGS} --lead_disable_simple_visual_anchor"
write_run "lead" "${GPU0}" "lead" "--cot_prompt_mode orign ${LEAD_PAPER_ARGS}"

cat > "${BASE_DIR}/queue_gpu${GPU0}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/cot_orign_greedy_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/lead_force_normal_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/lead_gpu${GPU0}/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu${GPU1}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/cot_step_greedy_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/lead_no_anchor_gpu${GPU1}/run_command.sh"
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
    ("cot_orign_greedy", "${GPU0}"),
    ("cot_step_greedy", "${GPU1}"),
    ("lead_force_normal", "${GPU0}"),
    ("lead_no_anchor", "${GPU1}"),
    ("lead", "${GPU0}"),
]

patterns = [
    r"[Tt]he\\s+(?:correct\\s+)?answer\\s+is\\s*[:\\s]*\\(?([A-Da-d])\\)?",
    r"[Aa]nswer\\s*[:\\s]+\\(?([A-Da-d])\\)?",
    r"\\\\boxed\\{([A-Da-d])\\}",
    r"\\*\\*([A-Da-d])\\*\\*",
    r"(?:^|\\n)\\s*([A-Da-d])\\s*$",
]

def extract(text):
    if not text:
        return None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    letters = re.findall(r"\\b([A-D])\\b", text[-200:])
    return letters[-1].upper() if letters else None

def load_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "results.jsonl").open("r", encoding="utf-8") if line.strip()]

summary = {}
correct_sets = {}
for run, gpu in runs:
    run_dir = base / f"{run}_gpu{gpu}"
    if not (run_dir / "results.jsonl").exists():
        print(f"{run}: MISSING")
        continue
    rows = load_rows(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8"))
    lens = [int(row.get("output_tokens") or 0) for row in rows]
    correct_ids = set()
    failed = 0
    for index, row in enumerate(rows):
        pred = extract(row.get("model_answer"))
        failed += int(pred is None)
        sample_id = row.get("id", index)
        if pred == str(row.get("answer", "")).strip().upper():
            correct_ids.add(sample_id)
    correct_sets[run] = correct_ids
    summary[run] = {
        "accuracy": report.get("accuracy"),
        "correct": report.get("correct"),
        "total": report.get("total"),
        "local_correct": len(correct_ids),
        "local_failed_extraction": failed,
        "avg_len": sum(lens) / len(lens) if lens else 0.0,
        "long_ge_256": sum(x >= 256 for x in lens),
        "maxed": sum(x >= 1024 for x in lens),
        "method": cfg.get("method"),
        "cot_prompt_mode": cfg.get("cot_prompt_mode"),
        "do_sample": cfg.get("do_sample"),
        "alpha": cfg.get("alpha"),
        "window_size": cfg.get("window_size"),
        "max_switch_count": cfg.get("max_switch_count"),
        "lead_disable_simple_visual_anchor": cfg.get("lead_disable_simple_visual_anchor"),
        "lead_force_normal": cfg.get("lead_force_normal"),
    }

print("# VStar LEAD/COT Sanity Matrix")
for run, _ in runs:
    if run not in summary:
        continue
    item = summary[run]
    acc = item["accuracy"]
    print(
        f"{run}: {item['correct']}/{item['total']} = {acc * 100:.2f}% | "
        f"local={item['local_correct']} failed={item['local_failed_extraction']} | "
        f"avg_len={item['avg_len']:.1f} long>=256={item['long_ge_256']} maxed={item['maxed']} | "
        f"do_sample={item['do_sample']} prompt={item['cot_prompt_mode']} "
        f"force_normal={item['lead_force_normal']} no_anchor={item['lead_disable_simple_visual_anchor']}"
    )

comparisons = [
    ("cot_orign_greedy", "lead_force_normal"),
    ("lead_force_normal", "lead_no_anchor"),
    ("lead_no_anchor", "lead"),
    ("cot_orign_greedy", "lead"),
    ("cot_step_greedy", "lead"),
]
print("\\n# Pairwise Correctness Deltas")
for left, right in comparisons:
    if left not in correct_sets or right not in correct_sets:
        continue
    left_only = sorted(correct_sets[left] - correct_sets[right])
    right_only = sorted(correct_sets[right] - correct_sets[left])
    both = len(correct_sets[left] & correct_sets[right])
    print(
        f"{left} -> {right}: {left}_only={len(left_only)} "
        f"{right}_only={len(right_only)} both={both} net={len(right_only) - len(left_only)}"
    )
    print(f"  {right}_only_ids={right_only[:50]}")
    print(f"  {left}_only_ids={left_only[:50]}")

(base / "sanity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

cat > "${BASE_DIR}/route_summary_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
for spec in "cot_orign_greedy:${GPU0}" "cot_step_greedy:${GPU1}" "lead_force_normal:${GPU0}" "lead_no_anchor:${GPU1}" "lead:${GPU0}"; do
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

#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar.jsonl"
BASELINE_RUN="${ROOT}/output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel/pure_soft_baseline_gpu0"
COOLDOWN8_RUN="${ROOT}/output/experiments/20260519_181159/pure_soft_format_cooldown_vstar_full/format_cooldown8_gpu0"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_format_cooldown_ablation_vstar_full"

mkdir -p "${BASE_DIR}"

write_run() {
  local steps="$1"
  local gpu="$2"
  local run_dir="${BASE_DIR}/format_cooldown${steps}_gpu${gpu}"
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
  --method pure_soft \\
  --cot_prompt_mode orign \\
  --max_new_tokens 1024 \\
  --temperature 0.6 \\
  --top_p 0.95 \\
  --top_k 20 \\
  --seed 42 \\
  --device cuda \\
  --no-do_sample \\
  --save_token_entropy \\
  --save_full_token_entropy \\
  --trace_topk 20 \\
  --pure_soft_format_cooldown \\
  --format_cooldown_steps ${steps}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run 2 0
write_run 4 1

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
import re
import statistics as st
from pathlib import Path
from lead.evaluator import evaluate_single

runs = {
    "baseline": Path("${BASELINE_RUN}"),
    "cooldown2": Path("${BASE_DIR}/format_cooldown2_gpu0"),
    "cooldown4": Path("${BASE_DIR}/format_cooldown4_gpu1"),
    "cooldown8": Path("${COOLDOWN8_RUN}"),
}

def format_anomalies(text):
    text = text or ""
    return {
        "empty_paren_answer": bool(re.search(r"Answer\\s*:\\s*\\(\\s*\\)", text, re.I)),
        "multiple_answer_lines": len(re.findall(r"\\bAnswer\\s*:", text, re.I)) > 1,
        "missing_answer_marker": not bool(re.search(r"\\bAnswer\\s*:", text, re.I)),
    }

rows = {}
for name, path in runs.items():
    if not (path / "results.jsonl").exists():
        print(f"SKIP {name}: missing results")
        continue
    cur = {}
    with (path / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ok, ex = evaluate_single(r.get("model_answer"), r.get("answer", ""))
            cur[r.get("id")] = {
                "ok": ok,
                "ex": ex,
                "len": r.get("output_tokens") or 0,
                "answer": r.get("model_answer"),
                "anom": format_anomalies(r.get("model_answer")),
                "subtopic": r.get("subtopic"),
            }
    rows[name] = cur

ids = sorted(rows["baseline"])
for name in ["baseline", "cooldown2", "cooldown4", "cooldown8"]:
    if name not in rows:
        continue
    vals = sorted(rows[name][i]["len"] for i in ids)
    correct = sum(rows[name][i]["ok"] for i in ids)
    print(f"== {name} ==")
    print(f"accuracy: {correct}/{len(ids)} = {correct/len(ids):.4f}")
    print(f"length mean={sum(vals)/len(vals):.2f} median={st.median(vals)} p90={vals[int(0.9*(len(vals)-1))]} max={vals[-1]}")
    print(f"long>=256: {sum(v >= 256 for v in vals)}")
    print(f"maxed1024: {sum(v >= 1024 for v in vals)}")
    for key in ["empty_paren_answer", "multiple_answer_lines", "missing_answer_marker"]:
        print(f"{key}: {sum(rows[name][i]['anom'][key] for i in ids)}")
    if name != "baseline":
        fixed = [i for i in ids if not rows["baseline"][i]["ok"] and rows[name][i]["ok"]]
        damaged = [i for i in ids if rows["baseline"][i]["ok"] and not rows[name][i]["ok"]]
        changed = [i for i in ids if rows["baseline"][i]["answer"] != rows[name][i]["answer"]]
        print(f"delta_vs_baseline: changed={len(changed)} fixed={len(fixed)} damaged={len(damaged)} net={len(fixed)-len(damaged)}")
        trace = runs[name] / "token_entropy_full.jsonl"
        if trace.exists():
            per_sample = []
            total = 0
            with trace.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    count = sum(1 for t in (r.get("tokens") or []) if t.get("format_cooldown_active"))
                    total += count
                    per_sample.append((r.get("id"), count))
            counts = sorted(c for _, c in per_sample)
            print(f"format_cooldown: total={total} samples={sum(c>0 for c in counts)}/{len(counts)} mean={sum(counts)/len(counts):.2f} median={st.median(counts)} p90={counts[int(0.9*(len(counts)-1))]} max={counts[-1]}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

for run_dir in "${BASE_DIR}/format_cooldown2_gpu0" "${BASE_DIR}/format_cooldown4_gpu1"; do
  setsid bash "${run_dir}/run_command.sh" > "${run_dir}/nohup.log" 2>&1 < /dev/null &
  echo $! > "${run_dir}/pid.txt"
done

echo "BASE_DIR=${BASE_DIR}"
echo "cooldown2 PID=$(cat "${BASE_DIR}/format_cooldown2_gpu0/pid.txt") GPU=0"
echo "cooldown4 PID=$(cat "${BASE_DIR}/format_cooldown4_gpu1/pid.txt") GPU=1"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"

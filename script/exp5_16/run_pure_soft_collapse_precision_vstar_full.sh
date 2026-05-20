#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar.jsonl"
BASELINE_RUN="${ROOT}/output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel/pure_soft_baseline_gpu0"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_collapse_precision_vstar_full"

mkdir -p "${BASE_DIR}"

write_run() {
  local name="$1"
  local gpu="$2"
  local extra_args="$3"
  local run_dir="${BASE_DIR}/${name}_gpu${gpu}"
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
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

COMMON="--pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_min_history 4 --collapse_min_entropy 1.0"
write_run "strict_threshold" 0 "${COMMON} --collapse_entropy_alpha 2.5 --collapse_low_conf_tau 0.12 --collapse_low_margin_tau 0.03"
write_run "patience2" 1 "${COMMON} --collapse_entropy_alpha 2.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_patience 2 --collapse_patience_window 16"
write_run "late64" 0 "${COMMON} --collapse_entropy_alpha 2.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64"
write_run "repeat_gate" 1 "${COMMON} --collapse_entropy_alpha 2.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"

run_one() {
  local name="$1"
  local gpu="$2"
  local run_dir="${BASE_DIR}/${name}_gpu${gpu}"
  echo "[start] ${name} gpu=${gpu} $(date '+%F %T')" | tee -a "${BASE_DIR}/manager.log"
  setsid bash "${run_dir}/run_command.sh" > "${run_dir}/nohup.log" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${run_dir}/pid.txt"
  while ps -p "${pid}" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[done] ${name} gpu=${gpu} $(date '+%F %T')" | tee -a "${BASE_DIR}/manager.log"
}

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
import statistics as st
from pathlib import Path
from lead.evaluator import evaluate_single

base = Path("${BASE_DIR}")
baseline = Path("${BASELINE_RUN}")
runs = {
    "baseline": baseline,
    "strict_threshold": base / "strict_threshold_gpu0",
    "patience2": base / "patience2_gpu1",
    "late64": base / "late64_gpu0",
    "repeat_gate": base / "repeat_gate_gpu1",
}

rows = {}
for name, run in runs.items():
    path = run / "results.jsonl"
    if not path.exists():
        print(f"SKIP {name}: missing {path}")
        continue
    cur = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ok, ex = evaluate_single(r.get("model_answer"), r.get("answer", ""))
            cur[r.get("id")] = {
                "ok": ok,
                "ex": ex,
                "len": r.get("output_tokens") or 0,
                "answer": r.get("model_answer"),
            }
    rows[name] = cur

ids = sorted(rows["baseline"])
for name in runs:
    if name not in rows:
        continue
    vals = sorted(rows[name][i]["len"] for i in ids)
    correct = sum(rows[name][i]["ok"] for i in ids)
    print(f"== {name} ==")
    print(f"accuracy: {correct}/{len(ids)} = {correct/len(ids):.4f}")
    print(f"length mean={sum(vals)/len(vals):.2f} median={st.median(vals)} p90={vals[int(0.9*(len(vals)-1))]} max={vals[-1]}")
    print(f"long>=256: {sum(v >= 256 for v in vals)}")
    print(f"maxed1024: {sum(v >= 1024 for v in vals)}")
    if name == "baseline":
        continue
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
                count = sum(1 for t in (r.get("tokens") or []) if t.get("collapse_on_diffuse"))
                total += count
                per_sample.append((r.get("id"), count))
        counts = sorted(c for _, c in per_sample)
        print(f"collapse: total={total} samples={sum(c > 0 for c in counts)}/{len(counts)} mean={sum(counts)/len(counts):.2f} median={st.median(counts)} p90={counts[int(0.9*(len(counts)-1))]} max={counts[-1]}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

echo "BASE_DIR=${BASE_DIR}"
echo "Manager log: ${BASE_DIR}/manager.log"
echo "Compare after done: bash ${BASE_DIR}/compare_after_done.sh"

(
  run_one "strict_threshold" 0 &
  run_one "patience2" 1 &
  wait
  run_one "late64" 0 &
  run_one "repeat_gate" 1 &
  wait
  echo "[all_done] $(date '+%F %T')" | tee -a "${BASE_DIR}/manager.log"
) &
echo $! > "${BASE_DIR}/manager.pid"
echo "Manager PID=$(cat "${BASE_DIR}/manager.pid")"

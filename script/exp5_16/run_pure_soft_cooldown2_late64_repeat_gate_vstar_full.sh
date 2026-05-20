#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar.jsonl"
BASELINE_RUN="${ROOT}/output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel/pure_soft_baseline_gpu0"
COOLDOWN2_RUN="${ROOT}/output/experiments/20260519_234017/pure_soft_format_cooldown_ablation_vstar_full/format_cooldown2_gpu0"
LATE_REPEAT_RUN="${ROOT}/output/experiments/20260518_200744/pure_soft_collapse_late64_repeat_gate_vstar_full/late64_repeat_gate_gpu0"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_cooldown2_late64_repeat_gate_vstar_full"
RUN_DIR="${BASE_DIR}/cooldown2_late64_repeat_gate_gpu0"

mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "${DATASET}" \\
  --output_dir "${RUN_DIR}" \\
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
  --format_cooldown_steps 2 \\
  --pure_soft_collapse_on_diffuse \\
  --collapse_entropy_window 16 \\
  --collapse_entropy_alpha 2.0 \\
  --collapse_min_history 4 \\
  --collapse_min_entropy 1.0 \\
  --collapse_low_conf_tau 0.20 \\
  --collapse_low_margin_tau 0.05 \\
  --collapse_min_step 64 \\
  --collapse_require_repeat_degen \\
  --collapse_repeat_ngram 3 \\
  --collapse_recent_repeat_window 32 \\
  --collapse_recent_repeat_tau 0.35
EOF
chmod +x "${RUN_DIR}/run_command.sh"

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
    "cooldown2": Path("${COOLDOWN2_RUN}"),
    "late64_repeat_gate": Path("${LATE_REPEAT_RUN}"),
    "combo": Path("${RUN_DIR}"),
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
for name in ["baseline", "late64_repeat_gate", "cooldown2", "combo"]:
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

trace = runs["combo"] / "token_entropy_full.jsonl"
cooldown_total = collapse_total = 0
cooldown_samples = collapse_samples = 0
per_sample = []
with trace.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        toks = r.get("tokens") or []
        cooldown_count = sum(1 for t in toks if t.get("format_cooldown_active"))
        collapse_count = sum(1 for t in toks if t.get("collapse_on_diffuse"))
        cooldown_total += cooldown_count
        collapse_total += collapse_count
        cooldown_samples += int(cooldown_count > 0)
        collapse_samples += int(collapse_count > 0)
        per_sample.append((r.get("id"), cooldown_count, collapse_count))
cool_counts = sorted(x[1] for x in per_sample)
col_counts = sorted(x[2] for x in per_sample)
print("== combo_routes ==")
print(f"format_cooldown total={cooldown_total} samples={cooldown_samples}/{len(per_sample)} mean={sum(cool_counts)/len(cool_counts):.2f} median={st.median(cool_counts)} p90={cool_counts[int(0.9*(len(cool_counts)-1))]} max={cool_counts[-1]}")
print(f"collapse total={collapse_total} samples={collapse_samples}/{len(per_sample)} mean={sum(col_counts)/len(col_counts):.2f} median={st.median(col_counts)} p90={col_counts[int(0.9*(len(col_counts)-1))]} max={col_counts[-1]}")
print(f"top_collapse={sorted(per_sample, key=lambda x: x[2], reverse=True)[:20]}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

setsid bash "${RUN_DIR}/run_command.sh" > "${RUN_DIR}/nohup.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pid.txt"

echo "BASE_DIR=${BASE_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "PID=$(cat "${RUN_DIR}/pid.txt")"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"

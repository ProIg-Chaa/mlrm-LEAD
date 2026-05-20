#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar.jsonl"
BASELINE_RUN="${ROOT}/output/experiments/20260516_183300/exp1_vstar_spike_type_parallel/lead_gpu1"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/lead_soft_veto_late64_repeat_gate_vstar_full"
RUN_DIR="${BASE_DIR}/lead_soft_veto_late64_repeat_gate_gpu0"

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
  --method lead \\
  --cot_prompt_mode orign \\
  --alpha 0.4 \\
  --max_switch_count 5 \\
  --window_size 128 \\
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
  --lead_soft_veto_on_diffuse \\
  --lead_veto_entropy_window 16 \\
  --lead_veto_entropy_alpha 2.0 \\
  --lead_veto_min_history 4 \\
  --lead_veto_min_entropy 1.0 \\
  --lead_veto_low_conf_tau 0.20 \\
  --lead_veto_low_margin_tau 0.05 \\
  --lead_veto_min_step 64 \\
  --lead_veto_require_repeat_degen \\
  --lead_veto_repeat_ngram 3 \\
  --lead_veto_recent_repeat_window 32 \\
  --lead_veto_recent_repeat_tau 0.35
EOF
chmod +x "${RUN_DIR}/run_command.sh"

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
import statistics as st
from pathlib import Path
from lead.evaluator import evaluate_single

baseline = Path("${BASELINE_RUN}")
run = Path("${RUN_DIR}")
runs = {"lead_baseline": baseline, "lead_soft_veto": run}

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
                "subtopic": r.get("subtopic"),
                "question": r.get("question"),
                "gt": r.get("answer"),
            }
    rows[name] = cur

ids = sorted(rows["lead_baseline"])
for name in ["lead_baseline", "lead_soft_veto"]:
    vals = sorted(rows[name][i]["len"] for i in ids)
    correct = sum(rows[name][i]["ok"] for i in ids)
    print(f"== {name} ==")
    print(f"accuracy: {correct}/{len(ids)} = {correct/len(ids):.4f}")
    print(f"length mean={sum(vals)/len(vals):.2f} median={st.median(vals)} p90={vals[int(0.9*(len(vals)-1))]} max={vals[-1]}")
    print(f"long>=256: {sum(v >= 256 for v in vals)}")
    print(f"maxed1024: {sum(v >= 1024 for v in vals)}")
    for sub in sorted({rows[name][i]['subtopic'] for i in ids}):
        sub_ids = [i for i in ids if rows[name][i]["subtopic"] == sub]
        c = sum(rows[name][i]["ok"] for i in sub_ids)
        print(f"{sub}: {c}/{len(sub_ids)} = {c/len(sub_ids):.4f}")

fixed = [i for i in ids if not rows["lead_baseline"][i]["ok"] and rows["lead_soft_veto"][i]["ok"]]
damaged = [i for i in ids if rows["lead_baseline"][i]["ok"] and not rows["lead_soft_veto"][i]["ok"]]
changed = [i for i in ids if rows["lead_baseline"][i]["answer"] != rows["lead_soft_veto"][i]["answer"]]
print("== delta_vs_lead_baseline ==")
print(f"changed={len(changed)}")
print(f"fixed={len(fixed)} {fixed}")
print(f"damaged={len(damaged)} {damaged}")
print(f"net={len(fixed)-len(damaged)}")
for label, arr in [("fixed", fixed), ("damaged", damaged)]:
    print(f"== {label}_details ==")
    for i in arr:
        b = rows["lead_baseline"][i]
        r = rows["lead_soft_veto"][i]
        print(json.dumps({
            "id": i,
            "question": b["question"],
            "answer": b["gt"],
            "subtopic": b["subtopic"],
            "base_ex": b["ex"],
            "run_ex": r["ex"],
            "base_len": b["len"],
            "run_len": r["len"],
        }, ensure_ascii=False))

trace = run / "token_entropy_full.jsonl"
per_sample = []
total = 0
with trace.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        count = sum(1 for t in (r.get("tokens") or []) if t.get("lead_soft_veto"))
        total += count
        per_sample.append((r.get("id"), count))
counts = sorted(c for _, c in per_sample)
print("== lead_soft_veto ==")
print(f"total={total}")
print(f"samples={sum(c > 0 for c in counts)}/{len(counts)}")
print(f"mean={sum(counts)/len(counts):.2f} median={st.median(counts)} p90={counts[int(0.9*(len(counts)-1))]} max={counts[-1]}")
print(f"top_samples={sorted(per_sample, key=lambda x: x[1], reverse=True)[:20]}")
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

#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar.jsonl"
BASELINE_RUN="${ROOT}/output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel/pure_soft_baseline_gpu0"
COOLDOWN2_RUN="${ROOT}/output/experiments/20260519_234017/pure_soft_format_cooldown_ablation_vstar_full/format_cooldown2_gpu0"
LATE_REPEAT_RUN="${ROOT}/output/experiments/20260518_200744/pure_soft_collapse_late64_repeat_gate_vstar_full/late64_repeat_gate_gpu0"
BEST_COMBO_RUN="${ROOT}/output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0"
ANSWER_ZONE_RUN="${ROOT}/output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full/answer_zone_discrete_gpu1"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_multisignal_mix_vstar_full"

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

COMMON_COOLDOWN="--pure_soft_format_cooldown \\
  --format_cooldown_steps 2"

LATE_REPEAT="--pure_soft_collapse_on_diffuse \\
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
  --collapse_recent_repeat_tau 0.35"

write_run "cooldown2_answer_zone" 0 "${COMMON_COOLDOWN} \\
  --pure_soft_answer_zone_discrete"

write_run "cooldown2_late64_repeat_answer_zone" 1 "${COMMON_COOLDOWN} \\
  ${LATE_REPEAT} \\
  --pure_soft_answer_zone_discrete"

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
    "best_combo": Path("${BEST_COMBO_RUN}"),
    "answer_zone": Path("${ANSWER_ZONE_RUN}"),
    "cooldown2_answer_zone": Path("${BASE_DIR}/cooldown2_answer_zone_gpu0"),
    "cooldown2_late64_repeat_answer_zone": Path("${BASE_DIR}/cooldown2_late64_repeat_answer_zone_gpu1"),
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
    result_path = path / "results.jsonl"
    if not result_path.exists():
        print(f"SKIP {name}: missing {result_path}")
        continue
    cur = {}
    with result_path.open(encoding="utf-8") as f:
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
order = [
    "baseline",
    "late64_repeat_gate",
    "answer_zone",
    "cooldown2",
    "best_combo",
    "cooldown2_answer_zone",
    "cooldown2_late64_repeat_answer_zone",
]
for name in order:
    if name not in rows:
        continue
    vals = sorted(rows[name][i]["len"] for i in ids)
    correct = sum(rows[name][i]["ok"] for i in ids)
    print(f"== {name} ==")
    print(f"accuracy: {correct}/{len(ids)} = {correct/len(ids):.4f}")
    print(f"length mean={sum(vals)/len(vals):.2f} median={st.median(vals)} p90={vals[int(0.9*(len(vals)-1))]} max={vals[-1]}")
    print(f"long>=256: {sum(v >= 256 for v in vals)}")
    print(f"maxed1024: {sum(v >= 1024 for v in vals)}")
    by_sub = {}
    for i in ids:
        sub = rows[name][i]["subtopic"]
        by_sub.setdefault(sub, [0, 0])
        by_sub[sub][1] += 1
        by_sub[sub][0] += int(rows[name][i]["ok"])
    for sub, (c, n) in sorted(by_sub.items()):
        print(f"subtopic {sub}: {c}/{n} = {c/n:.4f}")
    for key in ["empty_paren_answer", "multiple_answer_lines", "missing_answer_marker"]:
        print(f"{key}: {sum(rows[name][i]['anom'][key] for i in ids)}")
    if name != "baseline":
        fixed = [i for i in ids if not rows["baseline"][i]["ok"] and rows[name][i]["ok"]]
        damaged = [i for i in ids if rows["baseline"][i]["ok"] and not rows[name][i]["ok"]]
        changed = [i for i in ids if rows["baseline"][i]["answer"] != rows[name][i]["answer"]]
        print(f"delta_vs_baseline: changed={len(changed)} fixed={len(fixed)} damaged={len(damaged)} net={len(fixed)-len(damaged)}")

for target, base in [
    ("cooldown2_answer_zone", "cooldown2"),
    ("cooldown2_late64_repeat_answer_zone", "best_combo"),
    ("cooldown2_late64_repeat_answer_zone", "cooldown2_answer_zone"),
    ("cooldown2_late64_repeat_answer_zone", "cooldown2"),
]:
    if target not in rows or base not in rows:
        continue
    fixed = [i for i in ids if not rows[base][i]["ok"] and rows[target][i]["ok"]]
    damaged = [i for i in ids if rows[base][i]["ok"] and not rows[target][i]["ok"]]
    changed = [i for i in ids if rows[base][i]["answer"] != rows[target][i]["answer"]]
    print(f"== {target}_vs_{base} ==")
    print(f"changed={len(changed)} fixed={len(fixed)} damaged={len(damaged)} net={len(fixed)-len(damaged)}")
    print(f"fixed_ids={fixed}")
    print(f"damaged_ids={damaged}")

for name in ["cooldown2_answer_zone", "cooldown2_late64_repeat_answer_zone"]:
    path = runs[name] / "token_entropy_full.jsonl"
    if not path.exists():
        continue
    counts = {
        "format_cooldown_active": [],
        "collapse_on_diffuse": [],
        "answer_zone_discrete_active": [],
    }
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = r.get("tokens") or []
            for key in counts:
                counts[key].append(sum(1 for t in toks if t.get(key)))
    print(f"== {name}_routes ==")
    for key, vals in counts.items():
        vals_sorted = sorted(vals)
        print(
            f"{key}: total={sum(vals)} samples={sum(v > 0 for v in vals)}/{len(vals)} "
            f"mean={sum(vals)/len(vals):.2f} median={st.median(vals)} "
            f"p90={vals_sorted[int(0.9*(len(vals_sorted)-1))]} max={vals_sorted[-1]}"
        )
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

for run_dir in \
  "${BASE_DIR}/cooldown2_answer_zone_gpu0" \
  "${BASE_DIR}/cooldown2_late64_repeat_answer_zone_gpu1"; do
  setsid bash "${run_dir}/run_command.sh" > "${run_dir}/nohup.log" 2>&1 < /dev/null &
  echo $! > "${run_dir}/pid.txt"
done

echo "BASE_DIR=${BASE_DIR}"
echo "cooldown2_answer_zone PID=$(cat "${BASE_DIR}/cooldown2_answer_zone_gpu0/pid.txt") GPU=0"
echo "cooldown2_late64_repeat_answer_zone PID=$(cat "${BASE_DIR}/cooldown2_late64_repeat_answer_zone_gpu1/pid.txt") GPU=1"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"

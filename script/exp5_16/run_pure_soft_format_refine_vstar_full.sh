#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar.jsonl"
BASELINE_RUN="${ROOT}/output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel/pure_soft_baseline_gpu0"
COOLDOWN2_RUN="${ROOT}/output/experiments/20260519_234017/pure_soft_format_cooldown_ablation_vstar_full/format_cooldown2_gpu0"
BEST_COMBO_RUN="${ROOT}/output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_format_refine_vstar_full"

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
  --pure_soft_format_cooldown \\
  --format_cooldown_steps 2 \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run "format_highrisk_only_cooldown2" 0 "--format_cooldown_highrisk_only"
write_run "format_cooldown2_min_step32" 1 "--format_cooldown_min_step 32"

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
    "best_combo": Path("${BEST_COMBO_RUN}"),
    "highrisk_only": Path("${BASE_DIR}/format_highrisk_only_cooldown2_gpu0"),
    "min_step32": Path("${BASE_DIR}/format_cooldown2_min_step32_gpu1"),
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
order = ["baseline", "cooldown2", "best_combo", "highrisk_only", "min_step32"]
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
    ("highrisk_only", "cooldown2"),
    ("min_step32", "cooldown2"),
    ("highrisk_only", "best_combo"),
    ("min_step32", "best_combo"),
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

for name in ["highrisk_only", "min_step32"]:
    path = runs[name] / "token_entropy_full.jsonl"
    if not path.exists():
        continue
    per_sample = []
    format_token_total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = r.get("tokens") or []
            cooldown_count = sum(1 for t in toks if t.get("format_cooldown_active"))
            format_token_count = sum(1 for t in toks if t.get("format_token"))
            format_token_total += format_token_count
            per_sample.append((r.get("id"), cooldown_count, format_token_count))
    cooldown_counts = sorted(x[1] for x in per_sample)
    token_counts = sorted(x[2] for x in per_sample)
    print(f"== {name}_routes ==")
    print(
        f"format_cooldown_active: total={sum(cooldown_counts)} samples={sum(v > 0 for v in cooldown_counts)}/{len(cooldown_counts)} "
        f"mean={sum(cooldown_counts)/len(cooldown_counts):.2f} median={st.median(cooldown_counts)} "
        f"p90={cooldown_counts[int(0.9*(len(cooldown_counts)-1))]} max={cooldown_counts[-1]}"
    )
    print(
        f"format_token: total={format_token_total} samples={sum(v > 0 for v in token_counts)}/{len(token_counts)} "
        f"mean={sum(token_counts)/len(token_counts):.2f} median={st.median(token_counts)} "
        f"p90={token_counts[int(0.9*(len(token_counts)-1))]} max={token_counts[-1]}"
    )
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

for run_dir in \
  "${BASE_DIR}/format_highrisk_only_cooldown2_gpu0" \
  "${BASE_DIR}/format_cooldown2_min_step32_gpu1"; do
  setsid bash "${run_dir}/run_command.sh" > "${run_dir}/nohup.log" 2>&1 < /dev/null &
  echo $! > "${run_dir}/pid.txt"
done

echo "BASE_DIR=${BASE_DIR}"
echo "format_highrisk_only_cooldown2 PID=$(cat "${BASE_DIR}/format_highrisk_only_cooldown2_gpu0/pid.txt") GPU=0"
echo "format_cooldown2_min_step32 PID=$(cat "${BASE_DIR}/format_cooldown2_min_step32_gpu1/pid.txt") GPU=1"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"

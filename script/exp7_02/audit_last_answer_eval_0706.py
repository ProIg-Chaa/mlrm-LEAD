import json
import re
from pathlib import Path

base = Path(
    "output/experiments/20260705_integrated_cot_lead_baselines/"
    "integrated_repo_cot_lead_baselines/r1_onevision_7b"
)

patterns = [
    re.compile(r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?"),
    re.compile(r"\\boxed\{([A-Da-d])\}"),
    re.compile(r"\*\*([A-Da-d])\*\*"),
    re.compile(r"(?:^|\n)\s*([A-Da-d])\s*$"),
]


def extract_last(text: str):
    hits = []
    for pat in patterns:
        for match in pat.finditer(text or ""):
            hits.append((match.start(), match.group(1).upper()))
    if hits:
        return sorted(hits)[-1][1]
    last = re.findall(r"\b([A-D])\b", (text or "")[-200:])
    return last[-1].upper() if last else None


def eval_run(run: Path):
    total = correct = failed = 0
    for line in (run / "results.jsonl").open():
        row = json.loads(line)
        total += 1
        pred = extract_last(row.get("model_answer") or "")
        if pred is None:
            failed += 1
        if pred == str(row.get("answer", "")).strip().upper():
            correct += 1
    return total, correct, failed


for dataset in [
    "vstar",
    "realworldqa_fixed200",
    "visulogic300",
    "vmcbench_dev",
    "mmk12_math",
    "mmk12_physics",
    "mmk12_chemistry",
    "mmk12_biology",
]:
    parts = [dataset]
    for run_name in ["cot_orign_greedy_gpu0", "lead_gpu1"]:
        run_dir = base / dataset / run_name
        if (run_dir / "results.jsonl").exists():
            total, correct, failed = eval_run(run_dir)
            parts.append(f"{run_name}: {correct}/{total}={correct/total:.3f} fail={failed}")
    print(" | ".join(parts))

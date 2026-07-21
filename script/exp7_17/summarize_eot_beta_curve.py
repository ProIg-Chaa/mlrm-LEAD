#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


BETAS = [0.40, 0.55, 0.70, 0.85, 1.00]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer(text: str | None) -> str | None:
    if not text:
        return None
    marks = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[marks[-1].start():] if marks else text[-1800:]
    hits = []
    for pattern in (r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}", r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?", r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?"):
        hits.extend((m.start(), m.group(1).upper()) for m in re.finditer(pattern, region, re.I))
    if hits:
        return max(hits)[1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-200:])
    return letters[-1].upper() if letters else None


def scored(run: Path, dataset: str) -> dict[str, dict]:
    special = run / "specialized_eval_rows.jsonl"
    if dataset == "mmvp" and special.is_file():
        return {str(r["id"]): {"pred": r.get("specialized_pred"), "correct": bool(r.get("specialized_is_correct"))} for r in read_jsonl(special)}
    out = {}
    for row in read_jsonl(run / "results.jsonl"):
        guess = answer(row.get("model_answer"))
        out[str(row["id"])] = {"pred": guess, "correct": guess == str(row.get("answer", "")).strip().upper()[:1]}
    return out


def mcnemar(fixed: int, damaged: int) -> float:
    n = fixed + damaged
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixed, damaged) + 1)) / 2**n)


def delta(base: dict[str, dict], cur: dict[str, dict]) -> dict:
    ids = sorted(set(base) & set(cur))
    fixed = [i for i in ids if not base[i]["correct"] and cur[i]["correct"]]
    damaged = [i for i in ids if base[i]["correct"] and not cur[i]["correct"]]
    rng = random.Random(42)
    samples = []
    for _ in range(5000):
        draw = [rng.choice(ids) for _ in ids]
        samples.append(sum(int(cur[i]["correct"]) - int(base[i]["correct"]) for i in draw) / len(draw))
    samples.sort()
    return {"fixed": len(fixed), "damaged": len(damaged), "net": len(fixed)-len(damaged), "mcnemar_p": mcnemar(len(fixed), len(damaged),), "bootstrap_95ci": [samples[124], samples[4874]]}


def pair_acc(run: Path) -> float | None:
    path = run / "specialized_eval_report.json"
    return json.loads(path.read_text(encoding="utf-8")).get("pair_accuracy") if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline-root", required=True)
    args = parser.parse_args()
    root, base = Path(args.root), Path(args.baseline_root)
    records = []
    for dataset in ("vstar", "mmvp"):
        eot = scored(base / dataset / "original_eot_bridge_step1", dataset)
        direct = scored(base / dataset / "direct_token_step1", dataset)
        for beta in BETAS:
            label = f"beta{beta:.2f}".replace(".", "p")
            run = base / dataset / ("original_eot_bridge_step1" if beta == 0.70 else "direct_token_step1") if beta in {0.70, 1.00} else root / dataset / label
            cur = scored(run, dataset)
            record = {"dataset": dataset, "beta": beta, "anchor_weight": 1-beta, "run_dir": str(run), "accuracy": sum(x["correct"] for x in cur.values()) / len(cur), "failed_extraction": sum(x["pred"] is None for x in cur.values()), "vs_eot_beta070": delta(eot, cur), "vs_direct_beta100": delta(direct, cur)}
            if dataset == "mmvp":
                record["pair_accuracy"] = pair_acc(run)
            records.append(record)
    (root / "beta_curve_summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# EOT Bridge Strength Curve", "", "All new runs use the original EOT anchor and force the early handoff at step 1. Beta weights the current-token source; anchor weight is 1-beta.", "", "| Dataset | beta | EOT weight | Accuracy | Pair acc | Failed | Net vs beta=.70 | Net vs beta=1.00 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        pair = "-" if row.get("pair_accuracy") is None else f"{100*row['pair_accuracy']:.2f}%"
        lines.append(f"| {row['dataset']} | {row['beta']:.2f} | {row['anchor_weight']:.2f} | {100*row['accuracy']:.2f}% | {pair} | {row['failed_extraction']} | {row['vs_eot_beta070']['net']} | {row['vs_direct_beta100']['net']} |")
    lines += ["", "Interpret this as a sensitivity analysis of the inherited EOT bridge, not a claim that one beta is universally optimal. VStar and MMVP are both reported before any external selection."]
    (root / "beta_curve_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

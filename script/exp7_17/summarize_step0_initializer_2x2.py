#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


NAMES = ["hard_no_newline", "hard_with_newline", "soft_no_newline", "soft_with_newline"]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pred(text: str | None) -> str | None:
    if not text:
        return None
    marks = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[marks[-1].start():] if marks else text[-1800:]
    hits = []
    for p in (r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}", r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?", r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?"):
        hits.extend((m.start(), m.group(1).upper()) for m in re.finditer(p, region, re.I))
    return max(hits)[1] if hits else (re.findall(r"\b([A-Da-d])\b", region[-200:]) or [None])[-1]


def scored(run: Path, dataset: str) -> dict[str, dict]:
    special = run / "specialized_eval_rows.jsonl"
    if dataset == "mmvp" and special.is_file():
        return {str(r["id"]): {"pred": r.get("specialized_pred"), "correct": bool(r.get("specialized_is_correct"))} for r in jsonl(special)}
    return {str(r["id"]): {"pred": pred(r.get("model_answer")), "correct": pred(r.get("model_answer")) == str(r.get("answer", "")).strip().upper()[:1]} for r in jsonl(run / "results.jsonl")}


def compare(base: dict[str, dict], cur: dict[str, dict]) -> dict:
    ids = sorted(set(base) & set(cur))
    fixed = [i for i in ids if not base[i]["correct"] and cur[i]["correct"]]
    damaged = [i for i in ids if base[i]["correct"] and not cur[i]["correct"]]
    n = len(fixed) + len(damaged)
    p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(len(fixed), len(damaged)) + 1)) / 2**n)
    return {"fixed": len(fixed), "damaged": len(damaged), "net": len(fixed)-len(damaged), "mcnemar_p": p}


def pair(run: Path) -> float | None:
    path = run / "specialized_eval_report.json"
    return json.loads(path.read_text(encoding="utf-8")).get("pair_accuracy") if path.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--full-run-root", required=True)
    ap.add_argument("--cot-root", required=True)
    a = ap.parse_args()
    root, full, cotroot = Path(a.root), Path(a.full_run_root), Path(a.cot_root)
    records = []
    for ds in ("vstar", "mmvp"):
        runs = {
            "hard_no_newline": root / ds / "hard_no_newline",
            "hard_with_newline": root / ds / "hard_with_newline",
            "soft_no_newline": root / ds / "soft_no_newline",
            "soft_with_newline": full / ds / "direct_token_step1",
        }
        baseline = scored(runs["hard_no_newline"], ds)
        cot = scored(cotroot / ds / "cot_orign_greedy", ds)
        acc = {}
        for name, run in runs.items():
            data = scored(run, ds)
            acc[name] = sum(x["correct"] for x in data.values()) / len(data)
            row = {"dataset": ds, "condition": name, "run_dir": str(run), "accuracy": acc[name], "failed_extraction": sum(x["pred"] is None for x in data.values()), "vs_hard_no_newline": compare(baseline, data), "vs_cot": compare(cot, data)}
            if ds == "mmvp": row["pair_accuracy"] = pair(run)
            records.append(row)
        interaction = acc["soft_with_newline"] - acc["soft_no_newline"] - acc["hard_with_newline"] + acc["hard_no_newline"]
        records.append({"dataset": ds, "interaction_soft_x_newline_pp": 100 * interaction})
    (root / "step0_initializer_2x2_summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Step-0 Initializer 2x2", "", "All conditions force a step-1 direct hard handoff (beta=1) and then use normal greedy COT. The full soft+newline cell is reused from the matched direct-hard run.", "", "| Dataset | Step-0 state | Newline | Accuracy | Pair acc | Failed | Net vs hard/no-newline |", "|---|---|---|---:|---:|---:|---:|"]
    for row in records:
        if "condition" not in row:
            continue
        soft, newline = ("soft", "on") if row["condition"] == "soft_with_newline" else ("soft", "off") if row["condition"] == "soft_no_newline" else ("hard", "on") if row["condition"] == "hard_with_newline" else ("hard", "off")
        pair_acc = "-" if row.get("pair_accuracy") is None else f"{100*row['pair_accuracy']:.2f}%"
        lines.append(f"| {row['dataset']} | {soft} | {newline} | {100*row['accuracy']:.2f}% | {pair_acc} | {row['failed_extraction']} | {row['vs_hard_no_newline']['net']} |")
    lines += ["", "## Interaction", ""]
    for row in records:
        if "interaction_soft_x_newline_pp" in row:
            lines.append(f"- {row['dataset']}: soft x newline interaction = {row['interaction_soft_x_newline_pp']:.2f} pp.")
    lines += ["", "Interpretation requires the full cell to exceed both single-factor cells; otherwise do not claim a soft/newline synergy."]
    (root / "step0_initializer_2x2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

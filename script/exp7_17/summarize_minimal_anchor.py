#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pred(text: str | None) -> str | None:
    if not text:
        return None
    region = (list(re.finditer(r"answer\s*[:.]", text, re.I)) or [None])[-1]
    region = text[region.start():] if region else text[-1800:]
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
    return {"fixed": len(fixed), "damaged": len(damaged), "net": len(fixed)-len(damaged), "mcnemar_p": p, "agreement": sum(base[i]["pred"] == cur[i]["pred"] for i in ids)/len(ids)}


def pair(run: Path) -> float | None:
    p = run / "specialized_eval_report.json"
    return json.loads(p.read_text(encoding="utf-8")).get("pair_accuracy") if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline-root", required=True)
    a = ap.parse_args()
    root, base = Path(a.root), Path(a.baseline_root)
    out = []
    for ds in ("vstar", "mmvp"):
        direct_run = base / ds / "direct_token_step1"
        eot_run = base / ds / "original_eot_bridge_step1"
        direct, eot = scored(direct_run, ds), scored(eot_run, ds)
        for name, run in [("direct_hard", direct_run), ("end_thinking", eot_run), ("start_thinking", root / ds / "start_thinking"), ("newline", root / ds / "newline")]:
            data = scored(run, ds)
            row = {"dataset": ds, "anchor": name, "accuracy": sum(x["correct"] for x in data.values())/len(data), "failed_extraction": sum(x["pred"] is None for x in data.values()), "vs_direct": compare(direct, data), "vs_eot": compare(eot, data), "run_dir": str(run)}
            if ds == "mmvp": row["pair_accuracy"] = pair(run)
            out.append(row)
    (root / "minimal_anchor_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Minimal Anchor Identity Control", "", "All four conditions share the step-0 initializer, beta=0.7, forced step-1 handoff, greedy decoding, seed 42, and max_new_tokens=1024.", "", "| Dataset | Anchor | Accuracy | Pair acc | Failed | Fixed/Damaged vs direct | Net vs EOT | McNemar vs EOT |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in out:
        pair_acc = "-" if row.get("pair_accuracy") is None else f"{100*row['pair_accuracy']:.2f}%"
        lines.append(f"| {row['dataset']} | {row['anchor']} | {100*row['accuracy']:.2f}% | {pair_acc} | {row['failed_extraction']} | {row['vs_direct']['fixed']}/{row['vs_direct']['damaged']} | {row['vs_eot']['net']} | {row['vs_eot']['mcnemar_p']:.4f} |")
    lines += ["", "`</think>` and `<think>` are represented by the first subtokens resolved by the inherited LEAD implementation; this is an implementation-level identity control, not evidence about complete lexical tags."]
    (root / "minimal_anchor_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

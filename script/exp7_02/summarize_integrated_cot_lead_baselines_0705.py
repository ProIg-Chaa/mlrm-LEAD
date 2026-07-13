#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path


DATASET_TARGETS = {
    "mmhal": 96,
    "vstar": 191,
    "realworldqa_fixed200": 200,
    "mmvp": 300,
    "visulogic300": 300,
    "mathvista200": 200,
    "vmcbench_dev": 1000,
    "mmk12_math": 500,
    "mmk12_physics": 500,
    "mmk12_chemistry": 500,
    "mmk12_biology": 500,
    "pope_random": 3000,
    "pope_popular": 3000,
    "pope_adversarial": 3000,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_letter(text, letters="ABCDE"):
    if not text:
        return None
    tail = text[-1200:]
    patterns = [
        rf"[Aa]nswer\s*[:\s]+\(?([{letters}a-e])\)?",
        rf"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([{letters}a-e])\)?",
        rf"\\boxed\{{\s*([{letters}a-e])\s*\}}",
        rf"\(([{letters}a-e])\)",
    ]
    for pat in patterns:
        m = re.search(pat, tail)
        if m:
            return m.group(1).upper()
    found = re.findall(rf"\b([{letters}])\b", tail[-300:])
    return found[-1].upper() if found else None


def is_correct(row):
    pred = row.get("extracted_answer") or extract_letter(row.get("model_answer"))
    gt = str(row.get("answer") or "").strip().upper()[:1]
    return bool(pred) and pred.upper() == gt


def pairwise(cot_rows, lead_rows):
    cot = {str(r.get("id")): r for r in cot_rows}
    lead = {str(r.get("id")): r for r in lead_rows}
    ids = sorted(set(cot) & set(lead))
    out = {"fixed": 0, "damaged": 0, "unchanged_correct": 0, "unchanged_wrong": 0, "total": len(ids)}
    fixed_ids, damaged_ids = [], []
    for sid in ids:
        c = is_correct(cot[sid])
        l = is_correct(lead[sid])
        if (not c) and l:
            out["fixed"] += 1
            fixed_ids.append(sid)
        elif c and (not l):
            out["damaged"] += 1
            damaged_ids.append(sid)
        elif c and l:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1
    out["fixed_ids_head"] = fixed_ids[:100]
    out["damaged_ids_head"] = damaged_ids[:100]
    return out


def pope_yesno(rows):
    tp = tn = fp = fn = failed = 0
    for row in rows:
        pred = extract_letter(row.get("model_answer"), "AB")
        gt = str(row.get("answer") or "").strip().upper()[:1]
        if pred is None:
            failed += 1
            continue
        py, gy = pred == "A", gt == "A"
        if py and gy:
            tp += 1
        elif py and not gy:
            fp += 1
        elif (not py) and gy:
            fn += 1
        else:
            tn += 1
    total = len(rows)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    valid = total - failed
    return {
        "accuracy": (tp + tn) / total if total else 0,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "yes_ratio": (tp + fp) / valid if valid else 0,
        "failed": failed,
    }


def maybe_run(cmd, cwd: Path):
    try:
        return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=300)
    except Exception as exc:
        return None


def summarize(base_dir: Path, root: Path):
    summary = {"base_dir": str(base_dir), "models": {}}
    pairwise_out = {}
    for model_dir in sorted(p for p in base_dir.iterdir() if p.is_dir() and p.name not in {"logs", "datasets"}):
        model_summary = {}
        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            dataset = dataset_dir.name
            ds_summary = {}
            for run_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                report_path = run_dir / "eval_report.json"
                results_path = run_dir / "results.jsonl"
                config_path = run_dir / "config.json"
                method = "lead" if run_dir.name.startswith("lead") else "cot"
                rows = load_jsonl(results_path)
                report = load_json(report_path) if report_path.exists() else {}
                config = load_json(config_path) if config_path.exists() else {}
                ds_summary[method] = {
                    "run_dir": str(run_dir),
                    "rows": len(rows),
                    "target_rows": DATASET_TARGETS.get(dataset),
                    "accuracy": report.get("accuracy"),
                    "correct": report.get("correct"),
                    "total": report.get("total"),
                    "failed_extraction": report.get("failed_extraction"),
                    "config": {
                        k: config.get(k)
                        for k in ["model_name", "method", "alpha", "max_switch_count", "window_size", "cot_prompt_mode", "do_sample", "temperature", "top_p", "top_k", "seed", "max_new_tokens"]
                    },
                }
                if dataset.startswith("pope_"):
                    ds_summary[method]["pope_yesno"] = pope_yesno(rows)

            if "cot" in ds_summary and "lead" in ds_summary:
                cot_rows = load_jsonl(Path(ds_summary["cot"]["run_dir"]) / "results.jsonl")
                lead_rows = load_jsonl(Path(ds_summary["lead"]["run_dir"]) / "results.jsonl")
                pw = pairwise(cot_rows, lead_rows)
                ds_summary["pairwise"] = {k: v for k, v in pw.items() if not k.endswith("_head")}
                pairwise_out.setdefault(model_dir.name, {})[dataset] = pw

            model_summary[dataset] = ds_summary
        summary["models"][model_dir.name] = model_summary

    (base_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (base_dir / "pairwise_deltas.json").write_text(json.dumps(pairwise_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(base_dir / "summary.md", summary)
    return summary


def fmt_pct(x):
    return "NA" if x is None else f"{100*x:.2f}%"


def write_markdown(path: Path, summary):
    lines = ["# Integrated Repo COT/LEAD Baselines", ""]
    for model, datasets in summary["models"].items():
        lines += [f"## {model}", "", "| Dataset | COT | LEAD | Delta | Rows | Failed COT/LEAD | Fixed | Damaged |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for dataset, ds in datasets.items():
            cot = ds.get("cot", {})
            lead = ds.get("lead", {})
            ca, la = cot.get("accuracy"), lead.get("accuracy")
            delta = None if ca is None or la is None else la - ca
            pw = ds.get("pairwise", {})
            rows = f"{cot.get('rows', 0)}/{lead.get('rows', 0)}"
            failed = f"{cot.get('failed_extraction')}/{lead.get('failed_extraction')}"
            lines.append(
                f"| {dataset} | {fmt_pct(ca)} | {fmt_pct(la)} | {fmt_pct(delta)} | {rows} | {failed} | {pw.get('fixed','')} | {pw.get('damaged','')} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--root", default="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD")
    args = ap.parse_args()
    summary = summarize(Path(args.base_dir), Path(args.root))
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()


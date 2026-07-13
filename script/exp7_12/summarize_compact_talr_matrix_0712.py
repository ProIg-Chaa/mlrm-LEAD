#!/usr/bin/env python3
"""Build a continuously refreshable summary for the compact TALR matrix."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

MODELS = {
    "r1_onevision_7b": ["vstar", "realworldqa_fixed200", "mmvp", "visulogic300", "vmcbench_dev", "pope_adversarial", "mmk12_physics"],
    "vision_r1_7b": ["vstar", "realworldqa_fixed200", "mmvp", "visulogic300", "vmcbench_dev", "pope_adversarial", "mmk12_physics"],
    "openvlthinker_7b": ["vstar", "mmvp"],
}
METHODS = ["cot_orign_greedy", "lead", "initial_transition_only", "transition_preserving_quota05_guard_min2"]
LABEL = {"cot_orign_greedy": "COT", "lead": "LEAD", "initial_transition_only": "Initial transition", "transition_preserving_quota05_guard_min2": "TALR"}
PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}"),
    re.compile(r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?"),
    re.compile(r"\*\*([A-Da-d])\*\*"),
    re.compile(r"(?:^|\n)\s*([A-Da-d])\s*$"),
]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_gold(value):
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    if text in {"YES", "NO"}:
        return text
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else None


def extract_mcq(text, options=""):
    text = str(text or "")
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[markers[-1].start():] if markers else text[-1500:]
    hits = []
    for pattern in PATTERNS:
        for match in pattern.finditer(region):
            hits.append((match.start(), match.group(1).upper()))
    if hits:
        return sorted(hits)[-1][1]
    last_letters = re.findall(r"\b([A-D])\b", region[-200:])
    return last_letters[-1].upper() if last_letters else None


def pope_label(value, options=""):
    value = str(value or "").strip().upper()
    if value in {"YES", "NO"}:
        return value
    pred = extract_mcq(value, options)
    choices = {letter.upper(): text.strip().lower() for letter, text in re.findall(r"\(([A-D])\)\s*([^\n]+)", str(options), re.I)}
    if pred and choices.get(pred) in {"yes", "no"}:
        return choices[pred].upper()
    hits = re.findall(r"\b(yes|no)\b", value.lower())
    return hits[-1].upper() if hits else None


def paired_groups(cot_rows, method_rows, dataset):
    def scored(rows):
        out = {}
        for row in rows:
            if row.get("error_type"):
                out[str(row.get("id"))] = None
                continue
            gold = pope_label(row.get("answer"), row.get("options")) if dataset.startswith("pope_") else normalize_gold(row.get("answer"))
            pred = pope_label(row.get("model_answer"), row.get("options")) if dataset.startswith("pope_") else extract_mcq(row.get("model_answer"), row.get("options"))
            out[str(row.get("id"))] = bool(gold and pred and gold == pred)
        return out
    a, b = scored(cot_rows), scored(method_rows)
    groups = defaultdict(list)
    for key in sorted(set(a) & set(b)):
        if a[key] is None or b[key] is None:
            groups["runtime_error"].append(key)
        elif not a[key] and b[key]:
            groups["fixed"].append(key)
        elif a[key] and not b[key]:
            groups["damaged"].append(key)
        elif a[key]:
            groups["both_correct"].append(key)
        else:
            groups["both_wrong"].append(key)
    return {k: {"count": len(v), "ids": v} for k, v in groups.items()}


def summarize_run(run_dir: Path, dataset: str, expected: int):
    rows = read_jsonl(run_dir / "results.jsonl")
    traces = read_jsonl(run_dir / "token_entropy.jsonl")
    complete = len(rows) == expected and all((run_dir / name).exists() for name in ("config.json", "eval_report.json", "token_entropy.jsonl"))
    correct = failed = runtime = 0
    preds = []
    for row in rows:
        if row.get("error_type"):
            runtime += 1
            preds.append(None)
            continue
        if dataset.startswith("pope_"):
            gold = pope_label(row.get("answer"), row.get("options")); pred = pope_label(row.get("model_answer"), row.get("options"))
        else:
            gold = normalize_gold(row.get("answer")); pred = extract_mcq(row.get("model_answer"), row.get("options"))
        preds.append(pred)
        failed += int(pred is None)
        correct += int(gold is not None and pred == gold)
    lengths = [int(r.get("output_tokens") or 0) for r in rows if not r.get("error_type")]
    summaries = [t.get("entropy_summary") or {} for t in traces]
    def avg(key):
        vals = [s.get(key) for s in summaries if s.get(key) is not None]
        return mean(vals) if vals else None
    result = {
        "status": "complete" if complete else ("partial" if rows else "pending"), "run_dir": str(run_dir),
        "rows": len(rows), "expected_rows": expected, "accuracy": correct / len(rows) if rows else None,
        "correct": correct, "failed_extraction": failed, "runtime_errors": runtime,
        "avg_output_tokens": mean(lengths) if lengths else None, "long_ge_256": sum(x >= 256 for x in lengths), "maxed_1024": sum(x >= 1024 for x in lengths),
        "mean_soft_ratio": avg("soft_ratio"), "mean_switch_count": avg("switch_count"),
        "mean_format_trigger_count": avg("format_trigger_count"), "mean_format_active_steps": avg("format_cooldown_active_steps"),
        "mean_diffuse_veto_count": avg("lead_soft_veto_count"),
    }
    if dataset == "mmvp" and rows:
        flags = []
        for row, pred in zip(rows, preds):
            flags.append(pred is not None and pred == normalize_gold(row.get("answer")))
        pairs = [flags[i] and flags[i + 1] for i in range(0, len(flags) - 1, 2)]
        result["mmvp_pair_accuracy"] = mean(pairs) if pairs else None
    if dataset.startswith("pope_") and rows:
        tp = tn = fp = fn = 0
        for row, pred in zip(rows, preds):
            gold = pope_label(row.get("answer"), row.get("options"))
            if gold == "YES" and pred == "YES": tp += 1
            elif gold == "NO" and pred == "NO": tn += 1
            elif gold == "NO" and pred == "YES": fp += 1
            elif gold == "YES" and pred == "NO": fn += 1
        precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
        result["pope"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None,
                          "yes_ratio": sum(p == "YES" for p in preds) / len(preds)}
    return result, rows


def fmt(value, pct=False):
    if value is None: return "-"
    return f"{100 * value:.2f}%" if pct else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    base = args.root / "uniform_multimodel_full_matrix"; out = args.root / "compact_talr_summary"; out.mkdir(parents=True, exist_ok=True)
    data_counts = {"vstar": 191, "realworldqa_fixed200": 200, "mmvp": 300, "visulogic300": 300, "vmcbench_dev": 1000,
                   "pope_adversarial": 3000, "mmk12_physics": 500}
    manifest = []; summaries = []; row_cache = {}; pairwise = {}
    for model, datasets in MODELS.items():
        for dataset in datasets:
            for method in METHODS:
                run_dir = base / model / dataset / method
                item, rows = summarize_run(run_dir, dataset, data_counts[dataset]); item.update(model=model, dataset=dataset, method=method)
                manifest.append({k: item[k] for k in ("model", "dataset", "method", "status", "run_dir", "rows", "expected_rows")})
                summaries.append(item); row_cache[(model, dataset, method)] = rows
            cot = row_cache[(model, dataset, "cot_orign_greedy")]
            for method in METHODS[1:]:
                pairwise[f"{model}/{dataset}/{method}"] = paired_groups(cot, row_cache[(model, dataset, method)], dataset)
    (out / "compact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "pairwise_deltas.json").write_text(json.dumps(pairwise, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# TALR 紧凑主矩阵", "", "主指标采用 corrected last-answer evaluator；partial run 不进入最终结论。", "", "| Model | Dataset | Method | Status | Accuracy | Failed | Runtime | Avg tokens | Soft ratio | Switch |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['model']} | {s['dataset']} | {LABEL[s['method']]} | {s['status']} ({s['rows']}/{s['expected_rows']}) | {fmt(s['accuracy'], True)} | {s['failed_extraction']} | {s['runtime_errors']} | {fmt(s['avg_output_tokens'])} | {fmt(s['mean_soft_ratio'], True)} | {fmt(s['mean_switch_count'])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    trigger = [{k: s[k] for k in ("model", "dataset", "method", "status", "avg_output_tokens", "long_ge_256", "maxed_1024", "mean_soft_ratio", "mean_switch_count", "mean_format_trigger_count", "mean_format_active_steps", "mean_diffuse_veto_count")} for s in summaries]
    (out / "trigger_and_length_stats.json").write_text(json.dumps(trigger, ensure_ascii=False, indent=2), encoding="utf-8")
    tlines = ["# Trigger 与长度统计", "", "| Model | Dataset | Method | Avg tokens | Long | Maxed | Soft | Switch | Format triggers | Veto |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in trigger:
        tlines.append(f"| {s['model']} | {s['dataset']} | {LABEL[s['method']]} | {fmt(s['avg_output_tokens'])} | {s['long_ge_256']} | {s['maxed_1024']} | {fmt(s['mean_soft_ratio'], True)} | {fmt(s['mean_switch_count'])} | {fmt(s['mean_format_trigger_count'])} | {fmt(s['mean_diffuse_veto_count'])} |")
    (out / "trigger_and_length_stats.md").write_text("\n".join(tlines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

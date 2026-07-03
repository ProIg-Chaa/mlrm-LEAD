#!/usr/bin/env python3
"""Summarize the minimal paper-main reproduction runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_text(text: str | None) -> str:
    text = (text or "").lower()
    text = text.replace("\\boxed", " ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9.\-]+", " ", text)
    return " ".join(text.split())


def tail_answer_region(text: str | None) -> str:
    text = text or ""
    matches = list(re.finditer(r"(?:answer|final answer|therefore|thus)\s*[:is]*", text, re.I))
    if matches:
        return text[matches[-1].start() :]
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text[-1500:]


def extract_boxed(text: str | None) -> str | None:
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def extract_letter(text: str | None, letters: str = "ABCDE") -> str | None:
    if not text:
        return None
    tail = tail_answer_region(text)
    escaped = re.escape(letters)
    patterns = [
        rf"\\boxed\{{\s*([{escaped}a-e])\s*\}}",
        rf"[Aa]nswer\s*[:\s]+\(?([{escaped}a-e])\)?",
        rf"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([{escaped}a-e])\)?",
        rf"\(([{escaped}a-e])\)",
    ]
    for pattern in patterns:
        m = re.search(pattern, tail)
        if m:
            return m.group(1).upper()
    found = re.findall(rf"\b([{letters}])\b", tail[-300:])
    return found[-1].upper() if found else None


def extract_number(text: str | None) -> float | None:
    if not text:
        return None
    boxed = extract_boxed(text)
    region = boxed if boxed is not None else tail_answer_region(text)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", region.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def parse_gold_choice(answer: str | None) -> str | None:
    answer = answer or ""
    m = re.match(r"\s*([A-E])(?:\.|\)|:|\s|$)", answer.strip(), re.I)
    return m.group(1).upper() if m else None


def parse_gold_number(answer: str | None) -> float | None:
    answer = answer or ""
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def numeric_equal(pred: float | None, gold: float | None) -> bool:
    if pred is None or gold is None:
        return False
    tol = max(1e-3, abs(gold) * 1e-3)
    return abs(pred - gold) <= tol


def evaluate_mathvista(dataset_rows: list[dict], result_rows: list[dict]) -> tuple[dict, list[dict]]:
    by_id = {int(r["id"]): r for r in dataset_rows}
    rows = []
    correct = 0
    failed = 0
    for result in result_rows:
        sample = by_id[int(result["id"])]
        pred_text = result.get("model_answer") or ""
        gold_raw = str(sample.get("answer") or "")
        gold_choice = parse_gold_choice(gold_raw)
        gold_num = parse_gold_number(gold_raw) if gold_choice is None else None
        pred_value = None
        method = "none"
        ok = False
        if gold_choice is not None:
            pred_choice = extract_letter(pred_text, "ABCDE")
            pred_value = pred_choice
            method = "choice_letter"
            ok = pred_choice == gold_choice
        elif gold_num is not None:
            pred_num = extract_number(pred_text)
            pred_value = pred_num
            method = "numeric"
            ok = numeric_equal(pred_num, gold_num)
        else:
            boxed = extract_boxed(pred_text)
            region = norm_text(boxed or tail_answer_region(pred_text))
            gold_norm = norm_text(gold_raw)
            pred_value = region
            method = "normalized_text"
            ok = bool(gold_norm) and (gold_norm == region or gold_norm in region)
        if pred_value is None or pred_value == "":
            failed += 1
        correct += int(ok)
        enriched = dict(result)
        enriched.update(
            {
                "rule_gold": gold_raw,
                "rule_pred": pred_value,
                "rule_method": method,
                "rule_is_correct": ok,
            }
        )
        rows.append(enriched)
    total = len(result_rows)
    return (
        {
            "mode": "mathvista200_rule_based",
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "failed_extraction": failed,
            "note": "Lightweight rule-based proxy, not official MathVista evaluation.",
        },
        rows,
    )


def mmhal_similarity(prediction: str | None, gold: str | None) -> tuple[float, str | None]:
    pred_region = norm_text(tail_answer_region(prediction))
    gold_norm = norm_text(gold)
    if not pred_region:
        return 0.0, None
    if not gold_norm:
        return 0.0, pred_region
    if gold_norm in pred_region:
        return 1.0, pred_region
    pred_tokens = set(pred_region.split())
    gold_tokens = set(gold_norm.split())
    overlap = len(pred_tokens & gold_tokens) / max(1, len(gold_tokens))
    seq = SequenceMatcher(None, pred_region, gold_norm).ratio()
    return max(overlap, seq), pred_region


def evaluate_mmhal(dataset_rows: list[dict], result_rows: list[dict]) -> tuple[dict, list[dict]]:
    by_id = {int(r["id"]): r for r in dataset_rows}
    rows = []
    correct = 0
    failed = 0
    scores = []
    topic_stats = defaultdict(lambda: {"correct": 0, "total": 0, "score_sum": 0.0})
    for result in result_rows:
        sample = by_id[int(result["id"])]
        score, pred_region = mmhal_similarity(result.get("model_answer"), sample.get("gt_answer") or sample.get("answer"))
        ok = score >= 0.62
        if pred_region is None:
            failed += 1
        correct += int(ok)
        scores.append(score)
        subtopic = sample.get("subtopic") or sample.get("question_type") or "unknown"
        topic_stats[subtopic]["total"] += 1
        topic_stats[subtopic]["correct"] += int(ok)
        topic_stats[subtopic]["score_sum"] += score
        enriched = dict(result)
        enriched.update(
            {
                "mmhal_proxy_gold": sample.get("gt_answer") or sample.get("answer"),
                "mmhal_proxy_pred_region": pred_region,
                "mmhal_proxy_score": score,
                "mmhal_proxy_is_correct": ok,
            }
        )
        rows.append(enriched)
    total = len(result_rows)
    by_subtopic = {
        k: {
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
            "mean_proxy_score": v["score_sum"] / v["total"] if v["total"] else 0.0,
        }
        for k, v in sorted(topic_stats.items())
    }
    return (
        {
            "mode": "mmhal_rule_based_proxy",
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "failed_extraction": failed,
            "mean_proxy_score": mean(scores) if scores else 0.0,
            "by_subtopic": by_subtopic,
            "note": "Lightweight lexical proxy, not official MMHal GPT-based score.",
        },
        rows,
    )


def dataset_key_from_config(config: dict, run_dir: Path) -> str:
    dataset = str(config.get("dataset") or "").lower()
    if "realworldqa_fixed" in dataset:
        return "realworldqa_fixed200"
    if "mmvp" in dataset:
        return "mmvp"
    if "vstar" in dataset:
        return "vstar"
    if "mmhal" in dataset:
        return "mmhal"
    if "math_vista" in dataset:
        return "mathvista200"
    return run_dir.parent.name


def run_eval_if_needed(root: Path, run_dir: Path, dataset_key: str, dataset_path: str) -> tuple[dict, list[dict], str]:
    result_rows = load_jsonl(run_dir / "results.jsonl")
    if dataset_key == "mmvp":
        out = run_dir / "specialized_eval_report.json"
        rows_path = run_dir / "specialized_eval_rows.jsonl"
        if not out.exists():
            subprocess.run(
                [
                    sys.executable,
                    "script/evaluate_specialized_results.py",
                    "--dataset",
                    dataset_path,
                    "--results",
                    str(run_dir / "results.jsonl"),
                    "--output_json",
                    str(out),
                    "--output_results_jsonl",
                    str(rows_path),
                ],
                cwd=root,
                check=True,
            )
        return load_json(out), load_jsonl(rows_path), "specialized_mmvp"
    if dataset_key == "realworldqa_fixed200":
        out = run_dir / "realworldqa_mcq_eval.json"
        rows_path = run_dir / "realworldqa_mcq_eval_rows.jsonl"
        if not out.exists():
            subprocess.run(
                [
                    sys.executable,
                    "script/evaluate_realworldqa_mcq.py",
                    "--dataset",
                    dataset_path,
                    "--results",
                    str(run_dir / "results.jsonl"),
                    "--output_json",
                    str(out),
                    "--output_results_jsonl",
                    str(rows_path),
                ],
                cwd=root,
                check=True,
            )
        return load_json(out), load_jsonl(rows_path), "realworldqa_mcq"
    if dataset_key == "mathvista200":
        dataset_rows = load_jsonl(Path(dataset_path))
        report, rows = evaluate_mathvista(dataset_rows, result_rows)
        write_json(run_dir / "mathvista_rule_eval.json", report)
        write_jsonl(run_dir / "mathvista_rule_eval_rows.jsonl", rows)
        return report, rows, "mathvista_rule_based"
    if dataset_key == "mmhal":
        dataset_rows = load_jsonl(Path(dataset_path))
        report, rows = evaluate_mmhal(dataset_rows, result_rows)
        write_json(run_dir / "mmhal_rule_eval.json", report)
        write_jsonl(run_dir / "mmhal_rule_eval_rows.jsonl", rows)
        return report, rows, "mmhal_rule_based_proxy"
    return load_json(run_dir / "eval_report.json"), result_rows, "default"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def stats(values: list[float]) -> dict:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": None, "p90": None, "max": None}
    return {"mean": mean(values), "p90": quantile(values, 0.9), "max": max(values)}


def trace_metrics(run_dir: Path) -> tuple[float | None, float | None]:
    summary_ratios = []
    for row in load_jsonl(run_dir / "token_entropy.jsonl"):
        value = ((row.get("entropy_summary") or {}).get("soft_ratio"))
        if isinstance(value, (int, float)):
            summary_ratios.append(float(value))
    latent_ratio = mean(summary_ratios) if summary_ratios else None

    switch_counts = []
    full_rows = load_jsonl(run_dir / "token_entropy_full.jsonl")
    full_ratios = []
    for row in full_rows:
        tokens = row.get("tokens") or []
        if not tokens:
            continue
        switch_counts.append(
            sum(1 for t in tokens if bool(t.get("to_normal")) or bool(t.get("to_soft")))
        )
        full_ratios.append(sum(1 for t in tokens if t.get("mode") == "soft") / len(tokens))
    if latent_ratio is None and full_ratios:
        latent_ratio = mean(full_ratios)
    switch_avg = mean(switch_counts) if switch_counts else None
    return switch_avg, latent_ratio


def correctness_map(eval_type: str, eval_rows: list[dict], result_rows: list[dict]) -> tuple[dict[int, bool], int]:
    if eval_type == "specialized_mmvp":
        return (
            {int(r["id"]): bool(r.get("specialized_is_correct")) for r in eval_rows},
            sum(1 for r in eval_rows if r.get("specialized_pred") is None),
        )
    if eval_type == "realworldqa_mcq":
        return (
            {int(r["id"]): bool(r.get("realworldqa_is_correct")) for r in eval_rows},
            sum(1 for r in eval_rows if r.get("realworldqa_pred") is None),
        )
    if eval_type == "mathvista_rule_based":
        return (
            {int(r["id"]): bool(r.get("rule_is_correct")) for r in eval_rows},
            sum(1 for r in eval_rows if r.get("rule_pred") in (None, "")),
        )
    if eval_type == "mmhal_rule_based_proxy":
        return (
            {int(r["id"]): bool(r.get("mmhal_proxy_is_correct")) for r in eval_rows},
            sum(1 for r in eval_rows if r.get("mmhal_proxy_pred_region") in (None, "")),
        )
    out = {}
    failed = 0
    for row in result_rows:
        pred = extract_letter(row.get("model_answer"), "ABCDE")
        failed += int(pred is None)
        gold = parse_gold_choice(str(row.get("answer") or "")) or str(row.get("answer") or "").strip().upper()[:1]
        out[int(row["id"])] = bool(pred and pred == gold)
    return out, failed


def summarize_run(root: Path, run_dir: Path) -> dict | None:
    if not (run_dir / "config.json").exists() or not (run_dir / "results.jsonl").exists():
        return None
    config = load_json(run_dir / "config.json")
    result_rows = load_jsonl(run_dir / "results.jsonl")
    dataset_key = dataset_key_from_config(config, run_dir)
    report, eval_rows, eval_type = run_eval_if_needed(root, run_dir, dataset_key, str(config.get("dataset")))
    correctness, extracted_failed = correctness_map(eval_type, eval_rows, result_rows)
    runtime_errors = sum(1 for r in result_rows if r.get("error_type"))
    total = int(report.get("total") or len(result_rows))
    correct = int(report.get("correct") or 0)
    failed = int(report.get("failed_extraction", extracted_failed) or 0)
    wrong = max(0, total - correct - failed - runtime_errors)
    switch_avg, latent_ratio = trace_metrics(run_dir)
    lengths = [int(r.get("output_tokens") or 0) for r in result_rows]
    latencies = [float(r.get("latency_sec")) for r in result_rows if isinstance(r.get("latency_sec"), (int, float))]
    cuda_alloc = [
        float(r.get("cuda_peak_allocated_mb"))
        for r in result_rows
        if isinstance(r.get("cuda_peak_allocated_mb"), (int, float))
    ]
    cuda_reserved = [
        float(r.get("cuda_peak_reserved_mb"))
        for r in result_rows
        if isinstance(r.get("cuda_peak_reserved_mb"), (int, float))
    ]
    return {
        "model_key": run_dir.parent.parent.name,
        "dataset": dataset_key,
        "run": re.sub(r"_gpu\d+$", "", run_dir.name),
        "run_dir": str(run_dir),
        "eval_type": eval_type,
        "accuracy": report.get("accuracy"),
        "score": report.get("mean_proxy_score", report.get("accuracy")),
        "correct": correct,
        "total": total,
        "pair_accuracy": report.get("pair_accuracy"),
        "pair_correct": report.get("pair_correct"),
        "pair_total": report.get("pair_total"),
        "failed_extraction": failed,
        "no_answer_rate": failed / total if total else 0.0,
        "wrong_answer_rate": wrong / total if total else 0.0,
        "runtime_error_count": runtime_errors,
        "avg_output_tokens": mean(lengths) if lengths else 0.0,
        "long_ge_256": sum(x >= 256 for x in lengths),
        "maxed_1024": sum(x >= 1024 for x in lengths),
        "avg_latency_sec": mean(latencies) if latencies else None,
        "lead_switch_count_avg": switch_avg,
        "latent_step_ratio": latent_ratio,
        "cuda_peak_allocated_mb": stats(cuda_alloc),
        "cuda_peak_reserved_mb": stats(cuda_reserved),
        "by_subtopic": report.get("by_subtopic"),
        "note": report.get("note"),
        "_correctness": correctness,
    }


def build_pairwise(entries: list[dict]) -> dict:
    grouped = defaultdict(dict)
    for entry in entries:
        grouped[(entry["model_key"], entry["dataset"])][entry["run"]] = entry
    out = {}
    for (model_key, dataset), runs in grouped.items():
        if "cot_orign_greedy" not in runs or "lead" not in runs:
            continue
        ref = runs["cot_orign_greedy"]["_correctness"]
        cur = runs["lead"]["_correctness"]
        ids = sorted(set(ref) | set(cur))
        fixed = [i for i in ids if (not ref.get(i, False)) and cur.get(i, False)]
        damaged = [i for i in ids if ref.get(i, False) and (not cur.get(i, False))]
        out[f"{model_key}/{dataset}/cot_orign_greedy__vs__lead"] = {
            "fixed_ids": fixed,
            "damaged_ids": damaged,
            "fixed": len(fixed),
            "damaged": len(damaged),
            "net": len(fixed) - len(damaged),
        }
    return out


def pct(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.2f}%"


def fmt(value, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def write_summary_md(path: Path, entries: list[dict], pairwise: dict) -> None:
    clean_entries = [{k: v for k, v in e.items() if k != "_correctness"} for e in entries]
    grouped = defaultdict(list)
    for entry in clean_entries:
        grouped[(entry["model_key"], entry["dataset"])].append(entry)
    lines = [
        "# Paper Main Minimal Reproduction Summary",
        "",
        "主表只比较 `cot_orign_greedy` 与 paper-style `lead`。MMHal 与 MathVista200 使用轻量 rule-based proxy，不能当作官方分数。",
        "",
    ]
    for (model_key, dataset), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: 0 if r["run"] == "cot_orign_greedy" else 1)
        cot = next((r for r in rows if r["run"] == "cot_orign_greedy"), None)
        lines.append(f"## {model_key} / {dataset}")
        lines.append("")
        lines.append(
            "| run | eval | acc/score | delta vs COT | pair acc | avg tokens | avg latency | no-answer | wrong | switch avg | latent ratio | CUDA alloc mean/p90/max MB | CUDA reserved mean/p90/max MB | long>=256 | maxed1024 | errors |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            delta = None
            if cot is not None and row.get("accuracy") is not None and cot.get("accuracy") is not None:
                delta = float(row["accuracy"]) - float(cot["accuracy"])
            alloc = row["cuda_peak_allocated_mb"]
            reserved = row["cuda_peak_reserved_mb"]
            lines.append(
                "| {run} | {eval_type} | {acc} | {delta} | {pair} | {tokens} | {lat} | {noans} | {wrong} | {switch} | {latent} | {alloc_s} | {res_s} | {long} | {maxed} | {err} |".format(
                    run=row["run"],
                    eval_type=row["eval_type"],
                    acc=pct(row.get("accuracy")),
                    delta=("NA" if delta is None else f"{delta * 100:+.2f}"),
                    pair=pct(row.get("pair_accuracy")),
                    tokens=fmt(row.get("avg_output_tokens"), 1),
                    lat=fmt(row.get("avg_latency_sec"), 2),
                    noans=pct(row.get("no_answer_rate")),
                    wrong=pct(row.get("wrong_answer_rate")),
                    switch=fmt(row.get("lead_switch_count_avg"), 2),
                    latent=pct(row.get("latent_step_ratio")),
                    alloc_s=f"{fmt(alloc.get('mean'),1)}/{fmt(alloc.get('p90'),1)}/{fmt(alloc.get('max'),1)}",
                    res_s=f"{fmt(reserved.get('mean'),1)}/{fmt(reserved.get('p90'),1)}/{fmt(reserved.get('max'),1)}",
                    long=row.get("long_ge_256"),
                    maxed=row.get("maxed_1024"),
                    err=row.get("runtime_error_count"),
                )
            )
        key = f"{model_key}/{dataset}/cot_orign_greedy__vs__lead"
        if key in pairwise:
            p = pairwise[key]
            lines.append("")
            lines.append(f"Pairwise COT -> LEAD: fixed={p['fixed']}, damaged={p['damaged']}, net={p['net']}")
        notes = [r.get("note") for r in rows if r.get("note")]
        for note in sorted(set(notes)):
            lines.append(f"Note: {note}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    base = Path(args.base_dir)
    root = Path(args.root)
    entries = []
    for config in sorted(base.glob("*/*/*/config.json")):
        entry = summarize_run(root, config.parent)
        if entry:
            entries.append(entry)
    pairwise = build_pairwise(entries)
    clean_entries = [{k: v for k, v in e.items() if k != "_correctness"} for e in entries]
    write_json(base / "summary.json", {"runs": clean_entries})
    write_json(base / "pairwise_deltas.json", pairwise)
    write_summary_md(base / "summary.md", entries, pairwise)
    print(f"Wrote {base / 'summary.md'}")


if __name__ == "__main__":
    main()

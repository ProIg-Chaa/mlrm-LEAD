#!/usr/bin/env python3
"""Summarize the early path-dependence rerun matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_float(value):
    return float(value) if isinstance(value, (int, float)) else None


def run_eval_if_needed(root: Path, run_dir: Path, dataset_key: str, dataset_path: str) -> None:
    results = run_dir / "results.jsonl"
    if not results.exists():
        return
    if dataset_key == "mmvp":
        out = run_dir / "specialized_eval_report.json"
        rows = run_dir / "specialized_eval_rows.jsonl"
        if not out.exists():
            subprocess.run(
                [
                    sys.executable,
                    "script/evaluate_specialized_results.py",
                    "--dataset",
                    dataset_path,
                    "--results",
                    str(results),
                    "--output_json",
                    str(out),
                    "--output_results_jsonl",
                    str(rows),
                ],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    elif dataset_key == "realworldqa_fixed200":
        out = run_dir / "realworldqa_mcq_eval.json"
        rows = run_dir / "realworldqa_mcq_eval_rows.jsonl"
        if not out.exists():
            subprocess.run(
                [
                    sys.executable,
                    "script/evaluate_realworldqa_mcq.py",
                    "--dataset",
                    dataset_path,
                    "--results",
                    str(results),
                    "--output_json",
                    str(out),
                    "--output_results_jsonl",
                    str(rows),
                ],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )


def dataset_key_from_config(config: dict) -> str:
    dataset = str(config.get("dataset") or "").lower()
    if "mmvp" in dataset:
        return "mmvp"
    if "visulogic" in dataset:
        return "visulogic300" if config.get("limit") == 300 else "visulogic"
    if "realworldqa_fixed_mcq_random200_seed42" in dataset:
        return "realworldqa_fixed200"
    if "vstar" in dataset:
        return "vstar"
    return "unknown"


def run_name_from_dir(run_dir: Path) -> str:
    return re.sub(r"_gpu\d+$", "", run_dir.name)


def extract_letter(text: str | None) -> str | None:
    if not text:
        return None
    tail = text[-1600:]
    patterns = [
        r"\\boxed\{\s*([A-Da-d])\s*\}",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"(?:^|\n)\s*([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, tail)
        if match:
            return match.group(1).upper()
    region = tail.split("</think>")[-1]
    letters = re.findall(r"\b([A-D])\b", region[-300:])
    return letters[-1].upper() if letters else None


def local_correctness(run_dir: Path, dataset_key: str) -> tuple[dict[int, bool], int]:
    if dataset_key == "mmvp":
        rows = load_jsonl(run_dir / "specialized_eval_rows.jsonl")
        if rows:
            return {int(r["id"]): bool(r.get("specialized_is_correct")) for r in rows}, sum(
                r.get("specialized_pred") is None for r in rows
            )
    if dataset_key == "realworldqa_fixed200":
        rows = load_jsonl(run_dir / "realworldqa_mcq_eval_rows.jsonl")
        if rows:
            return {int(r["id"]): bool(r.get("realworldqa_is_correct")) for r in rows}, sum(
                r.get("realworldqa_pred") is None for r in rows
            )
    rows = load_jsonl(run_dir / "results.jsonl")
    correct = {}
    failed = 0
    for row in rows:
        pred = extract_letter(row.get("model_answer"))
        failed += int(pred is None)
        gold = str(row.get("answer") or "").strip().upper()[:1]
        correct[int(row["id"])] = bool(pred and pred == gold)
    return correct, failed


def soft_ratio_from_trace(run_dir: Path) -> float | None:
    ratios = []
    for row in load_jsonl(run_dir / "token_entropy.jsonl"):
        summary = row.get("entropy_summary") or {}
        value = safe_float(summary.get("soft_ratio"))
        if value is not None:
            ratios.append(value)
    if ratios:
        return sum(ratios) / len(ratios)

    per_sample = []
    for row in load_jsonl(run_dir / "token_entropy_full.jsonl"):
        tokens = row.get("tokens") or []
        total = len(tokens)
        if not total:
            continue
        soft = sum(1 for t in tokens if t.get("mode") == "soft")
        per_sample.append(soft / total)
    return sum(per_sample) / len(per_sample) if per_sample else None


def choose_report(run_dir: Path, dataset_key: str) -> tuple[dict, str]:
    if dataset_key == "mmvp" and (run_dir / "specialized_eval_report.json").exists():
        return load_json(run_dir / "specialized_eval_report.json"), "specialized_mmvp"
    if dataset_key == "realworldqa_fixed200" and (run_dir / "realworldqa_mcq_eval.json").exists():
        return load_json(run_dir / "realworldqa_mcq_eval.json"), "realworldqa_mcq"
    return load_json(run_dir / "eval_report.json"), "default"


def summarize_run(root: Path, run_dir: Path) -> dict | None:
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    eval_path = run_dir / "eval_report.json"
    if not (config_path.exists() and results_path.exists() and eval_path.exists()):
        return None
    config = load_json(config_path)
    dataset_key = dataset_key_from_config(config)
    dataset_path = str(config.get("dataset") or "")
    run_eval_if_needed(root, run_dir, dataset_key, dataset_path)
    report, report_type = choose_report(run_dir, dataset_key)
    rows = load_jsonl(results_path)
    lengths = [int(r.get("output_tokens") or 0) for r in rows]
    correctness, local_failed = local_correctness(run_dir, dataset_key)
    errors = {}
    for row in rows:
        err = row.get("error_type")
        if err:
            errors[err] = errors.get(err, 0) + 1
    return {
        "phase": run_dir.parent.parent.name,
        "dataset": dataset_key,
        "run": run_name_from_dir(run_dir),
        "run_dir": str(run_dir),
        "report_type": report_type,
        "accuracy": report.get("accuracy"),
        "correct": report.get("correct"),
        "total": report.get("total"),
        "failed_extraction": report.get("failed_extraction", local_failed),
        "pair_accuracy": report.get("pair_accuracy"),
        "pair_correct": report.get("pair_correct"),
        "pair_total": report.get("pair_total"),
        "output_length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "long_ge_256": sum(x >= 256 for x in lengths),
        "maxed_1024": sum(x >= 1024 for x in lengths),
        "mean_soft_ratio": soft_ratio_from_trace(run_dir),
        "error_type_counts": errors,
        "by_subtopic": report.get("by_subtopic"),
        "config_flags": {
            key: config.get(key)
            for key in [
                "method",
                "cot_prompt_mode",
                "lead_force_normal",
                "lead_initial_soft_only",
                "lead_initial_transition_only",
                "lead_initial_transition_delay_steps",
                "lead_disable_to_normal_transition",
                "lead_disable_step0_linebreak_mix",
                "lead_disable_simple_visual_anchor",
                "lead_soft_quota_ratio",
                "lead_format_cooldown",
                "lead_soft_veto_on_diffuse",
            ]
        },
        "_correctness": correctness,
    }


def pairwise(entries: list[dict]) -> dict:
    grouped = {}
    for entry in entries:
        grouped.setdefault((entry["phase"], entry["dataset"]), {})[entry["run"]] = entry
    out = {}
    for key, runs in grouped.items():
        phase, dataset = key
        out_key = f"{phase}/{dataset}"
        out[out_key] = {}
        refs = [r for r in ["cot_orign_greedy", "lead"] if r in runs]
        for ref in refs:
            ref_map = runs[ref]["_correctness"]
            ref_correct = {i for i, ok in ref_map.items() if ok}
            for run, entry in sorted(runs.items()):
                if run == ref:
                    continue
                cur_map = entry["_correctness"]
                cur_correct = {i for i, ok in cur_map.items() if ok}
                all_ids = sorted(set(ref_map) | set(cur_map))
                fixed = sorted(cur_correct - ref_correct)
                damaged = sorted(ref_correct - cur_correct)
                unchanged_correct = [i for i in all_ids if ref_map.get(i) and cur_map.get(i)]
                unchanged_wrong = [i for i in all_ids if not ref_map.get(i) and not cur_map.get(i)]
                out[out_key][f"{ref}__vs__{run}"] = {
                    "reference": ref,
                    "run": run,
                    "fixed_ids": fixed,
                    "damaged_ids": damaged,
                    "unchanged_correct_ids": unchanged_correct,
                    "unchanged_wrong_ids": unchanged_wrong,
                    "fixed": len(fixed),
                    "damaged": len(damaged),
                    "net": len(fixed) - len(damaged),
                    "delta_source": "specialized_or_local_extraction_for_pairwise_only",
                }
    return out


def strip_private(entries: list[dict]) -> list[dict]:
    clean = []
    for entry in entries:
        item = dict(entry)
        item.pop("_correctness", None)
        clean.append(item)
    return clean


def write_markdown(base_dir: Path, entries: list[dict], deltas: dict) -> None:
    lines = [
        "# Early Trajectory Commitment Rerun Summary",
        "",
        "主指标优先使用各数据集 official evaluator；MMVP 使用 specialized evaluator 并报告 pair accuracy；RealWorldQA 只使用 fixed200 MCQ evaluator。Pairwise fixed/damaged 用 enriched evaluator rows 或本地抽取，仅作为配对分析，不作为主 accuracy。",
        "",
    ]
    by_group = {}
    for entry in entries:
        by_group.setdefault((entry["phase"], entry["dataset"]), []).append(entry)
    for (phase, dataset), group in sorted(by_group.items()):
        lines.append(f"## {phase} / {dataset}")
        lines.append("")
        lines.append("| run | acc | pair acc | fixed/damaged vs COT | fixed/damaged vs LEAD | len mean | long>=256 | maxed1024 | failed | soft ratio | errors |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        delta_group = deltas.get(f"{phase}/{dataset}", {})
        for entry in sorted(group, key=lambda x: x["run"]):
            acc = entry["accuracy"]
            pair_acc = entry["pair_accuracy"]
            vs_cot = delta_group.get(f"cot_orign_greedy__vs__{entry['run']}", {})
            vs_lead = delta_group.get(f"lead__vs__{entry['run']}", {})
            soft = entry["mean_soft_ratio"]
            errors = ",".join(f"{k}:{v}" for k, v in sorted(entry["error_type_counts"].items())) or "0"
            lines.append(
                "| {run} | {acc} | {pair} | {cot} | {lead} | {length:.1f} | {long} | {maxed} | {failed} | {soft} | {errors} |".format(
                    run=entry["run"],
                    acc="NA" if acc is None else f"{acc*100:.2f}%",
                    pair="NA" if pair_acc is None else f"{pair_acc*100:.2f}%",
                    cot="NA" if not vs_cot else f"{vs_cot['fixed']}/{vs_cot['damaged']}",
                    lead="NA" if not vs_lead else f"{vs_lead['fixed']}/{vs_lead['damaged']}",
                    length=entry["output_length_mean"],
                    long=entry["long_ge_256"],
                    maxed=entry["maxed_1024"],
                    failed=entry["failed_extraction"],
                    soft="NA" if soft is None else f"{soft*100:.2f}%",
                    errors=errors,
                )
            )
        lines.append("")
        if any(e.get("by_subtopic") for e in group):
            lines.append("### by_subtopic")
            for entry in sorted(group, key=lambda x: x["run"]):
                by_subtopic = entry.get("by_subtopic") or {}
                if not by_subtopic:
                    continue
                compact = ", ".join(
                    f"{k}:{v.get('correct')}/{v.get('total')}"
                    for k, v in sorted(by_subtopic.items())
                )
                lines.append(f"- {entry['run']}: {compact}")
            lines.append("")
    (base_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    root = Path(args.root)
    entries = []
    for config in sorted(base_dir.glob("*/*/*/config.json")):
        entry = summarize_run(root, config.parent)
        if entry:
            entries.append(entry)
    deltas = pairwise(entries)
    clean = strip_private(entries)
    (base_dir / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (base_dir / "pairwise_deltas.json").write_text(
        json.dumps(deltas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(base_dir, clean, deltas)
    print(f"Wrote {base_dir / 'summary.json'}")
    print(f"Wrote {base_dir / 'summary.md'}")
    print(f"Wrote {base_dir / 'pairwise_deltas.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


DATASETS = {"vstar": 191, "mmvp": 300, "realworldqa_fixed200": 200, "visulogic300": 300}
METHODS = ["cot_orign_greedy", "lead", "initial_transition_only", "talr"]
# Legacy quota+guard runs did not preserve the early transition. They remain
# auditable historical evidence but must not be selected as the TALR main row.
ALIASES = {"talr": ["talr_early_quota05_guard_min2"]}
LABELS = {"cot_orign_greedy": "COT", "lead": "Full LEAD", "initial_transition_only": "Initial Transition", "talr": "TALR"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract(text: str | None) -> str | None:
    if not text:
        return None
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[markers[-1].start():] if markers else text[-1800:]
    patterns = [
        r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}",
        r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*$",
    ]
    hits = []
    for pattern in patterns:
        hits.extend((match.start(), match.group(1).upper()) for match in re.finditer(pattern, region, re.I | re.M))
    if hits:
        return sorted(hits)[-1][1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-200:])
    return letters[-1].upper() if letters else None


def locate(root: Path, dataset: str, method: str) -> Path:
    names = ALIASES.get(method, [method])
    for name in names:
        candidate = root / dataset / name
        if (candidate / "results.jsonl").is_file():
            return candidate
    return root / dataset / names[0]


def audit_config(run: Path, model_label: str, method: str) -> list[str]:
    path = run / "config.json"
    if not path.is_file():
        return ["missing config.json"]
    config = read_json(path)
    issues = []
    if Path(str(config.get("model_name", ""))).name != model_label:
        issues.append(f"model={config.get('model_name')}")
    if config.get("cot_prompt_mode") != "orign":
        issues.append(f"cot_prompt_mode={config.get('cot_prompt_mode')}")
    if bool(config.get("do_sample", False)):
        issues.append("do_sample=true")
    if int(config.get("seed", 42)) != 42 or int(config.get("max_new_tokens", 1024)) != 1024:
        issues.append("seed/max_new_tokens mismatch")
    if method == "cot_orign_greedy" and config.get("method") != "cot_greedy":
        issues.append(f"method={config.get('method')}")
    if method == "initial_transition_only" and not config.get("lead_initial_transition_only"):
        issues.append("initial transition flag missing")
    if method == "talr":
        expected = {
            "lead_initial_transition_with_refinement": True,
            "lead_soft_quota_ratio": 0.05,
            "lead_format_cooldown": True,
            "format_cooldown_steps": 2,
            "format_cooldown_min_step": 2,
            "lead_soft_veto_on_diffuse": True,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                issues.append(f"{key}={config.get(key)!r}")
    return issues


def evaluated_items(run: Path, dataset: str, results: list[dict]) -> tuple[list[dict], dict, str]:
    if dataset == "mmvp" and (run / "specialized_eval_rows.jsonl").is_file():
        rows = read_jsonl(run / "specialized_eval_rows.jsonl")
        items = [{"id": str(row["id"]), "pred": row.get("specialized_pred"), "gold": row.get("specialized_gold"), "correct": bool(row.get("specialized_is_correct"))} for row in rows]
        return items, read_json(run / "specialized_eval_report.json"), "MMVP specialized"
    if dataset == "realworldqa_fixed200" and (run / "realworldqa_mcq_rows.jsonl").is_file():
        rows = read_jsonl(run / "realworldqa_mcq_rows.jsonl")
        items = [{"id": str(row["id"]), "pred": row.get("realworldqa_pred"), "gold": row.get("realworldqa_gold"), "correct": bool(row.get("realworldqa_is_correct"))} for row in rows]
        return items, read_json(run / "realworldqa_mcq_eval.json"), "RealWorldQA specialized"
    items = []
    for row in results:
        pred = extract(row.get("model_answer"))
        gold = str(row.get("answer", "")).strip().upper()[:1]
        items.append({"id": str(row["id"]), "pred": pred, "gold": gold, "correct": pred == gold})
    report = {
        "accuracy": sum(item["correct"] for item in items) / len(items),
        "correct": sum(item["correct"] for item in items),
        "total": len(items),
        "failed_extraction": sum(item["pred"] is None for item in items),
    }
    return items, report, "corrected last-answer"


def trace_stats(run: Path) -> dict:
    path = run / "token_entropy.jsonl"
    if not path.is_file():
        return {}
    summaries = [row.get("entropy_summary") or {} for row in read_jsonl(path)]
    keys = ["soft_ratio", "switch_count", "format_trigger_count", "format_cooldown_active_steps", "lead_soft_veto_count"]
    output = {}
    for key in keys:
        values = [float(summary[key]) for summary in summaries if summary.get(key) is not None]
        output[f"mean_{key}"] = statistics.fmean(values) if values else None
    return output


def pairwise(base: list[dict], method: list[dict]) -> dict:
    left = {row["id"]: row for row in base}
    right = {row["id"]: row for row in method}
    ids = sorted(set(left) & set(right))
    fixed = [sample_id for sample_id in ids if not left[sample_id]["correct"] and right[sample_id]["correct"]]
    damaged = [sample_id for sample_id in ids if left[sample_id]["correct"] and not right[sample_id]["correct"]]
    return {"total": len(ids), "fixed": len(fixed), "damaged": len(damaged), "net": len(fixed) - len(damaged), "fixed_ids": fixed, "damaged_ids": damaged}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-root", required=True)
    parser.add_argument("--vision-root", required=True)
    parser.add_argument("--rl-talr-root")
    parser.add_argument("--vision-talr-root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [
        ("R1-Onevision-7B-RL", Path(args.rl_root), Path(args.rl_talr_root) if args.rl_talr_root else None),
        ("Vision-R1-7B", Path(args.vision_root), Path(args.vision_talr_root) if args.vision_talr_root else None),
    ]
    runs, item_map, audit = [], {}, []
    for model, root, talr_root in models:
        for dataset, expected in DATASETS.items():
            for method in METHODS:
                run = locate(talr_root if method == "talr" and talr_root else root, dataset, method)
                result_path = run / "results.jsonl"
                if not result_path.is_file():
                    audit.append({"model": model, "dataset": dataset, "method": method, "status": "missing"})
                    continue
                results = read_jsonl(result_path)
                issues = audit_config(run, model, method)
                if len(results) != expected:
                    issues.append(f"rows={len(results)}/{expected}")
                items, report, evaluator = evaluated_items(run, dataset, results)
                lengths = [int(row.get("output_tokens") or 0) for row in results]
                record = {
                    "model": model, "dataset": dataset, "method": method, "run_dir": str(run),
                    "accuracy": float(report["accuracy"]), "correct": int(report["correct"]),
                    "total": int(report["total"]), "failed_extraction": int(report.get("failed_extraction", 0)),
                    "pair_accuracy": report.get("pair_accuracy"), "evaluator": evaluator,
                    "runtime_errors": sum(bool(row.get("error_type")) for row in results),
                    "mean_output_tokens": statistics.fmean(lengths), "long_ge_256": sum(value >= 256 for value in lengths),
                    "maxed_1024": sum(value >= 1024 for value in lengths), **trace_stats(run),
                }
                runs.append(record)
                item_map[(model, dataset, method)] = items
                audit.append({"model": model, "dataset": dataset, "method": method, "status": "valid" if not issues else "invalid", "issues": issues, "run_dir": str(run)})

    pairs = []
    for model, _, _ in models:
        for dataset in DATASETS:
            for baseline in ["cot_orign_greedy", "lead"]:
                base = item_map.get((model, dataset, baseline))
                if not base:
                    continue
                for method in METHODS:
                    candidate = item_map.get((model, dataset, method))
                    if candidate and method != baseline:
                        pairs.append({"model": model, "dataset": dataset, "baseline": baseline, "method": method, **pairwise(base, candidate)})

    decisions = {}
    for model, _, _ in models:
        deltas = []
        noninferior = 0
        for dataset in DATASETS:
            cot = next((row for row in runs if row["model"] == model and row["dataset"] == dataset and row["method"] == "cot_orign_greedy"), None)
            talr = next((row for row in runs if row["model"] == model and row["dataset"] == dataset and row["method"] == "talr"), None)
            if cot and talr:
                delta = talr["accuracy"] - cot["accuracy"]
                deltas.append(delta)
                noninferior += int(delta >= 0)
        decisions[model] = {"datasets": len(deltas), "talr_noninferior_count": noninferior, "mean_delta_vs_cot": statistics.fmean(deltas) if deltas else None, "qualifies_as_main_method": len(deltas) == 4 and noninferior >= 3 and statistics.fmean(deltas) > 0}

    (out / "talr_core_main_table.json").write_text(json.dumps({"runs": runs, "decision": decisions, "audit": audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "pairwise_fixed_damaged.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "trigger_and_length_stats.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# TALR 两模型四数据集统一主表", "", "主指标使用 specialized 或 corrected last-answer evaluator。", "", "| Model | Dataset | Method | Acc | Pair | Failed | Avg tokens | Long | Maxed | Soft | Switch | Format trigger | Veto |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        def number(key: str, scale: float = 1.0) -> str:
            value = row.get(key)
            return "-" if value is None else f"{value * scale:.2f}"
        pair = "-" if row["pair_accuracy"] is None else f"{100 * float(row['pair_accuracy']):.2f}%"
        lines.append(f"| {row['model']} | {row['dataset']} | {LABELS[row['method']]} | {100 * row['accuracy']:.2f}% | {pair} | {row['failed_extraction']} | {row['mean_output_tokens']:.1f} | {row['long_ge_256']} | {row['maxed_1024']} | {number('mean_soft_ratio', 100)}% | {number('mean_switch_count')} | {number('mean_format_trigger_count')} | {number('mean_lead_soft_veto_count')} |")
    lines += ["", "## TALR 判定", ""]
    for model, decision in decisions.items():
        delta = decision["mean_delta_vs_cot"]
        delta_text = "NA" if delta is None else f"{100 * delta:+.2f} pp"
        lines.append(f"- {model}: non-inferior {decision['talr_noninferior_count']}/{decision['datasets']}, mean delta {delta_text}, main-method gate={decision['qualifies_as_main_method']}.")
    (out / "talr_core_main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    invalid = [row for row in audit if row["status"] != "valid"]
    if invalid:
        print(json.dumps({"invalid_runs": invalid}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(decisions, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

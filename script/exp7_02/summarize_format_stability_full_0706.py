import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

METHOD_ORDER = [
    "cot_orign_greedy",
    "lead",
    "pure_soft_format2",
    "pure_soft_guard",
    "quota05",
    "quota05_format2",
    "quota05_guard",
    "lead_format2",
    "lead_guard",
    "pure_soft",
    "pure_soft_diffuse_collapse",
    "answer_zone_discrete",
    "format_cooldown4",
    "highrisk_only_cooldown2",
]

BASELINE_RUNS = {
    "cot_orign_greedy": "cot_orign_greedy_gpu0",
    "lead": "lead_gpu1",
}

PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}"),
    re.compile(r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?"),
    re.compile(r"\*\*([A-Da-d])\*\*"),
    re.compile(r"(?:^|\n)\s*([A-Da-d])\s*$"),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def answer_region(text: str) -> str:
    markers = list(re.finditer(r"answer\s*[:.]", text or "", re.I))
    if markers:
        return text[markers[-1].start():]
    return (text or "")[-1500:]


def extract_last(text: str) -> str | None:
    if not text:
        return None
    region = answer_region(text)
    hits = []
    for pattern in PATTERNS:
        for match in pattern.finditer(region):
            hits.append((match.start(), match.group(1).upper()))
    if hits:
        return sorted(hits)[-1][1]
    last_letters = re.findall(r"\b([A-D])\b", region[-200:])
    return last_letters[-1].upper() if last_letters else None


def extract_yes_no(text: str) -> str | None:
    pred = extract_last(text)
    if pred == "A":
        return "yes"
    if pred == "B":
        return "no"
    region = answer_region(text).lower()
    hits = [(m.start(), "yes") for m in re.finditer(r"\byes\b", region)]
    hits += [(m.start(), "no") for m in re.finditer(r"\bno\b", region)]
    return sorted(hits)[-1][1] if hits else None


def gold_yes_no(answer: str) -> str | None:
    ans = (answer or "").strip().upper()
    if ans == "A":
        return "yes"
    if ans == "B":
        return "no"
    low = (answer or "").lower()
    if "yes" in low:
        return "yes"
    if "no" in low:
        return "no"
    return None


def eval_mcq(rows: list[dict]) -> tuple[dict, list[dict]]:
    total = correct = failed = runtime_errors = maxed = long = 0
    by_subtopic = defaultdict(lambda: {"correct": 0, "total": 0})
    items = []
    for row in rows:
        total += 1
        pred = extract_last(row.get("model_answer") or "")
        gold = str(row.get("answer", "")).strip().upper()
        ok = pred is not None and pred == gold
        failed += int(pred is None)
        runtime_errors += int(bool(row.get("error_type")))
        out_len = int(row.get("output_tokens") or 0)
        maxed += int(out_len >= 1024)
        long += int(out_len >= 256)
        correct += int(ok)
        sub = row.get("subtopic") or row.get("benchmark") or "unknown"
        by_subtopic[sub]["total"] += 1
        by_subtopic[sub]["correct"] += int(ok)
        items.append({"id": str(row.get("id")), "gold": gold, "pred": pred, "is_correct": ok})
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed,
        "runtime_errors": runtime_errors,
        "long_ge_256": long,
        "maxed_1024": maxed,
        "by_subtopic": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for k, v in sorted(by_subtopic.items())
        },
    }, items


def eval_pope(rows: list[dict]) -> tuple[dict, list[dict]]:
    total = correct = failed = tp = tn = fp = fn = 0
    items = []
    for row in rows:
        total += 1
        pred = extract_yes_no(row.get("model_answer") or "")
        gold = gold_yes_no(row.get("answer", ""))
        ok = pred is not None and pred == gold
        failed += int(pred is None or gold is None)
        correct += int(ok)
        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "no" and pred == "no":
            tn += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "yes" and pred == "no":
            fn += 1
        items.append({"id": str(row.get("id")), "gold": gold, "pred": pred, "is_correct": ok})
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / total if total else 0.0,
    }, items


def specialized_items(dataset: str, run: Path) -> tuple[dict | None, list[dict] | None]:
    if dataset == "mmvp" and (run / "specialized_eval_report.json").exists():
        report = load_json(run / "specialized_eval_report.json")
        rows = [
            {"id": str(r.get("id")), "gold": r.get("specialized_gold"), "pred": r.get("specialized_pred"), "is_correct": bool(r.get("specialized_is_correct"))}
            for r in load_jsonl(run / "specialized_eval_results.jsonl")
        ]
        return report, rows
    if dataset == "realworldqa_fixed200" and (run / "realworldqa_mcq_eval.json").exists():
        report = load_json(run / "realworldqa_mcq_eval.json")
        rows = [
            {"id": str(r.get("id")), "gold": r.get("realworldqa_gold"), "pred": r.get("realworldqa_pred"), "is_correct": bool(r.get("realworldqa_is_correct"))}
            for r in load_jsonl(run / "realworldqa_mcq_results.jsonl")
        ]
        return report, rows
    return None, None


def pairwise(base_items: list[dict], run_items: list[dict]) -> dict:
    b = {x["id"]: x for x in base_items}
    r = {x["id"]: x for x in run_items}
    ids = sorted(set(b) & set(r))
    fixed = [i for i in ids if not b[i]["is_correct"] and r[i]["is_correct"]]
    damaged = [i for i in ids if b[i]["is_correct"] and not r[i]["is_correct"]]
    return {
        "paired_total": len(ids),
        "fixed": len(fixed),
        "damaged": len(damaged),
        "net": len(fixed) - len(damaged),
        "unchanged_correct": sum(1 for i in ids if b[i]["is_correct"] and r[i]["is_correct"]),
        "unchanged_wrong": sum(1 for i in ids if not b[i]["is_correct"] and not r[i]["is_correct"]),
        "fixed_ids": fixed[:500],
        "damaged_ids": damaged[:500],
    }


def q(vals, quant):
    vals = sorted(vals)
    if not vals:
        return None
    return vals[int(round((len(vals) - 1) * quant))]


def trigger_stats(run: Path) -> dict:
    p = run / "token_entropy_full.jsonl"
    if not p.exists():
        return {}
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            toks = obj.get("tokens") or []
            n = len(toks)
            if not n:
                continue
            soft = sum(1 for t in toks if t.get("mode") == "soft")
            rows.append({
                "tokens": n,
                "soft": soft,
                "soft_ratio": soft / n,
                "switch": sum(1 for t in toks if t.get("to_soft") or t.get("to_normal")),
                "to_soft": sum(1 for t in toks if t.get("to_soft")),
                "to_normal": sum(1 for t in toks if t.get("to_normal")),
                "format_active": sum(1 for t in toks if t.get("format_cooldown_active")),
                "format_count": max([int(t.get("format_cooldown_active_count") or 0) for t in toks], default=0),
                "veto": sum(1 for t in toks if t.get("lead_soft_veto")),
                "veto_candidate": sum(1 for t in toks if t.get("lead_veto_candidate")),
            })
    if not rows:
        return {}
    return {
        "samples": len(rows),
        "mean_output_tokens": mean(r["tokens"] for r in rows),
        "mean_soft_ratio": mean(r["soft_ratio"] for r in rows),
        "mean_soft_tokens": mean(r["soft"] for r in rows),
        "mean_switch": mean(r["switch"] for r in rows),
        "p90_switch": q([r["switch"] for r in rows], 0.9),
        "mean_to_soft": mean(r["to_soft"] for r in rows),
        "mean_to_normal": mean(r["to_normal"] for r in rows),
        "mean_format_cooldown_active_tokens": mean(r["format_active"] for r in rows),
        "mean_format_cooldown_count": mean(r["format_count"] for r in rows),
        "mean_lead_soft_veto": mean(r["veto"] for r in rows),
        "mean_lead_veto_candidate": mean(r["veto_candidate"] for r in rows),
    }


def run_dir_for(format_base: Path, baseline_base: Path, model: str, dataset: str, method: str) -> Path | None:
    if method in BASELINE_RUNS:
        p = baseline_base / model / dataset / BASELINE_RUNS[method]
        return p if (p / "results.jsonl").exists() else None
    candidates = sorted((format_base / model / dataset).glob(f"{method}_gpu*"))
    for c in candidates:
        if (c / "results.jsonl").exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format_base", required=True)
    ap.add_argument("--baseline_base", required=True)
    args = ap.parse_args()
    format_base = Path(args.format_base)
    baseline_base = Path(args.baseline_base)

    model_names = sorted({p.name for p in format_base.iterdir() if p.is_dir()} | {p.name for p in baseline_base.iterdir() if p.is_dir()})
    summary = {}
    pairwise_all = {}
    trigger_all = {}
    md = ["# Format Stability Corrected Summary", "", "## Main Table", "", "| model | dataset | method | source | rows | acc | correct | failed | long>=256 | maxed1024 | avg_len | vs COT net | vs LEAD net |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    for model in model_names:
        datasets = sorted({p.name for p in (format_base / model).iterdir() if p.is_dir()} if (format_base / model).exists() else set())
        if (baseline_base / model).exists():
            datasets = sorted(set(datasets) | {p.name for p in (baseline_base / model).iterdir() if p.is_dir()})
        summary[model] = {}
        pairwise_all[model] = {}
        trigger_all[model] = {}
        item_cache = {}
        report_cache = {}
        for dataset in datasets:
            summary[model][dataset] = {}
            pairwise_all[model][dataset] = {}
            trigger_all[model][dataset] = {}
            for method in METHOD_ORDER:
                rd = run_dir_for(format_base, baseline_base, model, dataset, method)
                if rd is None:
                    continue
                rows = load_jsonl(rd / "results.jsonl")
                if not rows:
                    continue
                if dataset.startswith("pope_"):
                    corrected, items = eval_pope(rows)
                    source = "pope_yes_no_local"
                else:
                    corrected, items = eval_mcq(rows)
                    source = "corrected_last_answer"
                spec_report, spec_items = specialized_items(dataset, rd)
                if spec_report:
                    main_report = spec_report
                    items = spec_items or items
                    source = "specialized_mmvp_local" if dataset == "mmvp" else "realworldqa_mcq"
                else:
                    main_report = corrected
                avg_len = sum(float(r.get("output_tokens") or 0) for r in rows) / len(rows)
                rec = {
                    "run_dir": str(rd),
                    "source": source,
                    "main_eval": main_report,
                    "corrected_eval": corrected,
                    "old_eval": load_json(rd / "eval_report.json"),
                    "avg_output_tokens": avg_len,
                }
                summary[model][dataset][method] = rec
                item_cache[(dataset, method)] = items
                report_cache[(dataset, method)] = main_report
                trig = trigger_stats(rd)
                if trig:
                    trigger_all[model][dataset][method] = trig

            cot_items = item_cache.get((dataset, "cot_orign_greedy"))
            lead_items = item_cache.get((dataset, "lead"))
            for method, items in list(item_cache.items()):
                ds, meth = method
                if ds != dataset:
                    continue
                pw_cot = pairwise(cot_items, items) if cot_items else {}
                pw_lead = pairwise(lead_items, items) if lead_items else {}
                pairwise_all[model][dataset][meth] = {"vs_cot": pw_cot, "vs_lead": pw_lead}
                rep = report_cache[(dataset, meth)]
                total = rep.get("total", rep.get("sample_total", 0))
                correct = rep.get("correct", rep.get("sample_correct", 0))
                acc = rep.get("accuracy", rep.get("sample_accuracy", 0.0))
                failed = rep.get("failed_extraction", summary[model][dataset][meth]["corrected_eval"].get("failed_extraction", 0))
                corr = summary[model][dataset][meth]["corrected_eval"]
                avg_len = summary[model][dataset][meth]["avg_output_tokens"]
                md.append(f"| {model} | {dataset} | {meth} | {summary[model][dataset][meth]['source']} | {total} | {acc:.4f} | {correct} | {failed} | {corr.get('long_ge_256', 0)} | {corr.get('maxed_1024', 0)} | {avg_len:.1f} | {pw_cot.get('net', '')} | {pw_lead.get('net', '')} |")

    out_json = format_base / "corrected_eval_summary.json"
    out_pairwise = format_base / "pairwise_deltas.json"
    out_trigger = format_base / "format_trigger_stats.json"
    out_md = format_base / "corrected_eval_summary.md"
    out_trigger_md = format_base / "format_trigger_stats.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_pairwise.write_text(json.dumps(pairwise_all, ensure_ascii=False, indent=2), encoding="utf-8")
    out_trigger.write_text(json.dumps(trigger_all, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    tmd = ["# Format Trigger Stats", "", "| model | dataset | method | samples | mean soft ratio | mean switch | mean format active tokens | mean format count | mean veto |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for model, dsets in trigger_all.items():
        for dataset, methods in dsets.items():
            for method, s in methods.items():
                tmd.append(f"| {model} | {dataset} | {method} | {s.get('samples', 0)} | {s.get('mean_soft_ratio', 0):.4f} | {s.get('mean_switch', 0):.2f} | {s.get('mean_format_cooldown_active_tokens', 0):.2f} | {s.get('mean_format_cooldown_count', 0):.2f} | {s.get('mean_lead_soft_veto', 0):.2f} |")
    out_trigger_md.write_text("\n".join(tmd) + "\n", encoding="utf-8")
    print(out_md)
    print(out_trigger_md)


if __name__ == "__main__":
    main()

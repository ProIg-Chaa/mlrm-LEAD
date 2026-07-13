import json
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(
    "output/experiments/20260705_integrated_cot_lead_baselines/"
    "integrated_repo_cot_lead_baselines/r1_onevision_7b"
)
OUT_DIR = BASE.parent

METHODS = {
    "cot": "cot_orign_greedy_gpu0",
    "lead": "lead_gpu1",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}"),
    re.compile(r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?"),
    re.compile(r"\*\*([A-Da-d])\*\*"),
    re.compile(r"(?:^|\n)\s*([A-Da-d])\s*$"),
]


def answer_region(text: str) -> str:
    markers = list(re.finditer(r"answer\s*[:.]", text or "", re.I))
    if markers:
        return text[markers[-1].start() :]
    return (text or "")[-1500:]


def extract_first(text: str) -> str | None:
    # Mirrors lead/evaluator.py behavior but with the same pattern set used here.
    if not text:
        return None
    for pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    last_letters = re.findall(r"\b([A-D])\b", text[-200:])
    return last_letters[-1].upper() if last_letters else None


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
    yes = list(re.finditer(r"\byes\b", region))
    no = list(re.finditer(r"\bno\b", region))
    hits = [(m.start(), "yes") for m in yes] + [(m.start(), "no") for m in no]
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


def eval_mcq(rows: list[dict], extractor) -> tuple[dict, list[dict]]:
    total = correct = failed = runtime_errors = 0
    by_subtopic = defaultdict(lambda: {"correct": 0, "total": 0})
    enriched = []
    for row in rows:
        total += 1
        if row.get("error_type"):
            runtime_errors += 1
        pred = extractor(row.get("model_answer") or "")
        gold = str(row.get("answer", "")).strip().upper()
        ok = pred is not None and pred == gold
        failed += int(pred is None)
        correct += int(ok)
        sub = row.get("subtopic") or row.get("benchmark") or "unknown"
        by_subtopic[sub]["total"] += 1
        by_subtopic[sub]["correct"] += int(ok)
        item = {
            "id": str(row.get("id")),
            "gold": gold,
            "pred": pred,
            "is_correct": ok,
            "output_tokens": row.get("output_tokens"),
            "error_type": row.get("error_type"),
        }
        enriched.append(item)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed,
        "runtime_errors": runtime_errors,
        "by_subtopic": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            }
            for k, v in sorted(by_subtopic.items())
        },
    }, enriched


def eval_pope(rows: list[dict]) -> tuple[dict, list[dict]]:
    total = correct = failed = tp = tn = fp = fn = 0
    enriched = []
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
        enriched.append(
            {
                "id": str(row.get("id")),
                "gold": gold,
                "pred": pred,
                "is_correct": ok,
            }
        )
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }, enriched


def pairwise(cot_items: list[dict], lead_items: list[dict]) -> dict:
    c = {x["id"]: x for x in cot_items}
    l = {x["id"]: x for x in lead_items}
    ids = sorted(set(c) & set(l))
    fixed = [i for i in ids if not c[i]["is_correct"] and l[i]["is_correct"]]
    damaged = [i for i in ids if c[i]["is_correct"] and not l[i]["is_correct"]]
    unchanged_correct = [i for i in ids if c[i]["is_correct"] and l[i]["is_correct"]]
    unchanged_wrong = [i for i in ids if not c[i]["is_correct"] and not l[i]["is_correct"]]
    return {
        "paired_total": len(ids),
        "fixed": len(fixed),
        "damaged": len(damaged),
        "net": len(fixed) - len(damaged),
        "unchanged_correct": len(unchanged_correct),
        "unchanged_wrong": len(unchanged_wrong),
        "fixed_ids": fixed[:200],
        "damaged_ids": damaged[:200],
    }


def specialized_items(dataset: str, run: Path) -> list[dict] | None:
    if dataset == "mmvp":
        path = run / "specialized_eval_results.jsonl"
        if not path.exists():
            return None
        return [
            {
                "id": str(row.get("id")),
                "gold": row.get("specialized_gold"),
                "pred": row.get("specialized_pred"),
                "is_correct": bool(row.get("specialized_is_correct")),
            }
            for row in load_jsonl(path)
        ]
    if dataset == "realworldqa_fixed200":
        path = run / "realworldqa_mcq_results.jsonl"
        if not path.exists():
            return None
        return [
            {
                "id": str(row.get("id")),
                "gold": row.get("realworldqa_gold"),
                "pred": row.get("realworldqa_pred"),
                "is_correct": bool(row.get("realworldqa_is_correct")),
            }
            for row in load_jsonl(path)
        ]
    return None


def metric_kind(dataset: str) -> str:
    if dataset.startswith("pope_"):
        return "pope_yes_no_local"
    if dataset == "mmvp":
        return "mmvp_specialized_if_available"
    if dataset == "realworldqa_fixed200":
        return "realworldqa_specialized_if_available"
    if dataset in {"mmhal", "mathvista200"}:
        return "proxy_mcq_last_answer_not_official"
    return "mcq_last_answer"


def main():
    datasets = sorted(p.name for p in BASE.iterdir() if p.is_dir())
    summary = {}
    rows_md = [
        "| dataset | metric kind | method | rows | main acc | correct | failed extraction | metric source | old acc | corrected last-answer acc | first-match acc | avg output tokens |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    pairwise_md = [
        "| dataset | paired | fixed | damaged | net | unchanged correct | unchanged wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        ds_report = {"metric_kind": metric_kind(dataset), "methods": {}}
        enriched_by_method = {}
        for label, run_name in METHODS.items():
            run = BASE / dataset / run_name
            results_path = run / "results.jsonl"
            if not results_path.exists():
                continue
            rows = load_jsonl(results_path)
            if dataset.startswith("pope_"):
                corrected, enriched = eval_pope(rows)
                first_match = None
            else:
                corrected, enriched = eval_mcq(rows, extract_last)
                first_match, _ = eval_mcq(rows, extract_first)
            old = load_json(run / "eval_report.json")
            specialized = {}
            metric_source = "corrected_last_answer"
            main_eval = corrected
            if dataset == "mmvp" and (run / "specialized_eval_report.json").exists():
                specialized = load_json(run / "specialized_eval_report.json")
                main_eval = specialized
                metric_source = "specialized_mmvp_local"
            elif dataset == "realworldqa_fixed200" and (run / "realworldqa_mcq_eval.json").exists():
                specialized = load_json(run / "realworldqa_mcq_eval.json")
                main_eval = specialized
                metric_source = "realworldqa_mcq"
            avg_len = sum(float(r.get("output_tokens") or 0) for r in rows) / len(rows)
            ds_report["methods"][label] = {
                "run_dir": str(run),
                "old_eval": old,
                "corrected_eval": corrected,
                "first_match_eval": first_match,
                "specialized_eval": specialized,
                "main_eval": main_eval,
                "metric_source": metric_source,
                "avg_output_tokens": avg_len,
            }
            enriched_by_method[label] = specialized_items(dataset, run) or enriched
            rows_md.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        ds_report["metric_kind"],
                        label,
                        str(main_eval.get("total", main_eval.get("sample_total", corrected["total"]))),
                        f"{main_eval.get('accuracy', main_eval.get('sample_accuracy', 0.0)):.4f}",
                        str(main_eval.get("correct", main_eval.get("sample_correct", corrected["correct"]))),
                        str(main_eval.get("failed_extraction", corrected["failed_extraction"])),
                        metric_source,
                        f"{old.get('accuracy', 0.0):.4f}" if old else "NA",
                        f"{corrected['accuracy']:.4f}",
                        f"{first_match['accuracy']:.4f}" if first_match else "NA",
                        f"{avg_len:.1f}",
                    ]
                )
                + " |"
            )
        if "cot" in enriched_by_method and "lead" in enriched_by_method:
            pw = pairwise(enriched_by_method["cot"], enriched_by_method["lead"])
            ds_report["pairwise_cot_vs_lead"] = pw
            pairwise_md.append(
                f"| {dataset} | {pw['paired_total']} | {pw['fixed']} | {pw['damaged']} | {pw['net']} | {pw['unchanged_correct']} | {pw['unchanged_wrong']} |"
            )
        summary[dataset] = ds_report

    (OUT_DIR / "corrected_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Corrected Evaluation Summary",
        "",
        "Notes:",
        "- `old acc` is the run-time `eval_report.json` accuracy.",
        "- `acc` uses corrected extraction: answer-region/last-answer priority, and counts `pred is None` as failed extraction.",
        "- `mmvp`, `mmhal`, and `mathvista200` are marked as proxy when official/LLM judging is not used.",
        "- POPE uses deterministic yes/no mapping: A=yes, B=no plus answer-region fallback.",
        "",
        "## Main Table",
        "",
        *rows_md,
        "",
        "## Pairwise COT vs LEAD",
        "",
        *pairwise_md,
        "",
    ]
    (OUT_DIR / "corrected_eval_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(OUT_DIR / "corrected_eval_summary.md")
    print("\n".join(rows_md))
    print("\n".join(pairwise_md))


if __name__ == "__main__":
    main()

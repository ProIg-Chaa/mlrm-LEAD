#!/usr/bin/env python3
"""Deterministic evaluators for newly added Origin-LEAD benchmarks.

Supported modes:
- mcq: A/B/C/D/E extraction with option-text fallback
- yes_no: yes/no extraction plus hallucination metrics
- mathvision: MCQ/numeric normalized matching for MathVision-style rows

Official evaluators should be preferred when available; this script is the
stable local fallback for smoke tests and deterministic first-pass reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: str | None) -> str:
    text = (text or "").lower()
    text = text.replace("\\boxed", " ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9.\-]+", " ", text)
    return " ".join(text.split())


def answer_region(text: str | None) -> str:
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    matches = list(re.finditer(r"(?:final\s+)?answer\s*[:：]|therefore|thus", text, re.I))
    if matches:
        return text[matches[-1].start():]
    return text[-1500:]


def parse_letter_options(options: str | None) -> dict[str, str]:
    options = options or ""
    parsed: dict[str, str] = {}
    patterns = [
        re.compile(r"(?ms)(?:^|\n)\s*([A-E])[\.\):]\s*(.*?)(?=(?:\n\s*[A-E][\.\):]\s)|\Z)"),
        re.compile(r"(?ms)\(([A-Ea-e])\)\s*([^()]+?)(?=(?:\s+\([A-Ea-e]\))|\Z)"),
    ]
    for pattern in patterns:
        for letter, body in pattern.findall(options):
            parsed[letter.upper()] = body.strip()
        if parsed:
            break
    return parsed


def extract_letter(text: str | None, letters: str = "ABCDE") -> str | None:
    region = answer_region(text)
    if not region:
        return None
    cls = re.escape(letters)
    patterns = [
        rf"\\boxed\{{\s*([{cls}a-e])\s*\}}",
        rf"(?:answer|choice)\s*[:：]?\s*\(?([{cls}a-e])\)?",
        rf"final\s+(?:answer|choice)\s*(?:is)?\s*[:：]?\s*\(?([{cls}a-e])\)?",
        rf"\(([{cls}a-e])\)",
        rf"(?:^|\n)\s*([{cls}])\s*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, region, re.I)
        if m:
            return m.group(1).upper()
    found = re.findall(rf"\b([{letters}])\b", region[-300:])
    return found[-1].upper() if found else None


def extract_yes_no(text: str | None) -> str | None:
    region = answer_region(text)
    if not region:
        return None
    patterns = [
        r"\\boxed\{\s*(yes|no)\s*\}",
        r"(?:answer|final answer)\s*[:：]?\s*(yes|no)\b",
        r"\b(yes|no)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, region, re.I)
        if m:
            return m.group(1).lower()
    return None


def extract_number(text: str | None) -> float | None:
    region = answer_region(text)
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", region)
    if boxed:
        region = boxed[-1]
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", region.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def parse_gold_letter(answer: str | None) -> str | None:
    answer = (answer or "").strip()
    m = re.match(r"(?:\(?([A-Ea-e])\)?|([A-Ea-e])\.)", answer)
    if m:
        return (m.group(1) or m.group(2)).upper()
    m = re.search(r"\(([A-Ea-e])\)", answer)
    return m.group(1).upper() if m else None


def parse_gold_number(answer: str | None) -> float | None:
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", (answer or "").replace(",", ""))
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


def option_text_fallback(sample: dict, prediction: str | None) -> tuple[str | None, str]:
    options = parse_letter_options(sample.get("options"))
    if not options:
        return None, "no_options"
    region_norm = normalize(answer_region(prediction))
    if not region_norm:
        return None, "empty_region"
    scored = []
    for letter, body in options.items():
        body_norm = normalize(body)
        if not body_norm:
            continue
        contain = 1.0 if body_norm in region_norm else 0.0
        overlap = len(set(region_norm.split()) & set(body_norm.split())) / max(1, len(set(body_norm.split())))
        seq = SequenceMatcher(None, region_norm, body_norm).ratio()
        scored.append((max(contain, overlap, seq), contain, overlap, letter))
    if not scored:
        return None, "no_option_score"
    scored.sort(reverse=True)
    best, contain, overlap, letter = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if contain == 1.0 or overlap == 1.0 or (best >= 0.78 and best - second >= 0.08):
        return letter, "option_text_match"
    return None, "ambiguous_option_match"


def infer_mcq(sample: dict, prediction: str | None) -> tuple[str | None, str]:
    pred = extract_letter(prediction)
    if pred is not None:
        return pred, "direct_letter"
    pred, method = option_text_fallback(sample, prediction)
    return pred, method


def evaluate(dataset_rows: list[dict], result_rows: list[dict], mode: str) -> tuple[dict, list[dict]]:
    by_id = {str(row["id"]): row for row in dataset_rows}
    total = correct = failed = 0
    yes_no_counts = defaultdict(int)
    by_subtopic = defaultdict(lambda: {"correct": 0, "total": 0})
    enriched = []
    for row in result_rows:
        sample = by_id[str(row["id"])]
        prediction = row.get("model_answer") or ""
        gold_raw = str(sample.get("answer") or sample.get("label") or "").strip()
        pred = None
        method = "none"
        ok = False
        if mode == "yes_no":
            pred = extract_yes_no(prediction)
            gold = "yes" if gold_raw.lower() in {"yes", "1", "true"} else "no" if gold_raw.lower() in {"no", "0", "false"} else gold_raw.lower()
            ok = pred is not None and pred == gold
            method = "yes_no"
            if pred in {"yes", "no"} and gold in {"yes", "no"}:
                yes_no_counts[f"{gold}->{pred}"] += 1
        elif mode in {"mcq", "mathvision"}:
            gold_letter = parse_gold_letter(gold_raw)
            if gold_letter is not None:
                pred, method = infer_mcq(sample, prediction)
                ok = pred is not None and pred == gold_letter
            else:
                gold_num = parse_gold_number(gold_raw)
                pred_num = extract_number(prediction)
                pred = pred_num
                method = "numeric"
                ok = numeric_equal(pred_num, gold_num)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        total += 1
        correct += int(ok)
        failed += int(pred is None)
        subtopic = sample.get("subtopic") or sample.get("category") or sample.get("subject") or "unknown"
        by_subtopic[subtopic]["total"] += 1
        by_subtopic[subtopic]["correct"] += int(ok)
        item = dict(row)
        item.update({
            "eval_mode": mode,
            "eval_gold": gold_raw,
            "eval_pred": pred,
            "eval_match_method": method,
            "eval_is_correct": ok,
        })
        enriched.append(item)

    report = {
        "mode": mode,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed,
        "by_subtopic": {
            k: {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in sorted(by_subtopic.items())
        },
    }
    if mode == "yes_no":
        tp = yes_no_counts["yes->yes"]
        fp = yes_no_counts["no->yes"]
        fn = yes_no_counts["yes->no"]
        tn = yes_no_counts["no->no"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report.update({
            "precision_yes": precision,
            "recall_yes": recall,
            "f1_yes": f1,
            "confusion": {"tp_yes": tp, "fp_yes": fp, "fn_yes": fn, "tn_yes": tn},
        })
    return report, enriched


def run_synthetic_tests() -> None:
    assert extract_yes_no("Answer: Yes.") == "yes"
    assert extract_yes_no("The final answer is no") == "no"
    assert extract_letter("The answer is (C)") == "C"
    assert extract_letter("\\boxed{D}") == "D"
    assert numeric_equal(extract_number("Final answer: \\boxed{1.20}"), 1.2)
    sample = {"options": "A. cat\nB. dog\nC. bird\nD. fish"}
    assert option_text_fallback(sample, "The answer is dog.")[0] == "B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--results")
    parser.add_argument("--mode", choices=["mcq", "yes_no", "mathvision"])
    parser.add_argument("--output_json")
    parser.add_argument("--output_results_jsonl")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_synthetic_tests()
        print("self_test_ok")
        return 0
    if not args.dataset or not args.results or not args.mode:
        parser.error("--dataset, --results and --mode are required unless --self_test")

    report, enriched = evaluate(load_jsonl(Path(args.dataset)), load_jsonl(Path(args.results)), args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_results_jsonl:
        out = Path(args.output_results_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in enriched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

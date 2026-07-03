#!/usr/bin/env python3
"""Evaluate RealWorldQA multiple-choice results with option-text fallback."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\\boxed", " ")
    text = re.sub(r"\\[()\\[\\]]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def answer_region(text: str) -> str:
    markers = list(re.finditer(r"answer\s*[:.]", text or "", re.I))
    if markers:
        return text[markers[-1].start():]
    return (text or "")[-1200:]


def parse_options(options: str) -> dict[str, str]:
    options = options or ""
    parsed = {}
    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*([A-D])\.\s*(.*?)(?=(?:\n\s*[A-D]\.\s)|\Z)"
    )
    for letter, body in pattern.findall(options):
        parsed[letter.upper()] = body.strip()
    return parsed


def extract_letter(text: str) -> str | None:
    region = answer_region(text)
    patterns = [
        r"\\boxed\{\s*([A-Da-d])\s*\}",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"\(([A-Da-d])\)",
        r"(?:^|\n)\s*([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, region)
        if match:
            return match.group(1).upper()
    last_letters = re.findall(r"\b([A-D])\b", region[-200:])
    return last_letters[-1].upper() if last_letters else None


def infer_choice(sample: dict, prediction: str) -> tuple[str | None, str]:
    direct = extract_letter(prediction)
    options = parse_options(sample.get("options", ""))
    if direct in options:
        return direct, "direct_letter"
    if direct is not None and not options:
        return direct, "direct_letter_no_options"

    region_norm = normalize(answer_region(prediction))
    if not region_norm or not options:
        return direct, "no_option_match"

    scored = []
    for letter, body in options.items():
        label = normalize(body)
        if not label:
            continue
        seq = SequenceMatcher(None, region_norm, label).ratio()
        contain = 1.0 if label in region_norm else 0.0
        region_tokens = set(region_norm.split())
        label_tokens = set(label.split())
        overlap = len(region_tokens & label_tokens) / len(label_tokens) if label_tokens else 0.0
        scored.append((max(seq, contain, overlap), contain, overlap, letter))
    if not scored:
        return direct, "no_option_score"
    scored.sort(reverse=True)
    best, contain, overlap, letter = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if contain == 1.0 or overlap == 1.0 or (best >= 0.72 and best - second >= 0.08):
        return letter, "option_text_match"
    return direct, "ambiguous_option_match"


def evaluate(dataset_rows: list[dict], result_rows: list[dict]) -> tuple[dict, list[dict]]:
    dataset_by_id = {int(row["id"]): row for row in dataset_rows}
    total = correct = failed = 0
    enriched = []
    by_answer: dict[str, dict[str, int]] = {}
    for row in result_rows:
        sample = dataset_by_id[int(row["id"])]
        gold = (sample.get("answer") or "").strip().upper()
        pred, method = infer_choice(sample, row.get("model_answer") or "")
        ok = pred == gold
        total += 1
        correct += int(ok)
        failed += int(pred is None)
        by_answer.setdefault(gold, {"correct": 0, "total": 0})
        by_answer[gold]["total"] += 1
        by_answer[gold]["correct"] += int(ok)
        item = dict(row)
        item["realworldqa_gold"] = gold
        item["realworldqa_pred"] = pred
        item["realworldqa_match_method"] = method
        item["realworldqa_is_correct"] = ok
        enriched.append(item)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed,
        "by_answer": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            }
            for k, v in sorted(by_answer.items())
        },
    }, enriched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_results_jsonl", default=None)
    args = parser.parse_args()
    report, enriched = evaluate(
        load_jsonl(Path(args.dataset)),
        load_jsonl(Path(args.results)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_results_jsonl:
        out = Path(args.output_results_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in enriched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

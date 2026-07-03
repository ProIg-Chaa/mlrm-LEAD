#!/usr/bin/env python3
"""Strict RealWorldQA MCQ evaluator.

This evaluator avoids broad matches such as any "(A)" in the reasoning trace.
It first searches answer-like regions near the end of the output, then falls
back to conservative option-text matching only for sufficiently distinctive
option texts.
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ANSWER_MARKER = re.compile(
    r"(?i)(final\s+(?:answer|choice)|correct\s+(?:answer|option|choice)|"
    r"answer|choice|option|i\s+choose|therefore|so)\s*[:：]?"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\\boxed", " ")
    text = re.sub(r"\\[()\\[\\]]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def strip_think(text: str) -> str:
    text = text or ""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def answer_region(text: str) -> str:
    tail = strip_think(text)[-700:]
    matches = list(ANSWER_MARKER.finditer(tail))
    if matches:
        return tail[matches[-1].start():]
    return tail[-500:]


def parse_options(options: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*([A-D])\.\s*(.*?)(?=(?:\n\s*[A-D]\.\s)|\Z)"
    )
    for letter, body in pattern.findall(options or ""):
        parsed[letter.upper()] = body.strip()
    return parsed


def extract_letter(text: str) -> tuple[str | None, str]:
    region = answer_region(text)
    candidates: list[tuple[int, str, str]] = []
    patterns = [
        (r"\\boxed\{\s*([A-Da-d])\s*\}", "boxed"),
        (
            r"(?i)(?:final\s+(?:answer|choice)|correct\s+(?:answer|option|choice)|"
            r"answer|choice|option|i\s+choose)\s*(?:is|would\s+be)?\s*[:：]?\s*"
            r"(?:option|choice)?\s*([A-D])\b",
            "explicit_marker",
        ),
        (r"(?i)(?:therefore|so|thus)[^.\n]{0,80}?\b(?:answer\s+is\s*)?([A-D])\b", "therefore"),
        (r"(?m)^\s*(?:\*\*)?\s*([A-D])\s*(?:[.)]|$)", "line_start"),
    ]
    for pattern, method in patterns:
        for match in re.finditer(pattern, region):
            candidates.append((match.start(), match.group(1).upper(), method))
    if candidates:
        _, letter, method = sorted(candidates, key=lambda x: x[0])[-1]
        return letter, method
    return None, "no_direct_letter"


def option_text_match(options: dict[str, str], text: str) -> tuple[str | None, str]:
    region = answer_region(text)
    region_norm = normalize(region)
    if not region_norm or not options:
        return None, "no_option_region"

    scored = []
    region_tokens = set(region_norm.split())
    for letter, body in options.items():
        label = normalize(body)
        if not label:
            continue
        label_tokens = set(label.split())
        if len(label_tokens) <= 1:
            continue
        contain = label in region_norm
        overlap = len(region_tokens & label_tokens) / len(label_tokens) if label_tokens else 0.0
        seq = SequenceMatcher(None, region_norm, label).ratio()
        score = max(1.0 if contain else 0.0, overlap, seq)
        scored.append((score, contain, overlap, seq, letter))
    if not scored:
        return None, "no_distinctive_option"
    scored.sort(reverse=True)
    best, contain, overlap, seq, letter = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if contain and len(normalize(options[letter]).split()) >= 2:
        return letter, "option_text_contain"
    if overlap >= 0.92 and best - second >= 0.18:
        return letter, "option_text_overlap"
    if seq >= 0.78 and best - second >= 0.15:
        return letter, "option_text_seq"
    return None, "ambiguous_option_match"


def infer_choice(sample: dict, prediction: str) -> tuple[str | None, str]:
    options = parse_options(sample.get("options", ""))
    direct, method = extract_letter(prediction)
    if direct in options:
        return direct, method
    fallback, fallback_method = option_text_match(options, prediction)
    return fallback, fallback_method


def evaluate(dataset_rows: list[dict], result_rows: list[dict]) -> tuple[dict, list[dict]]:
    dataset_by_id = {int(row["id"]): row for row in dataset_rows}
    total = correct = failed = 0
    by_answer: dict[str, dict[str, int]] = {}
    by_method: dict[str, int] = {}
    enriched = []
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
        by_method[method] = by_method.get(method, 0) + 1
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
        "by_method": dict(sorted(by_method.items())),
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
    report, enriched = evaluate(load_jsonl(Path(args.dataset)), load_jsonl(Path(args.results)))
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

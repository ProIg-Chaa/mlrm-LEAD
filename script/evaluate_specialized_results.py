#!/usr/bin/env python3
"""Specialized evaluation for datasets whose answers are not plain A/B/C/D."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\\boxed", " ")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_tail_answer_region(text: str) -> str:
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    if markers:
        return text[markers[-1].start():]
    return text[-1500:]


def parse_mmvp_options(options_text: str) -> list[tuple[str, str]]:
    return [
        (letter.lower(), body.strip())
        for letter, body in re.findall(
            r"\(([abAB])\)\s*([^()]+?)(?=(?:\s+\([abAB]\))|$)",
            options_text or "",
        )
    ]


def parse_letter_options(options_text: str) -> dict[str, str]:
    options = {}
    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*([A-D])\.\s*(.*?)(?=(?:\n\s*[A-D]\.\s)|\Z)"
    )
    for letter, body in pattern.findall(options_text or ""):
        options[letter.upper()] = body.strip()
    return options


def extract_mcq_letter(text: str) -> str | None:
    if not text:
        return None

    patterns = [
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"\\boxed\{([A-Da-d])\}",
        r"\*\*([A-Da-d])\*\*",
        r"(?:^|\n)\s*([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    last_letters = re.findall(r"\b([A-D])\b", text[-200:])
    if last_letters:
        return last_letters[-1].upper()
    return None


def infer_physunibench_letter(sample: dict, prediction: str) -> tuple[str | None, str]:
    direct = extract_mcq_letter(prediction)
    if direct is not None:
        return direct, "direct_letter"

    options = parse_letter_options(sample.get("options", ""))
    if not options:
        return None, "no_options"

    answer_region = extract_tail_answer_region(prediction)
    answer_norm = normalize_text(answer_region)
    if not answer_norm:
        return None, "empty_prediction"

    scored = []
    for letter, body in options.items():
        option_norm = normalize_text(body)
        if not option_norm:
            continue
        seq_ratio = SequenceMatcher(None, answer_norm, option_norm).ratio()
        contain_ratio = 0.0
        if option_norm in answer_norm:
            contain_ratio = 1.0
        elif answer_norm in option_norm:
            contain_ratio = max(contain_ratio, 0.95)
        token_a = set(answer_norm.split())
        token_b = set(option_norm.split())
        overlap = len(token_a & token_b) / len(token_b) if token_b else 0.0
        score = max(seq_ratio, contain_ratio, overlap)
        scored.append((score, seq_ratio, overlap, letter))

    if not scored:
        return None, "no_option_score"

    scored.sort(reverse=True)
    best_score, _, _, best_letter = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.62 and best_score - second_score >= 0.08:
        return best_letter, "option_text_match"
    return None, "ambiguous_option_match"


def extract_mmvp_choice(sample: dict, prediction: str) -> tuple[str | None, str]:
    answer_region = extract_tail_answer_region(prediction)
    patterns = [
        r"\\boxed\{\s*\(?([abAB])\)?\s*\}",
        r"[Tt]he\s+(?:final\s+)?choice\s+(?:is\s*)?[:\s]*\(?([abAB])\)?(?=$|[\s\.,;:])",
        r"[Aa]nswer\s*[:\s]+\(?([abAB])\)?(?=$|[\s\.,;:])",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer_region)
        if match:
            return match.group(1).lower(), "direct_ab"

    answer_norm = normalize_text(answer_region)
    option_pairs = parse_mmvp_options(sample.get("options", ""))
    if option_pairs:
        scored = []
        for letter, body in option_pairs:
            label = normalize_text(body)
            if not label:
                continue
            seq_ratio = SequenceMatcher(None, answer_norm, label).ratio()
            contain_ratio = 1.0 if label in answer_norm else 0.0
            answer_tokens = set(answer_norm.split())
            label_tokens = set(label.split())
            overlap = len(answer_tokens & label_tokens) / len(label_tokens) if label_tokens else 0.0
            score = max(seq_ratio, contain_ratio, overlap)
            scored.append((score, contain_ratio, overlap, letter))
        scored.sort(reverse=True)
        best_score, best_contain, best_overlap, best_letter = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if (
            best_contain == 1.0
            or best_overlap == 1.0
            or (best_score >= 0.80 and best_score - second_score >= 0.08)
        ):
            return best_letter.lower(), "option_label_match"

    return None, "no_match"


def normalize_mmvp_gold(answer: str) -> str | None:
    match = re.search(r"\(([abAB])\)", answer or "")
    return match.group(1).lower() if match else None


def mode_from_dataset(path: Path) -> str:
    name = path.name.lower()
    if "mmvp" in name:
        return "mmvp"
    if "physunibench" in name:
        return "physunibench"
    raise ValueError(f"Cannot infer evaluation mode from dataset path: {path}")


def evaluate(dataset_rows: list[dict], result_rows: list[dict], mode: str) -> tuple[dict, list[dict]]:
    dataset_by_id = {int(row["id"]): row for row in dataset_rows}
    total = 0
    correct = 0
    failed_extraction = 0
    subtopic_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    language_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    enriched = []

    for row in result_rows:
        sample_id = int(row["id"])
        sample = dataset_by_id[sample_id]
        prediction = row.get("model_answer") or ""

        if mode == "mmvp":
            pred, method = extract_mmvp_choice(sample, prediction)
            gold = normalize_mmvp_gold(sample.get("answer", ""))
            is_correct = pred is not None and pred == gold
        elif mode == "physunibench":
            pred, method = infer_physunibench_letter(sample, prediction)
            gold = (sample.get("answer") or "").strip().upper() or None
            is_correct = pred is not None and pred == gold
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if pred is None:
            failed_extraction += 1

        total += 1
        correct += int(is_correct)

        subtopic = sample.get("subtopic", "unknown")
        difficulty = sample.get("difficulty", 0)
        language = sample.get("language", "unknown")
        subtopic_stats[subtopic]["total"] += 1
        difficulty_stats[difficulty]["total"] += 1
        language_stats[language]["total"] += 1
        if is_correct:
            subtopic_stats[subtopic]["correct"] += 1
            difficulty_stats[difficulty]["correct"] += 1
            language_stats[language]["correct"] += 1

        enriched_row = dict(row)
        enriched_row["specialized_gold"] = gold
        enriched_row["specialized_pred"] = pred
        enriched_row["specialized_match_method"] = method
        enriched_row["specialized_is_correct"] = is_correct
        enriched.append(enriched_row)

    def finalize(stats: dict) -> dict:
        return {
            key: {
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
                "correct": value["correct"],
                "total": value["total"],
            }
            for key, value in sorted(stats.items())
        }

    report = {
        "mode": mode,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed_extraction,
        "by_subtopic": finalize(subtopic_stats),
        "by_difficulty": finalize(difficulty_stats),
        "by_language": finalize(language_stats),
    }

    if mode == "mmvp":
        enriched.sort(key=lambda row: int(row["id"]))
        pair_total = 0
        pair_correct = 0
        for i in range(0, len(enriched), 2):
            pair = enriched[i:i + 2]
            if len(pair) < 2:
                continue
            pair_total += 1
            is_pair_correct = all(row["specialized_is_correct"] for row in pair)
            pair_correct += int(is_pair_correct)
            pair_index = pair_total - 1
            for row in pair:
                row["pair_index"] = pair_index
                row["pair_is_correct"] = is_pair_correct

        report["pair_accuracy"] = pair_correct / pair_total if pair_total else 0.0
        report["pair_correct"] = pair_correct
        report["pair_total"] = pair_total

    return report, enriched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--mode", choices=["auto", "mmvp", "physunibench"], default="auto")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_results_jsonl", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    results_path = Path(args.results)
    mode = mode_from_dataset(dataset_path) if args.mode == "auto" else args.mode

    dataset_rows = load_jsonl(dataset_path)
    result_rows = load_jsonl(results_path)
    report, enriched = evaluate(dataset_rows, result_rows, mode)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if args.output_results_jsonl:
        output_path = Path(args.output_results_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in enriched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

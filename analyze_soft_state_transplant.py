#!/usr/bin/env python3
"""Summarize paired answer changes in the soft-state transplant experiment."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Ea-e])\)?\s*\}"),
    re.compile(
        r"\**final\s+(?:answer|choice)\s*\**\s*(?:is)?\s*[:\s]*\**\s*\(?([A-Ea-e])\)?",
        re.I,
    ),
    re.compile(
        r"\**(?:the\s+correct\s+)?answer\s*\**\s+(?:is\s*)?[:\s]+\**\s*\(?([A-Ea-e])\)?",
        re.I,
    ),
    re.compile(r"\**answer\s*\**\s*[:\s]+\**\s*\(?([A-Ea-e])\)?", re.I),
    re.compile(r"\*\*([A-Ea-e])\*\*"),
    re.compile(r"(?:^|\n)\s*\(?([A-Ea-e])\)?\s*$"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_options(options: str | None) -> dict[str, str]:
    text = str(options or "")
    matches = list(
        re.finditer(r"(?:^|\s|\n)(?:\(([A-Ea-e])\)|([A-Ea-e])[\.:)])\s*", text)
    )
    parsed = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parsed[(match.group(1) or match.group(2)).upper()] = text[
            match.end() : end
        ].strip()
    return parsed


def gold_choice(row: dict[str, Any]) -> str | None:
    text = str(row.get("answer") or row.get("gold") or "").strip()
    match = re.search(r"(?:^|\()\s*([A-Ea-e])\s*\)?(?:\s|$)", text)
    if match:
        return match.group(1).upper()
    if text.upper() in {"YES", "TRUE"}:
        return "A"
    if text.upper() in {"NO", "FALSE"}:
        return "B"
    for letter, option in parse_options(row.get("options")).items():
        if text.casefold() == option.casefold():
            return letter
    return None


def answer_region(text: str) -> str:
    markers = list(re.finditer(r"(?:final\s+)?answer\s*[:.]", text, re.I))
    return text[markers[-1].start() :] if markers else text[-1800:]


def prediction(row: dict[str, Any]) -> str | None:
    text = str(row.get("model_answer") or "")
    if not text:
        return None
    region = answer_region(text)
    hits = []
    for pattern in PATTERNS:
        hits.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(region))
    if hits:
        return max(hits)[1]
    choices = re.findall(r"(?:^|\n)\s*\(?([A-Ea-e])\)?(?:\s|[.)])", region[-500:])
    if choices:
        return choices[-1].upper()
    letters = re.findall(r"\b([A-E])\b", region[-200:])
    if letters:
        return letters[-1].upper()
    option_hits = []
    for letter, option in parse_options(row.get("options")).items():
        if option:
            option_hits.extend(
                (match.start(), letter)
                for match in re.finditer(re.escape(option), region, re.I)
            )
    return max(option_hits)[1] if option_hits else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    branches = sorted({str(row["branch"]) for row in rows})
    enriched = []
    for row in rows:
        item = dict(row)
        item["pred"] = prediction(row)
        item["gold_choice"] = gold_choice(row)
        item["correct"] = item["pred"] is not None and item["pred"] == item["gold_choice"]
        enriched.append(item)
    by_event = collections.defaultdict(dict)
    for row in enriched:
        by_event[str(row["event_id"])][str(row["branch"])] = row

    def group_stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        correct = sum(bool(row["correct"]) for row in subset)
        failed = sum(row["pred"] is None for row in subset)
        return {
            "n": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
            "failed_extraction": failed,
        }

    result: dict[str, Any] = {
        "events": len(by_event),
        "rows": len(enriched),
        "branch_metrics": {},
        "by_dataset": {},
        "by_selection_class": {},
        "paired_against_hard": {},
        "paired_against_true_image": {},
    }
    for branch in branches:
        result["branch_metrics"][branch] = group_stats(
            [row for row in enriched if row["branch"] == branch]
        )
    for key, target in (("dataset", "by_dataset"), ("selection_class", "by_selection_class")):
        for value in sorted({str(row.get(key)) for row in enriched}):
            result[target][value] = {
                branch: group_stats(
                    [
                        row
                        for row in enriched
                        if str(row.get(key)) == value and row["branch"] == branch
                    ]
                )
                for branch in branches
            }

    for reference, target in (
        ("hard", "paired_against_hard"),
        ("true_image", "paired_against_true_image"),
    ):
        for branch in branches:
            if branch == reference:
                continue
            pairs = [
                (items[reference], items[branch])
                for items in by_event.values()
                if reference in items and branch in items
            ]
            fixed = sum(not left["correct"] and right["correct"] for left, right in pairs)
            damaged = sum(left["correct"] and not right["correct"] for left, right in pairs)
            changed = sum(left["pred"] != right["pred"] for left, right in pairs)
            result[target][branch] = {
                "n": len(pairs),
                "fixed": fixed,
                "damaged": damaged,
                "net_fixed_minus_damaged": fixed - damaged,
                "prediction_changed": changed,
                "prediction_agreement": 1.0 - changed / len(pairs) if pairs else None,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

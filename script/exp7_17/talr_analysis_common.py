#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Ea-e])\)?\s*\}"),
    re.compile(
        r"(?:final\s+)?(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Ea-e])\)?",
        re.I,
    ),
    re.compile(r"(?:^|\n)\s*\(?([A-Ea-e])\)?\s*$"),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_options(options: str) -> dict[str, str]:
    text = str(options or "")
    matches = list(
        re.finditer(
            r"(?:^|\s|\n)(?:\(([A-Ea-e])\)|([A-Ea-e])[\.:)])\s*",
            text,
        )
    )
    parsed = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parsed[(match.group(1) or match.group(2)).upper()] = text[
            match.end() : end
        ].strip()
    return parsed


def normalize_gold(value: str, options: str = "") -> str | None:
    text = str(value or "").strip()
    match = re.search(r"(?:^|\()\s*([A-Ea-e])\s*\)?(?:\s|$)", text)
    if match:
        return match.group(1).upper()
    if text.casefold() in {"yes", "true"}:
        return "A"
    if text.casefold() in {"no", "false"}:
        return "B"
    for letter, body in parse_options(options).items():
        if text.casefold() == body.casefold():
            return letter
    return None


def answer_region(text: str) -> str:
    markers = list(
        re.finditer(r"(?:final\s+)?(?:answer|choice)\s*[:.]", text or "", re.I)
    )
    return (text or "")[markers[-1].start() :] if markers else (text or "")[-1800:]


def extract_prediction(text: str, options: str = "") -> str | None:
    region = answer_region(str(text or ""))
    hits = []
    for pattern in ANSWER_PATTERNS:
        hits.extend((m.start(), m.group(1).upper()) for m in pattern.finditer(region))
    if hits:
        return max(hits)[1]
    letters = re.findall(r"\b([A-E])\b", region[-220:])
    if letters:
        return letters[-1]
    option_hits = []
    for letter, body in parse_options(options).items():
        if body:
            option_hits.extend(
                (match.start(), letter)
                for match in re.finditer(re.escape(body), region, re.I)
            )
    return max(option_hits)[1] if option_hits else None


def score_row(row: dict) -> dict:
    if row.get("error_type"):
        return {
            "pred": None,
            "gold": normalize_gold(row.get("answer"), row.get("options")),
            "correct": None,
            "runtime_error": True,
            "failed_extraction": False,
        }
    gold = normalize_gold(row.get("answer"), row.get("options"))
    pred = extract_prediction(row.get("model_answer"), row.get("options"))
    return {
        "pred": pred,
        "gold": gold,
        "correct": bool(gold is not None and pred == gold),
        "runtime_error": False,
        "failed_extraction": pred is None,
    }


def explicit_answers(text: str) -> list[str]:
    hits = []
    for pattern in ANSWER_PATTERNS:
        hits.extend((m.start(), m.group(1).upper()) for m in pattern.finditer(text or ""))
    values = []
    for _, value in sorted(hits):
        if not values or values[-1] != value:
            values.append(value)
    return values


def repeat_ratio(text: str, n: int = 3) -> float:
    words = re.findall(r"\w+", str(text or "").casefold())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def trace_by_id(run_dir: Path) -> dict[str, dict]:
    rows = load_jsonl(run_dir / "token_entropy_full.jsonl")
    return {str(row.get("id")): row for row in rows}


def summarize_trace(trace: dict | None) -> dict:
    tokens = (trace or {}).get("tokens") or []
    soft_positions = [
        int(token.get("step", index))
        for index, token in enumerate(tokens)
        if token.get("mode") in {"soft", "pure_soft"}
    ]
    refinement = [
        token
        for token in tokens
        if token.get("lead_refinement_active")
        or token.get("route_signal") == "talr_windowed_refinement"
    ]
    candidates = [token for token in tokens if token.get("lead_refinement_candidate")]
    format_steps = [
        int(token.get("step", index))
        for index, token in enumerate(tokens)
        if token.get("format_cooldown_active")
    ]
    veto_steps = [
        int(token.get("step", index))
        for index, token in enumerate(tokens)
        if token.get("lead_soft_veto")
    ]
    later_soft = [step for step in soft_positions if step > 1]
    first = refinement[0] if refinement else None
    return {
        "trace_available": bool(tokens),
        "token_count": len(tokens),
        "soft_positions": soft_positions,
        "later_soft_positions": later_soft,
        "refinement_candidate_positions": [
            int(token.get("step", 0)) for token in candidates
        ],
        "refinement_positions": [
            int(token.get("step", 0)) for token in refinement
        ],
        "format_positions": format_steps,
        "veto_positions": veto_steps,
        "first_refinement_entropy": (
            float(first.get("raw_entropy")) if first and first.get("raw_entropy") is not None else None
        ),
        "first_refinement_top1": (
            float(first.get("raw_top1_prob"))
            if first and first.get("raw_top1_prob") is not None
            else None
        ),
        "first_refinement_margin": (
            float(first.get("raw_margin"))
            if first and first.get("raw_margin") is not None
            else None
        ),
    }


def paired_groups(reference: dict[str, dict], method: dict[str, dict]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for sample_id in sorted(set(reference) & set(method), key=lambda value: (len(value), value)):
        left = score_row(reference[sample_id])
        right = score_row(method[sample_id])
        if left["runtime_error"] or right["runtime_error"]:
            group = "runtime_error"
        elif not left["correct"] and right["correct"]:
            group = "fixed"
        elif left["correct"] and not right["correct"]:
            group = "damaged"
        elif left["correct"]:
            group = "both_correct"
        else:
            group = "both_wrong"
        groups[group].append(sample_id)
    return dict(groups)


def mcnemar_exact(fixed: int, damaged: int) -> float:
    discordant = fixed + damaged
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) * (0.5**discordant)
        for k in range(0, min(fixed, damaged) + 1)
    )
    return min(1.0, 2.0 * tail)


def bootstrap_delta(
    reference_flags: list[bool],
    method_flags: list[bool],
    seed: int = 42,
    draws: int = 2000,
) -> list[float]:
    if not reference_flags or len(reference_flags) != len(method_flags):
        return [0.0, 0.0]
    rng = random.Random(seed)
    count = len(reference_flags)
    values = []
    for _ in range(draws):
        indices = [rng.randrange(count) for _ in range(count)]
        values.append(
            sum(
                int(method_flags[i]) - int(reference_flags[i])
                for i in indices
            )
            / count
        )
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def stratified_take(rows: list[dict], limit: int) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row.get("subtopic") or "unknown")].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: str(row.get("id")))
    selected = []
    while len(selected) < limit and buckets:
        for key in sorted(list(buckets)):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    return selected


#!/usr/bin/env python3
"""Analyze whether matched soft interventions occur near visual evidence.

The script enriches the merged Intervention Atlas with the original question,
image, hard/soft outputs, and token windows. It deliberately separates:

1. textual evidence that an intervention occurs while visual evidence is being
   verbalized; and
2. causal answer movement toward or away from the image-grounded gold answer.

It does not claim that a hidden state is image-derived without an image
ablation/swap control.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


WORD_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "there", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "would", "image", "picture",
    "photo", "shown", "following", "option", "answer",
}

VISUAL_REFERENCE = {
    "image", "picture", "photo", "photograph", "scene", "shown", "shows",
    "showing", "depict", "depicts", "depicted", "visible", "visibly", "look",
    "looks", "looking", "see", "seen", "appears", "wearing", "view",
}

VISUAL_ATTRIBUTE = {
    "black", "blue", "brown", "green", "grey", "gray", "orange", "pink",
    "purple", "red", "white", "yellow", "dark", "light", "bright", "color",
    "colour", "shape", "round", "square", "rectangular", "large", "small",
    "tall", "short", "standing", "sitting", "walking", "running", "squatting",
    "open", "closed", "empty", "full",
}

SPATIAL_RELATION = {
    "above", "below", "behind", "beside", "between", "bottom", "center",
    "centre", "front", "inside", "left", "near", "next", "outside", "over",
    "right", "top", "under", "underneath", "adjacent", "facing", "toward",
    "towards", "opposite", "overlapping",
}

COUNTING_TERMS = {
    "count", "number", "many", "few", "single", "double", "pair", "pairs",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "first", "second", "third",
}

REASONING_TERMS = {
    "because", "therefore", "thus", "hence", "implies", "indicates", "means",
    "since", "so", "conclude", "conclusion", "compare", "calculate", "total",
    "difference", "ratio", "likely", "must", "cannot",
}

STRUCTURAL_TERMS = {
    "think", "assistant", "analysis", "okay", "alright", "answer",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def words(text: str) -> list[str]:
    return [item.lower() for item in WORD_RE.findall(text or "")]


def content_words(text: str) -> set[str]:
    return {
        item for item in words(text)
        if item not in STOPWORDS and len(item) > 1 and not item.isdigit()
    }


def load_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row.get(key)): row for row in read_jsonl(path) if row.get(key) is not None}


def index_sources(roots: list[Path]) -> dict[str, dict[Any, dict[str, Any]]]:
    data_by_event: dict[str, dict[str, Any]] = {}
    base_results: dict[str, dict[str, Any]] = {}
    base_traces: dict[str, dict[str, Any]] = {}
    treatment_results: dict[tuple[str, str], dict[str, Any]] = {}
    treatment_traces: dict[tuple[str, str], dict[str, Any]] = {}

    for root in roots:
        for complete in root.rglob("SHARD_COMPLETE"):
            shard = complete.parent
            data_by_event.update(load_map(shard / "event_dataset.jsonl", "id"))
            base_results.update(load_map(shard / "hard_baseline" / "results.jsonl", "id"))
            base_traces.update(load_map(shard / "hard_baseline" / "token_entropy_full.jsonl", "id"))
            for treatment in ("contracted_soft_l095", "pure_soft_l100"):
                for key, value in load_map(shard / treatment / "results.jsonl", "id").items():
                    treatment_results[(key, treatment)] = value
                for key, value in load_map(
                    shard / treatment / "token_entropy_full.jsonl", "id"
                ).items():
                    treatment_traces[(key, treatment)] = value

    return {
        "data": data_by_event,
        "base_results": base_results,
        "base_traces": base_traces,
        "treatment_results": treatment_results,
        "treatment_traces": treatment_traces,
    }


def token_slice(tokens: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    if not tokens:
        return []
    return tokens[max(0, start):min(len(tokens), end)]


def token_text(tokens: list[dict[str, Any]]) -> str:
    return "".join(str(token.get("token_text") or "") for token in tokens)


def classify_context(
    event_step: int,
    tokens: list[dict[str, Any]],
    question: str,
    options: str,
) -> dict[str, Any]:
    near = token_slice(tokens, event_step - 4, event_step + 9)
    tight = token_slice(tokens, event_step - 2, event_step + 5)
    near_text = token_text(near)
    near_words = set(words(near_text))
    query_words = content_words(f"{question} {options}")
    overlap = sorted(near_words & query_words)
    relation_trace = any(bool(token.get("is_relation_token")) for token in near)

    visual_reference = bool(near_words & VISUAL_REFERENCE)
    visual_attribute = bool(near_words & VISUAL_ATTRIBUTE)
    spatial_relation = relation_trace or bool(near_words & SPATIAL_RELATION)
    counting = bool(near_words & COUNTING_TERMS)
    reasoning = bool(near_words & REASONING_TERMS)
    answer_zone = any(bool(token.get("token_is_answer_marker")) for token in near)
    marker_only = event_step <= 2 or (
        near_words
        and near_words.issubset(STRUCTURAL_TERMS)
    )

    score = 0
    score += 2 if visual_reference else 0
    score += 2 if spatial_relation else 0
    score += 1 if visual_attribute else 0
    score += 1 if counting else 0
    score += min(2, len(overlap))

    if marker_only:
        region = "structural"
        confidence = "low"
        multimodal_related = False
    elif answer_zone:
        region = "answer_zone"
        confidence = "low"
        multimodal_related = False
    elif score >= 3:
        region = "visual_evidence"
        confidence = "high"
        multimodal_related = True
    elif score >= 1:
        region = "mixed_visual_reasoning"
        confidence = "medium"
        multimodal_related = True
    elif reasoning:
        region = "abstract_reasoning"
        confidence = "low"
        multimodal_related = False
    else:
        region = "other"
        confidence = "low"
        multimodal_related = False

    return {
        "event_context": token_text(tight),
        "event_context_wide": near_text,
        "context_region": region,
        "multimodal_related": multimodal_related,
        "multimodal_confidence": confidence,
        "multimodal_score": score,
        "query_overlap": overlap,
        "visual_reference": visual_reference,
        "visual_attribute": visual_attribute,
        "spatial_relation": spatial_relation,
        "counting": counting,
        "reasoning": reasoning,
        "answer_zone": answer_zone,
    }


def classify_divergence(
    step: int | None,
    base_tokens: list[dict[str, Any]],
    treatment_tokens: list[dict[str, Any]],
    question: str,
    options: str,
) -> dict[str, Any]:
    if step is None:
        return {
            "divergence_context_base": "",
            "divergence_context_treatment": "",
            "divergence_visual": False,
            "divergence_region": "no_divergence",
        }
    base = classify_context(step, base_tokens, question, options)
    treatment = classify_context(step, treatment_tokens, question, options)
    return {
        "divergence_context_base": base["event_context"],
        "divergence_context_treatment": treatment["event_context"],
        "divergence_visual": bool(
            base["multimodal_related"] or treatment["multimodal_related"]
        ),
        "divergence_region": (
            "visual_or_multimodal"
            if base["multimodal_related"] or treatment["multimodal_related"]
            else "nonvisual_or_structural"
        ),
    }


def causal_direction(row: dict[str, Any]) -> str:
    if row.get("fixed"):
        return "toward_gold"
    if row.get("damaged"):
        return "away_from_gold"
    if row.get("answer_changed"):
        return "lateral_wrong_or_same_correctness"
    return "unchanged"


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    fixed = sum(bool(row.get("fixed")) for row in rows)
    damaged = sum(bool(row.get("damaged")) for row in rows)
    changed = sum(bool(row.get("answer_changed")) for row in rows)
    divergent = sum(row.get("first_divergence_step") is not None for row in rows)
    visual_divergent = sum(bool(row.get("divergence_visual")) for row in rows)
    return {
        "events": total,
        "fixed": fixed,
        "damaged": damaged,
        "net_fixed_minus_damaged": fixed - damaged,
        "net_per_100_events": round(100.0 * (fixed - damaged) / total, 3) if total else None,
        "answer_changed": changed,
        "answer_changed_rate": round(changed / total, 5) if total else None,
        "divergent": divergent,
        "visual_divergent": visual_divergent,
        "visual_share_of_divergent": (
            round(visual_divergent / divergent, 5) if divergent else None
        ),
    }


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: dict[str, Any] = {}
    for group_key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        name = " / ".join(str(part) for part in group_key)
        output[name] = summarize_group(items)
    return output


def select_cases(rows: list[dict[str, Any]], limit_per_bucket: int = 2) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        direction = row["causal_direction"]
        if direction not in {"toward_gold", "away_from_gold", "lateral_wrong_or_same_correctness"}:
            continue
        if not row["multimodal_related"]:
            continue
        buckets[(row["dataset"], row["treatment"], direction)].append(row)

    selected: list[dict[str, Any]] = []
    used_samples: set[tuple[str, str, str]] = set()
    for bucket, items in sorted(buckets.items()):
        ranked = sorted(
            items,
            key=lambda row: (
                bool(row.get("divergence_visual")),
                int(row.get("multimodal_score") or 0),
                -int(row.get("event_step") or 0),
            ),
            reverse=True,
        )
        count = 0
        for row in ranked:
            sample_key = (row["dataset"], row["original_id"], row["treatment"])
            if sample_key in used_samples:
                continue
            used_samples.add(sample_key)
            selected.append(row)
            count += 1
            if count >= limit_per_bucket:
                break
    return selected


def summarize_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["treatment"], row["original_id"])].append(row)

    by_dataset_treatment: dict[tuple[str, str], collections.Counter[str]] = (
        collections.defaultdict(collections.Counter)
    )
    for (dataset, treatment, _), items in grouped.items():
        has_fixed = any(item.get("fixed") for item in items)
        has_damaged = any(item.get("damaged") for item in items)
        has_changed = any(item.get("answer_changed") for item in items)
        if has_fixed and has_damaged:
            bucket = "mixed_fixed_and_damaged"
        elif has_fixed:
            bucket = "beneficial_only"
        elif has_damaged:
            bucket = "harmful_only"
        elif has_changed:
            bucket = "changed_without_correctness_flip"
        else:
            bucket = "inert"
        by_dataset_treatment[(dataset, treatment)][bucket] += 1

    output: dict[str, Any] = {}
    for key, counter in sorted(by_dataset_treatment.items()):
        output[" / ".join(key)] = {
            "samples": sum(counter.values()),
            **dict(counter),
        }
    return output


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output_dir / "selected_images"
    assets_dir.mkdir(exist_ok=True)

    source = index_sources(args.roots)
    canonical = list(read_jsonl(args.atlas))
    enriched: list[dict[str, Any]] = []
    missing = collections.Counter()

    for row in canonical:
        event_id = str(row["event_id"])
        original_id = str(row["original_id"])
        treatment = str(row["treatment"])
        data = source["data"].get(event_id)
        base_result = source["base_results"].get(original_id)
        base_trace = source["base_traces"].get(original_id)
        treatment_result = source["treatment_results"].get((event_id, treatment))
        treatment_trace = source["treatment_traces"].get((event_id, treatment))

        for name, value in (
            ("data", data),
            ("base_result", base_result),
            ("base_trace", base_trace),
            ("treatment_result", treatment_result),
            ("treatment_trace", treatment_trace),
        ):
            if value is None:
                missing[name] += 1

        sample_record = data or base_result or {}
        question = str(sample_record.get("question") or "")
        options = str(sample_record.get("options") or "")
        base_tokens = list((base_trace or {}).get("tokens") or [])
        treatment_tokens = list((treatment_trace or {}).get("tokens") or [])
        event_step = int(row.get("event_step") or 0)
        context = classify_context(event_step, base_tokens, question, options)
        divergence = classify_divergence(
            row.get("first_divergence_step"),
            base_tokens,
            treatment_tokens,
            question,
            options,
        )

        item = dict(row)
        item.update(
            {
                "question": question,
                "options": options,
                "image": sample_record.get("image"),
                "base_model_answer": (base_result or {}).get("model_answer"),
                "treatment_model_answer": (treatment_result or {}).get("model_answer"),
                "causal_direction": causal_direction(row),
            }
        )
        item.update(context)
        item.update(divergence)
        enriched.append(item)

    selected = select_cases(enriched)
    selected_compact: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        image = Path(str(row.get("image") or ""))
        image_copy = None
        if image.is_file():
            image_copy = assets_dir / (
                f"{index:02d}_{safe_name(row['dataset'])}_"
                f"{safe_name(row['original_id'])}{image.suffix.lower()}"
            )
            shutil.copy2(image, image_copy)
        compact = {
            key: row.get(key)
            for key in (
                "event_id", "original_id", "dataset", "subtopic", "event_type",
                "event_step", "treatment", "gold", "base_pred", "treatment_pred",
                "causal_direction", "question", "options", "image",
                "event_context", "event_context_wide", "context_region",
                "multimodal_score", "query_overlap", "divergence_context_base",
                "divergence_context_treatment", "divergence_visual",
                "first_divergence_step", "base_model_answer",
                "treatment_model_answer",
            )
        }
        compact["local_image"] = str(image_copy) if image_copy else None
        selected_compact.append(compact)

    summary = {
        "scope": {
            "events": len(enriched),
            "unique_samples": len({row["original_id"] for row in enriched}),
            "datasets": dict(collections.Counter(row["dataset"] for row in enriched)),
            "treatments": dict(collections.Counter(row["treatment"] for row in enriched)),
            "missing_source_records": dict(missing),
        },
        "overall_by_context_region": aggregate(enriched, ["context_region"]),
        "overall_by_multimodal_related": aggregate(enriched, ["multimodal_related"]),
        "by_dataset_and_multimodal": aggregate(
            enriched, ["dataset", "multimodal_related"]
        ),
        "by_treatment_and_multimodal": aggregate(
            enriched, ["treatment", "multimodal_related"]
        ),
        "by_multimodal_and_causal_direction": aggregate(
            enriched, ["multimodal_related", "causal_direction"]
        ),
        "by_multimodal_direction_and_divergence": aggregate(
            enriched,
            ["multimodal_related", "causal_direction", "divergence_region"],
        ),
        "by_event_type_and_multimodal": aggregate(
            enriched, ["event_type", "multimodal_related"]
        ),
        "by_dataset_event_type": aggregate(
            enriched, ["dataset", "event_type"]
        ),
        "by_dataset_event_type_and_multimodal": aggregate(
            enriched, ["dataset", "event_type", "multimodal_related"]
        ),
        "gold_direction_at_multimodal_events": aggregate(
            [row for row in enriched if row["multimodal_related"]],
            ["dataset", "treatment"],
        ),
        "sample_level_at_multimodal_events": summarize_samples(
            [row for row in enriched if row["multimodal_related"]]
        ),
        "selected_case_count": len(selected_compact),
    }

    write_jsonl(args.output_dir / "multimodal_event_labels.jsonl", enriched)
    write_jsonl(args.output_dir / "selected_cases.jsonl", selected_compact)
    with (args.output_dir / "multimodal_event_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary["scope"], ensure_ascii=False))
    print(f"selected_cases={len(selected_compact)}")


if __name__ == "__main__":
    main()

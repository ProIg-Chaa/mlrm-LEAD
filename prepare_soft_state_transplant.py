#!/usr/bin/env python3
"""Prepare a balanced manifest for image-source soft-state transplants."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(row[key]): row
        for row in read_jsonl(path)
        if row.get(key) is not None
    }


def text_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def index_sources(roots: list[Path]) -> dict[str, Any]:
    data: dict[str, dict[str, Any]] = {}
    base_results: dict[str, dict[str, Any]] = {}
    base_traces: dict[str, dict[str, Any]] = {}
    true_results: dict[str, dict[str, Any]] = {}
    for root in roots:
        for complete in root.rglob("SHARD_COMPLETE"):
            shard = complete.parent
            data.update(load_map(shard / "event_dataset.jsonl", "id"))
            base_results.update(
                load_map(shard / "hard_baseline" / "results.jsonl", "id")
            )
            base_traces.update(
                load_map(
                    shard / "hard_baseline" / "token_entropy_full.jsonl", "id"
                )
            )
            true_results.update(
                load_map(
                    shard / "contracted_soft_l095" / "results.jsonl", "id"
                )
            )
    return {
        "data": data,
        "base_results": base_results,
        "base_traces": base_traces,
        "true_results": true_results,
    }


def outcome_class(row: dict[str, Any]) -> str | None:
    if row.get("fixed"):
        return "fixed"
    if row.get("damaged"):
        return "damaged"
    if row.get("answer_changed"):
        return "lateral_wrong"
    if row.get("first_divergence_step") is not None:
        return "unchanged_divergent"
    return None


def pick_balanced(
    rows: list[dict[str, Any]],
    per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    targets = ("fixed", "damaged", "lateral_wrong", "unchanged_divergent")
    datasets = ("mmvp", "realworldqa", "visulogic", "vstar")
    selected: list[dict[str, Any]] = []
    used_samples: set[str] = set()

    for target in targets:
        by_dataset: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            if outcome_class(row) != target:
                continue
            by_dataset[str(row["dataset"])].append(row)
        for dataset in datasets:
            rng.shuffle(by_dataset[dataset])
            by_dataset[dataset].sort(
                key=lambda row: (
                    int(row.get("multimodal_score") or 0),
                    row.get("event_type") in {"fixed_4", "fixed_8", "fixed_16", "fixed_32"},
                ),
                reverse=True,
            )

        while sum(outcome_class(row) == target for row in selected) < per_class:
            progress = False
            for dataset in datasets:
                while by_dataset[dataset]:
                    row = by_dataset[dataset].pop(0)
                    if row["original_id"] in used_samples:
                        continue
                    selected.append(row)
                    used_samples.add(row["original_id"])
                    progress = True
                    break
                if sum(outcome_class(row) == target for row in selected) >= per_class:
                    break
            if not progress:
                raise RuntimeError(f"Not enough candidates for {target}")
    return selected


def choose_donor(
    receiver: dict[str, Any],
    pool: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    candidates = [
        item
        for item in pool
        if item["original_id"] != receiver["original_id"]
        and item.get("image")
        and Path(str(item["image"])).is_file()
        and str(item.get("answer")) != str(receiver.get("answer"))
        and (
            not receiver.get("subtopic")
            or not item.get("subtopic")
            or item.get("subtopic") == receiver.get("subtopic")
        )
    ]
    if not candidates:
        candidates = [
            item
            for item in pool
            if item["original_id"] != receiver["original_id"]
            and item.get("image")
            and Path(str(item["image"])).is_file()
            and str(item.get("answer")) != str(receiver.get("answer"))
        ]
    if not candidates:
        candidates = [
            item
            for item in pool
            if item["original_id"] != receiver["original_id"]
            and item.get("image")
            and Path(str(item["image"])).is_file()
        ]
    if not candidates:
        raise RuntimeError(f"No donor found for {receiver['original_id']}")
    candidates.sort(key=lambda item: item["original_id"])
    return candidates[rng.randrange(len(candidates))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-atlas", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = index_sources(args.roots)
    candidates = [
        row
        for row in read_jsonl(args.enriched_atlas)
        if row.get("treatment") == "contracted_soft_l095"
        and row.get("prefix_match")
        and row.get("multimodal_related")
        and row.get("event_type") not in {"fixed_1", "fixed_2"}
        and not row.get("base_runtime_error")
        and not row.get("treatment_runtime_error")
    ]
    selected = pick_balanced(candidates, args.per_class, args.seed)

    sample_pool: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_pool: set[str] = set()
    for original_id, result in source["base_results"].items():
        dataset = original_id.split("::", 1)[0]
        if original_id in seen_pool or not result.get("image"):
            continue
        seen_pool.add(original_id)
        sample_pool[dataset].append(
            {
                "original_id": original_id,
                "image": result.get("image"),
                "answer": result.get("answer"),
                "subtopic": result.get("subtopic"),
            }
        )

    rng = random.Random(args.seed)
    manifest: list[dict[str, Any]] = []
    for row in selected:
        event_id = str(row["event_id"])
        original_id = str(row["original_id"])
        event_step = int(row["event_step"])
        base = source["base_results"].get(original_id)
        trace = source["base_traces"].get(original_id)
        true_result = source["true_results"].get(event_id)
        event_data = source["data"].get(event_id)
        sample = event_data or base
        if not all((base, trace, true_result, sample)):
            raise RuntimeError(f"Incomplete source records for {event_id}")
        tokens = list(trace.get("tokens") or [])
        prefix_ids = [int(token["token_id"]) for token in tokens[: event_step + 1]]
        if len(prefix_ids) != event_step + 1:
            raise RuntimeError(f"Short baseline prefix for {event_id}")

        receiver = {
            "original_id": original_id,
            "image": sample.get("image"),
            "answer": sample.get("answer"),
            "subtopic": sample.get("subtopic"),
        }
        donor = choose_donor(receiver, sample_pool[row["dataset"]], rng)
        manifest.append(
            {
                "event_id": event_id,
                "original_id": original_id,
                "dataset": row["dataset"],
                "selection_class": outcome_class(row),
                "event_type": row["event_type"],
                "event_step": event_step,
                "multimodal_score": row.get("multimodal_score"),
                "event_context": row.get("event_context"),
                "question": sample.get("question"),
                "options": sample.get("options"),
                "answer": sample.get("answer"),
                "subtopic": sample.get("subtopic"),
                "true_image": sample.get("image"),
                "donor_original_id": donor["original_id"],
                "donor_image": donor["image"],
                "donor_answer": donor.get("answer"),
                "prefix_ids": prefix_ids,
                "expected_hard_text_sha256": text_hash(base.get("model_answer")),
                "expected_true_l095_text_sha256": text_hash(
                    true_result.get("model_answer")
                ),
                "expected_hard_pred": row.get("base_pred"),
                "expected_true_l095_pred": row.get("treatment_pred"),
                "gold": row.get("gold"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "samples": len(manifest),
        "classes": dict(
            collections.Counter(item["selection_class"] for item in manifest)
        ),
        "datasets": dict(collections.Counter(item["dataset"] for item in manifest)),
        "seed": args.seed,
    }
    with args.output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

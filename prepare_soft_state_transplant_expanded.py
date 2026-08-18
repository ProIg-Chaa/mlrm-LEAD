#!/usr/bin/env python3
"""Prepare an outcome-agnostic, checkpoint-paired transplant manifest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


EVENT_TYPES = (
    "fixed_1",
    "fixed_2",
    "fixed_4",
    "fixed_8",
    "fixed_16",
    "fixed_32",
    "entropy_top1",
    "random_control",
)
DATASETS = ("mmvp", "realworldqa", "visulogic", "vstar")


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


def index_sources(roots: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed = {
        "data": {},
        "base_results": {},
        "base_traces": {},
        "true_results": {},
    }
    for root in roots:
        for complete in root.rglob("SHARD_COMPLETE"):
            shard = complete.parent
            indexed["data"].update(load_map(shard / "event_dataset.jsonl", "id"))
            indexed["base_results"].update(
                load_map(shard / "hard_baseline" / "results.jsonl", "id")
            )
            indexed["base_traces"].update(
                load_map(
                    shard / "hard_baseline" / "token_entropy_full.jsonl", "id"
                )
            )
            indexed["true_results"].update(
                load_map(
                    shard / "contracted_soft_l095" / "results.jsonl", "id"
                )
            )
    return indexed


def choose_samples(
    events: dict[str, dict[str, dict[str, Any]]],
    samples_per_dataset: int,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    selected = {}
    for dataset in DATASETS:
        eligible = [
            original_id
            for original_id, event_map in events.items()
            if original_id.startswith(dataset + "::")
            and all(event_type in event_map for event_type in EVENT_TYPES)
        ]
        buckets: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
        for original_id in eligible:
            representative = events[original_id][EVENT_TYPES[0]]
            buckets[
                (
                    str(representative.get("subtopic") or ""),
                    str(representative.get("gold") or ""),
                )
            ].append(original_id)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        picked = []
        while buckets and len(picked) < samples_per_dataset:
            for key in sorted(list(buckets)):
                if buckets[key] and len(picked) < samples_per_dataset:
                    picked.append(buckets[key].pop())
                if not buckets[key]:
                    del buckets[key]
        if len(picked) != samples_per_dataset:
            raise RuntimeError(
                f"{dataset}: requested {samples_per_dataset}, found {len(picked)}"
            )
        selected[dataset] = picked
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
            or receiver.get("subtopic") == item.get("subtopic")
        )
    ]
    if not candidates:
        candidates = [
            item
            for item in pool
            if item["original_id"] != receiver["original_id"]
            and item.get("image")
            and Path(str(item["image"])).is_file()
        ]
    candidates.sort(key=lambda item: item["original_id"])
    if not candidates:
        raise RuntimeError(f"No donor for {receiver['original_id']}")
    return candidates[rng.randrange(len(candidates))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=32)
    parser.add_argument("--seed", type=int, default=420731)
    args = parser.parse_args()

    source = index_sources(args.roots)
    valid_atlas = {}
    for row in read_jsonl(args.atlas):
        if row.get("treatment") != "contracted_soft_l095":
            continue
        if row.get("event_type") not in EVENT_TYPES or not row.get("prefix_match"):
            continue
        if row.get("base_runtime_error") or row.get("treatment_runtime_error"):
            continue
        event_id = str(row["event_id"])
        original_id = str(row["original_id"])
        if (
            event_id not in source["data"]
            or event_id not in source["true_results"]
            or original_id not in source["base_results"]
            or original_id not in source["base_traces"]
        ):
            continue
        valid_atlas[event_id] = row

    by_sample: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in valid_atlas.values():
        by_sample[str(row["original_id"])][str(row["event_type"])] = row
    selected = choose_samples(by_sample, args.samples_per_dataset, args.seed)

    donor_pool: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for original_id, result in source["base_results"].items():
        donor_pool[original_id.split("::", 1)[0]].append(
            {
                "original_id": original_id,
                "image": result.get("image"),
                "answer": result.get("answer"),
                "subtopic": result.get("subtopic"),
            }
        )

    rng = random.Random(args.seed + 1)
    manifest = []
    for dataset in DATASETS:
        for original_id in selected[dataset]:
            for event_type in EVENT_TYPES:
                row = by_sample[original_id][event_type]
                event_id = str(row["event_id"])
                event_step = int(row["event_step"])
                sample = source["data"][event_id]
                base = source["base_results"][original_id]
                trace = source["base_traces"][original_id]
                true_result = source["true_results"][event_id]
                prefix_ids = [
                    int(token["token_id"])
                    for token in list(trace.get("tokens") or [])[: event_step + 1]
                ]
                if len(prefix_ids) != event_step + 1:
                    raise RuntimeError(f"Short prefix for {event_id}")
                receiver = {
                    "original_id": original_id,
                    "image": sample.get("image"),
                    "answer": sample.get("answer"),
                    "subtopic": sample.get("subtopic"),
                }
                donor = choose_donor(receiver, donor_pool[dataset], rng)
                manifest.append(
                    {
                        "event_id": event_id,
                        "original_id": original_id,
                        "dataset": dataset,
                        "selection_class": "outcome_agnostic",
                        "event_type": event_type,
                        "event_step": event_step,
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
                        "expected_hard_correct": row.get("base_correct"),
                        "expected_true_l095_correct": row.get("treatment_correct"),
                        "gold": row.get("gold"),
                    }
                )

    expected = len(DATASETS) * args.samples_per_dataset * len(EVENT_TYPES)
    if len(manifest) != expected:
        raise RuntimeError(f"Expected {expected} events, got {len(manifest)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "events": len(manifest),
        "unique_samples": len({row["original_id"] for row in manifest}),
        "samples_per_dataset": args.samples_per_dataset,
        "event_types": list(EVENT_TYPES),
        "dataset_events": dict(collections.Counter(row["dataset"] for row in manifest)),
        "seed": args.seed,
        "selection": "outcome agnostic; paired checkpoints within each sample",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

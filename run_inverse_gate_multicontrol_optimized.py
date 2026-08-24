#!/usr/bin/env python3
"""Collect the exact locked multi-control gate with duplicate probes removed."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lead import get_math_symbols_ids
from lead.generation_utils import generate_lead
from run_talr_visual_residual_expansion import (
    create_mean_mask,
    patch_segmented_vision_sdpa,
    prepare,
    talr_kwargs,
)


HORIZONS = (4, 8)
BRANCHES = ("baseline", "action")
CONTEXTS = ("true", "mask", "swap1", "swap2", "swap3")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def run_probe(
    model,
    processor,
    tokenizer,
    sample: Dict[str, Any],
    image_path: str,
    prefix_ids: List[int],
    event_step: int,
    gold_choice: str,
    math_ids_tensor,
):
    inputs, _ = prepare(
        processor, sample, image_path, next(model.parameters()).device
    )
    trace: List[Dict[str, Any]] = []
    kwargs = talr_kwargs(math_ids_tensor, event_step + 1)
    kwargs.update({
        "forced_prefix_ids": prefix_ids,
        "token_trace": trace,
        "trace_topk": 0,
        "trace_event_steps": [event_step],
        "trace_route_override_step": event_step,
        "trace_route_override_kind": "hard",
        "trace_forced_answer_probe": True,
        "trace_probe_gold_choice": gold_choice,
        "trace_probe_choice_case": "upper",
    })
    started = time.perf_counter()
    with torch.no_grad():
        generate_lead(model, tokenizer, **inputs, **kwargs)
    elapsed = time.perf_counter() - started
    matches = [record for record in trace if int(record.get("step", -1)) == event_step]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one trace record at step %d, got %d"
            % (event_step, len(matches))
        )
    probe = matches[0].get("forced_answer_probe")
    if not probe or not probe.get("available", False):
        raise RuntimeError("Forced-answer probe unavailable: %r" % (probe,))
    return probe, matches[0], elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for index, row in enumerate(read_jsonl(args.manifest)) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[:args.limit]
    results_path = args.output_dir / "optimized_gate_probe_results.jsonl"
    completed = {(str(row["dataset"]), str(row["id"])) for row in read_jsonl(results_path) if row.get("error_type") is None} if results_path.is_file() else set()

    patch_segmented_vision_sdpa()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, attn_implementation="sdpa", device_map="auto", torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    math_ids = get_math_symbols_ids(tokenizer)
    device = next(model.parameters()).device
    math_ids_tensor = torch.tensor(list(math_ids), device=device) if math_ids else None
    masks_dir = args.output_dir / "masks"

    for index, row in enumerate(rows, start=1):
        key = (str(row["dataset"]), str(row["id"]))
        if key in completed:
            continue
        print("[%d/%d] %s:%s eligible=%s" % (index, len(rows), key[0], key[1], row["probe_eligible"]), flush=True)
        base = {key: value for key, value in row.items() if key not in {"baseline_generated_token_ids", "action_generated_token_ids"}}
        if not row["probe_eligible"]:
            append_jsonl(results_path, dict(base, probes={}, latency_seconds={}, probe_count=0, error_type=None))
            continue
        sample = {"id": row["id"], "image": row["image"], "question": row["question"], "options": row.get("options"), "answer": row.get("answer")}
        event_step = int(row["event_step"])
        swaps = row["swaps"]
        context_paths = {
            "true": str(row["image"]),
            "mask": create_mean_mask(str(row["image"]), masks_dir / ("%s_%s.png" % key)),
            "swap1": str(swaps[0]["image"]), "swap2": str(swaps[1]["image"]), "swap3": str(swaps[2]["image"]),
        }
        trajectories = {
            "baseline": [int(value) for value in row["baseline_generated_token_ids"]],
            "action": [int(value) for value in row["action_generated_token_ids"]],
        }
        try:
            probes: Dict[str, Any] = {}
            latencies: Dict[str, float] = {}
            for horizon in HORIZONS:
                for branch in BRANCHES:
                    end = event_step + 1 + horizon
                    prefix_ids = trajectories[branch][:end]
                    if len(prefix_ids) != end:
                        raise RuntimeError("Missing horizon %d prefix for %s" % (horizon, branch))
                    checkpoint_step = len(prefix_ids) - 1
                    for context in CONTEXTS:
                        name = "h%d_%s_%s" % (horizon, branch, context)
                        probe, _, elapsed = run_probe(
                            model, processor, tokenizer, sample,
                            context_paths[context], prefix_ids, checkpoint_step,
                            row["gold_choice"], math_ids_tensor,
                        )
                        probes[name] = probe
                        latencies[name] = elapsed
            output = dict(base, probes=probes, latency_seconds=latencies, probe_count=len(probes), error_type=None)
        except Exception as exc:
            output = dict(base, error_type=type(exc).__name__, error_message=str(exc))
        append_jsonl(results_path, output)
        gc.collect()
        torch.cuda.empty_cache()

    config = {
        "model": str(args.model), "manifest": str(args.manifest), "rows": len(rows),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "horizons": list(HORIZONS), "branches": list(BRANCHES), "contexts": list(CONTEXTS),
        "unique_probes_per_eligible_sample": 20, "legacy_repeated_probes_per_sample": 36,
        "probe_reduction": 16 / 36.0,
        "policy": "reject iff median swap transient >= 0 and mask transient >= 0",
        "optimization": "deduplicate repeated true/mask probes; branch semantics unchanged",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [row for row in read_jsonl(results_path) if row.get("error_type")]
    if failures:
        raise RuntimeError("%d optimized gate samples failed" % len(failures))
    (args.output_dir / "RUN_COMPLETE").touch()
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

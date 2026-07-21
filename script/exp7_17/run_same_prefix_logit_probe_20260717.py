#!/usr/bin/env python3
"""Measure step-2 distribution differences under an identical visible prefix."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lead import format_prompt_from_sample, load_dataset
from lead.generation_utils import generate_lead
from lead.inference import prepare_inputs


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trace_tokens(path: Path) -> dict[str, list[int]]:
    return {
        str(row["id"]): [int(token["token_id"]) for token in row.get("tokens", [])]
        for row in read_jsonl(path)
    }


def select_rows(rows: list[dict], traces: dict[str, list[int]], limit: int) -> list[dict]:
    buckets: dict[tuple[str, str], deque[dict]] = defaultdict(deque)
    for row in sorted(rows, key=lambda x: (str(x.get("subtopic", "unknown")), str(x.get("answer", "")), str(x["id"]))):
        if len(traces.get(str(row["id"]), [])) >= 2:
            buckets[(str(row.get("subtopic", "unknown")), str(row.get("answer", "")))].append(row)
    selected: list[dict] = []
    while buckets and len(selected) < limit:
        for key in sorted(list(buckets)):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].popleft())
            if not buckets[key]:
                del buckets[key]
    return selected


def run_route(model, tokenizer, processor, sample: dict, prefix: list[int], hard_boundary: bool, args: argparse.Namespace) -> tuple[list[int], dict, torch.Tensor]:
    prompt = format_prompt_from_sample(sample, use_cot=(args.cot_prompt_mode == "cot"))
    messages = [{"role": "user", "content": [{"type": "image", "image": sample["image"]}, {"type": "text", "text": prompt}]}]
    inputs = prepare_inputs(processor, messages, torch.device(args.device))
    prompt_len = int(inputs["input_ids"].shape[1])
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(args.device)
    sink: list[dict] = []
    kwargs = dict(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_new_tokens=args.max_new_tokens, do_sample=False,
        alpha_0=0.4, max_switch_count=5, window_size=128,
        lead_initial_transition_only=True,
        lead_force_initial_transition_step1=True,
        lead_transition_source="soft",
        lead_transition_anchor="end_thinking",
        lead_transition_beta0=0.7,
        forced_prefix_ids=prefix,
        capture_logits_steps=[2],
        capture_logits_sink=sink,
        **inputs,
    )
    if hard_boundary:
        kwargs["lead_initial_transition_hard_boundary_only"] = True
    with torch.no_grad():
        output = generate_lead(model, tokenizer, **kwargs)[0]
    generated = [int(token) for token in output[prompt_len:].tolist()]
    geometry = next((row for row in sink if row["kind"] == "step0_geometry"), None)
    logits = next((row["probs"] for row in sink if row["kind"] == "logits" and row["step"] == 2), None)
    if geometry is None or logits is None:
        raise RuntimeError("Missing capture record from generator")
    return generated, geometry, logits


def metrics(hard: torch.Tensor, transition: torch.Tensor) -> dict:
    eps = 1e-12
    hard, transition = hard.float(), transition.float()
    midpoint = 0.5 * (hard + transition)
    js = 0.5 * ((hard * (hard.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum() + (transition * (transition.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum())
    k = min(20, hard.numel())
    hard_top = torch.topk(hard, k=k)
    transition_top = torch.topk(transition, k=k)
    overlap = len(set(hard_top.indices.tolist()) & set(transition_top.indices.tolist())) / k
    return {
        "js_divergence": float(js.item()),
        "top1_hard": int(hard_top.indices[0].item()),
        "top1_transition": int(transition_top.indices[0].item()),
        "top1_same": bool(hard_top.indices[0].item() == transition_top.indices[0].item()),
        "top20_overlap": overlap,
        "top1_probability_hard": float(hard_top.values[0].item()),
        "top1_probability_transition": float(transition_top.values[0].item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cot-trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cot-prompt-mode", default="orign")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    root, out = Path(args.root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces = trace_tokens(Path(args.cot_trace))
    rows = select_rows(load_dataset(args.dataset, str(root / "data")), traces, args.limit)
    (out / "selection.json").write_text(json.dumps([{"id": str(row["id"]), "answer": row.get("answer"), "subtopic": row.get("subtopic")} for row in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    records = []
    for index, sample in enumerate(rows, start=1):
        sample_id = str(sample["id"])
        prefix = traces[sample_id][:2]
        hard_tokens, hard_geometry, hard_probs = run_route(model, tokenizer, processor, sample, prefix, True, args)
        transition_tokens, transition_geometry, transition_probs = run_route(model, tokenizer, processor, sample, prefix, False, args)
        row = {
            "id": sample_id,
            "gold": sample.get("answer"),
            "subtopic": sample.get("subtopic"),
            "forced_prefix_ids": prefix,
            "hard_tokens": hard_tokens,
            "transition_tokens": transition_tokens,
            "first_free_token_same": len(hard_tokens) > 2 and len(transition_tokens) > 2 and hard_tokens[2] == transition_tokens[2],
            "hard_step0": hard_geometry,
            "transition_step0": transition_geometry,
            **metrics(hard_probs, transition_probs),
        }
        records.append(row)
        print(f"[{index}/{len(rows)}] id={sample_id} js={row['js_divergence']:.6f}", flush=True)
    with (out / "same_prefix_logit_probe.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "samples": len(records),
        "mean_js": sum(row["js_divergence"] for row in records) / len(records),
        "top1_different": sum(not row["top1_same"] for row in records),
        "mean_top20_overlap": sum(row["top20_overlap"] for row in records) / len(records),
        "first_free_token_different": sum(not row["first_free_token_same"] for row in records),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

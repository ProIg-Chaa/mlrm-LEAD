#!/usr/bin/env python3
"""Replay matched VStar samples with COT's first token forced in two routes."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lead import format_prompt_from_sample, load_dataset
from lead.generation_utils import generate_lead
from lead.inference import prepare_inputs


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choice(text: str | None) -> str | None:
    if not text:
        return None
    tail = text[-1800:]
    patterns = [
        r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}",
        r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, tail, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    letters = re.findall(r"\b([A-Da-d])\b", tail.split("</think>")[-1])
    return letters[-1].upper() if letters else None


def correct(row: dict) -> bool:
    return choice(row.get("model_answer")) == str(row.get("answer", "")).strip().upper()[:1]


def trace_tokens(rows: list[dict]) -> dict[int, list[int]]:
    return {
        int(row["id"]): [int(token["token_id"]) for token in row.get("tokens", [])]
        for row in rows
    }


def stratified_take(rows: list[dict], limit: int) -> list[dict]:
    buckets: dict[tuple[str, str], deque[dict]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: (str(item.get("subtopic", "")), str(item.get("answer", "")), int(item["id"]))):
        buckets[(str(row.get("subtopic", "")), str(row.get("answer", "")))].append(row)
    selected = []
    while buckets and len(selected) < limit:
        for key in list(sorted(buckets)):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].popleft())
            if not buckets[key]:
                del buckets[key]
    return selected


def make_selection(cot_rows: list[dict], transition_rows: list[dict], cot_tokens: dict[int, list[int]]) -> list[dict]:
    cot = {int(row["id"]): row for row in cot_rows}
    transition = {int(row["id"]): row for row in transition_rows}
    fixed, damaged, controls = [], [], []
    for sample_id in sorted(set(cot) & set(transition) & set(cot_tokens)):
        base, method = cot[sample_id], transition[sample_id]
        base_ok, method_ok = correct(base), correct(method)
        if not cot_tokens[sample_id]:
            continue
        if not base_ok and method_ok:
            fixed.append(base)
        elif base_ok and not method_ok:
            damaged.append(base)
        else:
            copy = dict(base)
            copy["control_group"] = "both_correct" if base_ok else "both_wrong"
            controls.append(copy)
    selected = []
    for label, rows, limit in [("fixed", fixed, 20), ("damaged", damaged, 20), ("control", controls, 20)]:
        for row in stratified_take(rows, limit):
            row = dict(row)
            row["selection_group"] = label if label != "control" else row["control_group"]
            selected.append(row)
    return selected


def generate(model, tokenizer, processor, sample: dict, route: str, forced_token: int, args) -> tuple[list[int], str]:
    prompt = format_prompt_from_sample(sample, use_cot=(args.cot_prompt_mode == "cot"))
    messages = [{"role": "user", "content": [{"type": "image", "image": sample["image"]}, {"type": "text", "text": prompt}]}]
    inputs = prepare_inputs(processor, messages, torch.device(args.device))
    prompt_len = int(inputs["input_ids"].shape[1])
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(args.device)
    kwargs = dict(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_new_tokens=args.max_new_tokens, do_sample=False,
        alpha_0=0.4, max_switch_count=5, window_size=128,
        forced_prefix_ids=[forced_token], **inputs,
    )
    if route == "hard":
        kwargs["lead_force_normal"] = True
    elif route == "transition":
        kwargs["lead_initial_transition_only"] = True
    else:
        raise ValueError(route)
    with torch.no_grad():
        output = generate_lead(model, tokenizer, **kwargs)[0]
    token_ids = [int(token) for token in output[prompt_len:].tolist()]
    text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return token_ids, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cot-results", required=True)
    parser.add_argument("--cot-trace", required=True)
    parser.add_argument("--transition-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cot-prompt-mode", default="orign")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cot_rows = read_jsonl(Path(args.cot_results))
    transition_rows = read_jsonl(Path(args.transition_results))
    cot_trace = trace_tokens(read_jsonl(Path(args.cot_trace)))
    selected = make_selection(cot_rows, transition_rows, cot_trace)
    data = {int(row["id"]): row for row in load_dataset(str(root / "data/vstar.jsonl"), str(root / "data"))}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    rows, sanity_failed = [], []
    for index, source_row in enumerate(selected, 1):
        sample_id = int(source_row["id"])
        forced_token = cot_trace[sample_id][0]
        sample = data[sample_id]
        hard_tokens, hard_text = generate(model, tokenizer, processor, sample, "hard", forced_token, args)
        transition_tokens, transition_text = generate(model, tokenizer, processor, sample, "transition", forced_token, args)
        sanity = hard_tokens == cot_trace[sample_id]
        if not sanity:
            sanity_failed.append(sample_id)
        first_divergence = next((i for i, pair in enumerate(zip(hard_tokens, transition_tokens)) if pair[0] != pair[1]), None)
        rows.append({
            "id": sample_id,
            "selection_group": source_row["selection_group"],
            "gold": sample.get("answer"),
            "forced_step0_token": forced_token,
            "force_sanity_pass": sanity,
            "hard_tokens": hard_tokens,
            "transition_tokens": transition_tokens,
            "hard_answer": hard_text,
            "transition_answer": transition_text,
            "hard_pred": choice(hard_text),
            "transition_pred": choice(transition_text),
            "hard_correct": choice(hard_text) == str(sample.get("answer", "")).upper()[:1],
            "transition_correct": choice(transition_text) == str(sample.get("answer", "")).upper()[:1],
            "first_divergence_step": first_divergence,
        })
        print(f"[{index}/{len(selected)}] id={sample_id} sanity={sanity}", flush=True)

    with (out_dir / "same_token_replay.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "selected": len(rows), "force_sanity_failed": sanity_failed,
        "valid": sum(row["force_sanity_pass"] for row in rows),
        "transition_differs_after_same_token": sum(
            row["force_sanity_pass"] and row["first_divergence_step"] is not None for row in rows
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

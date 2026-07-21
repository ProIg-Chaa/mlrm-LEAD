#!/usr/bin/env python3
"""Replay matched VStar samples with identical COT prefixes in two routes."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

# The runner may invoke this file outside the repository working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def generate(
    model,
    tokenizer,
    processor,
    sample: dict,
    route: str,
    forced_prefix: list[int],
    args,
) -> tuple[list[int], str]:
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
        forced_prefix_ids=forced_prefix, **inputs,
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
    parser.add_argument("--prefix-lengths", default="1,2,4")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cot_rows = read_jsonl(Path(args.cot_results))
    transition_rows = read_jsonl(Path(args.transition_results))
    cot_trace = trace_tokens(read_jsonl(Path(args.cot_trace)))
    selected = make_selection(cot_rows, transition_rows, cot_trace)
    prefix_lengths = sorted({int(value) for value in args.prefix_lengths.split(",") if value.strip()})
    if not prefix_lengths or prefix_lengths[0] <= 0:
        raise ValueError("prefix lengths must be positive")
    data = {int(row["id"]): row for row in load_dataset(str(root / "data/vstar.jsonl"), str(root / "data"))}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    rows, sanity_failed = [], []
    total_replays = sum(
        len(cot_trace[int(source_row["id"])]) >= prefix_len
        for source_row in selected
        for prefix_len in prefix_lengths
    )
    replay_index = 0
    for source_row in selected:
        sample_id = int(source_row["id"])
        sample = data[sample_id]
        for prefix_len in prefix_lengths:
            if len(cot_trace[sample_id]) < prefix_len:
                continue
            replay_index += 1
            forced_prefix = cot_trace[sample_id][:prefix_len]
            hard_tokens, hard_text = generate(
                model, tokenizer, processor, sample, "hard", forced_prefix, args
            )
            transition_tokens, transition_text = generate(
                model, tokenizer, processor, sample, "transition", forced_prefix, args
            )
            sanity = hard_tokens == cot_trace[sample_id]
            if not sanity:
                sanity_failed.append({"id": sample_id, "prefix_len": prefix_len})
            first_divergence = next(
                (i for i, pair in enumerate(zip(hard_tokens, transition_tokens)) if pair[0] != pair[1]),
                None,
            )
            rows.append({
                "id": sample_id,
                "selection_group": source_row["selection_group"],
                "prefix_len": prefix_len,
                "gold": sample.get("answer"),
                "forced_prefix_ids": forced_prefix,
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
                "diverged_after_forced_prefix": first_divergence is not None and first_divergence >= prefix_len,
            })
            print(
                f"[{replay_index}/{total_replays}] id={sample_id} prefix={prefix_len} sanity={sanity}",
                flush=True,
            )

    with (out_dir / "same_prefix_replay.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_prefix = {}
    by_group = {}
    for prefix_len in prefix_lengths:
        subset = [row for row in rows if row["prefix_len"] == prefix_len]
        valid = [row for row in subset if row["force_sanity_pass"]]
        by_prefix[str(prefix_len)] = {
            "total": len(subset),
            "valid": len(valid),
            "diverged_after_prefix": sum(row["diverged_after_forced_prefix"] for row in valid),
            "answer_disagreement": sum(row["hard_pred"] != row["transition_pred"] for row in valid),
            "transition_correct": sum(row["transition_correct"] for row in valid),
            "hard_correct": sum(row["hard_correct"] for row in valid),
        }
        for group in ["fixed", "damaged", "both_correct", "both_wrong"]:
            grouped = [row for row in valid if row["selection_group"] == group]
            by_group[f"prefix{prefix_len}:{group}"] = {
                "total": len(grouped),
                "diverged_after_prefix": sum(row["diverged_after_forced_prefix"] for row in grouped),
                "answer_disagreement": sum(row["hard_pred"] != row["transition_pred"] for row in grouped),
                "transition_correct": sum(row["transition_correct"] for row in grouped),
                "hard_correct": sum(row["hard_correct"] for row in grouped),
            }
    report = {
        "selected_samples": len(selected),
        "replays": len(rows),
        "force_sanity_failed": sanity_failed,
        "by_prefix": by_prefix,
        "by_group": by_group,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

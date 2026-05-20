#!/usr/bin/env python3
"""Replay finished COT outputs with an eager-attention observer.

This script never participates in token selection. It loads fixed model outputs
from results.jsonl, reconstructs the prompt, replays the already-generated token
sequence, and records each generated token's attention to prompt visual tokens.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lead.inference import prepare_inputs
from lead.prompts import format_prompt_from_sample
from lead.generation_utils import _build_visual_token_mask, _summarize_visual_attention


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_token_ids(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    by_id = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "id" not in rec:
                continue
            toks = rec.get("tokens") or []
            by_id[rec["id"]] = [int(t["token_id"]) for t in toks if "token_id" in t]
    return by_id


def replay_one(
    model,
    processor,
    tokenizer,
    sample,
    token_ids,
    device,
    cot_prompt_mode,
    attn_last_k,
):
    prompt = format_prompt_from_sample(
        sample,
        use_cot=(cot_prompt_mode == "step"),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    model_inputs = prepare_inputs(processor, messages, device)
    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask", None)
    prompt_len = input_ids.shape[1]
    visual_token_mask = _build_visual_token_mask(input_ids, tokenizer)
    cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
            cache_position=cache_position,
        )
    past_key_values = outputs.past_key_values
    cache_position = cache_position[-1:] + 1

    records = []
    for step, token_id in enumerate(token_ids):
        cur_ids = torch.tensor([[int(token_id)]], device=device, dtype=torch.long)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
        with torch.no_grad():
            outputs = model(
                input_ids=cur_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
                cache_position=cache_position,
            )
        past_key_values = outputs.past_key_values
        cache_position = cache_position[-1:] + 1

        summary = _summarize_visual_attention(
            attn_layers=outputs.attentions,
            visual_token_mask=visual_token_mask,
            prompt_len=prompt_len,
            attn_last_k=attn_last_k,
        )
        rec = {
            "step": step,
            "token_id": int(token_id),
            "token_text": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "visual_attn_available": bool(summary["available"][0].item()),
            "visual_attn_mass": float(summary["mass"][0].item()),
            "visual_attn_top1": float(summary["top1"][0].item()),
            "visual_attn_top4_sum": float(summary["top4_sum"][0].item()),
            "visual_attn_entropy": (
                float(summary["entropy"][0].item())
                if summary["available"][0].item()
                else None
            ),
            "visual_attn_token_count": int(summary["token_count"][0].item()),
        }
        records.append(rec)
    return records, prompt_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--token_entropy_full", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cot_prompt_mode", choices=["orign", "step"], default="orign")
    parser.add_argument("--attn_last_k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / "sidecar_visual_attention.jsonl"
    config_path = Path(args.output_dir) / "sidecar_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device(args.device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows = load_jsonl(args.results)
    if args.limit is not None:
        rows = rows[: args.limit]
    token_ids_by_id = load_token_ids(args.token_entropy_full)

    with out_path.open("w", encoding="utf-8") as out:
        for idx, sample in enumerate(rows):
            sid = sample.get("id")
            token_ids = token_ids_by_id.get(sid)
            if token_ids is None:
                token_ids = tokenizer.encode(
                    sample.get("model_answer") or "",
                    add_special_tokens=False,
                )
            try:
                records, prompt_len = replay_one(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    sample=sample,
                    token_ids=token_ids,
                    device=device,
                    cot_prompt_mode=args.cot_prompt_mode,
                    attn_last_k=args.attn_last_k,
                )
                error_type = None
                error_message = None
            except Exception as exc:
                records = []
                prompt_len = None
                error_type = type(exc).__name__
                error_message = str(exc)
                torch.cuda.empty_cache()

            out.write(json.dumps({
                "sample_index": idx,
                "id": sid,
                "answer": sample.get("answer"),
                "prompt_tokens": prompt_len,
                "output_tokens": len(token_ids),
                "error_type": error_type,
                "error_message": error_message,
                "tokens": records,
            }, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{idx + 1}/{len(rows)}] id={sid} tokens={len(records)} error={error_type}")


if __name__ == "__main__":
    main()

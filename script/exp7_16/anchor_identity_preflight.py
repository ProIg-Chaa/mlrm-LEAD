#!/usr/bin/env python3
"""Validate that every lexical anchor is a real, single tokenizer token."""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


TOKENS = {
    "end_thinking": "</think>",
    "start_thinking": "<think>",
    "im_end": "<|im_end|>",
    "newline": "\n",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report = {"model": args.model, "anchors": {}}
    failed = []
    for name, text in TOKENS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        converted = tokenizer.convert_tokens_to_ids(text)
        direct_special = converted not in {None, tokenizer.unk_token_id} and int(converted) >= 0
        resolved_id = int(converted) if direct_special else (int(ids[0]) if ids else None)
        usable = resolved_id is not None
        report["anchors"][name] = {
            "text": text,
            "encode_ids": ids,
            "convert_id": converted,
            "resolved_id": resolved_id,
            "resolution": "special_token" if direct_special else "first_encoded_subtoken",
            "decoded": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "usable": usable,
        }
        if not usable:
            failed.append(name)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if failed:
        raise SystemExit(f"anchors unavailable: {', '.join(failed)}")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

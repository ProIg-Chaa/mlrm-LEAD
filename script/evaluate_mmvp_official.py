#!/usr/bin/env python3
"""Evaluate MMVP results using the official repository's grading protocol.

This script mirrors the upstream MMVP flow:
1. Convert model results into the official answer-file schema.
2. Ask an OpenAI judge whether each response is correct.
3. Score pair accuracy: a pair is correct only when both items are judged correct.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a helpful and precise assistant for checking the quality of the "
    "answer. Please answer in only yes or no."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_results_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in sorted(rows, key=lambda x: int(x["id"])):
        converted.append(
            {
                "question_id": int(row["id"]) + 1,
                "prompt": f'{row.get("question", "").strip()} {row.get("options", "").strip()}'.strip(),
                "answer": row.get("answer"),
                "response": row.get("model_answer", ""),
                "source_id": int(row["id"]),
            }
        )
    return converted


def normalize_yes_no(text: str) -> str | None:
    if re.fullmatch(r"\s*yes\s*", text, re.I):
        return "yes"
    if re.fullmatch(r"\s*no\s*", text, re.I):
        return "no"
    return None


def judge_one(
    client: OpenAI,
    model: str,
    prompt: str,
    gold: str,
    response: str,
    sleep_seconds: float,
) -> tuple[str, str]:
    judge_prompt = (
        f"Given the following question {prompt}, the correct answer is {gold}. "
        f"Does the following answer correctly answers the question, answer:{response}?"
    )
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            normalized = normalize_yes_no(content)
            if normalized is None:
                return "invalid", content
            return normalized, content
        except Exception:
            time.sleep(sleep_seconds)


def grade_answer_rows(
    rows: list[dict],
    client: OpenAI,
    model: str,
    sleep_seconds: float,
) -> tuple[dict, list[dict]]:
    judged_rows = []
    pair_correct = 0
    num_total_pairs = 0
    current_pair_correct = 0
    current_pair_items = 0

    for row in rows:
        verdict, raw_judge = judge_one(
            client=client,
            model=model,
            prompt=row["prompt"],
            gold=row["answer"],
            response=row["response"],
            sleep_seconds=sleep_seconds,
        )
        is_correct = verdict == "yes"

        enriched = dict(row)
        enriched["judge_verdict"] = verdict
        enriched["judge_raw_response"] = raw_judge
        enriched["judge_is_correct"] = is_correct
        judged_rows.append(enriched)

        current_pair_items += 1
        if is_correct:
            current_pair_correct += 1

        if current_pair_items == 2:
            num_total_pairs += 1
            if current_pair_correct == 2:
                pair_correct += 1
            current_pair_items = 0
            current_pair_correct = 0

    report = {
        "pair_accuracy": pair_correct / num_total_pairs if num_total_pairs else 0.0,
        "pair_correct": pair_correct,
        "pair_total": num_total_pairs,
        "sample_accuracy": (
            sum(row["judge_is_correct"] for row in judged_rows) / len(judged_rows)
            if judged_rows
            else 0.0
        ),
        "sample_correct": sum(row["judge_is_correct"] for row in judged_rows),
        "sample_total": len(judged_rows),
        "judge_model": model,
        "official_protocol_note": (
            "Pair accuracy follows the MMVP official repository: a pair counts as "
            "correct only when both questions are judged correct."
        ),
    }
    return report, judged_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Project results.jsonl path")
    parser.add_argument("--answer_file", required=True, help="Converted official-format answer file")
    parser.add_argument("--judge_output", required=True, help="Per-sample judged jsonl output")
    parser.add_argument("--report_json", required=True, help="Final report json")
    parser.add_argument("--judge_model", default="gpt-4-0314")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--sleep_seconds", type=float, default=10.0)
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="Only convert results to official answer-file format",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.convert_only and not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is required for official MMVP judging. "
            "Use --convert_only to only write the answer file."
        )

    result_rows = load_jsonl(Path(args.results))
    answer_rows = convert_results_rows(result_rows)
    save_jsonl(Path(args.answer_file), answer_rows)

    if args.convert_only:
        report = {
            "status": "converted_only",
            "answer_file": args.answer_file,
            "sample_total": len(answer_rows),
        }
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.report_json).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return 0

    client_kwargs = {"api_key": api_key}
    if args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"):
        client_kwargs["base_url"] = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    client = OpenAI(**client_kwargs)

    report, judged_rows = grade_answer_rows(
        rows=answer_rows,
        client=client,
        model=args.judge_model,
        sleep_seconds=args.sleep_seconds,
    )
    save_jsonl(Path(args.judge_output), judged_rows)
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.report_json).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

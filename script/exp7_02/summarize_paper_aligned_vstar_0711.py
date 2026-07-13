#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}"),
    re.compile(r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?", re.I),
    re.compile(r"(?:the\s+correct\s+)?answer\s+(?:is\s*)?[:\s]+\(?([A-Da-d])\)?", re.I),
    re.compile(r"answer\s*[:\s]+\(?([A-Da-d])\)?", re.I),
]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_last(text: str):
    text = text or ""
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[markers[-1].start():] if markers else text[-1500:]
    hits = []
    for pattern in PATTERNS:
        hits.extend((m.start(), m.group(1).upper()) for m in pattern.finditer(region))
    if hits:
        return max(hits)[1]
    letters = re.findall(r"\b([A-D])\b", region[-300:])
    return letters[-1] if letters else None


def evaluate(rows):
    correct = failed = errors = 0
    output_tokens = []
    for row in rows:
        prediction = extract_last(row.get("model_answer") or "")
        gold = str(row.get("answer") or "").strip().upper()[:1]
        correct += int(prediction is not None and prediction == gold)
        failed += int(prediction is None)
        errors += int(bool(row.get("error_type")))
        if isinstance(row.get("output_tokens"), (int, float)):
            output_tokens.append(row["output_tokens"])
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "failed_extraction": failed,
        "runtime_errors": errors,
        "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "maxed_25600": sum(x >= 25600 for x in output_tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for results in sorted(args.base_dir.glob("*/vstar/*/*/results.jsonl")):
        rel = results.relative_to(args.base_dir)
        model, _, scope, method_dir, _ = rel.parts
        summary.setdefault(model, {}).setdefault(scope, {})[method_dir] = {
            **evaluate(load_jsonl(results)),
            "run_dir": str(results.parent),
        }
    (args.base_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Paper-aligned VStar reproduction", ""]
    for model, scopes in summary.items():
        lines += [f"## {model}", "", "| scope | method | accuracy | correct | failed | avg tokens |", "|---|---|---:|---:|---:|---:|"]
        for scope, methods in scopes.items():
            for method, metric in methods.items():
                acc = "NA" if metric["accuracy"] is None else f'{100 * metric["accuracy"]:.2f}%'
                avg = "NA" if metric["avg_output_tokens"] is None else f'{metric["avg_output_tokens"]:.1f}'
                lines.append(f'| {scope} | {method} | {acc} | {metric["correct"]}/{metric["total"]} | {metric["failed_extraction"]} | {avg} |')
        lines.append("")
    (args.base_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

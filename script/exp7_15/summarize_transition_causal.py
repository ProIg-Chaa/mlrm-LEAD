#!/usr/bin/env python3
"""Summarize matched transition controls without treating proxy extraction as official MMVP scoring."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def pick(text: str | None) -> str | None:
    if not text:
        return None
    tail = text[-1800:]
    for pattern in [r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}", r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?"]:
        found = re.findall(pattern, tail, flags=re.IGNORECASE)
        if found:
            return found[-1].upper()
    found = re.findall(r"\b([A-Da-d])\b", tail.split("</think>")[-1])
    return found[-1].upper() if found else None


def local_correct(row: dict) -> bool:
    return pick(row.get("model_answer")) == str(row.get("answer", "")).strip().upper()[:1]


def exact_mcnemar(fixed: int, damaged: int) -> float:
    n = fixed + damaged
    if not n:
        return 1.0
    k = min(fixed, damaged)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def bootstrap_delta(base: dict[int, bool], method: dict[int, bool], draws: int = 2000) -> list[float]:
    ids = sorted(set(base) & set(method))
    rng = random.Random(42)
    if not ids:
        return [0.0, 0.0]
    samples = []
    for _ in range(draws):
        chosen = [rng.choice(ids) for _ in ids]
        samples.append(sum(int(method[i]) - int(base[i]) for i in chosen) / len(chosen))
    samples.sort()
    return [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]]


def run_correctness(run_dir: Path, dataset: str) -> tuple[dict[int, bool], dict[int, dict], str]:
    if dataset == "mmvp" and (run_dir / "specialized_eval_rows.jsonl").exists():
        data = rows(run_dir / "specialized_eval_rows.jsonl")
        return {int(row["id"]): bool(row.get("specialized_is_correct")) for row in data}, {int(row["id"]): row for row in data}, "specialized"
    data = rows(run_dir / "results.jsonl")
    return {int(row["id"]): local_correct(row) for row in data}, {int(row["id"]): row for row in data}, "corrected_local"


def trace_map(run_dir: Path) -> dict[int, list[int]]:
    path = run_dir / "token_entropy_full.jsonl"
    if not path.exists():
        return {}
    return {int(row["id"]): [int(token["token_id"]) for token in row.get("tokens", [])] for row in rows(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    methods = [
        "lead_force_normal", "initial_soft_only", "initial_transition_only",
        "initial_transition_no_to_normal", "initial_transition_no_linebreak", "initial_transition_cache_rebuild",
    ]
    summary, deltas, trace_summary = [], [], []
    for dataset in ["vstar", "mmvp"]:
        cot_dir = root / dataset / "cot_orign_greedy"
        if not (cot_dir / "results.jsonl").exists():
            continue
        base_correct, base_rows, evaluator = run_correctness(cot_dir, dataset)
        base_trace = trace_map(cot_dir)
        base_result_rows = {int(row["id"]): row for row in rows(cot_dir / "results.jsonl")}
        summary.append({
            "dataset": dataset, "method": "cot_orign_greedy", "evaluator": evaluator,
            "total": len(base_correct),
            "accuracy": sum(base_correct.values()) / len(base_correct) if base_correct else None,
            "failed_extraction": sum(pick(row.get("model_answer")) is None for row in base_result_rows.values()), "fixed": 0, "damaged": 0,
            "net": 0, "mcnemar_p": 1.0, "bootstrap_delta_95ci": [0.0, 0.0],
        })
        for method in methods:
            method_dir = root / dataset / method
            if not (method_dir / "results.jsonl").exists():
                continue
            method_correct, method_rows, method_evaluator = run_correctness(method_dir, dataset)
            ids = sorted(set(base_correct) & set(method_correct))
            fixed = [i for i in ids if not base_correct[i] and method_correct[i]]
            damaged = [i for i in ids if base_correct[i] and not method_correct[i]]
            result_rows = {int(row["id"]): row for row in rows(method_dir / "results.jsonl")}
            failed = sum(pick(result_rows[i].get("model_answer")) is None for i in ids if i in result_rows)
            summary.append({
                "dataset": dataset, "method": method, "evaluator": method_evaluator,
                "total": len(ids), "accuracy": sum(method_correct[i] for i in ids) / len(ids) if ids else None,
                "failed_extraction": failed, "fixed": len(fixed), "damaged": len(damaged),
                "net": len(fixed) - len(damaged), "mcnemar_p": exact_mcnemar(len(fixed), len(damaged)),
                "bootstrap_delta_95ci": bootstrap_delta(base_correct, method_correct),
            })
            deltas.append({"dataset": dataset, "method": method, "fixed_ids": fixed, "damaged_ids": damaged})
            method_trace = trace_map(method_dir)
            common = sorted(set(base_trace) & set(method_trace))
            for step in [1, 2, 4, 8, 16, 32]:
                comparable = [i for i in common if len(base_trace[i]) > step and len(method_trace[i]) > step]
                diverged = sum(base_trace[i][:step] != method_trace[i][:step] for i in comparable)
                trace_summary.append({"dataset": dataset, "method": method, "prefix_tokens": step, "n": len(comparable), "diverged": diverged})

    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (root / "pairwise_deltas.json").write_text(json.dumps(deltas, indent=2) + "\n", encoding="utf-8")
    (root / "trace_summary.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace_summary), encoding="utf-8")
    lines = ["# Transition Causal Decomposition", "", "| dataset | method | evaluator | acc | fixed | damaged | net | McNemar p | bootstrap 95% CI |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        acc = "NA" if row["accuracy"] is None else f"{row['accuracy'] * 100:.2f}%"
        ci = ", ".join(f"{value * 100:.2f}%" for value in row["bootstrap_delta_95ci"])
        lines.append(f"| {row['dataset']} | {row['method']} | {row['evaluator']} | {acc} | {row['fixed']} | {row['damaged']} | {row['net']} | {row['mcnemar_p']:.4f} | [{ci}] |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = [
        "# Transition Mechanism Report",
        "",
        "This report applies the preregistered interpretation rules to matched greedy runs.",
        "",
        "- `initial_soft_only` isolates the first soft input.",
        "- `initial_transition_no_to_normal` removes the step-1 transition mix.",
        "- `initial_transition_no_linebreak` removes the step-0 newline mixture.",
        "- `initial_transition_cache_rebuild` preserves the first two emitted tokens but removes the persistent soft KV history.",
        "- same-token replay compares hard and transition routes after forcing COT's first greedy token.",
        "",
        "Interpret the effects using `summary.json`, `pairwise_deltas.json`, `trace_summary.jsonl`, and `vstar/same_token_replay/summary.json`. Do not make a causal trajectory claim when the matched control confidence intervals cross zero.",
    ]
    (root / "mechanism_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

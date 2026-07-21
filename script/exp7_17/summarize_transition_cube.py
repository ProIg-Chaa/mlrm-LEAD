#!/usr/bin/env python3
"""Summarize the controlled soft/newline/EOT factorial experiment."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


CONDITIONS = {
    (0, 0, 0): ("direct", "hard_no_newline"),
    (0, 1, 0): ("direct", "hard_with_newline"),
    (1, 0, 0): ("direct", "soft_no_newline"),
    (1, 1, 0): ("direct", "soft_with_newline"),
    (0, 0, 1): ("eot", "eot_hard_no_newline"),
    (0, 1, 1): ("eot", "eot_hard_with_newline"),
    (1, 0, 1): ("eot", "eot_soft_no_newline"),
    (1, 1, 1): ("eot", "eot_soft_with_newline"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract(text: str | None) -> str | None:
    if not text:
        return None
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[markers[-1].start():] if markers else text[-1800:]
    patterns = [
        r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}",
        r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
    ]
    hits = []
    for pattern in patterns:
        hits.extend((m.start(), m.group(1).upper()) for m in re.finditer(pattern, region, re.I))
    if hits:
        return max(hits)[1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-200:])
    return letters[-1].upper() if letters else None


def items(run: Path, dataset: str) -> dict[str, dict]:
    special = run / "specialized_eval_rows.jsonl"
    if dataset == "mmvp" and special.is_file():
        return {str(row["id"]): {"pred": row.get("specialized_pred"), "correct": bool(row.get("specialized_is_correct"))} for row in read_jsonl(special)}
    out = {}
    for row in read_jsonl(run / "results.jsonl"):
        prediction = extract(row.get("model_answer"))
        out[str(row["id"])] = {"pred": prediction, "correct": prediction == str(row.get("answer", "")).strip().upper()[:1]}
    return out


def pair_accuracy(run: Path) -> float | None:
    path = run / "specialized_eval_report.json"
    return json.loads(path.read_text(encoding="utf-8")).get("pair_accuracy") if path.is_file() else None


def effect(values: dict[tuple[int, int, int], int], index: int, ids: list[str]) -> float:
    signed = []
    for sample_id in ids:
        hi = [values[(a, b, c)][sample_id] for a, b, c in CONDITIONS if (a, b, c)[index] == 1]
        lo = [values[(a, b, c)][sample_id] for a, b, c in CONDITIONS if (a, b, c)[index] == 0]
        signed.append(sum(hi) / len(hi) - sum(lo) / len(lo))
    return sum(signed) / len(signed)


def factorial(values: dict[tuple[int, int, int], dict[str, int]], ids: list[str]) -> dict[str, float]:
    # Effect-coded factorial contrasts. Main effects are high-minus-low averages.
    results = {}
    for name, factors in [("soft", (0,)), ("newline", (1,)), ("eot", (2,)), ("soft_x_newline", (0, 1)), ("soft_x_eot", (0, 2)), ("newline_x_eot", (1, 2)), ("soft_x_newline_x_eot", (0, 1, 2))]:
        per_id = []
        for sample_id in ids:
            total = 0.0
            for key in CONDITIONS:
                sign = 1
                for factor in factors:
                    sign *= 1 if key[factor] else -1
                total += sign * values[key][sample_id]
            per_id.append(total / 4 if len(factors) == 1 else total / 2 ** (3 - len(factors)))
        results[name] = sum(per_id) / len(per_id)
    return results


def bootstrap(values: dict[tuple[int, int, int], dict[str, int]], ids: list[str], draws: int = 4000) -> dict[str, list[float]]:
    rng = random.Random(42)
    rows = {name: [] for name in factorial(values, ids)}
    for _ in range(draws):
        sample = [rng.choice(ids) for _ in ids]
        current = factorial(values, sample)
        for name, value in current.items():
            rows[name].append(value)
    return {name: [sorted(values)[int(0.025 * (draws - 1))], sorted(values)[int(0.975 * (draws - 1))]] for name, values in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-root", required=True)
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--eot-root", required=True)
    args = parser.parse_args()
    direct, full, eot = Path(args.direct_root), Path(args.full_root), Path(args.eot_root)
    all_records, factor_records = [], []
    for dataset in ("vstar", "mmvp"):
        condition_items = {}
        condition_runs = {}
        for key, (group, name) in CONDITIONS.items():
            if key == (1, 1, 0):
                run = full / dataset / "direct_token_step1"
            else:
                root = direct if group == "direct" else eot
                run = root / dataset / name
            condition_runs[key] = run
            condition_items[key] = items(run, dataset)
            data = condition_items[key]
            row = {
                "dataset": dataset, "soft": bool(key[0]), "newline": bool(key[1]), "eot": bool(key[2]),
                "condition": name, "run_dir": str(run), "accuracy": sum(x["correct"] for x in data.values()) / len(data),
                "failed_extraction": sum(x["pred"] is None for x in data.values()),
            }
            if dataset == "mmvp":
                row["pair_accuracy"] = pair_accuracy(run)
            all_records.append(row)
        ids = sorted(set.intersection(*(set(value) for value in condition_items.values())))
        binary = {key: {sample_id: int(value[sample_id]["correct"]) for sample_id in ids} for key, value in condition_items.items()}
        effects = factorial(binary, ids)
        factor_records.append({"dataset": dataset, "n": len(ids), "effects": effects, "bootstrap_95ci": bootstrap(binary, ids)})
    out = eot
    (out / "transition_cube_summary.json").write_text(json.dumps({"cells": all_records, "factorial": factor_records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Transition Cube", "", "All cells force the step-1 transition and disable later routing. Direct uses beta=1; EOT uses beta=0.7 with the inherited EOT anchor. This is a controlled bridge decomposition, not a replacement for the natural exact-original audit.", "", "| Dataset | Step-0 | Newline | Step-1 | Accuracy | Pair | Failed |", "|---|---|---|---|---:|---:|---:|"]
    for row in all_records:
        pair = "-" if row.get("pair_accuracy") is None else f"{100 * row['pair_accuracy']:.2f}%"
        lines.append(f"| {row['dataset']} | {'soft' if row['soft'] else 'hard'} | {'on' if row['newline'] else 'off'} | {'EOT' if row['eot'] else 'direct'} | {100 * row['accuracy']:.2f}% | {pair} | {row['failed_extraction']} |")
    lines += ["", "## Factorial Effects", "", "Effects are signed differences in accuracy probability; intervals are paired bootstrap 95% CIs.", ""]
    for record in factor_records:
        lines.append(f"### {record['dataset']} (n={record['n']})")
        for name, value in record["effects"].items():
            low, high = record["bootstrap_95ci"][name]
            lines.append(f"- {name}: {100 * value:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}]")
        lines.append("")
    (out / "transition_cube_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/root/gushuo/outputs/experiments/20260716_token_anchored_transition")
PAIRS = [
    ("original_eot_bridge_step1", "hard_eot_bridge_step1"),
    ("direct_token_step1", "token_anchored_transition_step1"),
    ("original_eot_bridge_step1", "direct_token_step1"),
    ("original_eot_bridge_step1", "token_anchored_transition_step1"),
]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_id(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in rows(path)}


def guess(text: str | None) -> str | None:
    if not text:
        return None
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[markers[-1].start():] if markers else text[-1800:]
    hits = []
    for pattern in (
        r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}",
        r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
    ):
        hits.extend((m.start(), m.group(1).upper()) for m in re.finditer(pattern, region, re.I))
    if hits:
        return max(hits)[1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-200:])
    return letters[-1].upper() if letters else None


def correctness(dataset: str, name: str) -> dict[str, bool]:
    run = ROOT / dataset / name
    special = run / "specialized_eval_rows.jsonl"
    if dataset == "mmvp" and special.is_file():
        return {str(row["id"]): bool(row.get("specialized_is_correct")) for row in rows(special)}
    return {
        str(row["id"]): guess(row.get("model_answer")) == str(row.get("answer", "")).strip().upper()[:1]
        for row in rows(run / "results.jsonl")
    }


def main() -> None:
    report: dict[str, object] = {}
    for dataset in ("vstar", "mmvp"):
        dataset_report: dict[str, object] = {}
        for left, right in PAIRS:
            left_results = by_id(ROOT / dataset / left / "results.jsonl")
            right_results = by_id(ROOT / dataset / right / "results.jsonl")
            ids = sorted(set(left_results) & set(right_results))
            output_equal = sum(
                left_results[idx].get("model_answer") == right_results[idx].get("model_answer")
                for idx in ids
            )
            left_correct, right_correct = correctness(dataset, left), correctness(dataset, right)
            fixed = [idx for idx in ids if not left_correct[idx] and right_correct[idx]]
            damaged = [idx for idx in ids if left_correct[idx] and not right_correct[idx]]
            left_trace = by_id(ROOT / dataset / left / "token_entropy_full.jsonl")
            right_trace = by_id(ROOT / dataset / right / "token_entropy_full.jsonl")
            trace_equal = 0
            source_l1 = []
            for idx in ids:
                lt = next((t for t in left_trace[idx].get("tokens", []) if t.get("step") == 1), {})
                rt = next((t for t in right_trace[idx].get("tokens", []) if t.get("step") == 1), {})
                if lt == rt:
                    trace_equal += 1
                if lt.get("transition_source_norm") is not None:
                    source_l1.append(abs(float(lt["transition_source_norm"]) - float(rt["transition_source_norm"])))
            dataset_report[f"{left}__vs__{right}"] = {
                "samples": len(ids),
                "identical_model_answer": output_equal,
                "fixed_vs_left": len(fixed),
                "damaged_vs_left": len(damaged),
                "net_vs_left": len(fixed) - len(damaged),
                "identical_step1_trace_record": trace_equal,
                "mean_abs_source_norm_difference": sum(source_l1) / len(source_l1) if source_l1 else None,
                "first_left_step1": next((t for t in left_trace[ids[0]].get("tokens", []) if t.get("step") == 1), {}),
                "first_right_step1": next((t for t in right_trace[ids[0]].get("tokens", []) if t.get("step") == 1), {}),
            }
        report[dataset] = dataset_report
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

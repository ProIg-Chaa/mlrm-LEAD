#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


METHODS = [
    ("original_eot_bridge_step1", "soft", "end_thinking"),
    ("hard_eot_bridge_step1", "hard", "end_thinking"),
    ("direct_token_step1", "hard", "generated_token"),
    ("token_anchored_transition_step1", "soft", "generated_token"),
]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer(text: str | None) -> str | None:
    if not text:
        return None
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[markers[-1].start():] if markers else text[-1800:]
    patterns = [
        r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}",
        r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
    ]
    hits = [(m.start(), m.group(1).upper()) for p in patterns for m in re.finditer(p, region, re.I)]
    if hits:
        return max(hits)[1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-200:])
    return letters[-1].upper() if letters else None


def evaluated(run: Path, dataset: str) -> dict[str, dict]:
    special = run / "specialized_eval_rows.jsonl"
    if dataset == "mmvp" and special.is_file():
        return {
            str(row["id"]): {"pred": row.get("specialized_pred"), "gold": row.get("specialized_gold"), "correct": bool(row.get("specialized_is_correct"))}
            for row in jsonl(special)
        }
    return {
        str(row["id"]): {
            "pred": answer(row.get("model_answer")),
            "gold": str(row.get("answer", "")).strip().upper()[:1],
            "correct": answer(row.get("model_answer")) == str(row.get("answer", "")).strip().upper()[:1],
        }
        for row in jsonl(run / "results.jsonl")
    }


def mcnemar(fixed: int, damaged: int) -> float:
    n = fixed + damaged
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixed, damaged) + 1)) / 2**n)


def comparison(base: dict[str, dict], method: dict[str, dict]) -> dict:
    ids = sorted(set(base) & set(method))
    fixed = [i for i in ids if not base[i]["correct"] and method[i]["correct"]]
    damaged = [i for i in ids if base[i]["correct"] and not method[i]["correct"]]
    rng = random.Random(42)
    deltas = []
    for _ in range(5000):
        sample = [rng.choice(ids) for _ in ids]
        deltas.append(sum(int(method[i]["correct"]) - int(base[i]["correct"]) for i in sample) / len(sample))
    deltas.sort()
    return {
        "paired": len(ids), "fixed": len(fixed), "damaged": len(damaged), "net": len(fixed) - len(damaged),
        "fixed_ids": fixed, "damaged_ids": damaged, "mcnemar_exact_p": mcnemar(len(fixed), len(damaged)),
        "bootstrap_delta_95ci": [deltas[124], deltas[4874]],
        "prediction_agreement": sum(base[i]["pred"] == method[i]["pred"] for i in ids) / len(ids),
    }


def trace_audit(run: Path, source: str, anchor: str) -> dict:
    trace_rows = jsonl(run / "token_entropy_full.jsonl")
    events = []
    failures = []
    for row in trace_rows:
        hits = [t for t in row.get("tokens", []) if t.get("step") == 1]
        if len(hits) != 1:
            failures.append({"id": row.get("id"), "reason": "step1_missing_or_duplicate"})
            continue
        t = hits[0]
        if not (t.get("to_normal") and t.get("forced_transition_step1") and t.get("lead_transition_source") == source and t.get("lead_transition_anchor") == anchor):
            failures.append({"id": row.get("id"), "reason": "handoff_mismatch", "trace": {k: t.get(k) for k in ("to_normal", "forced_transition_step1", "lead_transition_source", "lead_transition_anchor")}})
            continue
        events.append(t)
    means = {}
    for key in ("transition_source_norm", "transition_anchor_norm", "transition_bridge_norm", "transition_source_anchor_cos"):
        values = [float(t[key]) for t in events if t.get(key) is not None]
        means[key] = sum(values) / len(values) if values else None
    return {"trace_samples": len(trace_rows), "valid_step1_handoffs": len(events), "failures": failures, "transition_means": means}


def stats(run: Path) -> dict:
    rows = jsonl(run / "results.jsonl")
    lengths = [
        int(r.get("output_tokens", r.get("output_length", len(str(r.get("model_answer", "")).split()))))
        for r in rows
    ]
    errors = sum(bool(r.get("error_type")) for r in rows)
    return {"runtime_errors": errors, "mean_output_tokens": sum(lengths) / len(lengths), "long_ge_256": sum(x >= 256 for x in lengths), "maxed_1024": sum(x >= 1024 for x in lengths)}


def mmvp_pair(report: Path) -> float | None:
    if not report.is_file():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    for key in ("pair_accuracy", "pair_acc", "pairwise_accuracy"):
        if key in data:
            return data[key]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    source_root, root = Path(args.source_root), Path(args.experiment_root)
    all_rows, deltas = [], {}
    for dataset in ("vstar", "mmvp"):
        cot = evaluated(source_root / dataset / "cot_orign_greedy", dataset)
        for name, source, anchor in METHODS:
            run = root / dataset / name
            result = evaluated(run, dataset)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config_ok = config.get("lead_force_initial_transition_step1") is True and config.get("lead_transition_source") == source and config.get("lead_transition_anchor") == anchor
            row = {
                "dataset": dataset, "method": name, "source": source, "anchor": anchor, "run_dir": str(run),
                "total": len(result), "accuracy": sum(x["correct"] for x in result.values()) / len(result),
                "failed_extraction": sum(x["pred"] is None for x in result.values()), "config_ok": config_ok,
                "stats": stats(run), "trace_audit": trace_audit(run, source, anchor), "vs_cot": comparison(cot, result),
            }
            if dataset == "mmvp":
                row["mmvp_pair_accuracy"] = mmvp_pair(run / "specialized_eval_report.json")
            all_rows.append(row)
            deltas[f"{dataset}/{name}"] = row["vs_cot"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "bridge_factorial_summary.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "pairwise_deltas.json").write_text(json.dumps(deltas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Token-Anchored Transition: Bridge Factorial", "", "All four rows retain the same step-0 soft/newline initializer and force the handoff at step 1.", "", "| Dataset | Bridge | Source | Anchor | Accuracy | Pair acc | Failed | Fixed/Damaged vs COT | Net | McNemar p | Step-1 audit |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        vs, audit = row["vs_cot"], row["trace_audit"]
        pair = "-" if row.get("mmvp_pair_accuracy") is None else f"{100 * row['mmvp_pair_accuracy']:.2f}%"
        lines.append(f"| {row['dataset']} | {row['method']} | {row['source']} | {row['anchor']} | {100 * row['accuracy']:.2f}% | {pair} | {row['failed_extraction']} | {vs['fixed']}/{vs['damaged']} | {vs['net']} | {vs['mcnemar_exact_p']:.4f} | {audit['valid_step1_handoffs']}/{audit['trace_samples']} |")
    (root / "bridge_factorial_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = ["# Token-Anchored Transition Mechanism Report", "", "This experiment factors the step-1 bridge while holding the step-0 soft/newline initialization, greedy decoding, model, prompt, and sample order fixed.", "", "- `original_eot_bridge_step1`: soft semantic state plus the model-native `</think>` boundary embedding.", "- `hard_eot_bridge_step1`: keeps only the boundary prior; it removes the probability-weighted soft source at the bridge.", "- `direct_token_step1`: a hard direct cut to the generated token embedding.", "- `token_anchored_transition_step1`: soft semantic state handed off to the embedding of the actual generated token, with no `</think>` embedding.", "", "Interpret the Token-Anchored row against original EOT for the necessity of the fixed boundary anchor, and against direct-token for the value of the soft semantic source. The forced-step trace audit is a prerequisite for every comparison; failed audits invalidate causal interpretation.", "", "The report intentionally does not claim that an observed accuracy gap identifies a long-lived latent state. It tests only the local bridge mechanism at the externally controlled handoff step."]
    (root / "token_anchor_mechanism_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "output": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

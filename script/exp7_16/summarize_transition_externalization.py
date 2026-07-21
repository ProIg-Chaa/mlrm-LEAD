#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pred(text: str | None) -> str | None:
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
        hits.extend((match.start(), match.group(1).upper()) for match in re.finditer(pattern, region, re.I))
    if hits:
        return sorted(hits)[-1][1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-200:])
    return letters[-1].upper() if letters else None


def items(run: Path, dataset: str) -> dict[str, dict]:
    specialized = run / "specialized_eval_rows.jsonl"
    if dataset == "mmvp" and specialized.is_file():
        return {
            str(row["id"]): {
                "pred": row.get("specialized_pred"),
                "gold": row.get("specialized_gold"),
                "correct": bool(row.get("specialized_is_correct")),
            }
            for row in read_jsonl(specialized)
        }
    output = {}
    for row in read_jsonl(run / "results.jsonl"):
        guess = pred(row.get("model_answer"))
        gold = str(row.get("answer", "")).strip().upper()[:1]
        output[str(row["id"])] = {"pred": guess, "gold": gold, "correct": guess == gold}
    return output


def exact_mcnemar(fixed: int, damaged: int) -> float:
    total = fixed + damaged
    if not total:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(fixed, damaged) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def bootstrap_delta(base: dict[str, dict], method: dict[str, dict], draws: int = 2000) -> list[float]:
    ids = sorted(set(base) & set(method))
    rng = random.Random(42)
    values = []
    for _ in range(draws):
        sample = [rng.choice(ids) for _ in ids]
        values.append(sum(int(method[i]["correct"]) - int(base[i]["correct"]) for i in sample) / len(sample))
    values.sort()
    return [values[int(0.025 * (draws - 1))], values[int(0.975 * (draws - 1))]]


def compare(base: dict[str, dict], method: dict[str, dict]) -> dict:
    ids = sorted(set(base) & set(method))
    fixed = [i for i in ids if not base[i]["correct"] and method[i]["correct"]]
    damaged = [i for i in ids if base[i]["correct"] and not method[i]["correct"]]
    agreement = sum(base[i]["pred"] == method[i]["pred"] for i in ids)
    return {
        "total": len(ids), "fixed": len(fixed), "damaged": len(damaged), "net": len(fixed) - len(damaged),
        "prediction_agreement": agreement / len(ids), "mcnemar_p": exact_mcnemar(len(fixed), len(damaged)),
        "bootstrap_delta_95ci": bootstrap_delta(base, method), "fixed_ids": fixed, "damaged_ids": damaged,
    }


def summarize_run(name: str, run: Path, dataset: str, cot: dict[str, dict], transition: dict[str, dict]) -> dict:
    data = items(run, dataset)
    return {
        "dataset": dataset, "method": name, "run_dir": str(run), "total": len(data),
        "accuracy": sum(row["correct"] for row in data.values()) / len(data),
        "failed_extraction": sum(row["pred"] is None for row in data.values()),
        "vs_cot": compare(cot, data), "vs_initial_transition": compare(transition, data),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--rl-main-root", required=True)
    parser.add_argument("--timing-summary", required=True)
    args = parser.parse_args()
    source = Path(args.source_root)
    exp = Path(args.experiment_root)
    rl_main = Path(args.rl_main_root)

    summaries = []
    for dataset in ["vstar", "mmvp"]:
        cot = items(source / dataset / "cot_orign_greedy", dataset)
        transition = items(source / dataset / "initial_transition_only", dataset)
        candidates = [
            ("initial_transition", source / dataset / "initial_transition_only"),
            ("cache_rebuild_prefix1", exp / dataset / "cache_rebuild_prefix1"),
            ("cache_rebuild_prefix2", source / dataset / "initial_transition_cache_rebuild"),
        ]
        if dataset == "vstar":
            candidates.append(("hard_boundary_only", exp / dataset / "hard_boundary_only"))
        for name, run in candidates:
            if (run / "results.jsonl").is_file():
                summaries.append(summarize_run(name, run, dataset, cot, transition))

    same_prefix_path = exp / "vstar" / "same_prefix_replay" / "summary.json"
    same_prefix = read_json(same_prefix_path) if same_prefix_path.is_file() else {}
    timing_path = Path(args.timing_summary)
    timing_rows = []
    if timing_path.is_file():
        for row in read_json(timing_path):
            if row.get("phase") in {"phase2_timing_curve", "phase2_timing_curve_cross"} and row.get("run") in {"transition_step0", "transition_step4", "transition_step16", "transition_step32"}:
                timing_rows.append({key: row.get(key) for key in ["phase", "dataset", "run", "accuracy", "pair_accuracy", "failed_extraction"]})

    late_utility = []
    for dataset in ["vstar", "mmvp", "realworldqa_fixed200", "visulogic300"]:
        lead_run = rl_main / dataset / "lead"
        transition_run = rl_main / dataset / "initial_transition_only"
        if (lead_run / "results.jsonl").is_file() and (transition_run / "results.jsonl").is_file():
            lead_items = items(lead_run, dataset)
            transition_items = items(transition_run, dataset)
            late_utility.append({"dataset": dataset, **compare(transition_items, lead_items)})

    (exp / "externalization_curve.json").write_text(json.dumps({"runs": summaries, "same_prefix": same_prefix, "timing": timing_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (exp / "late_routing_utility.json").write_text(json.dumps(late_utility, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Transition Externalization Curve", "", "| Dataset | Control | Accuracy | Failed | Fixed/Damaged vs COT | Agreement with transition |", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['method']} | {100 * row['accuracy']:.2f}% | {row['failed_extraction']} | {row['vs_cot']['fixed']}/{row['vs_cot']['damaged']} | {100 * row['vs_initial_transition']['prediction_agreement']:.2f}% |")
    lines += ["", "## Same-prefix replay", ""]
    for prefix_len, row in (same_prefix.get("by_prefix") or {}).items():
        lines.append(f"- Prefix {prefix_len}: valid={row['valid']}, diverged={row['diverged_after_prefix']}, answer disagreement={row['answer_disagreement']}, transition/hard correct={row['transition_correct']}/{row['hard_correct']}.")
    (exp / "externalization_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    late_lines = ["# Late Routing Utility", "", "Full LEAD is compared against Initial Transition; fixed means late routing repairs a sample and damaged means it breaks one.", "", "| Dataset | Fixed | Damaged | Net | Agreement | McNemar p |", "|---|---:|---:|---:|---:|---:|"]
    for row in late_utility:
        late_lines.append(f"| {row['dataset']} | {row['fixed']} | {row['damaged']} | {row['net']} | {100 * row['prediction_agreement']:.2f}% | {row['mcnemar_p']:.4f} |")
    (exp / "late_routing_utility.md").write_text("\n".join(late_lines) + "\n", encoding="utf-8")

    report = [
        "# Early Transition Mechanism Report", "",
        "The causal claim is split into a transient latent-state channel and its later externalization into visible discrete tokens.", "",
        "- Same-prefix replay tests whether routes diverge despite identical visible prefixes.",
        "- Prefix-length cache rebuild tests when persistent soft/mixed KV history stops being necessary.",
        "- Hard boundary-only retains newline and reasoning-boundary steering while removing probability-weighted soft semantics.",
        "- Do not claim a persistent hidden-state basin when prefix-2 cache rebuild preserves the effect.", "",
        "Use `externalization_curve.json`, `late_routing_utility.json`, and the four-quadrant replay rows for the final interpretation.",
    ]
    (exp / "transition_mechanism_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"controls": len(summaries), "same_prefix_replays": same_prefix.get("replays"), "late_datasets": len(late_utility)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

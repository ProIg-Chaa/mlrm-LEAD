#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from talr_analysis_common import (
    bootstrap_delta,
    explicit_answers,
    load_jsonl,
    mcnemar_exact,
    paired_groups,
    repeat_ratio,
    score_row,
    stratified_take,
    summarize_trace,
    trace_by_id,
    write_json,
    write_jsonl,
)


def index_rows(path: Path) -> dict[str, dict]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def output_metrics(row: dict) -> dict:
    text = str(row.get("model_answer") or "")
    answers = explicit_answers(text)
    return {
        "output_tokens": int(row.get("output_tokens") or 0),
        "repeat_ratio_3gram": repeat_ratio(text),
        "long_ge_256": int(row.get("output_tokens") or 0) >= 256,
        "maxed_1024": int(row.get("output_tokens") or 0) >= 1024,
        "answer_sequence": answers,
        "answer_reversal": len(set(answers)) > 1,
    }


def comparison_stats(reference: dict[str, dict], method: dict[str, dict]) -> dict:
    ids = sorted(set(reference) & set(method), key=lambda value: (len(value), value))
    left = [score_row(reference[sample_id]) for sample_id in ids]
    right = [score_row(method[sample_id]) for sample_id in ids]
    valid = [
        index
        for index in range(len(ids))
        if not left[index]["runtime_error"] and not right[index]["runtime_error"]
    ]
    left_flags = [left[index]["correct"] for index in valid]
    right_flags = [right[index]["correct"] for index in valid]
    groups = paired_groups(reference, method)
    fixed = len(groups.get("fixed", []))
    damaged = len(groups.get("damaged", []))
    return {
        "paired_samples": len(ids),
        "valid_samples": len(valid),
        "reference_accuracy": mean(left_flags) if left_flags else None,
        "method_accuracy": mean(right_flags) if right_flags else None,
        "delta": (
            mean(right_flags) - mean(left_flags) if left_flags else None
        ),
        "fixed": fixed,
        "damaged": damaged,
        "net_fixed": fixed - damaged,
        "mcnemar_exact_p": mcnemar_exact(fixed, damaged),
        "bootstrap_95_ci": bootstrap_delta(left_flags, right_flags),
        "groups": {key: len(value) for key, value in groups.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-per-group", type=int, default=20)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    all_cards = []
    selected_cards = []
    statistics = {}
    refinement_utility = defaultdict(Counter)
    guard_utility = defaultdict(Counter)

    for entry in manifest["comparisons"]:
        model = entry["model"]
        dataset = entry["dataset"]
        runs = {key: Path(value) for key, value in entry["runs"].items()}
        rows = {
            key: index_rows(path / "results.jsonl")
            for key, path in runs.items()
            if (path / "results.jsonl").exists()
        }
        traces = {
            key: trace_by_id(path)
            for key, path in runs.items()
            if (path / "token_entropy_full.jsonl").exists()
        }
        for talr_key in ("legacy_talr", "true_talr", "talr"):
            if talr_key not in rows:
                continue
            for reference_key in ("full_lead", "initial_transition"):
                if reference_key not in rows:
                    continue
                comparison_key = f"{model}/{dataset}/{reference_key}_vs_{talr_key}"
                groups = paired_groups(rows[reference_key], rows[talr_key])
                statistics[comparison_key] = comparison_stats(
                    rows[reference_key], rows[talr_key]
                )
                cards_for_comparison = []
                for group, ids in groups.items():
                    if group == "runtime_error":
                        continue
                    for sample_id in ids:
                        method_row = rows[talr_key][sample_id]
                        reference_row = rows[reference_key][sample_id]
                        trace = summarize_trace(
                            traces.get(talr_key, {}).get(sample_id)
                        )
                        card = {
                            "model": model,
                            "dataset": dataset,
                            "comparison": comparison_key,
                            "reference": reference_key,
                            "method": talr_key,
                            "group": group,
                            "id": sample_id,
                            "subtopic": method_row.get("subtopic"),
                            "question": method_row.get("question"),
                            "options": method_row.get("options"),
                            "gold": score_row(method_row)["gold"],
                            "reference_pred": score_row(reference_row)["pred"],
                            "method_pred": score_row(method_row)["pred"],
                            "reference_output": output_metrics(reference_row),
                            "method_output": output_metrics(method_row),
                            "trace": trace,
                        }
                        cards_for_comparison.append(card)
                        all_cards.append(card)
                        has_refinement = bool(trace["later_soft_positions"])
                        has_format = bool(trace["format_positions"])
                        has_veto = bool(trace["veto_positions"])
                        refinement_utility[comparison_key][
                            f"{group}/has_refinement={has_refinement}"
                        ] += 1
                        guard_utility[comparison_key][
                            f"{group}/format={has_format}/veto={has_veto}"
                        ] += 1
                for group in ("fixed", "damaged", "both_correct", "both_wrong"):
                    candidates = [
                        card
                        for card in cards_for_comparison
                        if card["group"] == group
                    ]
                    selected_cards.extend(
                        stratified_take(candidates, args.selected_per_group)
                    )

    write_json(output_dir / "talr_component_diagnosis.json", statistics)
    write_jsonl(output_dir / "talr_fixed_damaged_samples.jsonl", all_cards)
    write_jsonl(output_dir / "selected_talr_samples.jsonl", selected_cards)
    write_json(
        output_dir / "refinement_event_utility.json",
        {key: dict(value) for key, value in refinement_utility.items()},
    )
    write_json(
        output_dir / "guard_event_utility.json",
        {key: dict(value) for key, value in guard_utility.items()},
    )
    refinement_lines = [
        "# Refinement Event Utility",
        "",
        "These counts are observational and grouped by paired outcome.",
        "",
        "| Comparison | Outcome/event | Samples |",
        "|---|---|---:|",
    ]
    for key, counts in sorted(refinement_utility.items()):
        for label, count in sorted(counts.items()):
            refinement_lines.append(f"| {key} | {label} | {count} |")
    (output_dir / "refinement_event_utility.md").write_text(
        "\n".join(refinement_lines) + "\n", encoding="utf-8"
    )
    guard_lines = [
        "# Guard Event Utility",
        "",
        "A guard event is counted only when it changes or suppresses a proposed route.",
        "",
        "| Comparison | Outcome/event | Samples |",
        "|---|---|---:|",
    ]
    for key, counts in sorted(guard_utility.items()):
        for label, count in sorted(counts.items()):
            guard_lines.append(f"| {key} | {label} | {count} |")
    (output_dir / "guard_event_utility.md").write_text(
        "\n".join(guard_lines) + "\n", encoding="utf-8"
    )

    lines = [
        "# TALR Component Diagnosis",
        "",
        "All comparisons use paired sample IDs and the corrected last-answer extractor.",
        "Event counts are observational; only single-event replay may be interpreted causally.",
        "",
        "| Comparison | N | Ref acc | TALR acc | Delta | Fixed | Damaged | Net | McNemar p | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(statistics.items()):
        ci = value["bootstrap_95_ci"]
        lines.append(
            f"| {key} | {value['valid_samples']} | "
            f"{100 * value['reference_accuracy']:.2f}% | "
            f"{100 * value['method_accuracy']:.2f}% | "
            f"{100 * value['delta']:+.2f}pp | {value['fixed']} | "
            f"{value['damaged']} | {value['net_fixed']:+d} | "
            f"{value['mcnemar_exact_p']:.4f} | "
            f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]pp |"
        )
    (output_dir / "talr_component_diagnosis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote TALR diagnosis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

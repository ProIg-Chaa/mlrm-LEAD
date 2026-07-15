#!/usr/bin/env python3
"""Summarize staged Early Actual-Visual Anchor experiments."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.evaluate_realworldqa_mcq import evaluate, load_jsonl  # noqa: E402


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_path(dataset_rows: list[dict], path: Path) -> tuple[dict, dict[int, dict]]:
    dataset_ids = {int(row["id"]) for row in dataset_rows}
    result_rows = [
        row for row in load_jsonl(path) if int(row["id"]) in dataset_ids
    ]
    report, enriched = evaluate(dataset_rows, result_rows)
    report["runtime_errors"] = sum(bool(row.get("error_type")) for row in enriched)
    report["runtime_error_ids"] = [
        int(row["id"]) for row in enriched if row.get("error_type")
    ]
    report["failed_extraction_ids"] = [
        int(row["id"])
        for row in enriched
        if row.get("realworldqa_pred") is None
    ]
    report["mean_output_tokens"] = (
        sum(int(row.get("output_tokens") or 0) for row in enriched) / len(enriched)
        if enriched
        else 0.0
    )
    return report, {int(row["id"]): row for row in enriched}


def trace_summary(result_path: Path) -> dict:
    trace_path = result_path.parent / "token_entropy_full.jsonl"
    if not trace_path.exists():
        return {"available": False, "path": str(trace_path)}
    applied = 0
    similarities = []
    norm_ratios = []
    sources = set()
    rows = 0
    for row in load_jsonl(trace_path):
        rows += 1
        tokens = row.get("tokens") or row.get("token_trace") or []
        step0 = next((token for token in tokens if token.get("step") == 0), None)
        if not step0:
            continue
        if step0.get("early_visual_anchor_applied"):
            applied += 1
        source = step0.get("early_visual_anchor_source")
        if source:
            sources.add(source)
        value = step0.get("early_visual_anchor_query_similarity")
        if value is not None:
            similarities.append(float(value))
        value = step0.get("early_visual_anchor_norm_ratio")
        if value is not None:
            norm_ratios.append(float(value))

    def summarize(values: list[float]) -> dict:
        if not values:
            return {}
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "available": True,
        "path": str(trace_path),
        "rows": rows,
        "step0_applied": applied,
        "sources": sorted(sources),
        "query_similarity": summarize(similarities),
        "norm_ratio": summarize(norm_ratios),
    }


def compare(reference: dict[int, dict], method: dict[int, dict]) -> dict:
    common = sorted(set(reference) & set(method))
    fixed = [
        sample_id
        for sample_id in common
        if not reference[sample_id]["realworldqa_is_correct"]
        and method[sample_id]["realworldqa_is_correct"]
    ]
    damaged = [
        sample_id
        for sample_id in common
        if reference[sample_id]["realworldqa_is_correct"]
        and not method[sample_id]["realworldqa_is_correct"]
    ]
    return {
        "paired": len(common),
        "fixed": len(fixed),
        "damaged": len(damaged),
        "net": len(fixed) - len(damaged),
        "fixed_ids": fixed,
        "damaged_ids": damaged,
    }


def model_excerpt(row: dict, limit: int = 1200) -> str:
    text = row.get("model_answer") or ""
    return text if len(text) <= limit else text[:limit] + "..."


def write_case_cards(
    output_dir: Path,
    dataset_by_id: dict[int, dict],
    initial: dict[int, dict],
    static: dict[int, dict],
    actual: dict[int, dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_fixed = [
        sample_id
        for sample_id in sorted(actual)
        if sample_id in initial
        and not initial[sample_id]["realworldqa_is_correct"]
        and actual[sample_id]["realworldqa_is_correct"]
    ]
    for sample_id in actual_fixed[:12]:
        sample = dataset_by_id[sample_id]
        lines = [
            f"# Sample {sample_id}",
            "",
            f"- Gold: `{sample.get('answer')}`",
            f"- Static prediction: `{static.get(sample_id, {}).get('realworldqa_pred')}`",
            f"- Actual-visual prediction: `{actual[sample_id].get('realworldqa_pred')}`",
            f"- Image: `{sample.get('image')}`",
            "",
            "## Question",
            "",
            sample.get("question", ""),
            "",
            sample.get("options", ""),
            "",
            "## Initial Transition",
            "",
            model_excerpt(initial[sample_id]),
            "",
            "## Static Anchor",
            "",
            model_excerpt(static.get(sample_id, {})),
            "",
            "## Actual Visual Anchor",
            "",
            model_excerpt(actual[sample_id]),
            "",
        ]
        (output_dir / f"sample_{sample_id}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["hard", "control", "full"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-results", type=Path, required=True)
    parser.add_argument("--actual-results", type=Path, required=True)
    parser.add_argument("--static-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-dataset", type=Path)
    args = parser.parse_args()

    dataset_rows = load_jsonl(args.dataset)
    dataset_by_id = {int(row["id"]): row for row in dataset_rows}
    initial_report, initial = evaluate_path(dataset_rows, args.initial_results)
    actual_report, actual = evaluate_path(dataset_rows, args.actual_results)
    summary = {
        "stage": args.stage,
        "dataset": str(args.dataset),
        "initial_transition": initial_report,
        "actual_visual_anchor": actual_report,
        "actual_visual_anchor_trace": trace_summary(args.actual_results),
        "actual_vs_initial": compare(initial, actual),
    }

    static = {}
    if args.static_results:
        static_report, static = evaluate_path(dataset_rows, args.static_results)
        summary["static_anchor"] = static_report
        summary["static_anchor_trace"] = trace_summary(args.static_results)
        summary["static_vs_initial"] = compare(initial, static)
        summary["actual_vs_static"] = compare(static, actual)

    if args.stage == "hard":
        actual_fixed = summary["actual_vs_initial"]["fixed"]
        static_fixed = summary.get("static_vs_initial", {}).get("fixed", -1)
        summary["hard_gate_passed"] = bool(
            actual_fixed >= 5
            and actual_fixed > static_fixed
            and actual_report["runtime_errors"] == 0
            and actual_report["failed_extraction"] == 0
        )
        combined = []
        for sample_id in sorted(dataset_by_id):
            row = dict(dataset_by_id[sample_id])
            for name, source in (
                ("initial_transition", initial),
                ("static_anchor", static),
                ("actual_visual_anchor", actual),
            ):
                value = source.get(sample_id, {})
                row[name] = {
                    "pred": value.get("realworldqa_pred"),
                    "correct": value.get("realworldqa_is_correct"),
                    "output_tokens": value.get("output_tokens"),
                    "model_answer": value.get("model_answer"),
                }
            combined.append(row)
        dump_jsonl(args.output_dir / "hard54_predictions.jsonl", combined)
        dump_json(
            args.output_dir / "fixed_ids.json",
            {
                "static": summary.get("static_vs_initial", {}).get("fixed_ids", []),
                "actual_visual": summary["actual_vs_initial"]["fixed_ids"],
            },
        )
        dump_json(
            args.output_dir / "static_vs_visual_deltas.json",
            summary.get("actual_vs_static", {}),
        )
        write_case_cards(
            args.output_dir / "selected_case_cards",
            dataset_by_id,
            initial,
            static,
            actual,
        )
    elif args.stage == "control":
        cot_correct_ids = set(dataset_by_id)
        damaged = [
            sample_id
            for sample_id in sorted(cot_correct_ids)
            if not actual.get(sample_id, {}).get("realworldqa_is_correct", False)
        ]
        summary["cot_correct_control_damaged"] = len(damaged)
        summary["cot_correct_control_damaged_ids"] = damaged
        summary["control_gate_passed"] = bool(
            len(damaged) <= 2
            and actual_report["runtime_errors"] == 0
            and actual_report["failed_extraction"] == 0
        )

    dump_json(args.output_dir / f"{args.stage}_summary.json", summary)
    lines = [
        "# Early Actual-Visual Anchor 阶段结果",
        "",
        f"- 阶段：`{args.stage}`",
        f"- Initial Transition: {initial_report['correct']}/{initial_report['total']} ({initial_report['accuracy']:.2%})",
        f"- Actual visual anchor：{actual_report['correct']}/{actual_report['total']} ({actual_report['accuracy']:.2%})",
        f"- Actual vs Initial fixed/damaged：{summary['actual_vs_initial']['fixed']}/{summary['actual_vs_initial']['damaged']}",
        f"- Actual failed extraction/runtime error：{actual_report['failed_extraction']}/{actual_report['runtime_errors']}",
    ]
    if "static_anchor" in summary:
        lines.extend(
            [
                f"- Static anchor：{summary['static_anchor']['correct']}/{summary['static_anchor']['total']} ({summary['static_anchor']['accuracy']:.2%})",
                f"- Static vs Initial fixed/damaged：{summary['static_vs_initial']['fixed']}/{summary['static_vs_initial']['damaged']}",
            ]
        )
    if args.stage == "hard":
        lines.extend(
            [
                f"- Actual 独有修复：{summary['actual_vs_static']['fixed_ids']}",
                f"- Static 独有修复：{summary['actual_vs_static']['damaged_ids']}",
                f"- Hard gate passed：`{summary['hard_gate_passed']}`",
                "",
                "预注册门槛要求 actual 至少修复 5 题、严格超过 static，且 failed extraction/runtime error 均为 0。",
            ]
        )
    if args.stage == "control":
        lines.append(f"- COT-correct controls damaged: {summary['cot_correct_control_damaged']}")
        lines.append(f"- Control gate passed: `{summary['control_gate_passed']}`")
    (args.output_dir / f"{args.stage}_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    all_stages = {}
    for stage in ("hard", "control", "full"):
        stage_path = args.output_dir / f"{stage}_summary.json"
        if stage_path.exists():
            all_stages[stage] = json.loads(stage_path.read_text(encoding="utf-8"))
    dump_json(args.output_dir / "early_actual_visual_anchor_summary.json", all_stages)
    overview = [
        "# Early Actual-Visual Anchor 高优先级验证",
        "",
        "该实验只在 step 0 注入 question-conditioned visual-token hidden-state anchor，随后沿用 Initial Transition 并锁回 normal COT。",
        "静态 `<|image_pad|>` anchor 使用相同权重与范数对齐，控制一般 embedding 扰动效应。",
        "",
    ]
    for stage in ("hard", "control", "full"):
        if stage not in all_stages:
            continue
        value = all_stages[stage]
        overview.extend(
            [
                f"## {stage.title()} 阶段",
                "",
                f"- Actual：{value['actual_visual_anchor']['correct']}/{value['actual_visual_anchor']['total']} ({value['actual_visual_anchor']['accuracy']:.2%})",
                f"- 相对参照 fixed/damaged：{value['actual_vs_initial']['fixed']}/{value['actual_vs_initial']['damaged']}",
                "",
            ]
        )
        if stage == "hard":
            overview.extend(
                [
                    f"- Static：{value['static_anchor']['correct']}/{value['static_anchor']['total']} ({value['static_anchor']['accuracy']:.2%})",
                    f"- Actual/Static 独有修复：{value['actual_vs_static']['fixed']}/{value['actual_vs_static']['damaged']}",
                    f"- Actual failed extraction：{value['actual_visual_anchor']['failed_extraction']}",
                    f"- 预注册门槛：`{'通过' if value['hard_gate_passed'] else '未通过'}`",
                    "",
                    "Actual 与 Static 净收益相同，不能把修复归因于真实视觉内容；因此按计划停止 control20 与 fixed200 外推。",
                    "",
                ]
            )
    (args.output_dir / "early_actual_visual_anchor_summary.md").write_text(
        "\n".join(overview), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

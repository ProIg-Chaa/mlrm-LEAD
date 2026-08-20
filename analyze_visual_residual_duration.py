#!/usr/bin/env python3
"""Compare persistent visual residuals against matched hard and controls."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


BRANCHES = (
    "true_mask_residual",
    "random_residual",
    "reverse_mask_residual",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(path: Path, branch: str | None = None) -> list[dict]:
    rows = read_jsonl(path)
    if branch is not None:
        rows = [row for row in rows if row.get("branch") == branch]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--duration-root", type=Path, required=True)
    parser.add_argument("--p0-root", type=Path, required=True)
    parser.add_argument("--strength-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from analyze_visual_action_strength_atlas import (  # noqa: PLC0415
        compare,
        enrich,
        paired_branch,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "unit_warning": (
            "Event-level statistics; checkpoints are clustered within original_id. "
            "McNemar p-values are descriptive and not treated as sample-independent."
        ),
        "datasets": {},
    }

    specs = {
        "vstar": {
            "baseline": args.p0_root / "vstar_64events/results.jsonl",
            "l1": args.p0_root / "vstar_64events/results.jsonl",
            "duration_name": "vstar",
        },
        "realworldqa": {
            "baseline": args.p0_root / "realworldqa_64events/results.jsonl",
            "l1": args.p0_root / "realworldqa_64events/results.jsonl",
            "duration_name": "realworldqa",
        },
        "mmvp": {
            "baseline": args.strength_root / "hard/mmvp/results.jsonl",
            "l1": args.strength_root / "lambda_095/mmvp/results.jsonl",
            "duration_name": "mmvp_256events",
        },
    }

    for dataset, spec in specs.items():
        hard_rows = select(spec["baseline"], "hard")
        baseline = enrich(hard_rows)
        dataset_report = {
            "events": len(baseline),
            "independent_samples": len({row["original_id"] for row in hard_rows}),
            "durations": {},
        }
        for duration in (1, 2, 4):
            path = (
                spec["l1"]
                if duration == 1
                else args.duration_root
                / f"L{duration}"
                / spec["duration_name"]
                / "results.jsonl"
            )
            rows = read_jsonl(path)
            by_branch: dict[str, list[dict]] = defaultdict(list)
            for raw in rows:
                branch = str(raw.get("branch"))
                if branch not in BRANCHES:
                    continue
                by_branch[branch].append(enrich([raw])[str(raw["event_id"])])
            branch_maps = {
                branch: {row["event_id"]: row for row in items}
                for branch, items in by_branch.items()
            }
            duration_report = {
                "branches": {
                    branch: compare(items, baseline)
                    for branch, items in sorted(by_branch.items())
                },
                "paired_controls": {},
                "by_event_type": {},
            }
            true_map = branch_maps.get("true_mask_residual", {})
            for control in ("random_residual", "reverse_mask_residual"):
                if true_map and control in branch_maps:
                    duration_report["paired_controls"][
                        f"true_mask_residual_vs_{control}"
                    ] = paired_branch(true_map, branch_maps[control])
            grouped: dict[str, dict[str, list[dict]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for branch, items in by_branch.items():
                for item in items:
                    grouped[str(item["event_type"])][branch].append(item)
            for event_type, branch_rows in sorted(grouped.items()):
                duration_report["by_event_type"][event_type] = {
                    branch: compare(items, baseline)
                    for branch, items in sorted(branch_rows.items())
                }
            dataset_report["durations"][f"L{duration}"] = duration_report
        report["datasets"][dataset] = dataset_report

    json_path = args.output_dir / "visual_residual_duration_summary.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Visual Residual Duration Summary",
        "",
        report["unit_warning"],
        "",
        "| Dataset | Duration | Branch | Acc | Changed | Fixed | Damaged | Net | Failed |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_report in report["datasets"].items():
        for duration, duration_report in dataset_report["durations"].items():
            for branch, stats in duration_report["branches"].items():
                lines.append(
                    f"| {dataset} | {duration[1:]} | {branch} | "
                    f"{stats['accuracy']:.2%} | {stats['changed_vs_hard']} | "
                    f"{stats['fixed_vs_hard']} | {stats['damaged_vs_hard']} | "
                    f"{stats['net_vs_hard']:+d} | {stats['failed_extraction']} |"
                )
    lines.extend([
        "",
        "## True Residual Versus Controls",
        "",
        "| Dataset | Duration | Comparison | Target only | Control only | Net |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for dataset, dataset_report in report["datasets"].items():
        for duration, duration_report in dataset_report["durations"].items():
            for name, stats in duration_report["paired_controls"].items():
                lines.append(
                    f"| {dataset} | {duration[1:]} | {name} | "
                    f"{stats['target_only_correct']} | "
                    f"{stats['control_only_correct']} | {stats['net']:+d} |"
                )
    md_path = args.output_dir / "visual_residual_duration_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

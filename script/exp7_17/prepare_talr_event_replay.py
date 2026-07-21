#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from talr_analysis_common import load_jsonl, trace_by_id, write_json, write_jsonl


def quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def event_for_trace(trace: dict, branch: str) -> int | None:
    tokens = trace.get("tokens") or []
    if branch == "suppress_refinement":
        matches = [
            token
            for token in tokens
            if token.get("lead_refinement_active")
        ]
    elif branch == "bypass_guard":
        matches = [
            token
            for token in tokens
            if token.get("lead_refinement_candidate")
            and not token.get("lead_refinement_active")
            and (
                token.get("format_cooldown_active")
                or token.get("lead_soft_veto")
            )
        ]
    else:
        matches = [
            token
            for token in tokens
            if token.get("lead_refinement_candidate")
        ]
    return int(matches[0]["step"]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    locked = json.loads(args.locked_config.read_text(encoding="utf-8"))
    comparison = json.loads(
        args.comparison_manifest.read_text(encoding="utf-8")
    )
    selected = load_jsonl(args.selected_samples)
    selected_ids = {
        (row["dataset"], str(row["id"]))
        for row in selected
        if row["group"] in {"fixed", "damaged"}
        and row["method"] in {"talr", "true_talr"}
    }
    dataset_files = {
        "vstar": root / "data/vstar.jsonl",
        "mmvp": root / "data/mmvp.jsonl",
        "realworldqa_fixed200": root
        / "data/realworldqa_fixed_mcq_random200_seed42.jsonl",
        "visulogic300": root / "data/visulogic.jsonl",
    }
    method_args = list(locked["method_args"])
    jobs = []
    replay_manifest = []

    for entry in comparison["comparisons"]:
        if entry["model"] != "r1_onevision_7b_rl":
            continue
        dataset = entry["dataset"]
        talr_dir = Path(entry["runs"].get("talr", ""))
        if not talr_dir.exists():
            continue
        source = {
            str(row.get("id")): row
            for row in load_jsonl(dataset_files[dataset])
        }
        traces = trace_by_id(talr_dir)
        ids = sorted(
            sample_id
            for ds, sample_id in selected_ids
            if ds == dataset and sample_id in source and sample_id in traces
        )
        if not ids:
            continue
        subset_path = args.output_dir / "datasets" / f"{dataset}.jsonl"
        write_jsonl(subset_path, [source[sample_id] for sample_id in ids])
        for branch, override_kind in (
            ("actual", "none"),
            ("suppress_refinement", "hard"),
            ("bypass_guard", "method_soft"),
        ):
            event_map = {}
            for sample_id in ids:
                step = event_for_trace(traces[sample_id], branch)
                if step is not None:
                    event_map[sample_id] = step
            if not event_map:
                continue
            branch_subset = args.output_dir / "datasets" / f"{dataset}_{branch}.jsonl"
            write_jsonl(
                branch_subset,
                [source[sample_id] for sample_id in ids if sample_id in event_map],
            )
            event_path = args.output_dir / "events" / f"{dataset}_{branch}.json"
            write_json(event_path, event_map)
            run_dir = args.output_dir / "runs" / dataset / branch
            command = [
                str(args.python),
                "main.py",
                "--model_name",
                str(args.model),
                "--dataset",
                str(branch_subset),
                "--output_dir",
                str(run_dir),
                "--method",
                "lead",
                "--alpha",
                "0.4",
                "--max_switch_count",
                "5",
                "--window_size",
                "128",
                "--cot_prompt_mode",
                "orign",
                "--no-do_sample",
                "--temperature",
                "0.6",
                "--top_p",
                "0.95",
                "--top_k",
                "20",
                "--seed",
                "42",
                "--max_new_tokens",
                "1024",
                "--device",
                "cuda",
                "--save_token_entropy",
                "--save_full_token_entropy",
                "--trace_topk",
                "20",
                "--trace_route_override_manifest",
                str(event_path),
                "--trace_route_override_kind",
                override_kind,
                *method_args,
            ]
            jobs.append((command, run_dir))
            replay_manifest.append(
                {
                    "dataset": dataset,
                    "branch": branch,
                    "event_steps": str(event_path),
                    "reference_dir": str(talr_dir),
                    "run_dir": str(run_dir),
                }
            )

    write_json(args.output_dir / "replay_manifest.json", replay_manifest)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(root))}",
    ]
    for command, run_dir in jobs:
        lines.extend(
            [
                f"if [[ ! -f {shlex.quote(str(run_dir / 'eval_report.json'))} ]]; then",
                f"  {quote(command)}",
                "fi",
            ]
        )
    lines.append(
        f"{shlex.quote(str(args.python))} "
        "script/exp7_17/summarize_talr_event_replay.py "
        f"--replay-dir {shlex.quote(str(args.output_dir))}"
    )
    runner = args.output_dir / "run_talr_event_replay.sh"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runner.chmod(0o755)
    print(f"Wrote {runner} with {len(jobs)} replay runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

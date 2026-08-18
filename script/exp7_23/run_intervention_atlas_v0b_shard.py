#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


FIXED_STEPS = (1, 2, 4, 8, 16, 32)
TREATMENTS = {
    "contracted_soft_l095": ("contracted_soft", 0.95),
    "pure_soft_l100": ("raw_soft", 1.0),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def trace_by_id(path: Path) -> dict[str, dict]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def complete(run_dir: Path, expected: int) -> bool:
    try:
        rows = load_jsonl(run_dir / "results.jsonl")
    except (OSError, json.JSONDecodeError):
        return False
    required = (
        "eval_report.json",
        "config.json",
        "token_entropy.jsonl",
        "token_entropy_full.jsonl",
    )
    return (
        len(rows) == expected
        and all((run_dir / name).exists() for name in required)
        and not any(row.get("error_type") for row in rows)
    )


def event_steps(sample_id: str, tokens: list[dict]) -> list[tuple[str, int]]:
    legal = list(range(max(0, len(tokens) - 1)))
    if not legal:
        return []
    events: list[tuple[str, int]] = []
    used: set[int] = set()
    for step in FIXED_STEPS:
        if step in legal:
            events.append((f"fixed_{step}", step))
            used.add(step)

    entropy_ranked = sorted(
        legal,
        key=lambda step: float(tokens[step].get("raw_entropy") or 0.0),
        reverse=True,
    )
    entropy_step = next((step for step in entropy_ranked if step not in used), None)
    if entropy_step is not None:
        events.append(("entropy_top1", entropy_step))
        used.add(entropy_step)

    remaining = [step for step in legal if step not in used]
    if remaining:
        digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
        random_step = remaining[int.from_bytes(digest[:8], "big") % len(remaining)]
        events.append(("random_control", random_step))
    return events


def run_main(
    args: argparse.Namespace,
    dataset: Path,
    run_dir: Path,
    expected: int,
    treatment: tuple[str, float] | None,
    override_manifest: Path | None = None,
) -> None:
    if complete(run_dir, expected):
        print(f"SKIP complete {run_dir}", flush=True)
        return
    if run_dir.exists():
        backup = run_dir.with_name(
            run_dir.name + f".incomplete.{int(time.time())}"
        )
        shutil.move(run_dir, backup)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(args.repo / "main.py"),
        "--model_name",
        str(args.model),
        "--dataset",
        str(dataset),
        "--output_dir",
        str(run_dir),
        "--method",
        "cot_greedy",
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
        "5" if treatment is None else "0",
    ]
    if treatment is not None:
        override_kind, mix_lambda = treatment
        command += [
            "--trace_route_override_manifest",
            str(override_manifest),
            "--trace_route_override_kind",
            override_kind,
            "--trace_route_override_mix_lambda",
            str(mix_lambda),
        ]
    print("START", run_dir, flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=args.repo,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not complete(run_dir, expected):
        raise RuntimeError(f"FAILED {run_dir}, rc={result.returncode}")
    print("DONE", run_dir, flush=True)


def prepare_events(
    shard_rows: list[dict], baseline_dir: Path, output_dir: Path
) -> tuple[Path, Path, int]:
    traces = trace_by_id(baseline_dir / "token_entropy_full.jsonl")
    expanded_rows = []
    manifest: dict[str, int] = {}
    event_rows = []
    for source in shard_rows:
        original_id = str(source.get("id"))
        trace = traces.get(original_id)
        if not trace:
            continue
        tokens = trace.get("tokens") or []
        for event_type, step in event_steps(original_id, tokens):
            event_id = f"{original_id}::atlas::{event_type}::{step}"
            expanded = dict(source)
            expanded["id"] = event_id
            expanded["_atlas_original_id"] = original_id
            expanded["_atlas_event_type"] = event_type
            expanded["_atlas_event_step"] = step
            expanded_rows.append(expanded)
            manifest[event_id] = step
            event_rows.append(
                {
                    "event_id": event_id,
                    "original_id": original_id,
                    "dataset": source.get("_atlas_dataset"),
                    "event_type": event_type,
                    "event_step": step,
                    "gold": source.get("answer"),
                    "subtopic": source.get("subtopic") or source.get("subject"),
                }
            )
    event_dataset = output_dir / "event_dataset.jsonl"
    event_manifest = output_dir / "event_override_manifest.json"
    write_jsonl(event_dataset, expanded_rows)
    write_jsonl(output_dir / "event_manifest.jsonl", event_rows)
    event_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return event_dataset, event_manifest, len(expanded_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.model = args.model.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = load_jsonl(args.shard)
    baseline_dir = args.output_dir / "hard_baseline"
    run_main(
        args,
        args.shard,
        baseline_dir,
        len(shard_rows),
        treatment=None,
    )
    event_dataset, override_manifest, event_count = prepare_events(
        shard_rows, baseline_dir, args.output_dir
    )
    for name, treatment in TREATMENTS.items():
        run_main(
            args,
            event_dataset,
            args.output_dir / name,
            event_count,
            treatment=treatment,
            override_manifest=override_manifest,
        )
    subprocess.run(
        [
            args.python,
            str(args.repo / "script/exp7_23/summarize_intervention_atlas_v0b.py"),
            "--shard-dir",
            str(args.output_dir),
        ],
        cwd=args.repo,
        check=True,
    )
    (args.output_dir / "SHARD_COMPLETE").write_text(
        f"shard={args.shard_index}\nsamples={len(shard_rows)}\nevents={event_count}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

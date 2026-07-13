#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from pathlib import Path


PYTHON = "/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL = "/dev/shm/wangzixu_models/R1-Onevision-7B-RL"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def method_run_dir(root: Path, dataset: str, method: str) -> Path:
    if method == "initial_transition_only":
        return root / "output/experiments/20260602_220321/rerun_early_path_dependence_mechanism/phase3_cross_dataset_minimal" / dataset / "initial_transition_only_gpu0"
    if method == "pure_soft_format2":
        return root / "output/experiments/20260706_format_stability_full_baselines/format_stability_full_baselines/r1_onevision_7b" / dataset / "pure_soft_format2_gpu0"
    raise ValueError(method)


def dataset_path(root: Path, dataset: str) -> Path:
    names = {
        "vstar": "vstar.jsonl",
        "mmvp": "mmvp.jsonl",
        "visulogic300": "visulogic.jsonl",
        "realworldqa_fixed200": "realworldqa_fixed_mcq_random200_seed42.jsonl",
    }
    return root / "data" / names[dataset]


def event_steps(trace: dict, method: str) -> dict[str, int]:
    tokens = trace.get("tokens") or []
    if method == "initial_transition_only":
        exits = [int(token["step"]) for token in tokens if token.get("to_normal")]
        return {"step0": 0, **({"to_normal": exits[0]} if exits else {})}
    format_tokens = [token for token in tokens if token.get("format_cooldown_active")]
    if not format_tokens:
        return {}
    first = int(format_tokens[0]["step"])
    max_risk = int(max(format_tokens, key=lambda token: float(token.get("raw_entropy", 0.0)))["step"])
    return {"format_first": first, "format_maxrisk": max_risk}


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def method_args(method: str) -> list[str]:
    if method == "initial_transition_only":
        return ["--method", "lead", "--alpha", "0.4", "--max_switch_count", "5", "--window_size", "128", "--lead_initial_transition_only"]
    return ["--method", "pure_soft", "--pure_soft_format_cooldown", "--format_cooldown_steps", "2"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["vstar", "mmvp"])
    parser.add_argument("--methods", nargs="+", default=["initial_transition_only", "pure_soft_format2"])
    parser.add_argument("--branches", nargs="+", default=["none", "hard", "raw_soft", "method_soft"])
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()
    root, analysis_dir, out_dir = args.root.resolve(), args.analysis_dir.resolve(), args.output_dir.resolve()
    selected = load_jsonl(analysis_dir / "selected_rows.jsonl")
    commands, replay_manifest = [], []
    for dataset in args.datasets:
        source_rows = {str(row.get("id")): row for row in load_jsonl(dataset_path(root, dataset))}
        for method in args.methods:
            chosen = [row for row in selected if row["dataset"] == dataset and row["method"] == method and row["group"] in {"fixed", "damaged", "both_correct", "both_wrong"}]
            if args.limit_samples is not None:
                chosen = chosen[:args.limit_samples]
            chosen_ids = {str(row["id"]) for row in chosen}
            if not chosen_ids:
                continue
            run_dir = method_run_dir(root, dataset, method)
            traces = {str(row.get("id")): row for row in load_jsonl(run_dir / "token_entropy_full.jsonl")}
            subset = [source_rows[sid] for sid in chosen_ids if sid in source_rows]
            subset_path = out_dir / "datasets" / f"{dataset}_{method}_selected.jsonl"
            write_jsonl(subset_path, subset)
            event_maps = defaultdict(dict)
            for sid in sorted(chosen_ids):
                for event, step in event_steps(traces.get(sid, {}), method).items():
                    event_maps[event][sid] = step
            for event, mapping in sorted(event_maps.items()):
                event_path = out_dir / "event_manifests" / dataset / method / f"{event}.json"
                write_json(event_path, mapping)
                branch_kinds = args.branches
                for branch in branch_kinds:
                    branch_name = "actual" if branch == "none" else branch
                    branch_dir = out_dir / "runs" / dataset / method / event / branch_name
                    command = [
                        PYTHON, "main.py", "--model_name", MODEL,
                        "--dataset", str(subset_path), "--output_dir", str(branch_dir),
                        "--max_new_tokens", "1024", "--temperature", "0.6", "--top_p", "0.95", "--top_k", "20",
                        "--seed", "42", "--device", "cuda", "--no-do_sample", "--cot_prompt_mode", "orign",
                        "--save_token_entropy", "--save_full_token_entropy", "--trace_topk", "20", "--trace_event_geometry",
                        "--trace_route_override_manifest", str(event_path), "--trace_route_override_kind", branch,
                        "--trace_forced_answer_probe",
                        *method_args(method),
                    ]
                    commands.append({"dataset": dataset, "method": method, "event": event, "branch": branch_name, "run_dir": str(branch_dir), "command": command})
                    replay_manifest.append({"dataset": dataset, "method": method, "event": event, "branch": branch_name, "event_steps": str(event_path), "reference_dir": str(run_dir), "run_dir": str(branch_dir)})
    write_json(out_dir / "replay_manifest.json", replay_manifest)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(root))}", f"PYTHON={shlex.quote(PYTHON)}", "$PYTHON -m py_compile main.py lead/inference.py lead/generation_utils.py"]
    for item in commands:
        result_path = Path(item["run_dir"]) / "eval_report.json"
        lines += [
            f"if [[ -f {shlex.quote(str(result_path))} ]]; then",
            f"  echo '[SKIP] {item['dataset']}/{item['method']}/{item['event']}/{item['branch']}'",
            "else",
            f"  echo '[START] {item['dataset']}/{item['method']}/{item['event']}/{item['branch']}'",
            f"  {quote_command(item['command'])}",
            "fi",
        ]
    lines.append(f"$PYTHON script/exp7_11/summarize_counterfactual_replay.py --root {shlex.quote(str(root))} --replay-dir {shlex.quote(str(out_dir))}")
    runner = out_dir / "run_counterfactual_replay.sh"
    runner.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runner.chmod(0o755)
    print(f"Wrote {runner} with {len(commands)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

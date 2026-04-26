#!/usr/bin/env python3
"""Run a small method comparison experiment and summarize metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

from lead.evaluator import evaluate_dataset, extract_mcq_answer
from lead.utils import load_json


METHODS = ("cot", "cot_greedy", "lead")


def should_echo(line: str) -> bool:
    keep = (
        "INFO" in line
        or "WARNING" in line
        or "ERROR" in line
        or "EVALUATION REPORT" in line
        or "Overall Accuracy" in line
        or "Failed Extractions" in line
        or "Loading weights:" in line
    )
    noisy = (
        line.startswith("=== Prompt ===")
        or line.startswith("=== Model Output ===")
        or line.startswith("input_ids len:")
        or line.startswith("image_grid_thw:")
    )
    return keep and not noisy


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown_report(run_root: Path, config: dict, summaries: list[dict]) -> None:
    lines = [
        "# Method Comparison Report",
        "",
        "## Experiment",
        "",
        f"- Run root: `{config.get('run_root', run_root)}`",
        f"- Project dir: `{config.get('project_dir')}`",
        f"- Environment: `{config.get('env_name')}`",
        f"- Model: `{config.get('model_name')}`",
        f"- Dataset: `{config.get('dataset')}`",
        f"- Limit: `{config.get('limit')}`",
        f"- Max new tokens: `{config.get('max_new_tokens')}`",
        f"- GPU index: `{config.get('gpu_index')}`",
        f"- Methods: `{', '.join(config.get('methods', []))}`",
        f"- Temperature: `{config.get('temperature')}`",
        f"- Top-p: `{config.get('top_p')}`",
        f"- Top-k: `{config.get('top_k')}`",
        f"- Alpha: `{config.get('alpha')}`",
        f"- Max switch count: `{config.get('max_switch_count')}`",
        f"- Seed: `{config.get('seed')}`",
        f"- Save token entropy: `{config.get('save_token_entropy')}`",
        "",
        "## Results",
        "",
        "| Method | Status | Accuracy | Avg latency (s) | Total latency (s) | Avg output tokens | Total output tokens | Max CUDA reserved (MB) | GPU peak delta (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        total = summary.get("total")
        correct = summary.get("correct")
        accuracy = summary.get("accuracy")
        accuracy_text = "-"
        if accuracy is not None:
            accuracy_text = f"{accuracy * 100:.2f}% ({correct}/{total})"
        lines.append(
            "| {method} | {status} | {accuracy} | {avg_latency} | {total_latency} | "
            "{avg_tokens} | {total_tokens} | {max_reserved} | {gpu_delta} |".format(
                method=summary.get("method", "-"),
                status=summary.get("return_code", "-"),
                accuracy=accuracy_text,
                avg_latency=fmt(summary.get("avg_latency_sec")),
                total_latency=fmt(summary.get("total_latency_sec")),
                avg_tokens=fmt(summary.get("avg_output_tokens")),
                total_tokens=fmt(summary.get("total_output_tokens"), 0),
                max_reserved=fmt(summary.get("max_cuda_peak_reserved_mb")),
                gpu_delta=fmt(summary.get("gpu_peak_delta_mb"), 0),
            )
        )

    lines.extend(["", "## Error Types", ""])
    for summary in summaries:
        lines.append(f"### {summary.get('method', '-')}")
        lines.append("")
        error_types = summary.get("error_types", {})
        if not error_types:
            lines.append("- None")
        else:
            for name, count in sorted(error_types.items()):
                lines.append(f"- `{name}`: {count}")
        lines.extend(
            [
                f"- Output dir: `{summary.get('output_dir')}`",
                f"- Run log: `{summary.get('run_log')}`",
                f"- Token entropy: `{summary.get('token_entropy_path', '-')}`",
                "",
            ]
        )

    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def query_gpu_used_mb(gpu_index: int) -> int | None:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    try:
        return int(line)
    except ValueError:
        return None


def monitor_gpu(
    gpu_index: int,
    stop_event: threading.Event,
    samples: list[dict],
    interval_sec: float,
) -> None:
    while not stop_event.is_set():
        used_mb = query_gpu_used_mb(gpu_index)
        if used_mb is not None:
            samples.append({"time": time.time(), "memory_used_mb": used_mb})
        stop_event.wait(interval_sec)


def run_method(args: argparse.Namespace, method: str, run_root: Path) -> dict:
    method_dir = run_root / method
    method_dir.mkdir(parents=True, exist_ok=True)
    run_log = method_dir / "run.log"
    gpu_samples: list[dict] = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(args.gpu_index, stop_event, gpu_samples, args.gpu_poll_interval),
        daemon=True,
    )

    cmd = [
        "micromamba",
        "run",
        "-n",
        args.env_name,
        "python",
        "main.py",
        "--model_name",
        args.model_name,
        "--dataset",
        args.dataset,
        "--output_dir",
        str(method_dir),
        "--method",
        method,
        "--limit",
        str(args.limit),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--alpha",
        str(args.alpha),
        "--max_switch_count",
        str(args.max_switch_count),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
    ]
    if method == "cot_greedy":
        cmd.append("--no-do_sample")
    if args.save_token_entropy:
        cmd.append("--save_token_entropy")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    baseline_mb = query_gpu_used_mb(args.gpu_index)
    started = time.time()
    monitor.start()
    with run_log.open("w", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=args.project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            if should_echo(line):
                print(f"[{method}] {line}", end="")
        return_code = proc.wait()
    stop_event.set()
    monitor.join(timeout=5)
    finished = time.time()

    gpu_csv = method_dir / "gpu_memory_samples.csv"
    with gpu_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "memory_used_mb"])
        writer.writeheader()
        writer.writerows(gpu_samples)

    results_path = method_dir / "results.jsonl"
    eval_report_path = method_dir / "eval_report.json"
    token_entropy_path = method_dir / "token_entropy.jsonl"
    rows = read_jsonl(results_path) if results_path.exists() else []
    eval_report = load_json(str(eval_report_path)) if eval_report_path.exists() else {}
    eval_by_script = evaluate_dataset(rows) if rows else {}

    error_types = Counter()
    latencies = []
    output_tokens = []
    cuda_peaks = []
    for row in rows:
        if row.get("latency_sec") is not None:
            latencies.append(float(row["latency_sec"]))
        if row.get("output_tokens") is not None:
            output_tokens.append(int(row["output_tokens"]))
        if row.get("cuda_peak_reserved_mb") is not None:
            cuda_peaks.append(float(row["cuda_peak_reserved_mb"]))

        if row.get("error_type"):
            error_types[f"runtime:{row['error_type']}"] += 1
            continue
        answer = row.get("model_answer")
        extracted = extract_mcq_answer(answer or "")
        if extracted is None:
            error_types["no_answer_extracted"] += 1
        elif extracted.upper() == str(row.get("answer", "")).strip().upper():
            error_types["correct"] += 1
        else:
            error_types["wrong_answer"] += 1

    peak_gpu_mb = max((s["memory_used_mb"] for s in gpu_samples), default=None)
    summary = {
        "method": method,
        "return_code": return_code,
        "output_dir": str(method_dir),
        "run_log": str(run_log),
        "token_entropy_path": str(token_entropy_path) if token_entropy_path.exists() else None,
        "started_at": datetime.fromtimestamp(started).isoformat(),
        "finished_at": datetime.fromtimestamp(finished).isoformat(),
        "wall_time_sec": finished - started,
        "num_results": len(rows),
        "accuracy": eval_by_script.get("accuracy", eval_report.get("accuracy")),
        "correct": eval_by_script.get("correct", eval_report.get("correct")),
        "total": eval_by_script.get("total", eval_report.get("total")),
        "failed_extraction": eval_by_script.get(
            "failed_extraction", eval_report.get("failed_extraction")
        ),
        "avg_latency_sec": mean(latencies) if latencies else None,
        "total_latency_sec": sum(latencies) if latencies else None,
        "avg_output_tokens": mean(output_tokens) if output_tokens else None,
        "total_output_tokens": sum(output_tokens) if output_tokens else None,
        "avg_cuda_peak_reserved_mb": mean(cuda_peaks) if cuda_peaks else None,
        "max_cuda_peak_reserved_mb": max(cuda_peaks) if cuda_peaks else None,
        "gpu_index": args.gpu_index,
        "gpu_baseline_used_mb": baseline_mb,
        "gpu_peak_used_mb": peak_gpu_mb,
        "gpu_peak_delta_mb": (
            peak_gpu_mb - baseline_mb
            if peak_gpu_mb is not None and baseline_mb is not None
            else None
        ),
        "error_types": dict(error_types),
    }
    write_json(method_dir / "method_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--env_name", default="mlrm-lead")
    parser.add_argument(
        "--model_name",
        default="/share/home/wangzixu/liudinghao/gushuo/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--dataset", default="data/physunibench.jsonl")
    parser.add_argument("--output_root", default="output/experiments")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gpu_index", type=int, default=1)
    parser.add_argument("--gpu_poll_interval", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--max_switch_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_token_entropy", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project_dir = str(Path(args.project_dir).resolve())
    run_root = (
        Path(args.project_dir)
        / args.output_root
        / ("compare_methods_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    run_root.mkdir(parents=True, exist_ok=False)

    config = vars(args).copy()
    config["run_root"] = str(run_root)
    write_json(run_root / "experiment_config.json", config)

    summaries = []
    write_markdown_report(run_root, config, summaries)
    for method in args.methods:
        summary = run_method(args, method, run_root)
        summaries.append(summary)
        write_json(run_root / "summary.json", summaries)
        write_markdown_report(run_root, config, summaries)
        if summary["return_code"] != 0:
            break

    write_json(run_root / "summary.json", summaries)
    write_markdown_report(run_root, config, summaries)
    print(f"Experiment root: {run_root}")
    return 0 if all(s["return_code"] == 0 for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())

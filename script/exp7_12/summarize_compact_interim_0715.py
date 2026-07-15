#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


MIGRATED = Path("/root/gushuo/migrated_results/rl_compact_matrix_migration_20260713/reusable_results/r1_onevision_7b_rl")
NEW_RL = Path("/root/autodl-tmp/gushuo/outputs/experiments/20260712_uniform_multimodel_full_matrix/uniform_multimodel_full_matrix/r1_onevision_7b_rl")
NEW_VISION = Path("/root/autodl-tmp/gushuo/outputs/experiments/20260714_vision_r1_compact_matrix/vision_r1_7b")
OUTPUT = Path("/root/gushuo/proj/mlrm-LEAD/result/5-27")

DATASETS = [
    "vstar",
    "realworldqa_fixed200",
    "mmvp",
    "visulogic300",
    "vmcbench_dev",
    "pope_adversarial",
    "mmk12_physics",
]
METHODS = [
    "cot_orign_greedy",
    "lead",
    "initial_transition_only",
    "talr",
]
LABELS = {
    "cot_orign_greedy": "COT",
    "lead": "LEAD",
    "initial_transition_only": "Initial Transition",
    "talr": "TALR",
}
METHOD_ALIASES = {
    "cot_orign_greedy": "cot_orign_greedy",
    "lead": "lead",
    "initial_transition_only": "initial_transition_only",
    "talr": "talr",
    "transition_preserving_quota05_guard_min2": "talr",
}
EXPECTED = {
    "vstar": 191,
    "realworldqa_fixed200": 200,
    "mmvp": 300,
    "visulogic300": 300,
    "vmcbench_dev": 1000,
    "pope_adversarial": 3000,
    "mmk12_physics": 500,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def collect_run(run_dir: Path, model: str, dataset: str, method: str, source: str) -> dict | None:
    result_path = run_dir / "results.jsonl"
    eval_path = run_dir / "eval_report.json"
    if not result_path.is_file() or not eval_path.is_file():
        return None
    rows = read_jsonl(result_path)
    if len(rows) != EXPECTED[dataset]:
        return None
    specialized_path = run_dir / "specialized_eval_report.json"
    metric_path = specialized_path if dataset == "mmvp" and specialized_path.is_file() else eval_path
    report = json.loads(metric_path.read_text(encoding="utf-8"))
    lengths = [int(row.get("output_tokens") or 0) for row in rows]
    latencies = [float(row.get("latency_sec") or 0.0) for row in rows]
    errors = sum(bool(row.get("error_type")) for row in rows)
    return {
        "model": model,
        "dataset": dataset,
        "method": method,
        "source": source,
        "run_dir": str(run_dir),
        "accuracy": float(report["accuracy"]),
        "correct": int(report["correct"]),
        "total": int(report["total"]),
        "failed_extraction": int(report.get("failed_extraction", 0)),
        "pair_accuracy": report.get("pair_accuracy"),
        "evaluator": "MMVP specialized" if metric_path == specialized_path else "run evaluator",
        "runtime_errors": errors,
        "mean_output_tokens": statistics.fmean(lengths) if lengths else 0.0,
        "mean_latency_sec": statistics.fmean(latencies) if latencies else 0.0,
        "long_ge_256": sum(value >= 256 for value in lengths),
        "maxed_1024": sum(value >= 1024 for value in lengths),
    }


def scan_root(root: Path, model: str, source: str) -> dict[tuple[str, str], dict]:
    found = {}
    if not root.is_dir():
        return found
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name not in EXPECTED:
            continue
        for method_dir in dataset_dir.iterdir():
            method = METHOD_ALIASES.get(method_dir.name)
            if method is None:
                continue
            run = collect_run(method_dir, model, dataset_dir.name, method, source)
            if run:
                found[(dataset_dir.name, method)] = run
    return found


def fmt(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    rl = scan_root(MIGRATED, "R1-Onevision-7B-RL", "migrated")
    rl.update(scan_root(NEW_RL, "R1-Onevision-7B-RL", "NewGpu"))
    vision = scan_root(NEW_VISION, "Vision-R1-7B", "NewGpu")
    all_runs = list(rl.values()) + list(vision.values())

    for runs in (rl, vision):
        for dataset in DATASETS:
            cot = runs.get((dataset, "cot_orign_greedy"))
            for method in METHODS:
                run = runs.get((dataset, method))
                if run:
                    run["delta_vs_cot"] = run["accuracy"] - cot["accuracy"] if cot else None

    lines = [
        "# 紧凑主矩阵阶段性结果（2026-07-15）",
        "",
        "## 口径与覆盖",
        "",
        "本报告只纳入结果行数完整且同时存在 `eval_report.json` 的 run；正在运行的 partial run 不进入表格。R1-RL 合并配置匹配的历史迁移结果和 NewGpu 补跑结果，Vision-R1 使用 NewGpu 新结果。",
        "",
        f"- R1-Onevision-7B-RL：{len(rl)}/28 个紧凑矩阵 run 已完成。",
        f"- Vision-R1-7B：{len(vision)}/28 个紧凑矩阵 run 已完成。",
        f"- 合计：{len(all_runs)}/56。",
        "- 主生成口径：greedy、seed 42、max_new_tokens 1024、origin COT prompt。",
        "- MMVP 使用 specialized evaluator，并同时报告 sample accuracy 与 pair accuracy；POPE precision/recall/F1 和其他 corrected evaluator 将在最终统一汇总中补充。",
        "",
    ]

    for model, runs in [("R1-Onevision-7B-RL", rl), ("Vision-R1-7B", vision)]:
        lines += [f"## {model}", "", "| Dataset | Method | Accuracy | Pair acc | Delta vs COT | Failed | Avg tokens | Long>=256 | Maxed1024 | Runtime errors |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for dataset in DATASETS:
            for method in METHODS:
                run = runs.get((dataset, method))
                if not run:
                    continue
                delta = run.get("delta_vs_cot")
                delta_text = "-" if delta is None else f"{100 * delta:+.2f} pp"
                pair_text = "-" if run["pair_accuracy"] is None else fmt(float(run["pair_accuracy"]))
                lines.append(
                    f"| {dataset} | {LABELS[method]} | {fmt(run['accuracy'])} | {pair_text} | {delta_text} | "
                    f"{run['failed_extraction']} | {run['mean_output_tokens']:.1f} | {run['long_ge_256']} | "
                    f"{run['maxed_1024']} | {run['runtime_errors']} |"
                )
        lines.append("")

    lines += [
        "## 当前可读结论",
        "",
        "1. R1-RL 的跨数据集结果仍然是明显异质的：没有一种 latent 方法在所有 benchmark 上稳定优于 COT。",
        "2. Vision-R1 的 MMK12-Physics 上，LEAD/TALR 明显高于 COT，而 Initial Transition 下降；这说明 early transition 的收益并非无条件跨模型成立。",
        "3. Vision-R1 的 POPE-Adversarial 上四种方法几乎持平，说明该 benchmark 对这些路由改动不敏感，或其主要瓶颈不在生成轨迹初始化。",
        "4. Format/guard 更适合被解释为稳定组件；是否提高 reasoning accuracy 必须按模型和数据集分别验证。",
        "",
        "## 已知审计事项",
        "",
        "- R1-RL 的 POPE/TALR accuracy 与结果文件有效，但其 `token_entropy.jsonl` 曾被重复 worker 并发写入，不能用于触发次数或 soft-ratio 统计。",
        "- 当前报告是阶段性 sample-accuracy 汇总，不替代最终 corrected/specialized evaluator 主表。",
        "- 正在运行的 Vision-R1 MMVP 及后续数据集将在完整落盘后自动进入下一版报告。",
    ]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "compact_matrix_interim_results_20260715.json").write_text(
        json.dumps({"runs": all_runs}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "compact_matrix_interim_results_20260715.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"R1-RL={len(rl)}/28 Vision-R1={len(vision)}/28 total={len(all_runs)}/56")


if __name__ == "__main__":
    main()

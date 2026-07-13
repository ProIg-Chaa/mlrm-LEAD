#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("output/experiments/20260712_pure_soft_confidence_correctness")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def f3(value):
    return "-" if value is None else f"{value:.3f}"


def main():
    stats = load_json(ROOT / "confidence_correctness_summary.json")
    wrong_samples = load_jsonl(ROOT / "high_confidence_semantic_wrong_samples.jsonl")
    by_dataset = {item["dataset"]: item for item in stats}
    total = sum(item["summary"]["total"] for item in stats)
    failed = sum(item["summary"]["failed_extraction"] for item in stats)
    strict_below = sum(item["summary"]["score_stats"]["mean_raw_conf"]["auc"] < 0.5 for item in stats)
    semantic_nonpredictive = sum(item["summary"]["score_stats"]["mean_raw_conf"]["semantic_only_auc"] <= 0.55 for item in stats)

    lines = [
        "# Pure-Soft 推理中的高置信错误实验",
        "",
        "## 目录",
        "",
        "1. [为什么做这次实验](#1-为什么做这次实验)",
        "2. [实验对象与口径](#2-实验对象与口径)",
        "3. [指标解释](#3-指标解释)",
        "4. [主要结果](#4-主要结果)",
        "5. [排除抽取失败后的结果](#5-排除抽取失败后的结果)",
        "6. [置信度随推理阶段的变化](#6-置信度随推理阶段的变化)",
        "7. [高置信阈值组](#7-高置信阈值组)",
        "8. [长度控制](#8-长度控制)",
        "9. [代表性高置信错题](#9-代表性高置信错题)",
        "10. [结论与边界](#10-结论与边界)",
        "",
        "## 1. 为什么做这次实验",
        "",
        "此前 VStar50 和 MMVP300 的 pure-soft 分析发现，错题可能具有更高 token confidence、更低 entropy、更长输出。旧实验主要比较 correct/wrong 均值和最高置信样本，尚不足以回答“confidence 能否作为最终正确性的可靠排序信号”。本次实验用当前统一模型、greedy、seed42、1024-token 全量结果重新分析，并增加 AUROC、置信度十分位、risk-coverage、semantic-only 和长度分层控制。",
        "",
        "## 2. 实验对象与口径",
        "",
        "本实验只分析真正的 `pure_soft`，不包含 COT、LEAD、format2 或 guard。复用已完成的 5 个 full run，不重新生成，因此没有额外 sampling 噪声。",
        "",
        f"总计 `{total}` 个样本，覆盖 VStar、MMVP、VisuLogic300、VMCBench-dev、MMK12-Physics；其中严格口径下 failed extraction 共 `{failed}` 个。所有结果使用 corrected/specialized evaluator。",
        "",
        "置信度使用每一步所选 greedy token 在过滤前完整词表分布中的概率 `raw_selected_prob`。pure-soft greedy 下它等价于 raw top-1 probability。禁止使用 top-k/top-p 过滤后的 `selected_prob`，因为该值经常接近 1，会制造虚假高置信。",
        "",
        "## 3. 指标解释",
        "",
        "- **Strict accuracy/AUC**：抽取失败也算错误，回答“这种轨迹能否可靠地产生正确可用答案”。",
        "- **Semantic-only AUC**：排除抽取失败，只比较成功抽取答案中的语义正确与错误。",
        "- **Confidence AUROC**：用 sample mean raw confidence 排序最终正确性；0.5 表示随机，低于 0.5 表示错误样本反而更自信。",
        "- **Top10 delta**：最高置信 10% 的准确率减总体准确率。若 confidence 可靠，该值应明显为正。",
        "- **Length-controlled AUC**：在输出长度四分位内分别计算再加权，控制长输出退化的影响。",
        "- **Early32/Tail20**：分别观察开头 32 token 与最后 20 token，区分早期不确定和后期错误锁定。",
        "",
        "## 4. 主要结果",
        "",
        "| 数据集 | strict acc | failed | mean-conf AUC (95% CI) | top10 strict delta | wrong-conf - correct-conf |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        summary = item["summary"]
        score = summary["score_stats"]["mean_raw_conf"]
        ci = score["ci95"]
        lines.append(
            f"| {item['dataset']} | {pct(summary['accuracy'])} | {summary['failed_extraction']} | "
            f"{score['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{100*score['top_decile_delta']:+.2f}pp | {score['wrong_minus_correct']:+.4f} |"
        )
    lines += [
        "",
        f"五个数据集的 strict confidence AUC 全部低于 0.5（{strict_below}/5）。MMVP、VisuLogic、VMCBench 的 95% CI 上界也低于 0.5，表明在这些设置上，高 confidence 不仅不能预测正确，反而稳定地偏向错误。最高置信 10% 的准确率比总体低 15.33–48.20 个百分点。",
        "",
        "所有数据集的 wrong mean confidence 都高于 correct，差值为 +0.0058 到 +0.0327。这个方向与旧 VStar/MMVP 观察一致。",
        "",
        "## 5. 排除抽取失败后的结果",
        "",
        "| 数据集 | semantic baseline acc | semantic AUC (95% CI) | top10 semantic delta |",
        "|---|---:|---:|---:|",
    ]
    for item in stats:
        summary = item["summary"]
        score = summary["score_stats"]["mean_raw_conf"]
        ci = score["semantic_only_auc_ci95"]
        semantic_total = summary["total"] - summary["failed_extraction"] - summary["runtime_error"]
        semantic_acc = summary["correct"] / semantic_total if semantic_total else None
        lines.append(
            f"| {item['dataset']} | {pct(semantic_acc)} | {score['semantic_only_auc']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {100*score['semantic_top_decile_delta']:+.2f}pp |"
        )
    lines += [
        "",
        f"排除格式/抽取失败后，{semantic_nonpredictive}/5 个数据集的 AUC 仍不超过 0.55。VisuLogic 的 AUC=0.425，仍明显表现为语义错题更自信；MMVP 和 VMCBench 约为 0.49/0.48，基本没有预测力。",
        "",
        "VStar 是重要反例：semantic AUC=0.611，说明在能稳定抽取答案的样本中，confidence 有一定正相关。因此不能写成“pure-soft 在所有数据集上错题一定更自信”。正确表述是：confidence 不是跨数据集可靠的 correctness signal，并存在大量高置信语义错误。",
        "",
        "## 6. 置信度随推理阶段的变化",
        "",
        "| 数据集 | early32 strict/semantic AUC | mean strict/semantic AUC | tail20 strict/semantic AUC |",
        "|---|---:|---:|---:|",
    ]
    for item in stats:
        scores = item["summary"]["score_stats"]
        early, middle, tail = scores["early32_raw_conf"], scores["mean_raw_conf"], scores["tail20_raw_conf"]
        lines.append(
            f"| {item['dataset']} | {early['auc']:.3f}/{early['semantic_only_auc']:.3f} | "
            f"{middle['auc']:.3f}/{middle['semantic_only_auc']:.3f} | "
            f"{tail['auc']:.3f}/{tail['semantic_only_auc']:.3f} |"
        )
    lines += [
        "",
        "Early32 strict AUC 位于 0.440–0.536，几乎等于随机，说明模型在开头并没有一个可用于判断最终正确性的可靠 confidence signal。Tail20 strict AUC 降至 0.381–0.437：错误轨迹在后续展开中往往变得更确定，而不是持续保持高熵犹豫。",
        "",
        "这与 early trajectory commitment 相容：模型可能较早进入错误路径，随后 token distribution 逐渐收缩；尾部高 confidence 反映的是轨迹已经锁定，不代表该轨迹与图像或 gold 一致。",
        "",
        "## 7. 高置信阈值组",
        "",
        "| 数据集 | mean conf≥0.90 n/strict/semantic acc | tail20 conf≥0.95 n/strict/semantic acc |",
        "|---|---:|---:|",
    ]
    for item in stats:
        groups = item["summary"]["high_confidence_groups"]
        mean_group, tail_group = groups["mean_raw_conf_ge_090"], groups["tail20_raw_conf_ge_095"]
        lines.append(
            f"| {item['dataset']} | {mean_group['count']} / {pct(mean_group['accuracy'])} / {pct(mean_group['semantic_accuracy'])} | "
            f"{tail_group['count']} / {pct(tail_group['accuracy'])} / {pct(tail_group['semantic_accuracy'])} |"
        )
    lines += [
        "",
        "典型例子是 MMVP：mean confidence≥0.90 的 27 条样本 strict accuracy 只有 11.11%；即使排除抽取失败，6 条可抽取样本也只有 50% 正确。VMCBench 同一阈值组 strict accuracy 为 22.76%，semantic accuracy 为 54.90%，均明显低于其总体/semantic baseline。",
        "",
        "## 8. 长度控制",
        "",
        "| 数据集 | correct/wrong mean length | raw AUC | length-controlled AUC |",
        "|---|---:|---:|---:|",
    ]
    for item in stats:
        summary = item["summary"]
        score = summary["score_stats"]["mean_raw_conf"]
        lines.append(
            f"| {item['dataset']} | {summary['mean_output_tokens_correct']:.1f}/{summary['mean_output_tokens_wrong']:.1f} | "
            f"{score['auc']:.3f} | {score['length_stratified']['weighted_auc']:.3f} |"
        )
    lines += [
        "",
        "五个数据集的错题都更长。控制长度后，AUC 收敛到 0.454–0.546：VMCBench 的强反向关系主要由长输出/抽取退化放大；MMVP 和 VisuLogic 仍保留一定反向趋势。",
        "",
        "因此存在两种高置信错误：一类是语义上选错但输出格式正常；另一类是 soft trajectory 进入长输出/重复退化后，token distribution 极度收缩。两者都说明 token confidence 不能直接当成最终答案可靠性。",
        "",
        "## 9. 代表性高置信错题",
        "",
        "以下样本均已成功抽取答案，排除了纯格式失败：",
        "",
        "| 数据集 | id | pred/gold | mean conf | early32 | tail20 | entropy | tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in by_dataset:
        examples = [row for row in wrong_samples if row["dataset"] == dataset][:3]
        for row in examples:
            lines.append(
                f"| {dataset} | {row['id']} | {row['pred']}/{row['gold']} | {row['mean_raw_conf']:.4f} | "
                f"{row['early32_raw_conf']:.4f} | {row['tail20_raw_conf']:.4f} | "
                f"{row['mean_raw_entropy']:.4f} | {row['output_tokens']} |"
            )
    lines += [
        "",
        "例如 VStar id=158 的预测为 A、gold 为 B：early32 confidence 已达 0.9359，整段 mean confidence=0.9722，tail20=0.9996，最终仍然错误。这是较干净的“早期就高置信地走错”案例。",
        "",
        "## 10. 结论与边界",
        "",
        "1. **高 token confidence 不是 pure-soft 最终正确性的可靠信号。** 五个数据集 strict AUC 全部低于 0.5，最高置信 10% 反而显著更差。",
        "2. **模型确实会自信地犯错。** 排除抽取失败后仍保留 100 条最高置信语义错题；VisuLogic 上语义 AUC 显著低于 0.5。",
        "3. **高置信错误的一部分来自轨迹锁定和生成退化。** 错题普遍更长，尾部 confidence 比早期更反向；这不是简单的早期低置信。",
        "4. **结论不是‘confidence 永远反向’。** VStar semantic-only 显示一定正相关，MMK12-Physics 也有弱正趋势；真正可靠的主张是 confidence 缺乏跨任务校准性。",
        "5. **这里的 confidence 是 token distribution concentration，不是模型显式报告的最终答案置信度。** 因此论文表述应使用 ‘token-level confidence does not imply final correctness’ 或 ‘pure-soft can confidently follow an incorrect trajectory’。",
        "",
        "最终结论：",
        "",
        "> Pure-soft reasoning often becomes highly confident after committing to a trajectory, but this confidence is poorly calibrated to final-answer correctness. High confidence can indicate trajectory lock-in or degeneration rather than reliable multimodal reasoning.",
        "",
    ]
    report = "\n".join(lines)
    output = ROOT / "pure_soft_confidence_correctness_report_20260712.md"
    output.write_text(report, encoding="utf-8")
    result_dir = Path("result/5-27")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / output.name).write_text(report, encoding="utf-8")
    print(f"Wrote {output} and {result_dir / output.name}")


if __name__ == "__main__":
    main()

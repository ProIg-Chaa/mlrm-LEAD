"""
PhysUniBench 评测主入口。

使用 LEAD 或 CoT 方法，
在 PhysUniBench 基准上对视觉语言模型进行推理评测。

用法示例：
    python main.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --method lead
    python main.py --limit 10 --max_new_tokens 512   # 调试模式
    python main.py --eval_only --results output/results.jsonl  # 仅评估
"""

import argparse
import json
import os
import re

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)

from lead import (
    run_single_inference,
    load_dataset,
    get_dataset_statistics,
    evaluate_dataset,
    print_evaluation_report,
    save_evaluation_report,
    format_prompt_from_sample,
    setup_logger,
    save_jsonl,
    save_json,
    Timer,
)


def percentile(values, q):
    """Return a simple percentile from an in-memory list."""
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def entropy_stats(values):
    """Compact entropy statistics for a list of scalar values."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
            "high_gt_1_ratio": None,
            "high_gt_2_ratio": None,
        }
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "mean": sum(ordered) / count,
        "median": percentile(ordered, 0.5),
        "p90": percentile(ordered, 0.9),
        "max": ordered[-1],
        "high_gt_1_ratio": sum(v > 1.0 for v in ordered) / count,
        "high_gt_2_ratio": sum(v > 2.0 for v in ordered) / count,
    }


RELATION_MARKER_CATEGORIES = {
    "conclusion": {
        "therefore",
        "thus",
        "hence",
        "consequently",
        "accordingly",
    },
    "contrast": {
        "however",
        "but",
        "although",
        "though",
        "instead",
        "while",
        "whereas",
    },
    "causal_condition": {
        "because",
        "since",
        "so",
        "if",
        "when",
        "as",
        "given",
        "assuming",
        "implies",
        "means",
    },
    "sequence": {
        "then",
        "first",
        "second",
        "third",
        "next",
        "finally",
        "now",
        "also",
        "moreover",
        "furthermore",
    },
    "result": {
        "result",
        "results",
        "resulting",
        "thereby",
    },
}
RELATION_MARKERS = set().union(*RELATION_MARKER_CATEGORIES.values())


def normalize_token_text(text):
    """Normalize a decoded single-token text for relation-marker matching."""
    normalized = text.strip().lower()
    normalized = re.sub(r"^[^a-z]+|[^a-z]+$", "", normalized)
    return normalized


def annotate_token_trace(tokenizer, trace):
    """Annotate per-token traces with decoded text and reasoning/relation flags."""
    if not trace:
        return []

    token_texts = [
        tokenizer.decode(
            [token["token_id"]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token in trace
    ]
    spans = []
    full_text = ""
    for text in token_texts:
        start = len(full_text)
        full_text += text
        spans.append((start, len(full_text)))

    think_start = full_text.find("<think>")
    think_end = full_text.find("</think>", think_start + 1) if think_start >= 0 else -1
    reasoning_start = think_start if think_start >= 0 else None
    reasoning_end = (
        think_end + len("</think>")
        if think_end >= 0
        else len(full_text) if think_start >= 0 else None
    )

    annotated = []
    for token, token_text, (start, end) in zip(trace, token_texts, spans):
        is_reasoning = (
            reasoning_start is not None
            and reasoning_end is not None
            and end > reasoning_start
            and start < reasoning_end
        )
        normalized_text = normalize_token_text(token_text)
        relation_category = None
        for category, markers in RELATION_MARKER_CATEGORIES.items():
            if normalized_text in markers:
                relation_category = category
                break

        annotated_token = dict(token)
        annotated_token["token_text"] = token_text
        annotated_token["normalized_token_text"] = normalized_text
        annotated_token["is_reasoning_token"] = is_reasoning
        annotated_token["is_relation_token"] = relation_category is not None
        annotated_token["relation_category"] = relation_category
        annotated.append(annotated_token)

    return annotated


def build_entropy_summary(tokenizer, trace):
    """Build compact entropy stats, with robust <think> span detection."""
    if not trace:
        return {
            "token_count": 0,
            "think_opened": False,
            "think_closed": False,
            "reasoning_token_count": 0,
            "reasoning_token_ratio": None,
            "all_raw_entropy": entropy_stats([]),
            "reasoning_raw_entropy": entropy_stats([]),
            "non_reasoning_raw_entropy": entropy_stats([]),
            "all_filtered_entropy": entropy_stats([]),
            "reasoning_filtered_entropy": entropy_stats([]),
            "non_reasoning_filtered_entropy": entropy_stats([]),
            "soft_token_count": 0,
            "soft_ratio": None,
            "reasoning_soft_token_count": 0,
            "reasoning_soft_ratio": None,
            "relation_token_count": 0,
            "relation_token_ratio": None,
            "reasoning_relation_token_count": 0,
            "reasoning_relation_token_ratio": None,
            "relation_raw_entropy": entropy_stats([]),
            "non_relation_raw_entropy": entropy_stats([]),
            "reasoning_relation_raw_entropy": entropy_stats([]),
            "reasoning_non_relation_raw_entropy": entropy_stats([]),
            "relation_filtered_entropy": entropy_stats([]),
            "reasoning_relation_filtered_entropy": entropy_stats([]),
            "relation_marker_counts": {},
            "relation_category_stats": {},
        }

    annotated_trace = annotate_token_trace(tokenizer, trace)
    think_start = any(token["token_text"] == "<think>" for token in annotated_trace)
    think_closed = any("</think>" in token["token_text"] for token in annotated_trace)

    all_raw = []
    reasoning_raw = []
    non_reasoning_raw = []
    all_filtered = []
    reasoning_filtered = []
    non_reasoning_filtered = []
    soft_token_count = 0
    reasoning_soft_token_count = 0
    reasoning_token_count = 0
    relation_token_count = 0
    reasoning_relation_token_count = 0
    relation_raw = []
    non_relation_raw = []
    reasoning_relation_raw = []
    reasoning_non_relation_raw = []
    relation_filtered = []
    reasoning_relation_filtered = []
    relation_marker_counts = {}
    relation_category_raw = {
        category: [] for category in RELATION_MARKER_CATEGORIES
    }

    for token in annotated_trace:
        is_reasoning = token["is_reasoning_token"]
        normalized_text = token["normalized_token_text"]
        relation_category = token["relation_category"]
        is_relation_marker = token["is_relation_token"]

        is_soft = token.get("mode") == "soft"
        if is_reasoning:
            reasoning_token_count += 1
        if is_relation_marker:
            relation_token_count += 1
            relation_marker_counts[normalized_text] = (
                relation_marker_counts.get(normalized_text, 0) + 1
            )
            if is_reasoning:
                reasoning_relation_token_count += 1
        if is_soft:
            soft_token_count += 1
            if is_reasoning:
                reasoning_soft_token_count += 1

        raw_entropy = token.get("raw_entropy")
        if raw_entropy is not None:
            raw_entropy = float(raw_entropy)
            all_raw.append(raw_entropy)
            if is_reasoning:
                reasoning_raw.append(raw_entropy)
            else:
                non_reasoning_raw.append(raw_entropy)
            if is_relation_marker:
                relation_raw.append(raw_entropy)
                relation_category_raw[relation_category].append(raw_entropy)
                if is_reasoning:
                    reasoning_relation_raw.append(raw_entropy)
            else:
                non_relation_raw.append(raw_entropy)
                if is_reasoning:
                    reasoning_non_relation_raw.append(raw_entropy)

        filtered_entropy = token.get("filtered_entropy")
        if filtered_entropy is not None:
            filtered_entropy = float(filtered_entropy)
            all_filtered.append(filtered_entropy)
            if is_reasoning:
                reasoning_filtered.append(filtered_entropy)
            else:
                non_reasoning_filtered.append(filtered_entropy)
            if is_relation_marker:
                relation_filtered.append(filtered_entropy)
                if is_reasoning:
                    reasoning_relation_filtered.append(filtered_entropy)

    token_count = len(trace)
    return {
        "token_count": token_count,
        "think_opened": think_start,
        "think_closed": think_closed,
        "reasoning_token_count": reasoning_token_count,
        "reasoning_token_ratio": (
            reasoning_token_count / token_count if token_count else None
        ),
        "all_raw_entropy": entropy_stats(all_raw),
        "reasoning_raw_entropy": entropy_stats(reasoning_raw),
        "non_reasoning_raw_entropy": entropy_stats(non_reasoning_raw),
        "all_filtered_entropy": entropy_stats(all_filtered),
        "reasoning_filtered_entropy": entropy_stats(reasoning_filtered),
        "non_reasoning_filtered_entropy": entropy_stats(non_reasoning_filtered),
        "soft_token_count": soft_token_count,
        "soft_ratio": soft_token_count / token_count if token_count else None,
        "reasoning_soft_token_count": reasoning_soft_token_count,
        "reasoning_soft_ratio": (
            reasoning_soft_token_count / reasoning_token_count
            if reasoning_token_count
            else None
        ),
        "relation_token_count": relation_token_count,
        "relation_token_ratio": relation_token_count / token_count if token_count else None,
        "reasoning_relation_token_count": reasoning_relation_token_count,
        "reasoning_relation_token_ratio": (
            reasoning_relation_token_count / reasoning_token_count
            if reasoning_token_count
            else None
        ),
        "relation_raw_entropy": entropy_stats(relation_raw),
        "non_relation_raw_entropy": entropy_stats(non_relation_raw),
        "reasoning_relation_raw_entropy": entropy_stats(reasoning_relation_raw),
        "reasoning_non_relation_raw_entropy": entropy_stats(
            reasoning_non_relation_raw
        ),
        "relation_filtered_entropy": entropy_stats(relation_filtered),
        "reasoning_relation_filtered_entropy": entropy_stats(
            reasoning_relation_filtered
        ),
        "relation_marker_counts": dict(
            sorted(
                relation_marker_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "relation_category_stats": {
            category: entropy_stats(values)
            for category, values in relation_category_raw.items()
        },
    }


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="PhysUniBench Evaluation with LEAD / CoT",
    )

    # ---- 模型与数据 ----
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace 模型名或本地权重路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集 JSONL 文件路径（默认 data/physunibench.jsonl）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录（默认 output/）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅运行前 N 条样本（用于调试），不指定则运行全部",
    )

    # ---- 数据过滤 ----
    parser.add_argument(
        "--subtopics",
        type=str,
        nargs="*",
        default=None,
        help="按子领域过滤，如 --subtopics Mechanics Optics",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["english", "chinese"],
        help="按语言过滤",
    )
    parser.add_argument(
        "--min_difficulty",
        type=int,
        default=1,
        help="最低难度（默认 1）",
    )
    parser.add_argument(
        "--max_difficulty",
        type=int,
        default=5,
        help="最高难度（默认 5）",
    )

    # ---- 推理方法 ----
    parser.add_argument(
        "--method",
        type=str,
        default="lead",
        choices=["lead", "lead_attenachor", "lead_attenanchor", "cot", "cot_greedy", "cot_visual_reanchor", "pure_soft"],
        help="推理方法：lead / lead_attenachor / pure_soft / cot / cot_greedy / cot_visual_reanchor",
    )
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="LEAD alpha_0 参数")
    parser.add_argument("--max_switch_count", type=int, default=5,
                        help="LEAD 最大切换次数")
    parser.add_argument("--window_size", type=int, default=256,
                        help="LEAD 离散到潜在模式切换的最小持续步数")
    parser.add_argument("--visual_anchor_top_m", type=int, default=32,
                        help="lead_attenachor 中按当前 token 对视觉 token attention 选取的 top-m")
    parser.add_argument("--visual_anchor_attn_last_k", type=int, default=4,
                        help="lead_attenachor 中用于聚合的最后几层 attention；<=0 表示使用全部层")
    parser.add_argument("--visual_anchor_lambda_scale", type=float, default=1.0,
                        help="lead_attenachor 中视觉 anchor 融合系数的额外缩放")
    parser.add_argument("--visual_anchor_entropy_upper", type=float, default=None,
                        help="lead_attenachor 中若当前 token 原始熵高于该阈值，则跳过 anchor 注入")
    parser.add_argument("--visual_anchor_skip_nonword", action="store_true",
                        help="lead_attenachor 中若当前 token 解码后为空白/标点样式，则跳过 anchor 注入")
    parser.add_argument("--visual_anchor_single_use", action="store_true",
                        help="lead_attenachor 中每个样本最多只允许一次 anchor 注入")
    parser.add_argument("--soft_trigger_mode", type=str, default="legacy",
                        choices=["legacy", "dual_delta2"],
                        help="lead_attenachor 的 soft 触发逻辑：legacy 或 dual_delta2")
    parser.add_argument("--soft_warning_margin", type=float, default=0.4,
                        help="dual_delta2 模式下的预警阈值，相对 cur_ref_entropy 的边际增量")
    parser.add_argument("--soft_confirm_margin", type=float, default=0.6,
                        help="dual_delta2 模式下 armed 后确认切换的阈值，相对 cur_ref_entropy 的边际增量")
    parser.add_argument("--soft_delta2_threshold", type=float, default=0.25,
                        help="dual_delta2 模式下 Δ2 = H_t - mean(H_{t-3:t-1}) 的阈值")
    parser.add_argument("--soft_repeat_warning_boost", type=float, default=0.0,
                        help="dual_delta2 模式下，对第 2 次及以后 soft 触发追加的 warning margin")
    parser.add_argument("--soft_repeat_confirm_boost", type=float, default=0.0,
                        help="dual_delta2 模式下，对第 2 次及以后 soft 触发追加的 confirm margin")
    parser.add_argument("--soft_repeat_delta2_boost", type=float, default=0.0,
                        help="dual_delta2 模式下，对第 2 次及以后 soft 触发追加的 Δ2 阈值")
    parser.add_argument("--soft_repeat_cooldown", type=int, default=0,
                        help="dual_delta2 模式下，对第 2 次及以后 soft 触发追加的最小等待步数")
    parser.add_argument("--soft_post_reset_ref_margin", type=float, default=0.0,
                        help="首次 soft->normal 后，将 cur_ref_entropy 重置为当前熵加该边际")
    parser.add_argument("--soft_post_reset_cooldown", type=int, default=0,
                        help="首次 soft->normal 后，额外增加的冷却步数")
    parser.add_argument("--reanchor_entropy_threshold", type=float, default=1.0,
                        help="cot_visual_reanchor 中触发视觉增强的 raw entropy 下限")
    parser.add_argument("--reanchor_visual_attn_threshold", type=float, default=0.12,
                        help="cot_visual_reanchor 中触发视觉增强的 visual_attn_mass 上限")
    parser.add_argument("--reanchor_lambda", type=float, default=0.15,
                        help="cot_visual_reanchor 中视觉 anchor 混入下一步 embedding 的系数")
    parser.add_argument("--reanchor_top_m", type=int, default=4,
                        help="cot_visual_reanchor 中构造动态视觉 anchor 时使用的 top-m 视觉 token")
    parser.add_argument("--reanchor_attn_last_k", type=int, default=4,
                        help="cot_visual_reanchor 中构造动态视觉 anchor 时聚合的最后几层 attention")
    parser.add_argument("--reanchor_max_trigger_count", type=int, default=1,
                        help="cot_visual_reanchor 中每个样本最多触发几次视觉增强")
    parser.add_argument("--reanchor_cooldown", type=int, default=32,
                        help="cot_visual_reanchor 中两次视觉增强之间的最小冷却 token 数")
    parser.add_argument("--reanchor_min_step", type=int, default=None,
                        help="cot_visual_reanchor 中允许触发的最小 step（含）")
    parser.add_argument("--reanchor_max_step", type=int, default=None,
                        help="cot_visual_reanchor 中允许触发的最大 step（含）")
    parser.add_argument("--reanchor_anchor_mode", type=str, default="dynamic",
                        choices=["dynamic", "mean"],
                        help="cot_visual_reanchor 中 top-m 视觉 token 的聚合方式：dynamic 为原 latent 加权，mean 为简单平均")
    parser.add_argument(
        "--save_token_entropy",
        action="store_true",
        help="保存每个生成 token 的熵轨迹到 token_entropy.jsonl",
    )
    parser.add_argument(
        "--save_full_token_entropy",
        action="store_true",
        help="保存完整逐 token 熵轨迹到 token_entropy_full.jsonl，便于画曲线",
    )
    parser.add_argument(
        "--save_visual_attn_summary",
        action="store_true",
        help="在 cot/cot_greedy 推理时为每个生成 token 记录视觉注意力摘要",
    )
    parser.add_argument(
        "--visual_attn_summary_last_k",
        type=int,
        default=4,
        help="视觉注意力摘要聚合时使用最后几层 attention；<=0 表示使用全部层",
    )

    # ---- 采样参数 ----
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=25600)
    parser.add_argument(
        "--do_sample",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="输入张量所使用的设备，默认自动选择 CUDA 或 CPU",
    )

    # ---- 评估模式 ----
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅评估已有结果，不运行推理",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="已有结果文件路径（配合 --eval_only 使用）",
    )

    # ---- 兼容 inference.py 中使用的字段（由循环动态赋值） ----
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")

    return parser.parse_args()


def run_eval_only(args, logger):
    """
    仅评估模式：加载已有结果并输出评估报告。

    Args:
        args: 命令行参数。
        logger: logger 实例。
    """
    from lead import load_jsonl

    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(project_root, "output")
    results_path = args.results or os.path.join(output_dir, "results.jsonl")

    logger.info(f"Loading results from: {results_path}")
    dataset = load_jsonl(results_path)
    logger.info(f"Loaded {len(dataset)} samples")

    eval_results = evaluate_dataset(dataset)
    print_evaluation_report(eval_results)

    report_path = os.path.join(output_dir, "eval_report.json")
    save_evaluation_report(eval_results, report_path)
    logger.info(f"Evaluation report saved to: {report_path}")


def main():
    """主流程：加载模型 → 遍历数据集 → 逐样本推理 → 评估 → 保存结果。"""
    args = parse_args()

    # ---- 路径解析 ----
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    output_dir = args.output_dir or os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")

    logger = setup_logger("lead", log_dir=log_dir)

    # ---- 仅评估模式 ----
    if args.eval_only:
        run_eval_only(args, logger)
        return

    dataset_path = args.dataset or os.path.join(data_dir, "physunibench.jsonl")
    output_path = os.path.join(output_dir, "results.jsonl")
    report_path = os.path.join(output_dir, "eval_report.json")
    token_entropy_path = os.path.join(output_dir, "token_entropy.jsonl")
    full_token_entropy_path = os.path.join(output_dir, "token_entropy_full.jsonl")

    # ---- 加载模型（仅加载一次） ----
    logger.info(f"Loading model: {args.model_name}")
    with Timer("model_loading") as t:
        model_load_kwargs = {}
        if args.method in {"lead_attenachor", "lead_attenanchor"}:
            if args.device == "auto":
                model_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                model_device = args.device
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name,
                **model_load_kwargs,
            )
            model = model.to(model_device)
        else:
            model_load_kwargs["device_map"] = "auto"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name,
                **model_load_kwargs,
            )
        processor = AutoProcessor.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Model loaded in {t}")

    # ---- 加载数据集 ----
    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(
        dataset_path,
        data_dir,
        subtopics=args.subtopics,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        language=args.language,
    )
    if args.limit is not None:
        dataset = dataset[: args.limit]
        logger.info(f"Limited to first {len(dataset)} samples (--limit={args.limit})")

    stats = get_dataset_statistics(dataset)
    logger.info(f"Dataset: {stats['total']} samples, "
                f"subtopics={list(stats['subtopics'].keys())}, "
                f"languages={list(stats['languages'].keys())}")

    # ---- 保存运行配置 ----
    config = {
        "model_name": args.model_name,
        "method": args.method,
        "alpha": args.alpha,
        "max_switch_count": args.max_switch_count,
        "window_size": args.window_size,
        "visual_anchor_top_m": args.visual_anchor_top_m,
        "visual_anchor_attn_last_k": args.visual_anchor_attn_last_k,
        "visual_anchor_lambda_scale": args.visual_anchor_lambda_scale,
        "visual_anchor_entropy_upper": args.visual_anchor_entropy_upper,
        "visual_anchor_skip_nonword": args.visual_anchor_skip_nonword,
        "visual_anchor_single_use": args.visual_anchor_single_use,
        "soft_trigger_mode": args.soft_trigger_mode,
        "soft_warning_margin": args.soft_warning_margin,
        "soft_confirm_margin": args.soft_confirm_margin,
        "soft_delta2_threshold": args.soft_delta2_threshold,
        "soft_repeat_warning_boost": args.soft_repeat_warning_boost,
        "soft_repeat_confirm_boost": args.soft_repeat_confirm_boost,
        "soft_repeat_delta2_boost": args.soft_repeat_delta2_boost,
        "soft_repeat_cooldown": args.soft_repeat_cooldown,
        "soft_post_reset_ref_margin": args.soft_post_reset_ref_margin,
        "soft_post_reset_cooldown": args.soft_post_reset_cooldown,
        "reanchor_entropy_threshold": args.reanchor_entropy_threshold,
        "reanchor_visual_attn_threshold": args.reanchor_visual_attn_threshold,
        "reanchor_lambda": args.reanchor_lambda,
        "reanchor_top_m": args.reanchor_top_m,
        "reanchor_attn_last_k": args.reanchor_attn_last_k,
        "reanchor_max_trigger_count": args.reanchor_max_trigger_count,
        "reanchor_cooldown": args.reanchor_cooldown,
        "reanchor_min_step": args.reanchor_min_step,
        "reanchor_max_step": args.reanchor_max_step,
        "reanchor_anchor_mode": args.reanchor_anchor_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dataset": dataset_path,
        "num_samples": len(dataset),
        "save_token_entropy": args.save_token_entropy,
        "save_full_token_entropy": args.save_full_token_entropy,
        "save_visual_attn_summary": args.save_visual_attn_summary,
        "visual_attn_summary_last_k": args.visual_attn_summary_last_k,
    }
    if args.save_token_entropy:
        config["token_entropy_path"] = token_entropy_path
    if args.save_full_token_entropy:
        config["full_token_entropy_path"] = full_token_entropy_path
    save_json(config, os.path.join(output_dir, "config.json"))

    # ---- 逐样本推理 ----
    logger.info(f"Starting inference with method={args.method}")
    total_timer = Timer("total_inference").start()

    for idx, sample in enumerate(dataset):
        prompt = format_prompt_from_sample(
            sample,
            use_cot=args.method in {"cot", "cot_greedy"},
        )
        image_url = sample.get("image", "")

        logger.info(f"[{idx + 1}/{len(dataset)}] id={sample.get('id', '?')} "
                     f"image={os.path.basename(image_url)}")

        args.image = image_url
        args.prompt = prompt

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        args.token_entropy_trace = None
        args.prompt_tokens = None
        with Timer("sample") as sample_timer:
            try:
                sample["model_answer"] = run_single_inference(
                    model, processor, tokenizer, args,
                )
            except Exception as e:
                logger.warning(f"Sample {idx} failed: {e}")
                sample["model_answer"] = None
                sample["error_type"] = type(e).__name__
                sample["error_message"] = str(e)
                torch.cuda.empty_cache()

        sample["latency_sec"] = sample_timer.elapsed
        sample.setdefault("error_type", None)
        sample.setdefault("error_message", None)
        if sample.get("model_answer") is not None:
            sample["output_tokens"] = len(
                tokenizer.encode(sample["model_answer"], add_special_tokens=False)
            )
        if torch.cuda.is_available():
            sample["cuda_peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            )
            sample["cuda_peak_reserved_mb"] = (
                torch.cuda.max_memory_reserved() / 1024 / 1024
            )
        if args.save_token_entropy:
            token_trace = args.token_entropy_trace or []
            trace_record = {
                "sample_index": idx,
                "id": sample.get("id"),
                "method": args.method,
                "answer": sample.get("answer"),
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": sample.get("output_tokens"),
                "error_type": sample.get("error_type"),
                "entropy_summary": build_entropy_summary(
                    tokenizer,
                    token_trace,
                ),
            }
            with open(token_entropy_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            if args.save_full_token_entropy:
                full_record = {
                    "sample_index": idx,
                    "id": sample.get("id"),
                    "method": args.method,
                    "answer": sample.get("answer"),
                    "prompt_tokens": args.prompt_tokens,
                    "output_tokens": sample.get("output_tokens"),
                    "error_type": sample.get("error_type"),
                    "tokens": annotate_token_trace(tokenizer, token_trace),
                }
                with open(full_token_entropy_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(full_record, ensure_ascii=False) + "\n")

        logger.info(f"  Completed in {sample_timer}")

    total_timer.stop()
    logger.info(f"Inference completed in {total_timer}")

    # ---- 保存结果 ----
    save_jsonl(dataset, output_path)
    logger.info(f"Results saved to: {output_path}")

    # ---- 评估 ----
    logger.info("Running evaluation...")
    eval_results = evaluate_dataset(dataset)
    eval_results["config"] = config
    print_evaluation_report(eval_results)
    save_evaluation_report(eval_results, report_path)
    logger.info(f"Evaluation report saved to: {report_path}")

    logger.info(f"Done. Total samples: {len(dataset)}, "
                f"Accuracy: {eval_results['accuracy']:.2%}")


if __name__ == "__main__":
    main()

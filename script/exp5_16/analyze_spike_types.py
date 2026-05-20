#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from lead.evaluator import evaluate_single


VISUAL_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "brown", "gray", "grey",
    "orange", "purple", "pink", "silver", "gold", "golden", "dark", "light",
    "left", "right", "front", "back", "behind", "above", "below", "under",
    "over", "near", "next", "middle", "center", "top", "bottom", "side",
    "person", "people", "man", "woman", "child", "dog", "cat", "car", "bus",
    "truck", "bike", "bicycle", "motorcycle", "train", "chair", "bench", "table",
    "sign", "poster", "flag", "hat", "shirt", "jacket", "pants", "shoe", "bag",
    "box", "bucket", "glove", "door", "window", "building", "street", "road",
    "tree", "ground", "wall", "floor", "object", "item", "image", "picture",
    "wearing", "holding", "standing", "sitting", "lying", "hanging", "attached",
    "wood", "wooden", "metal", "plastic", "rubber", "cotton", "leather",
}

RELATION_WORDS = {
    "because", "therefore", "thus", "so", "since", "however", "but", "although",
    "while", "whereas", "then", "next", "first", "second", "finally", "given",
    "if", "unless", "also", "moreover", "hence", "consequently", "alternatively",
}

FORMAT_WORDS = {
    "\n", "\n\n", ".", ",", ":", ";", "-", "(", ")", "[", "]", "{", "}",
    "<", ">", "</", "<think", "think", "answer", "option", "**", "*",
}

ANSWER_WORDS = {"a", "b", "c", "d", "(a", "(b", "(c", "(d", "answer"}


def norm_text(text):
    text = (text or "").strip().lower()
    text = text.replace("▁", " ")
    text = re.sub(r"^[^a-z0-9<>/]+|[^a-z0-9<>/]+$", "", text)
    return text


def token_category(text):
    raw = text or ""
    t = norm_text(raw)
    if not t:
        return "format"
    if raw in FORMAT_WORDS or t in FORMAT_WORDS or re.fullmatch(r"[\W_]+", raw.strip() or ""):
        return "format"
    if t in ANSWER_WORDS or re.fullmatch(r"\(?[abcd]\)?", t):
        return "answer"
    if t in RELATION_WORDS:
        return "relation"
    if t in VISUAL_WORDS:
        return "visual"
    return "other"


def is_spike(entropies, idx, window, alpha, min_history, min_entropy):
    cur = entropies[idx]
    if cur is None or cur < min_entropy:
        return False
    start = max(0, idx - window)
    hist = [x for x in entropies[start:idx] if x is not None]
    if len(hist) < min_history:
        return False
    mean = sum(hist) / len(hist)
    var = sum((x - mean) ** 2 for x in hist) / len(hist)
    std = math.sqrt(var)
    return cur > mean + alpha * std


def topk_masses(token, topk_field):
    topk = token.get(topk_field) or []
    total = sum(float(x.get("prob") or 0.0) for x in topk)
    masses = Counter()
    if total <= 0:
        return masses, 0.0, 0.0, 0.0
    for item in topk:
        cat = token_category(item.get("token_text", ""))
        masses[cat] += float(item.get("prob") or 0.0) / total
    top1 = float(topk[0].get("prob") or 0.0) if topk else 0.0
    top2 = float(topk[1].get("prob") or 0.0) if len(topk) > 1 else 0.0
    return masses, total, top1, top1 - top2


def classify_spike(token, args):
    masses, _, top1, margin = topk_masses(token, args.topk_field)
    if masses["format"] >= args.format_tau:
        return "format_spike", masses, top1, margin
    if masses["answer"] >= args.answer_tau:
        return "answer_spike", masses, top1, margin
    if top1 < args.low_conf_tau or margin < args.low_margin_tau:
        return "diffuse_low_conf_spike", masses, top1, margin
    if masses["visual"] >= args.visual_tau:
        return "visual_spike", masses, top1, margin
    if masses["relation"] >= args.relation_tau:
        return "relation_spike", masses, top1, margin
    return "other", masses, top1, margin


def load_correctness(run_dir):
    path = run_dir / "results.jsonl"
    correct = {}
    if not path.exists():
        return correct
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ok, extracted = evaluate_single(rec.get("model_answer"), rec.get("answer", ""))
            correct[rec.get("id")] = {
                "correct": ok,
                "extracted": extracted,
                "output_tokens": rec.get("output_tokens"),
                "failed_extraction": extracted is None,
            }
    return correct


def analyze_run(label, run_dir, args):
    trace = run_dir / "token_entropy_full.jsonl"
    if not trace.exists():
        raise FileNotFoundError(trace)
    correctness = load_correctness(run_dir)
    counts = Counter()
    by_correct = defaultdict(Counter)
    by_mode = defaultdict(Counter)
    mass_sums = defaultdict(Counter)
    sample_count = 0
    token_count = 0
    spike_count = 0
    missing_topk = 0

    with trace.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sample_count += 1
            sid = rec.get("id")
            meta = correctness.get(sid, {})
            corr_label = (
                "correct" if meta.get("correct") is True
                else "wrong" if meta.get("correct") is False
                else "unknown"
            )
            tokens = rec.get("tokens") or []
            entropies = [t.get("raw_entropy") for t in tokens]
            for idx, token in enumerate(tokens):
                token_count += 1
                if not is_spike(entropies, idx, args.window, args.alpha, args.min_history, args.min_entropy):
                    continue
                spike_count += 1
                if not token.get(args.topk_field):
                    missing_topk += 1
                cat, masses, top1, margin = classify_spike(token, args)
                counts[cat] += 1
                by_correct[corr_label][cat] += 1
                by_mode[token.get("mode", "unknown")][cat] += 1
                for mass_cat, value in masses.items():
                    mass_sums[cat][mass_cat] += value

    return {
        "label": label,
        "run_dir": str(run_dir),
        "sample_count": sample_count,
        "token_count": token_count,
        "spike_count": spike_count,
        "missing_topk": missing_topk,
        "counts": counts,
        "by_correct": by_correct,
        "by_mode": by_mode,
        "mass_sums": mass_sums,
    }


def pct(n, d):
    return "0.0%" if d == 0 else f"{100.0 * n / d:.1f}%"


def render_report(results, args):
    cats = [
        "visual_spike",
        "relation_spike",
        "format_spike",
        "answer_spike",
        "diffuse_low_conf_spike",
        "other",
    ]
    lines = []
    lines.append("# Experiment 1 Spike Type Analysis")
    lines.append("")
    lines.append(f"- spike rule: `H_t > local_mean({args.window}) + {args.alpha} * local_std({args.window})`, min_history=`{args.min_history}`, min_entropy=`{args.min_entropy}`")
    lines.append(f"- topk field: `{args.topk_field}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("| method | samples | tokens | spikes | missing_topk | visual | relation | format | answer | diffuse_low_conf | other |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        total = r["spike_count"]
        c = r["counts"]
        lines.append(
            f"| {r['label']} | {r['sample_count']} | {r['token_count']} | {total} | {r['missing_topk']} | "
            f"{c['visual_spike']} ({pct(c['visual_spike'], total)}) | "
            f"{c['relation_spike']} ({pct(c['relation_spike'], total)}) | "
            f"{c['format_spike']} ({pct(c['format_spike'], total)}) | "
            f"{c['answer_spike']} ({pct(c['answer_spike'], total)}) | "
            f"{c['diffuse_low_conf_spike']} ({pct(c['diffuse_low_conf_spike'], total)}) | "
            f"{c['other']} ({pct(c['other'], total)}) |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['label']} By Correctness")
        lines.append("| group | spikes | visual | relation | format | answer | diffuse_low_conf | other |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for group in ["correct", "wrong", "unknown"]:
            c = r["by_correct"].get(group, Counter())
            total = sum(c.values())
            if total == 0:
                continue
            lines.append(
                f"| {group} | {total} | "
                f"{c['visual_spike']} ({pct(c['visual_spike'], total)}) | "
                f"{c['relation_spike']} ({pct(c['relation_spike'], total)}) | "
                f"{c['format_spike']} ({pct(c['format_spike'], total)}) | "
                f"{c['answer_spike']} ({pct(c['answer_spike'], total)}) | "
                f"{c['diffuse_low_conf_spike']} ({pct(c['diffuse_low_conf_spike'], total)}) | "
                f"{c['other']} ({pct(c['other'], total)}) |"
            )
        lines.append("")
        lines.append(f"## {r['label']} By Generation Mode")
        lines.append("| mode | spikes | visual | relation | format | answer | diffuse_low_conf | other |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for mode, c in sorted(r["by_mode"].items()):
            total = sum(c.values())
            lines.append(
                f"| {mode} | {total} | "
                f"{c['visual_spike']} ({pct(c['visual_spike'], total)}) | "
                f"{c['relation_spike']} ({pct(c['relation_spike'], total)}) | "
                f"{c['format_spike']} ({pct(c['format_spike'], total)}) | "
                f"{c['answer_spike']} ({pct(c['answer_spike'], total)}) | "
                f"{c['diffuse_low_conf_spike']} ({pct(c['diffuse_low_conf_spike'], total)}) | "
                f"{c['other']} ({pct(c['other'], total)}) |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--min_history", type=int, default=4)
    parser.add_argument("--min_entropy", type=float, default=1.0)
    parser.add_argument("--topk_field", choices=["raw_topk", "filtered_topk"], default="raw_topk")
    parser.add_argument("--visual_tau", type=float, default=0.35)
    parser.add_argument("--relation_tau", type=float, default=0.25)
    parser.add_argument("--format_tau", type=float, default=0.45)
    parser.add_argument("--answer_tau", type=float, default=0.35)
    parser.add_argument("--low_conf_tau", type=float, default=0.20)
    parser.add_argument("--low_margin_tau", type=float, default=0.05)
    args = parser.parse_args()

    results = [
        analyze_run(label, Path(run_dir), args)
        for label, run_dir in args.run
    ]
    report = render_report(results, args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

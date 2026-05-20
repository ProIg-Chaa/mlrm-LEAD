#!/usr/bin/env python3
"""Analyze online sidecar visual attention traces."""

import argparse
import json
import math
import re
from collections import defaultdict


ANSWER_PATTERNS = [
    re.compile(r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?"),
    re.compile(r"\\boxed\{([A-Da-d])\}"),
    re.compile(r"\*\*([A-Da-d])\*\*"),
    re.compile(r"(?:^|\n)\s*([A-Da-d])\s*$"),
]

BOILERPLATE = {
    "the", "image", "picture", "shows", "show", "i", "need", "determine",
    "okay", "so", "let", "look", "analyze", "based", "seen",
}


def extract_answer(text):
    if not text:
        return None
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    letters = re.findall(r"\b([A-D])\b", text[-200:])
    return letters[-1].upper() if letters else None


def load_results(path):
    by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pred = extract_answer(row.get("model_answer") or "")
            row["_pred"] = pred
            row["_correct"] = pred == row.get("answer")
            by_id[row["id"]] = row
    return by_id


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def stats(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": quantile(vals, 0.5),
        "p10": quantile(vals, 0.1),
        "p90": quantile(vals, 0.9),
    }


def norm_token(text):
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", (text or "").strip().lower())


def token_bucket(token):
    text = norm_token(token.get("token_text"))
    if not text:
        return "nonword"
    if text in BOILERPLATE:
        return "boilerplate"
    if token.get("is_reasoning_token"):
        return "reasoning_content"
    return "content_or_answer"


def step_bucket(step):
    if step <= 10:
        return "early_0_10"
    if step <= 50:
        return "mid_11_50"
    return "late_51_plus"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    args = parser.parse_args()

    results = load_results(args.results)
    rows = []
    with open(args.trace, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("id")
            meta = results.get(sid, {})
            for token in rec.get("tokens") or []:
                raw_entropy = token.get("raw_entropy")
                observed = token.get("sidecar_attn_observed") is True
                available = token.get("sidecar_visual_attn_available") is True
                rows.append({
                    "id": sid,
                    "correct": bool(meta.get("_correct")),
                    "subtopic": meta.get("subtopic", "unknown"),
                    "step": token.get("step"),
                    "step_bucket": step_bucket(token.get("step", 0)),
                    "token_bucket": token_bucket(token),
                    "token_text": token.get("token_text"),
                    "raw_entropy": raw_entropy,
                    "observed": observed,
                    "available": available,
                    "mass": token.get("sidecar_visual_attn_mass") if observed and available else None,
                    "top1": token.get("sidecar_visual_attn_top1") if observed and available else None,
                    "top4": token.get("sidecar_visual_attn_top4_sum") if observed and available else None,
                    "attn_entropy": token.get("sidecar_visual_attn_entropy") if observed and available else None,
                    "attn_entropy_norm": token.get("sidecar_visual_attn_entropy_norm") if observed and available else None,
                    "attn_concentration": token.get("sidecar_visual_attn_concentration") if observed and available else None,
                    "align_max": token.get("sidecar_hidden_visual_align_max") if observed else None,
                    "align_top4": token.get("sidecar_hidden_visual_align_top4_mean") if observed else None,
                })

    lines = []
    lines.append(f"# Online Sidecar Attention Analysis\n")
    lines.append(f"- results: `{args.results}`")
    lines.append(f"- trace: `{args.trace}`")
    lines.append(f"- total tokens: `{len(rows)}`")
    lines.append("")

    for threshold in args.thresholds:
        subset = [r for r in rows if r["raw_entropy"] is not None and r["raw_entropy"] >= threshold]
        observed = [r for r in subset if r["observed"]]
        missing = len(subset) - len(observed)
        lines.append(f"## H >= {threshold}")
        lines.append(f"- tokens: `{len(subset)}`")
        lines.append(f"- sidecar observed: `{len(observed)}`")
        lines.append(f"- missing attention: `{missing}`")
        lines.append(f"- visual mass: `{stats([r['mass'] for r in observed])}`")
        lines.append("")

        groups = [
            ("correct", lambda r: r["correct"]),
            ("wrong", lambda r: not r["correct"]),
            ("direct_attributes", lambda r: r["subtopic"] == "direct_attributes"),
            ("relative_position", lambda r: r["subtopic"] == "relative_position"),
            ("boilerplate", lambda r: r["token_bucket"] == "boilerplate"),
            ("reasoning_content", lambda r: r["token_bucket"] == "reasoning_content"),
            ("nonword", lambda r: r["token_bucket"] == "nonword"),
            ("early_0_10", lambda r: r["step_bucket"] == "early_0_10"),
            ("mid_11_50", lambda r: r["step_bucket"] == "mid_11_50"),
            ("late_51_plus", lambda r: r["step_bucket"] == "late_51_plus"),
        ]
        lines.append("| group | observed | mass_mean | mass_median | conc_mean | align_max_mean | align_top4_mean | top4_mean |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, pred in groups:
            cur = [r for r in observed if pred(r)]
            mass_stats = stats([r["mass"] for r in cur])
            conc_stats = stats([r["attn_concentration"] for r in cur])
            align_max_stats = stats([r["align_max"] for r in cur])
            align_top4_stats = stats([r["align_top4"] for r in cur])
            top4_stats = stats([r["top4"] for r in cur])
            lines.append(
                f"| {name} | {len(cur)} | "
                f"{mass_stats.get('mean', float('nan')):.4f} | "
                f"{mass_stats.get('median', float('nan')):.4f} | "
                f"{conc_stats.get('mean', float('nan')):.4f} | "
                f"{align_max_stats.get('mean', float('nan')):.4f} | "
                f"{align_top4_stats.get('mean', float('nan')):.4f} | "
                f"{top4_stats.get('mean', float('nan')):.4f} |"
            )
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()

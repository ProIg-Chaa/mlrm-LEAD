#!/usr/bin/env python3
"""Create publication-ready figures for the pure-soft confidence analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_LABELS = {
    "vstar": "VStar",
    "mmvp": "MMVP",
    "visulogic300": "VisuLogic300",
    "vmcbench_dev": "VMCBench-dev",
    "mmk12_physics": "MMK12-Physics",
}
COLORS = {
    "blue": "#277DA1",
    "cyan": "#43AA8B",
    "gold": "#F9C74F",
    "orange": "#F8961E",
    "red": "#D1495B",
    "gray": "#6B7280",
    "light": "#E5E7EB",
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D1D5DB",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def overview_figure(records, out_dir):
    labels = [DATASET_LABELS[x["dataset"]] for x in records]
    strict_auc = [x["summary"]["score_stats"]["mean_raw_conf"]["auc"] for x in records]
    semantic_auc = [
        x["summary"]["score_stats"]["mean_raw_conf"]["semantic_only_auc"]
        for x in records
    ]
    top_strict = [
        100 * x["summary"]["score_stats"]["mean_raw_conf"]["top_decile_delta"]
        for x in records
    ]
    top_sem = [
        100
        * x["summary"]["score_stats"]["mean_raw_conf"][
            "semantic_top_decile_delta"
        ]
        for x in records
    ]
    correct_conf = [
        x["summary"]["score_stats"]["mean_raw_conf"]["correct_mean"]
        for x in records
    ]
    wrong_conf = [
        x["summary"]["score_stats"]["mean_raw_conf"]["wrong_mean"]
        for x in records
    ]
    correct_len = [x["summary"]["mean_output_tokens_correct"] for x in records]
    wrong_len = [x["summary"]["mean_output_tokens_wrong"] for x in records]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.35

    ax = axes[0, 0]
    ax.bar(x - width / 2, strict_auc, width, label="Strict", color=COLORS["blue"])
    ax.bar(x + width / 2, semantic_auc, width, label="Semantic-only", color=COLORS["cyan"])
    ax.axhline(0.5, color=COLORS["red"], ls="--", lw=1.3, label="Random ranking")
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel("AUROC")
    ax.set_title("Can confidence rank final correctness?")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    ax.bar(x - width / 2, top_strict, width, label="Strict", color=COLORS["blue"])
    ax.bar(x + width / 2, top_sem, width, label="Semantic-only", color=COLORS["cyan"])
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("Top-confidence 10% vs. all samples")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    ax.bar(x - width / 2, correct_conf, width, label="Correct", color=COLORS["cyan"])
    ax.bar(x + width / 2, wrong_conf, width, label="Wrong (strict)", color=COLORS["red"])
    ax.set_ylim(0.70, 0.87)
    ax.set_ylabel("Mean raw token confidence")
    ax.set_title("Wrong trajectories are at least as confident")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    ax.bar(x - width / 2, correct_len, width, label="Correct", color=COLORS["cyan"])
    ax.bar(x + width / 2, wrong_len, width, label="Wrong (strict)", color=COLORS["red"])
    ax.set_ylabel("Mean generated tokens")
    ax.set_title("Wrong pure-soft trajectories are longer")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    add_panel_label(ax, "D")

    fig.suptitle("Pure-soft confidence does not imply correctness", fontsize=16, fontweight="bold")
    save(fig, out_dir, "figure1_pure_soft_confidence_overview")


def temporal_heatmap(records, out_dir):
    stages = ["early32_raw_conf", "mean_raw_conf", "tail20_raw_conf"]
    stage_labels = ["Early 32", "Full trajectory", "Tail 20"]
    strict = np.array(
        [[x["summary"]["score_stats"][s]["auc"] for s in stages] for x in records]
    )
    semantic = np.array(
        [
            [x["summary"]["score_stats"][s]["semantic_only_auc"] for s in stages]
            for x in records
        ]
    )
    labels = [DATASET_LABELS[x["dataset"]] for x in records]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.3), constrained_layout=True)
    for ax, matrix, title in zip(
        axes, [strict, semantic], ["Strict AUROC", "Semantic-only AUROC"]
    ):
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=0.35, vmax=0.65, aspect="auto")
        ax.set_xticks(range(3), stage_labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        ax.grid(False)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value - 0.5) > 0.105 else "#111827",
                    fontweight="bold",
                )
        ax.axvline(0.5, color="white", lw=0.5, alpha=0)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("AUROC (0.5 = random)")
    fig.suptitle("Confidence predictiveness across reasoning stages", fontsize=15, fontweight="bold")
    save(fig, out_dir, "figure2_temporal_confidence_auroc")


def decile_figure(records, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True, sharex=True)
    axes = axes.ravel()
    for ax, record in zip(axes, records):
        stats = record["summary"]["score_stats"]["mean_raw_conf"]
        strict = stats["deciles"]
        semantic = stats["semantic_deciles"]
        ax.plot(
            [x["bin"] for x in strict],
            [100 * x["accuracy"] for x in strict],
            marker="o",
            lw=2,
            color=COLORS["blue"],
            label="Strict",
        )
        ax.plot(
            [x["bin"] for x in semantic],
            [100 * x["accuracy"] for x in semantic],
            marker="s",
            lw=2,
            color=COLORS["cyan"],
            label="Semantic-only",
        )
        ax.axhline(
            100 * stats["overall_accuracy_on_eligible"],
            color=COLORS["blue"],
            ls=":",
            lw=1.1,
        )
        ax.axhline(
            100 * stats["semantic_overall_accuracy"],
            color=COLORS["cyan"],
            ls=":",
            lw=1.1,
        )
        ax.set_ylim(0, 105)
        ax.set_title(DATASET_LABELS[record["dataset"]])
        ax.set_xlabel("Confidence decile (low to high)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks([1, 2, 4, 6, 8, 10])
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].axis("off")
    axes[-1].text(
        0.05,
        0.72,
        "Solid lines: accuracy within each confidence decile\n"
        "Dotted lines: dataset-wide accuracy\n\n"
        "A calibrated confidence signal should rise toward\n"
        "the right. The curves are non-monotonic, and the\n"
        "highest-confidence strict bins often collapse.",
        va="top",
        fontsize=11,
        linespacing=1.5,
    )
    fig.suptitle("Accuracy is not monotonic in pure-soft confidence", fontsize=15, fontweight="bold")
    save(fig, out_dir, "figure3_confidence_decile_reliability")


def distribution_figure(rows, out_dir):
    datasets = list(DATASET_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)

    groups = {
        "Correct": [r for r in rows if r["correct"]],
        "Semantic wrong": [r for r in rows if not r["correct"] and not r["failed_extraction"]],
        "Extraction failure": [r for r in rows if r["failed_extraction"]],
    }
    group_colors = [COLORS["cyan"], COLORS["red"], COLORS["gold"]]
    positions = np.arange(len(datasets))
    offsets = [-0.24, 0, 0.24]
    for (name, group), color, offset in zip(groups.items(), group_colors, offsets):
        values = [
            [r["mean_raw_conf"] for r in group if r["dataset"] == dataset]
            for dataset in datasets
        ]
        parts = axes[0].violinplot(
            values,
            positions=positions + offset,
            widths=0.22,
            showmeans=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("white")
            body.set_alpha(0.85)
        parts["cmeans"].set_color("#111827")
        parts["cmeans"].set_linewidth(1.2)
        axes[0].plot([], [], color=color, lw=8, label=name)
    axes[0].set_xticks(positions, [DATASET_LABELS[x] for x in datasets], rotation=20, ha="right")
    axes[0].set_ylim(0.35, 1.02)
    axes[0].set_ylabel("Mean raw token confidence")
    axes[0].set_title("Confidence distributions")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    add_panel_label(axes[0], "A")

    for (name, group), color in zip(groups.items(), group_colors):
        lengths = np.array([r["output_tokens"] for r in group], dtype=float)
        confidence = np.array([r["mean_raw_conf"] for r in group], dtype=float)
        order = np.argsort(lengths)
        lengths, confidence = lengths[order], confidence[order]
        if len(lengths) >= 20:
            bins = np.array_split(np.arange(len(lengths)), 15)
            x = [np.median(lengths[b]) for b in bins if len(b)]
            y = [np.mean(confidence[b]) for b in bins if len(b)]
            axes[1].plot(x, y, marker="o", ms=3.5, lw=2, color=color, label=name)
    axes[1].set_xlabel("Generated tokens (binned median)")
    axes[1].set_ylabel("Mean raw token confidence")
    axes[1].set_title("Long trajectories become highly confident")
    axes[1].set_xlim(0, 1050)
    axes[1].set_ylim(0.6, 1.01)
    axes[1].legend(frameon=False, fontsize=8)
    add_panel_label(axes[1], "B")

    fig.suptitle("Two sources of high-confidence failure", fontsize=15, fontweight="bold")
    save(fig, out_dir, "figure4_confidence_failure_distributions")


def combined_figure(records, out_dir):
    labels = [DATASET_LABELS[x["dataset"]] for x in records]
    strict = [x["summary"]["score_stats"]["mean_raw_conf"]["auc"] for x in records]
    semantic = [x["summary"]["score_stats"]["mean_raw_conf"]["semantic_only_auc"] for x in records]
    top_delta = [100 * x["summary"]["score_stats"]["mean_raw_conf"]["top_decile_delta"] for x in records]
    early = [x["summary"]["score_stats"]["early32_raw_conf"]["auc"] for x in records]
    tail = [x["summary"]["score_stats"]["tail20_raw_conf"]["auc"] for x in records]
    gap = [x["summary"]["score_stats"]["mean_raw_conf"]["wrong_minus_correct"] for x in records]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(labels)); width = 0.35
    axes[0, 0].bar(x-width/2, strict, width, color=COLORS["blue"], label="Strict")
    axes[0, 0].bar(x+width/2, semantic, width, color=COLORS["cyan"], label="Semantic-only")
    axes[0, 0].axhline(.5, color=COLORS["red"], ls="--", lw=1.2)
    axes[0, 0].set_ylim(.3,.7); axes[0, 0].set_ylabel("AUROC")
    axes[0, 0].set_title("Confidence vs. correctness")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].bar(x, top_delta, color=[COLORS["red"] if v < 0 else COLORS["cyan"] for v in top_delta])
    axes[0, 1].axhline(0, color="#111827", lw=.9)
    axes[0, 1].set_ylabel("Accuracy change (pp)")
    axes[0, 1].set_title("Highest-confidence 10% vs. overall")

    axes[1, 0].plot(x, early, marker="o", lw=2, color=COLORS["gold"], label="Early 32")
    axes[1, 0].plot(x, strict, marker="s", lw=2, color=COLORS["blue"], label="Full")
    axes[1, 0].plot(x, tail, marker="^", lw=2, color=COLORS["red"], label="Tail 20")
    axes[1, 0].axhline(.5, color=COLORS["gray"], ls="--", lw=1)
    axes[1, 0].set_ylim(.3,.6); axes[1, 0].set_ylabel("Strict AUROC")
    axes[1, 0].set_title("Predictiveness over generation")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=3)

    axes[1, 1].bar(x, gap, color=COLORS["red"])
    axes[1, 1].axhline(0, color="#111827", lw=.9)
    axes[1, 1].set_ylabel("Wrong confidence - correct confidence")
    axes[1, 1].set_title("Wrong trajectories are more confident")
    for ax in axes.ravel():
        ax.set_xticks(x, labels, rotation=20, ha="right")
    for ax, label in zip(axes.ravel(), "ABCD"):
        add_panel_label(ax, label)
    fig.suptitle("Pure-soft models can be confidently wrong", fontsize=16, fontweight="bold")
    save(fig, out_dir, "figure5_pure_soft_confidently_wrong_summary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    records = load_json(args.input_dir / "confidence_correctness_summary.json")
    rows = load_jsonl(args.input_dir / "sample_confidence_metrics.jsonl")
    overview_figure(records, args.output_dir)
    temporal_heatmap(records, args.output_dir)
    decile_figure(records, args.output_dir)
    distribution_figure(rows, args.output_dir)
    combined_figure(records, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot token entropy curves from token_entropy_full.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def sanitize_text(text: str, limit: int = 36) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_records(input_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_question_types(dataset_path: Path) -> dict[int, str]:
    mapping = {}
    rows = load_records(dataset_path)
    for row in rows:
        sample_id = row.get("id")
        if sample_id is None:
            continue
        try:
            mapping[int(sample_id)] = row.get("question_type", "unknown")
        except (TypeError, ValueError):
            continue
    return mapping


def get_xy(tokens: list[dict], metric: str) -> tuple[list[int], list[float]]:
    xs = list(range(len(tokens)))
    ys = [float(token.get(metric) or 0.0) for token in tokens]
    return xs, ys


def plot_record(record: dict, output_dir: Path, metric: str) -> Path:
    sample_id = record.get("id")
    sample_index = record.get("sample_index")
    tokens = record.get("tokens", [])
    xs, ys = get_xy(tokens, metric)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(xs, ys, linewidth=1.5, color="#2563eb", label=metric)

    relation_x = [
        i for i, token in enumerate(tokens) if token.get("is_relation_token")
    ]
    relation_y = [ys[i] for i in relation_x]
    if relation_x:
        ax.scatter(
            relation_x,
            relation_y,
            color="#dc2626",
            s=20,
            label="relation token",
            zorder=3,
        )
        for i in relation_x:
            label = sanitize_text(tokens[i].get("normalized_token_text") or tokens[i].get("token_text") or "")
            ax.annotate(
                label,
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="#991b1b",
            )

    soft_x = [i for i, token in enumerate(tokens) if token.get("mode") == "soft"]
    soft_y = [ys[i] for i in soft_x]
    if soft_x:
        ax.scatter(
            soft_x,
            soft_y,
            color="#f59e0b",
            s=14,
            label="soft token",
            zorder=2,
        )

    ax.set_title(
        f"sample_index={sample_index}, id={sample_id}, metric={metric}, tokens={len(tokens)}"
    )
    ax.set_xlabel("Generated token index")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sample_{sample_index:03d}_id_{sample_id}_{metric}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def select_one_per_type(records: list[dict], id_to_type: dict[int, str]) -> list[tuple[str, dict]]:
    grouped = {}
    for record in sorted(records, key=lambda row: (row.get("sample_index", 0), row.get("id", 0))):
        sample_id = record.get("id")
        try:
            question_type = id_to_type.get(int(sample_id), "unknown")
        except (TypeError, ValueError):
            question_type = "unknown"
        if question_type not in grouped:
            grouped[question_type] = record
    return sorted(grouped.items(), key=lambda item: item[0])


def plot_overview_per_type(
    records: list[dict],
    id_to_type: dict[int, str],
    output_path: Path,
    metric: str,
) -> Path:
    selected = select_one_per_type(records, id_to_type)
    if not selected:
        raise ValueError("No records found for overview plot.")

    ncols = 2
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4.5 * nrows), sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    colors = [
        "#2563eb",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0f766e",
        "#dc2626",
        "#4f46e5",
        "#65a30d",
    ]

    max_y = 0.0
    for _, record in selected:
        _, ys = get_xy(record.get("tokens", []), metric)
        if ys:
            max_y = max(max_y, max(ys))
    max_y = max(1.0, max_y * 1.08)

    for idx, ((question_type, record), ax) in enumerate(zip(selected, axes_list)):
        tokens = record.get("tokens", [])
        xs, ys = get_xy(tokens, metric)
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, linewidth=1.35, color=color)

        relation_x = [i for i, token in enumerate(tokens) if token.get("is_relation_token")]
        relation_y = [ys[i] for i in relation_x]
        if relation_x:
            ax.scatter(relation_x, relation_y, color="#dc2626", s=18, zorder=3)
            for i in relation_x:
                label = sanitize_text(
                    tokens[i].get("normalized_token_text") or tokens[i].get("token_text") or "",
                    limit=16,
                )
                ax.annotate(
                    label,
                    (xs[i], ys[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=6,
                    color="#991b1b",
                )

        soft_x = [i for i, token in enumerate(tokens) if token.get("mode") == "soft"]
        soft_y = [ys[i] for i in soft_x]
        if soft_x:
            ax.scatter(soft_x, soft_y, color="#f59e0b", s=12, zorder=2)

        sample_id = record.get("id")
        sample_index = record.get("sample_index")
        ax.set_title(
            f"{question_type} | sample={sample_index}, id={sample_id}, n={len(tokens)}",
            fontsize=10,
        )
        ax.set_xlabel("Generated token index")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, max_y)

    for ax in axes_list[: len(selected)]:
        ax.set_ylabel("Entropy")

    for ax in axes_list[len(selected) :]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color="#2563eb", linewidth=1.5, label=metric),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=6, label="relation token"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f59e0b", markersize=5, label="soft token"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"MMHal one-sample-per-type entropy overview ({metric})", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("token_entropy_full")
    parser.add_argument(
        "--metric",
        choices=["raw_entropy", "filtered_entropy"],
        default="raw_entropy",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for png figures. Defaults next to the input file.",
    )
    parser.add_argument(
        "--dataset_jsonl",
        default=None,
        help="Dataset JSONL path used to recover question_type labels.",
    )
    parser.add_argument(
        "--overview_per_type",
        action="store_true",
        help="Create one 4x2 overview figure with one sample per question_type.",
    )
    args = parser.parse_args()

    input_path = Path(args.token_entropy_full)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_{args.metric}_plots"
    )

    records = load_records(input_path)
    if args.overview_per_type:
        if not args.dataset_jsonl:
            raise SystemExit("--overview_per_type requires --dataset_jsonl")
        dataset_path = Path(args.dataset_jsonl)
        id_to_type = load_question_types(dataset_path)
        output_path = output_dir / f"{input_path.stem}_{args.metric}_overview_per_type.png"
        plot_overview_per_type(records, id_to_type, output_path, args.metric)
        print(output_path)
    else:
        for record in records:
            plot_record(record, output_dir, args.metric)
        print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

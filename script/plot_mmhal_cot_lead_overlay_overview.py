#!/usr/bin/env python3
"""Overlay COT and LEAD token-entropy curves for one MMHal sample per type."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt


TYPE_ORDER = [
    "adversarial",
    "attribute",
    "comparison",
    "counting",
    "environment",
    "holistic",
    "other",
    "relation",
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_first_id_per_type(dataset_path: Path) -> OrderedDict[str, int]:
    rows = load_jsonl(dataset_path)
    selected: OrderedDict[str, int] = OrderedDict()
    for question_type in TYPE_ORDER:
        selected[question_type] = None  # type: ignore[assignment]
    for row in rows:
        question_type = row.get("question_type", "unknown")
        sample_id = row.get("id")
        if question_type in selected and selected[question_type] is None:
            selected[question_type] = sample_id
    return OrderedDict((k, v) for k, v in selected.items() if v is not None)


def index_by_id(records: list[dict]) -> dict[int, dict]:
    out = {}
    for record in records:
        sample_id = record.get("id")
        if sample_id is None:
            continue
        try:
            out[int(sample_id)] = record
        except (TypeError, ValueError):
            continue
    return out


def entropy_series(record: dict, metric: str) -> tuple[list[int], list[float]]:
    tokens = record.get("tokens", [])
    xs = list(range(len(tokens)))
    ys = [float(token.get(metric) or 0.0) for token in tokens]
    return xs, ys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", required=True, help="Path to COT token_entropy_full.jsonl")
    parser.add_argument("--lead", required=True, help="Path to LEAD token_entropy_full.jsonl")
    parser.add_argument("--dataset_jsonl", required=True, help="Balanced MMHal subset jsonl")
    parser.add_argument("--metric", choices=["raw_entropy", "filtered_entropy"], default="raw_entropy")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    cot_records = index_by_id(load_jsonl(Path(args.cot)))
    lead_records = index_by_id(load_jsonl(Path(args.lead)))
    selected = select_first_id_per_type(Path(args.dataset_jsonl))

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 18), sharey=True)
    axes_list = list(axes.flat)

    max_y = 0.0
    pairs = []
    for question_type, sample_id in selected.items():
        cot_record = cot_records.get(int(sample_id))
        lead_record = lead_records.get(int(sample_id))
        if cot_record is None or lead_record is None:
            continue
        pairs.append((question_type, int(sample_id), cot_record, lead_record))
        for record in [cot_record, lead_record]:
            _, ys = entropy_series(record, args.metric)
            if ys:
                max_y = max(max_y, max(ys))
    max_y = max(1.0, max_y * 1.08)

    for ax, (question_type, sample_id, cot_record, lead_record) in zip(axes_list, pairs):
        cot_x, cot_y = entropy_series(cot_record, args.metric)
        lead_x, lead_y = entropy_series(lead_record, args.metric)

        ax.plot(cot_x, cot_y, color="#2563eb", linewidth=1.35, label="cot")
        ax.plot(lead_x, lead_y, color="#dc2626", linewidth=1.35, alpha=0.9, label="lead")

        cot_relation_x = [i for i, token in enumerate(cot_record.get("tokens", [])) if token.get("is_relation_token")]
        cot_relation_y = [cot_y[i] for i in cot_relation_x]
        if cot_relation_x:
            ax.scatter(cot_relation_x, cot_relation_y, color="#1d4ed8", s=16, zorder=3)

        lead_relation_x = [i for i, token in enumerate(lead_record.get("tokens", [])) if token.get("is_relation_token")]
        lead_relation_y = [lead_y[i] for i in lead_relation_x]
        if lead_relation_x:
            ax.scatter(lead_relation_x, lead_relation_y, color="#b91c1c", s=16, zorder=3)

        lead_soft_x = [i for i, token in enumerate(lead_record.get("tokens", [])) if token.get("mode") == "soft"]
        lead_soft_y = [lead_y[i] for i in lead_soft_x]
        if lead_soft_x:
            ax.scatter(lead_soft_x, lead_soft_y, color="#f59e0b", s=10, alpha=0.9, zorder=2)

        ax.set_title(
            f"{question_type} | id={sample_id} | cot n={len(cot_x)} | lead n={len(lead_x)}",
            fontsize=10,
        )
        ax.set_xlabel("Generated token index")
        ax.set_ylabel("Entropy")
        ax.set_ylim(0.0, max_y)
        ax.grid(alpha=0.25)

    for ax in axes_list[len(pairs):]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color="#2563eb", linewidth=1.5, label="cot"),
        plt.Line2D([0], [0], color="#dc2626", linewidth=1.5, label="lead"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1d4ed8", markersize=6, label="cot relation"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#b91c1c", markersize=6, label="lead relation"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f59e0b", markersize=5, label="lead soft"),
    ]
    fig.suptitle(f"MMHal per-type overlay overview ({args.metric})", fontsize=15, y=0.992)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

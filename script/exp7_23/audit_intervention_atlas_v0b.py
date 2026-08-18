#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


TREATMENT_DIRS = {
    "contracted_soft_l095": "contracted_soft_l095",
    "pure_soft_l100": "pure_soft_l100",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def explicit_answers(text: str) -> list[str]:
    patterns = (
        r"(?i)(?:final\s+answer|answer|choice)\s*(?:is|:)?\s*[\(\[]?([A-E])[\)\]]?",
        r"(?i)\*\*answer\s*:\*\*\s*[\(\[]?([A-E])[\)\]]?",
        r"(?i)therefore[^\n]{0,120}?(?:option|choice)\s*[\(\[]?([A-E])[\)\]]?",
        r"(?i)(?:option|choice)\s*[\(\[]?([A-E])[\)\]]?\s*(?:is|would be)?\s*(?:correct|best)",
    )
    found: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            found.append((match.start(), match.group(1).upper()))
    return [answer for _, answer in sorted(found)]


def option_text_fallback(row: dict[str, Any]) -> str | None:
    text = str(row.get("model_answer") or "").lower()
    options = str(row.get("options") or "")
    matches = re.findall(
        r"(?:^|\n)\s*([A-E])[\.\):]\s*(.+?)(?=\n\s*[A-E][\.\):]|\Z)",
        options,
        flags=re.IGNORECASE | re.DOTALL,
    )
    hits = []
    for letter, option in matches:
        normalized = re.sub(r"\s+", " ", option.strip().lower())
        if len(normalized) >= 4 and normalized in text[-800:]:
            hits.append(letter.upper())
    return hits[-1] if hits else None


def trace_tokens(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        str(row.get("id")): list(row.get("tokens") or [])
        for row in load_jsonl(path)
    }


def shard_for_event(root: Path, event_id: str) -> Path | None:
    for shard in sorted(root.glob("shard_*")):
        manifest = shard / "event_override_manifest.json"
        if not manifest.exists():
            continue
        mapping = json.loads(manifest.read_text(encoding="utf-8"))
        if event_id in mapping:
            return shard
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = load_jsonl(args.atlas)
    failed_labels = [
        row for row in labels if bool(row.get("treatment_failed_extraction"))
    ]
    fixed1_labels = [row for row in labels if row.get("event_type") == "fixed_1"]

    cache: dict[tuple[str, str], tuple[dict, dict]] = {}

    def data_for(row: dict[str, Any]) -> tuple[dict, list[dict[str, Any]]]:
        event_id = str(row["event_id"])
        treatment = str(row["treatment"])
        shard = shard_for_event(args.shard_root, event_id)
        if shard is None:
            raise KeyError(f"No shard for {event_id}")
        key = (str(shard), treatment)
        if key not in cache:
            treatment_dir = shard / TREATMENT_DIRS[treatment]
            cache[key] = (
                by_id(treatment_dir / "results.jsonl"),
                trace_tokens(treatment_dir / "token_entropy_full.jsonl"),
            )
        results, traces = cache[key]
        return results[event_id], traces[event_id]

    failed_audit = []
    failed_counts = Counter()
    for label in failed_labels:
        result, _ = data_for(label)
        text = str(result.get("model_answer") or "")
        answers = explicit_answers(text)
        fallback = option_text_fallback(result)
        recovered = answers[-1] if answers else fallback
        gold = str(label.get("gold") or "").strip().upper()
        recovered_correct = recovered == gold if recovered is not None else None
        failed_counts["total"] += 1
        failed_counts["regex_recovered"] += int(bool(answers))
        failed_counts["option_text_recovered"] += int(not answers and fallback is not None)
        failed_counts["still_unresolved"] += int(recovered is None)
        failed_counts["recovered_correct"] += int(recovered_correct is True)
        failed_counts["recovered_wrong"] += int(recovered_correct is False)
        failed_audit.append(
            {
                "event_id": label["event_id"],
                "dataset": label["dataset"],
                "treatment": label["treatment"],
                "gold": gold,
                "old_pred": label.get("treatment_pred"),
                "explicit_answers": answers,
                "option_text_fallback": fallback,
                "recovered_pred": recovered,
                "recovered_correct": recovered_correct,
                "output_tail": text[-1000:],
            }
        )

    fixed1_audit = []
    fixed1_counts = Counter()
    relative_l2 = []
    cosine = []
    entropy = []
    for label in fixed1_labels:
        _, tokens = data_for(label)
        token = next(
            (item for item in tokens if int(item.get("step", -1)) == 1),
            None,
        )
        if token is None:
            fixed1_counts["missing_step1_trace"] += 1
            continue
        active = bool(token.get("route_override_active"))
        fixed1_counts["events"] += 1
        fixed1_counts["override_active"] += int(active)
        fixed1_counts[f"kind_{token.get('route_override_kind')}"] += 1
        fixed1_counts["answer_changed"] += int(bool(label.get("answer_changed")))
        value = token.get("soft_hard_relative_l2")
        if value is not None:
            relative_l2.append(float(value))
        value = token.get("soft_hard_cosine")
        if value is not None:
            cosine.append(float(value))
        value = token.get("raw_entropy")
        if value is not None:
            entropy.append(float(value))
        fixed1_audit.append(
            {
                "event_id": label["event_id"],
                "dataset": label["dataset"],
                "treatment": label["treatment"],
                "override_active": active,
                "override_kind": token.get("route_override_kind"),
                "mix_lambda": token.get("route_override_mix_lambda"),
                "raw_entropy": token.get("raw_entropy"),
                "raw_top1_prob": token.get("raw_top1_prob"),
                "soft_hard_relative_l2": token.get("soft_hard_relative_l2"),
                "soft_hard_cosine": token.get("soft_hard_cosine"),
                "answer_changed": bool(label.get("answer_changed")),
                "first_divergence_step": label.get("first_divergence_step"),
            }
        )

    fixed1_summary = dict(fixed1_counts)
    for name, values in (
        ("soft_hard_relative_l2", relative_l2),
        ("soft_hard_cosine", cosine),
        ("raw_entropy", entropy),
    ):
        fixed1_summary[f"{name}_mean"] = statistics.fmean(values) if values else None
        fixed1_summary[f"{name}_median"] = statistics.median(values) if values else None
        fixed1_summary[f"{name}_max"] = max(values) if values else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "failed_extraction": dict(failed_counts),
        "fixed1": fixed1_summary,
    }
    (args.output_dir / "atlas_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, rows in (
        ("failed_extraction_audit.jsonl", failed_audit),
        ("fixed1_override_audit.jsonl", fixed1_audit),
    ):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Intervention Atlas V0B Audit",
        "",
        "## Failed extraction",
        "",
        "```json",
        json.dumps(dict(failed_counts), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Fixed-step-1 override",
        "",
        "```json",
        json.dumps(fixed1_summary, ensure_ascii=False, indent=2),
        "```",
    ]
    (args.output_dir / "atlas_audit_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIXED_STEPS = (1, 2, 4, 8, 16, 32)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    samples = {
        str(row["id"]): row
        for row in load_jsonl(args.samples)
        if str(row["id"]).startswith("visulogic::")
    }
    traces = {
        str(row["id"]): row for row in load_jsonl(args.baseline_trace)
    }
    expanded_rows = []
    event_rows = []
    override = {}

    for sample_id, source in samples.items():
        tokens = (traces.get(sample_id) or {}).get("tokens") or []
        legal = list(range(max(0, len(tokens) - 1)))
        used = {step for step in FIXED_STEPS if step in legal}
        entropy_ranked = sorted(
            legal,
            key=lambda step: float(tokens[step].get("raw_entropy") or 0.0),
            reverse=True,
        )
        entropy_step = next(
            (step for step in entropy_ranked if step not in used), None
        )
        events = []
        if entropy_step is not None:
            events.append(("entropy_top1", entropy_step))
            used.add(entropy_step)
        remaining = [step for step in legal if step not in used]
        if remaining:
            digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
            random_step = remaining[
                int.from_bytes(digest[:8], "big") % len(remaining)
            ]
            events.append(("random_control", random_step))

        for event_type, step in events:
            event_id = f"{sample_id}::atlas::{event_type}::{step}"
            expanded = dict(source)
            expanded["id"] = event_id
            expanded["_atlas_original_id"] = sample_id
            expanded["_atlas_event_type"] = event_type
            expanded["_atlas_event_step"] = step
            expanded_rows.append(expanded)
            override[event_id] = step
            event_rows.append(
                {
                    "event_id": event_id,
                    "original_id": sample_id,
                    "dataset": "visulogic",
                    "event_type": event_type,
                    "event_step": step,
                    "gold": source.get("answer"),
                    "subtopic": source.get("subtopic") or source.get("subject"),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "event_dataset.jsonl", expanded_rows)
    write_jsonl(args.output_dir / "event_manifest.jsonl", event_rows)
    (args.output_dir / "event_override_manifest.json").write_text(
        json.dumps(override, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "samples": len(samples),
                "events": len(event_rows),
                "expected_events_per_sample": 2,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

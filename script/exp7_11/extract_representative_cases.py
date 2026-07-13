#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path


root = Path("output/experiments/20260711_fixed_damaged_mechanism_analysis/fixed_damaged_mechanism_analysis")
rows = [json.loads(line) for line in (root / "selected_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
core = {"vstar", "mmvp", "visulogic300", "realworldqa_fixed200"}
groups = defaultdict(list)
for row in rows:
    if row["dataset"] in core and row["group"] in {"fixed", "damaged"}:
        groups[(row["dataset"], row["group"])].append(row)

output = []
for key in sorted(groups):
    candidates = sorted(groups[key], key=lambda row: (row["method"], str(row["id"])))
    chosen = []
    for method in ("pure_soft_format2", "initial_transition_only"):
        chosen.extend([row for row in candidates if row["method"] == method][:3])
    chosen = chosen[:5]
    for row in chosen:
        output.append({
            "dataset": row["dataset"],
            "method": row["method"],
            "group": row["group"],
            "id": row["id"],
            "subtopic": row.get("subtopic"),
            "question": row.get("question"),
            "gold": row.get("gold"),
            "cot_pred": row.get("baseline_pred"),
            "method_pred": row.get("method_pred"),
            "extraction_only_flip": row.get("extraction_only_flip"),
            "answer_reversal": row.get("method_answer_reversal"),
            "cot_length": row.get("baseline_length"),
            "method_length": row.get("method_length"),
            "suggested_semantic_label": row.get("semantic_audit_candidate"),
            "audit_status": "pending_manual_review",
        })

with (root / "representative_cases_for_audit.jsonl").open("w", encoding="utf-8") as handle:
    for row in output:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
for row in output:
    print(json.dumps(row, ensure_ascii=False))

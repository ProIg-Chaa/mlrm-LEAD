#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.replay_dir / "counterfactual_branches.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    actual = {
        (row["dataset"], row["method"], row["event"], str(row["id"])): row
        for row in rows if row["branch"] == "actual"
    }
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["event"], row["branch"])].append(row)
    summary = {
        "rows": len(rows),
        "prefix_mismatch": sum(not row.get("prefix_match") for row in rows),
        "actual_trajectory_mismatch": sum(row["branch"] == "actual" and not row.get("trajectory_match") for row in rows),
        "probe_unavailable": sum(not (row.get("forced_answer_probe") or {}).get("available") for row in rows),
        "visual_attention_available": sum((row.get("diagnostic_availability") or {}).get("visual_attention_available", False) for row in rows),
        "hidden_visual_alignment_available": sum((row.get("diagnostic_availability") or {}).get("hidden_visual_alignment_available", False) for row in rows),
        "groups": [],
    }
    for key, group in sorted(groups.items()):
        changed = fixed = damaged = 0
        for row in group:
            reference = actual.get((row["dataset"], row["method"], row["event"], str(row["id"])))
            if reference is None or row["branch"] == "actual":
                continue
            changed += int(row.get("prediction") != reference.get("prediction"))
            ref_correct = reference.get("prediction") == reference.get("gold")
            branch_correct = row.get("prediction") == row.get("gold")
            fixed += int(not ref_correct and branch_correct)
            damaged += int(ref_correct and not branch_correct)
        summary["groups"].append({
            "dataset": key[0], "method": key[1], "event": key[2], "branch": key[3],
            "n": len(group),
            "next_token_changed": sum(bool(row.get("next_token_changed")) for row in group),
            "final_answer_changed_vs_actual": changed,
            "fixed_vs_actual": fixed,
            "damaged_vs_actual": damaged,
            "edit8_norm_mean": mean([row.get("token_edit_distance_8_normalized") for row in group]),
            "edit32_norm_mean": mean([row.get("token_edit_distance_32_normalized") for row in group]),
            "js_mean": mean([
                (row.get("actual_branch_logits_divergence") or {}).get("js_divergence")
                for row in group
                if (row.get("actual_branch_logits_divergence") or {}).get("available")
            ]),
            "forced_gold_margin_mean": mean([
                (row.get("forced_answer_probe") or {}).get("gold_margin") for row in group
            ]),
            "hidden_visual_align_mean": mean([
                (row.get("forced_answer_probe") or {}).get("event_hidden_visual_align_top4_mean")
                for row in group
            ]),
        })
    path = args.replay_dir / "replay_status_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

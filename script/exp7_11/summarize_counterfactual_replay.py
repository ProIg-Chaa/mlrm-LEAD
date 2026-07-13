#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract(text: str):
    patterns = [r"\\boxed\{\s*\(?([A-Ea-e])", r"final\s+(?:answer|choice).*?([A-Ea-e])", r"answer\s*[:\s]+\(?([A-Ea-e])"]
    region = (text or "").rsplit("</think>", 1)[-1]
    hits = []
    for pattern in patterns:
        hits.extend((match.start(), match.group(1).upper()) for match in re.finditer(pattern, region, re.I))
    return max(hits)[1] if hits else None


def normalize_gold(value):
    text = str(value or "").strip()
    match = re.search(r"(?:^|\()\s*([A-Ea-e])(?:\)|\b)", text)
    return match.group(1).upper() if match else text.upper() or None


def token_ids(trace):
    return [int(token["token_id"]) for token in (trace or {}).get("tokens", []) if "token_id" in token]


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row_index, left_value in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_value in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + int(left_value != right_value),
            ))
        previous = current
    return previous[-1]


def continuation_distances(reference, branch, event_step):
    reference_tail = reference[event_step + 1:]
    branch_tail = branch[event_step + 1:]
    result = {}
    for width in (8, 16, 32):
        left, right = reference_tail[:width], branch_tail[:width]
        distance = edit_distance(left, right)
        result[f"token_edit_distance_{width}"] = distance
        result[f"token_edit_distance_{width}_normalized"] = distance / max(1, len(left), len(right))
    return result


def token_record_at(trace, step):
    for token in (trace or {}).get("tokens", []):
        if int(token.get("step", -1)) == step:
            return token
    return None


def topk_distribution(topk):
    distribution = {int(item["token_id"]): max(0.0, float(item["prob"])) for item in (topk or [])}
    distribution["__other__"] = max(0.0, 1.0 - sum(distribution.values()))
    return distribution


def approximate_divergence(actual_topk, branch_topk):
    """Compute top-k plus residual-bucket divergence; this is not full-vocabulary KL."""
    if not actual_topk or not branch_topk:
        return {
            "available": False,
            "reason": "next_step_topk_missing",
            "scope": "top20_plus_residual_bucket",
        }
    left, right = topk_distribution(actual_topk), topk_distribution(branch_topk)
    keys = set(left) | set(right)
    epsilon = 1e-12
    p = {key: max(epsilon, left.get(key, 0.0)) for key in keys}
    q = {key: max(epsilon, right.get(key, 0.0)) for key in keys}
    p_total, q_total = sum(p.values()), sum(q.values())
    p = {key: value / p_total for key, value in p.items()}
    q = {key: value / q_total for key, value in q.items()}
    midpoint = {key: 0.5 * (p[key] + q[key]) for key in keys}

    def kl(left_dist, right_dist):
        return sum(left_dist[key] * math.log(left_dist[key] / right_dist[key]) for key in keys)

    return {
        "available": True,
        "scope": "top20_plus_residual_bucket",
        "kl_actual_to_branch": kl(p, q),
        "kl_branch_to_actual": kl(q, p),
        "js_divergence": 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_json(args.replay_dir / "replay_manifest.json")
    output = []
    event_traces = []
    for entry in manifest:
        run_dir, reference_dir = Path(entry["run_dir"]), Path(entry["reference_dir"])
        steps = {str(k): int(v) for k, v in load_json(Path(entry["event_steps"])).items()}
        rows = {str(row.get("id")): row for row in load_jsonl(run_dir / "results.jsonl")}
        traces = {str(row.get("id")): row for row in load_jsonl(run_dir / "token_entropy_full.jsonl")}
        refs = {str(row.get("id")): row for row in load_jsonl(reference_dir / "token_entropy_full.jsonl")}
        for sid, event_step in steps.items():
            branch_ids, ref_ids = token_ids(traces.get(sid)), token_ids(refs.get(sid))
            prefix_match = branch_ids[:event_step + 1] == ref_ids[:event_step + 1]
            divergence = first_divergence(ref_ids, branch_ids)
            row = rows.get(sid, {})
            event_token = token_record_at(traces.get(sid), event_step)
            next_token_record = token_record_at(traces.get(sid), event_step + 1)
            geometry = None
            forced_answer_probe = None
            if event_token:
                geometry = {key: event_token.get(key) for key in ["hard_emb_norm", "soft_emb_norm", "route_emb_norm", "soft_hard_cosine", "route_hard_cosine", "route_soft_cosine", "route_visual_anchor_cosine", "route_override_active", "route_override_kind"]}
                forced_answer_probe = event_token.get("forced_answer_probe")
            probe_record = forced_answer_probe or {}
            diagnostics = {
                "visual_attention_available": bool(probe_record.get("event_visual_attn_available")),
                "visual_attention_reason": probe_record.get("event_visual_attn_reason"),
                "hidden_visual_alignment_available": bool(probe_record.get("event_hidden_visual_align_available")),
                "hidden_visual_alignment_reason": probe_record.get("event_hidden_visual_align_reason"),
                "visual_attention_scope": "forced-answer event route probe, decoder last four layers",
            }
            output.append({
                **{key: entry[key] for key in ["dataset", "method", "event", "branch"]},
                "id": sid, "event_step": event_step, "prefix_match": prefix_match,
                "replay_mismatch": not prefix_match, "trajectory_match": branch_ids == ref_ids,
                "first_divergence": divergence,
                "next_token_changed": divergence == event_step + 1,
                "prediction": extract(row.get("model_answer")), "gold": normalize_gold(row.get("answer")),
                "output_tokens": row.get("output_tokens"), "event_geometry": geometry,
                "forced_answer_probe": forced_answer_probe,
                "next_step_raw_topk": (next_token_record or {}).get("raw_topk"),
                "diagnostic_availability": diagnostics,
                **continuation_distances(ref_ids, branch_ids, event_step),
            })
            event_traces.append({
                **{key: entry[key] for key in ["dataset", "method", "event", "branch"]},
                "id": sid,
                "event_step": event_step,
                "event_record": event_token,
                "next_step_raw_topk": (next_token_record or {}).get("raw_topk"),
                "event_geometry": geometry,
                "forced_answer_probe": forced_answer_probe,
                "diagnostic_availability": diagnostics,
            })

    actual_topk = {
        (row["dataset"], row["method"], row["event"], row["id"]): row.get("next_step_raw_topk")
        for row in output if row["branch"] == "actual"
    }
    for row in output:
        key = (row["dataset"], row["method"], row["event"], row["id"])
        row["actual_branch_logits_divergence"] = approximate_divergence(
            actual_topk.get(key), row.get("next_step_raw_topk")
        )
    out = args.replay_dir / "counterfactual_branches.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    event_out = args.replay_dir / "event_traces.jsonl"
    with event_out.open("w", encoding="utf-8") as handle:
        for row in event_traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {}
    for row in output:
        key = "/".join([row["dataset"], row["method"], row["event"], row["branch"]])
        item = summary.setdefault(key, {"n": 0, "prefix_match": 0, "next_token_changed": 0})
        item["n"] += 1
        item["prefix_match"] += int(row["prefix_match"])
        item["next_token_changed"] += int(row["next_token_changed"])
    (args.replay_dir / "counterfactual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} and {event_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

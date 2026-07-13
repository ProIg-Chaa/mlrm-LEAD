#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path


CORE_DATASETS = ["vstar", "mmvp", "visulogic300", "realworldqa_fixed200"]
EXTENDED_FORMAT_DATASETS = [
    "vmcbench_dev", "pope_random", "pope_popular", "pope_adversarial",
    "mmk12_math", "mmk12_physics", "mmk12_chemistry", "mmk12_biology",
]
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32)
ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Ea-e])\)?\s*\}"),
    re.compile(r"\**final\s+(?:answer|choice)\s*\**\s*(?:is)?\s*[:\s]*\**\s*\(?([A-Ea-e])\)?", re.I),
    re.compile(r"\**(?:the\s+correct\s+)?answer\s*\**\s+(?:is\s*)?[:\s]+\**\s*\(?([A-Ea-e])\)?", re.I),
    re.compile(r"\**answer\s*\**\s*[:\s]+\**\s*\(?([A-Ea-e])\)?", re.I),
    re.compile(r"\*\*([A-Ea-e])\*\*"),
    re.compile(r"(?:^|\n)\s*\(?([A-Ea-e])\)?\s*$"),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_gold(answer: str, options: str = "") -> str | None:
    text = str(answer or "").strip()
    match = re.search(r"(?:^|\()\s*([A-Ea-e])\s*\)?(?:\s|$)", text)
    if match:
        return match.group(1).upper()
    upper = text.upper()
    if upper in {"YES", "TRUE"}:
        return "A"
    if upper in {"NO", "FALSE"}:
        return "B"
    for letter, option_text in parse_options(options).items():
        if text.casefold() == option_text.casefold():
            return letter
    return None


def parse_options(options: str) -> dict[str, str]:
    text = str(options or "")
    matches = list(re.finditer(r"(?:^|\s|\n)(?:\(([A-Ea-e])\)|([A-Ea-e])[\.:)])\s*", text))
    parsed = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        letter = match.group(1) or match.group(2)
        parsed[letter.upper()] = text[match.end():end].strip()
    return parsed


def answer_region(text: str) -> str:
    markers = list(re.finditer(r"(?:final\s+)?answer\s*[:.]", text or "", re.I))
    if markers:
        return text[markers[-1].start():]
    return (text or "")[-1800:]


def extract_prediction(text: str, options: str = "", loose: bool = False) -> str | None:
    if not text:
        return None
    region = answer_region(text)
    hits = []
    for pattern in ANSWER_PATTERNS:
        hits.extend((m.start(), m.group(1).upper()) for m in pattern.finditer(region))
    if hits:
        return max(hits)[1]
    choices = re.findall(r"(?:^|\n)\s*\(?([A-Ea-e])\)?(?:\s|[.)])", region[-500:])
    if choices:
        return choices[-1].upper()
    last_letters = re.findall(r"\b([A-E])\b", region[-200:])
    if last_letters:
        return last_letters[-1].upper()
    option_hits = []
    for letter, option_text in parse_options(options).items():
        if not option_text:
            continue
        for match in re.finditer(re.escape(option_text), region, re.I):
            option_hits.append((match.start(), letter))
    if option_hits:
        return max(option_hits)[1]
    if loose:
        letters = re.findall(r"\b([A-Ea-e])\b", (text or "")[-600:])
        return letters[-1].upper() if letters else None
    return None


def explicit_answers(text: str) -> list[str]:
    hits = []
    for pattern in ANSWER_PATTERNS:
        hits.extend((m.start(), m.group(1).upper()) for m in pattern.finditer(text or ""))
    answers = []
    for _, value in sorted(hits):
        if not answers or answers[-1] != value:
            answers.append(value)
    return answers


def ngram_repeat_ratio(text: str, n: int = 3) -> float:
    words = re.findall(r"\w+", (text or "").casefold())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def answer_marker_position(text: str) -> int | None:
    markers = list(re.finditer(r"(?:final\s+)?answer\s*[:.]", text or "", re.I))
    return markers[-1].start() if markers else None


def semantic_candidate(row: dict, dataset: str) -> str:
    question = str(row.get("question") or "").casefold()
    subtopic = str(row.get("subtopic") or "").casefold()
    if dataset.startswith("pope_"):
        return "object_hallucination_yes_no"
    if "relative" in subtopic or any(x in question for x in ["left or right", "above or below", "closer to"]):
        return "spatial_relation"
    if any(x in question for x in ["color", "material", "shape", "wearing"]):
        return "visual_attribute"
    if any(x in question for x in ["how many", "number of"]):
        return "visual_counting"
    if any(x in question for x in ["price", "cost", "text", "read", "label"]):
        return "ocr_or_numeric_reading"
    if dataset.startswith("mmk12_"):
        return dataset.replace("mmk12_", "science_or_math_")
    if dataset == "visulogic300":
        return "visual_logic"
    return "visual_reasoning_other"


def parse_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["source"] = str(config_path)
        return config
    command_path = run_dir / "run_command.sh"
    if not command_path.exists():
        return {"source": None}
    content = command_path.read_text(encoding="utf-8")
    flat = re.sub(r"\\\s*\n", " ", content)
    tokens = shlex.split(flat)
    config = {"source": str(command_path)}
    value_flags = {
        "--model_name": "model_name", "--dataset": "dataset", "--method": "method",
        "--max_new_tokens": "max_new_tokens", "--temperature": "temperature",
        "--top_p": "top_p", "--top_k": "top_k", "--seed": "seed",
        "--cot_prompt_mode": "cot_prompt_mode", "--alpha": "alpha",
        "--window_size": "window_size", "--max_switch_count": "max_switch_count",
    }
    for i, token in enumerate(tokens[:-1]):
        if token in value_flags:
            config[value_flags[token]] = tokens[i + 1]
    config["do_sample"] = "--no-do_sample" not in tokens
    config["lead_initial_transition_only"] = "--lead_initial_transition_only" in tokens
    return config


def normalized_model(value: str | None) -> str:
    return Path(str(value or "")).name.casefold()


def validate_pair(dataset: str, baseline_dir: Path, method_dir: Path) -> dict:
    base = parse_run_config(baseline_dir)
    method = parse_run_config(method_dir)
    errors = []
    expected_dataset = {
        "visulogic300": "visulogic.jsonl",
        "realworldqa_fixed200": "realworldqa_fixed_mcq_random200_seed42.jsonl",
    }.get(dataset, f"{dataset}.jsonl")
    for label, config in [("baseline", base), ("method", method)]:
        if normalized_model(config.get("model_name")) != "r1-onevision-7b-rl":
            errors.append(f"{label}: unexpected model={config.get('model_name')}")
        if Path(str(config.get("dataset") or "")).name != expected_dataset:
            errors.append(f"{label}: unexpected dataset={config.get('dataset')}")
        if bool(config.get("do_sample", True)):
            errors.append(f"{label}: do_sample must be false")
        if int(config.get("max_new_tokens", -1)) != 1024:
            errors.append(f"{label}: max_new_tokens must be 1024")
        if int(config.get("seed", -1)) != 42:
            errors.append(f"{label}: seed must be 42")
        if config.get("cot_prompt_mode", "orign") != "orign":
            errors.append(f"{label}: cot_prompt_mode must be orign")
    if not (baseline_dir / "results.jsonl").exists():
        errors.append("baseline results missing")
    if not (method_dir / "results.jsonl").exists():
        errors.append("method results missing")
    return {"valid": not errors, "errors": errors, "baseline_config": base, "method_config": method}


def traces_by_id(run_dir: Path) -> dict[str, dict]:
    return {str(row.get("id")): row for row in load_jsonl(run_dir / "token_entropy_full.jsonl")}


def token_sequence(trace: dict | None) -> list[int]:
    return [int(token["token_id"]) for token in (trace or {}).get("tokens", []) if "token_id" in token]


def first_divergence(a: dict | None, b: dict | None) -> int | None:
    if not a or not b:
        return None
    left, right = token_sequence(a), token_sequence(b)
    for index, (x, y) in enumerate(zip(left, right)):
        if x != y:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def trace_summary(trace: dict | None) -> dict:
    if not trace:
        return {
            "available": False, "tokens": 0, "soft_steps": None, "soft_ratio": None,
            "to_soft": None, "to_normal": None, "format_triggers": None,
            "early_entropy_mean": None, "early_top1_mean": None,
        }
    tokens = (trace or {}).get("tokens", [])
    soft = sum(token.get("mode") in {"soft", "pure_soft"} for token in tokens)
    return {
        "available": True,
        "tokens": len(tokens),
        "soft_steps": soft,
        "soft_ratio": soft / len(tokens) if tokens else 0.0,
        "to_soft": sum(bool(token.get("to_soft")) for token in tokens),
        "to_normal": sum(bool(token.get("to_normal")) for token in tokens),
        "format_triggers": sum(bool(token.get("format_cooldown_active")) for token in tokens),
        "early_entropy_mean": mean([float(token.get("raw_entropy", 0.0)) for token in tokens[:32]]),
        "early_top1_mean": mean([float(token.get("raw_top1_prob", token.get("selected_prob", 0.0))) for token in tokens[:32]]),
    }


def mean(values: list[float]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def result_features(row: dict) -> dict:
    text = str(row.get("model_answer") or "")
    answers = explicit_answers(text)
    return {
        "output_tokens": int(row.get("output_tokens") or 0),
        "long_256": int(row.get("output_tokens") or 0) >= 256,
        "maxed_1024": int(row.get("output_tokens") or 0) >= 1024,
        "answer_marker_pos": answer_marker_position(text),
        "repeat_ngram3_ratio": ngram_repeat_ratio(text),
        "answer_reversal": len(answers) >= 2 and answers[0] != answers[-1],
        "runtime_error": bool(row.get("error_type")),
    }


def mcnemar_exact(fixed: int, damaged: int) -> float:
    n = fixed + damaged
    if n == 0:
        return 1.0
    k = min(fixed, damaged)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_net(effects: list[int], seed: int = 42, draws: int = 10000) -> list[float]:
    if not effects:
        return [0.0, 0.0]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(sum(rng.choice(effects) for _ in effects) / len(effects))
    samples.sort()
    return [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]]


def build_manifest(root: Path) -> list[dict]:
    baseline_root = root / "output/experiments/20260705_integrated_cot_lead_baselines/integrated_repo_cot_lead_baselines/r1_onevision_7b"
    format_root = root / "output/experiments/20260706_format_stability_full_baselines/format_stability_full_baselines/r1_onevision_7b"
    transition_root = root / "output/experiments/20260602_220321/rerun_early_path_dependence_mechanism/phase3_cross_dataset_minimal"
    manifest = []
    for dataset in CORE_DATASETS + EXTENDED_FORMAT_DATASETS:
        manifest.append({
            "dataset": dataset,
            "method": "pure_soft_format2",
            "baseline_dir": str(baseline_root / dataset / "cot_orign_greedy_gpu0"),
            "method_dir": str(format_root / dataset / "pure_soft_format2_gpu0"),
            "tier": "core" if dataset in CORE_DATASETS else "format_extension",
        })
    for dataset in CORE_DATASETS:
        manifest.append({
            "dataset": dataset,
            "method": "initial_transition_only",
            "baseline_dir": str(baseline_root / dataset / "cot_orign_greedy_gpu0"),
            "method_dir": str(transition_root / dataset / "initial_transition_only_gpu0"),
            "tier": "core",
        })
    combo_root = root / "output/experiments/20260711_fixed_damaged_mechanism_analysis/transition_preserving_combo"
    for dataset in CORE_DATASETS:
        combo_dir = combo_root / dataset / "transition_preserving_quota05_guard_min2"
        if (combo_dir / "results.jsonl").exists():
            manifest.append({
                "dataset": dataset,
                "method": "transition_preserving_quota05_guard_min2",
                "baseline_dir": str(baseline_root / dataset / "cot_orign_greedy_gpu0"),
                "method_dir": str(combo_dir),
                "tier": "core_combo",
            })
    return manifest


def subgroup(row: dict) -> str:
    if row["baseline_correct"] and row["method_correct"]:
        return "both_correct"
    if row["baseline_correct"] and not row["method_correct"]:
        return "damaged"
    if not row["baseline_correct"] and row["method_correct"]:
        return "fixed"
    return "both_wrong"


def row_map(run_dir: Path, dataset: str) -> dict[str, dict]:
    result_rows = {str(row.get("id")): row for row in load_jsonl(run_dir / "results.jsonl")}
    if dataset == "mmvp":
        specialized = load_jsonl(run_dir / "specialized_eval_results.jsonl")
        if not specialized:
            specialized = load_jsonl(run_dir / "specialized_eval_rows.jsonl")
        if specialized:
            for row in specialized:
                sid = str(row.get("id"))
                if sid in result_rows:
                    result_rows[sid]["_specialized_pred"] = row.get("specialized_pred")
                    result_rows[sid]["_specialized_correct"] = bool(row.get("specialized_is_correct"))
                    result_rows[sid]["_pair_index"] = row.get("pair_index")
                    result_rows[sid]["_pair_correct"] = row.get("pair_is_correct")
    return result_rows


def adjudicate(row: dict, dataset: str) -> tuple[str | None, bool, str | None]:
    if dataset == "mmvp" and "_specialized_correct" in row:
        pred = row.get("_specialized_pred")
        pred = str(pred).upper() if pred else None
        return pred, bool(row["_specialized_correct"]), pred
    gold = normalize_gold(row.get("answer"), row.get("options"))
    pred = extract_prediction(row.get("model_answer"), row.get("options"))
    loose = extract_prediction(row.get("model_answer"), row.get("options"), loose=True)
    return pred, bool(pred and gold and pred == gold), loose


def select_rows(rows: list[dict]) -> list[dict]:
    rng = random.Random(42)
    selected = []
    limits = {"fixed": 40, "damaged": 40, "both_correct": 20, "both_wrong": 20}
    for group, limit in limits.items():
        candidates = [row for row in rows if row["group"] == group]
        buckets = defaultdict(list)
        for row in candidates:
            length_bin = min(4, int(row["baseline_features"]["output_tokens"] // 128))
            buckets[(row["subtopic"], length_bin)].append(row)
        for values in buckets.values():
            rng.shuffle(values)
        ordered = []
        while buckets and len(ordered) < limit:
            for key in sorted(list(buckets)):
                values = buckets[key]
                if values:
                    ordered.append(values.pop())
                if not values:
                    del buckets[key]
                if len(ordered) >= limit:
                    break
        selected.extend(ordered)
    return selected


def trace_prefix(trace: dict | None, count: int) -> str:
    texts = [str(token.get("token_text") or "") for token in (trace or {}).get("tokens", [])[:count]]
    return "".join(texts)


def route_events(trace: dict | None) -> list[dict]:
    events = []
    for token in (trace or {}).get("tokens", []):
        if token.get("to_soft") or token.get("to_normal") or token.get("format_cooldown_active") or token.get("lead_soft_veto"):
            events.append({
                key: token.get(key) for key in [
                    "step", "token_text", "mode", "route_signal", "route_action", "to_soft", "to_normal",
                    "format_cooldown_active", "lead_soft_veto", "raw_entropy", "raw_top1_prob", "raw_margin",
                ]
            })
    return events[:30]


def compact_text(text: str, limit: int = 1200) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def write_card(path: Path, row: dict) -> None:
    lines = [
        f"# {row['dataset']} / {row['method']} / {row['group']} / id={row['id']}", "",
        f"- gold: `{row['gold']}`; COT: `{row['baseline_pred']}`; method: `{row['method_pred']}`",
        f"- subtopic: `{row['subtopic']}`; semantic audit candidate: `{row['semantic_audit_candidate']}`",
        f"- first token divergence: `{row['first_token_divergence']}`",
        f"- extraction-only flip: `{row['extraction_only_flip']}`",
        f"- image: `{row['image']}`", "",
        "## Question", "", str(row.get("question") or ""), "", str(row.get("options") or ""), "",
        "## Objective Signals", "", "```json", json.dumps({
            "baseline": row["baseline_features"], "method": row["method_features"],
            "baseline_trace": row["baseline_trace_summary"], "method_trace": row["method_trace_summary"],
        }, ensure_ascii=False, indent=2), "```", "",
        "## Prefixes", "",
    ]
    for checkpoint in [1, 2, 4, 8, 16, 32]:
        lines.append(f"- COT {checkpoint}: `{row['baseline_prefixes'][str(checkpoint)]}`")
        lines.append(f"- method {checkpoint}: `{row['method_prefixes'][str(checkpoint)]}`")
    lines += ["", "## Route Events", "", "```json", json.dumps(row["method_route_events"], ensure_ascii=False, indent=2), "```", "",
              "## COT Output", "", compact_text(row["baseline_output"]), "", "## Method Output", "", compact_text(row["method_output"]), "",
              "## Manual Semantic Audit", "", "- [ ] visual recognition", "- [ ] attribute", "- [ ] spatial relation", "- [ ] OCR/numeric", "- [ ] arithmetic/logic", "- [ ] hallucination", "- [ ] option mapping", "- [ ] other", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_comparison(entry: dict, out_dir: Path) -> tuple[list[dict], dict, list[dict]]:
    dataset, method = entry["dataset"], entry["method"]
    baseline_dir, method_dir = Path(entry["baseline_dir"]), Path(entry["method_dir"])
    validation = validate_pair(dataset, baseline_dir, method_dir)
    entry["validation"] = validation
    if not validation["valid"]:
        return [], {"dataset": dataset, "method": method, "valid": False, "errors": validation["errors"]}, []
    baseline_rows, method_rows = row_map(baseline_dir, dataset), row_map(method_dir, dataset)
    baseline_traces, method_traces = traces_by_id(baseline_dir), traces_by_id(method_dir)
    ids = sorted(set(baseline_rows) & set(method_rows), key=lambda value: (len(value), value))
    rows = []
    for sid in ids:
        base, current = baseline_rows[sid], method_rows[sid]
        base_pred, base_correct, base_loose = adjudicate(base, dataset)
        method_pred, method_correct, method_loose = adjudicate(current, dataset)
        gold = normalize_gold(base.get("answer"), base.get("options"))
        base_trace, current_trace = baseline_traces.get(sid), method_traces.get(sid)
        item = {
            "dataset": dataset, "method": method, "tier": entry["tier"], "id": sid,
            "image": base.get("image"), "question": base.get("question"), "options": base.get("options"),
            "answer": base.get("answer"), "gold": gold, "subtopic": base.get("subtopic") or base.get("benchmark") or "unknown",
            "baseline_pred": base_pred, "method_pred": method_pred,
            "baseline_loose_pred": base_loose, "method_loose_pred": method_loose,
            "baseline_correct": base_correct, "method_correct": method_correct,
            "mmvp_pair_index": base.get("_pair_index") if dataset == "mmvp" else None,
            "baseline_pair_correct": base.get("_pair_correct") if dataset == "mmvp" else None,
            "method_pair_correct": current.get("_pair_correct") if dataset == "mmvp" else None,
            "baseline_failed_extraction": base_pred is None, "method_failed_extraction": method_pred is None,
            "extraction_only_flip": (base_pred is None and base_loose == gold) or (method_pred is None and method_loose == gold),
            "baseline_features": result_features(base), "method_features": result_features(current),
            "baseline_trace_summary": trace_summary(base_trace), "method_trace_summary": trace_summary(current_trace),
            "first_token_divergence": first_divergence(base_trace, current_trace),
            "semantic_audit_candidate": semantic_candidate(base, dataset),
            "baseline_output": base.get("model_answer"), "method_output": current.get("model_answer"),
            "baseline_prefixes": {str(n): trace_prefix(base_trace, n) for n in [1, 2, 4, 8, 16, 32]},
            "method_prefixes": {str(n): trace_prefix(current_trace, n) for n in [1, 2, 4, 8, 16, 32]},
            "method_route_events": route_events(current_trace),
        }
        item["group"] = subgroup(item)
        rows.append(item)
    counts = Counter(row["group"] for row in rows)
    fixed, damaged = counts["fixed"], counts["damaged"]
    effects = [int(row["method_correct"]) - int(row["baseline_correct"]) for row in rows]
    stats = {
        "dataset": dataset, "method": method, "tier": entry["tier"], "valid": True,
        "total": len(rows), "counts": dict(counts), "net_gain_count": fixed - damaged,
        "net_gain_rate": (fixed - damaged) / len(rows) if rows else 0.0,
        "bootstrap_95ci_net_rate": bootstrap_net(effects), "mcnemar_exact_p": mcnemar_exact(fixed, damaged),
        "baseline_accuracy": mean([float(row["baseline_correct"]) for row in rows]),
        "method_accuracy": mean([float(row["method_correct"]) for row in rows]),
        "extraction_only_flips": sum(row["extraction_only_flip"] for row in rows),
        "group_metrics": {},
    }
    if dataset == "mmvp":
        pair_samples = {}
        for row in rows:
            pair_index = row.get("mmvp_pair_index")
            if pair_index is None:
                try:
                    pair_index = int(row["id"]) // 2
                except (TypeError, ValueError):
                    continue
            pair_samples.setdefault(int(pair_index), []).append(
                (bool(row["baseline_correct"]), bool(row["method_correct"]))
            )
        pair_rows = {
            pair_index: (
                len(samples) == 2 and all(baseline for baseline, _ in samples),
                len(samples) == 2 and all(method for _, method in samples),
            )
            for pair_index, samples in pair_samples.items()
        }
        pair_fixed = sum(not baseline and method for baseline, method in pair_rows.values())
        pair_damaged = sum(baseline and not method for baseline, method in pair_rows.values())
        stats["mmvp_pair_consistency"] = {
            "pairs": len(pair_rows),
            "baseline_pair_accuracy": mean([float(baseline) for baseline, _ in pair_rows.values()]),
            "method_pair_accuracy": mean([float(method) for _, method in pair_rows.values()]),
            "fixed_pairs": pair_fixed,
            "damaged_pairs": pair_damaged,
            "net_pair_gain": pair_fixed - pair_damaged,
            "mcnemar_exact_p": mcnemar_exact(pair_fixed, pair_damaged),
        }
    for group in ["fixed", "damaged", "both_correct", "both_wrong"]:
        group_rows = [row for row in rows if row["group"] == group]
        stats["group_metrics"][group] = {
            "count": len(group_rows),
            "first_divergence_mean": mean([row["first_token_divergence"] for row in group_rows if row["first_token_divergence"] is not None]),
            "baseline_length_mean": mean([row["baseline_features"]["output_tokens"] for row in group_rows]),
            "method_length_mean": mean([row["method_features"]["output_tokens"] for row in group_rows]),
            "method_soft_ratio_mean": mean([row["method_trace_summary"]["soft_ratio"] for row in group_rows]),
            "method_format_triggers_mean": mean([row["method_trace_summary"]["format_triggers"] for row in group_rows]),
            "method_answer_reversal": sum(row["method_features"]["answer_reversal"] for row in group_rows),
            "semantic_candidates": dict(Counter(row["semantic_audit_candidate"] for row in group_rows)),
        }
    selected = select_rows(rows)
    for row in selected:
        card = out_dir / "sample_cards" / dataset / method / row["group"] / f"id_{row['id']}.md"
        write_card(card, row)
        row["sample_card"] = str(card)
    return rows, stats, selected


def markdown_summary(stats: list[dict]) -> str:
    lines = ["# Fixed/Damaged Cross-Dataset Mechanism Analysis", "", "主表仅包含配置校验通过的 greedy、seed42、max_new_tokens=1024 配对结果。", "",
             "| dataset | method | n | COT acc | method acc | fixed | damaged | net | McNemar p | extraction flips |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in stats:
        if not item.get("valid"):
            lines.append(f"| {item['dataset']} | {item['method']} | INVALID | - | - | - | - | - | - | - |")
            continue
        counts = item["counts"]
        lines.append(
            f"| {item['dataset']} | {item['method']} | {item['total']} | {100*item['baseline_accuracy']:.2f}% | "
            f"{100*item['method_accuracy']:.2f}% | {counts.get('fixed', 0)} | {counts.get('damaged', 0)} | "
            f"{item['net_gain_count']:+d} | {item['mcnemar_exact_p']:.4f} | {item['extraction_only_flips']} |"
        )
    lines += ["", "## 解释边界", "", "- fixed/damaged 是配对相关性；只有 event branch replay 可作单次 intervention 的因果证据。", "- semantic audit candidate 是启发式候选，必须结合 sample card 人工确认。", "- sampled 论文复现不进入本表。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out_dir = args.root.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root)
    all_rows, all_stats, selected_rows = [], [], []
    for entry in manifest:
        print(f"[ANALYZE] {entry['dataset']} / {entry['method']}", flush=True)
        rows, stats, selected = analyze_comparison(entry, out_dir)
        all_rows.extend(rows)
        all_stats.append(stats)
        selected_rows.extend(selected)
    write_json(out_dir / "comparison_manifest.json", manifest)
    write_jsonl(out_dir / "pairwise_groups.jsonl", all_rows)
    write_json(out_dir / "group_statistics.json", all_stats)
    (out_dir / "group_statistics.md").write_text(markdown_summary(all_stats), encoding="utf-8")
    selected_ids = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in selected_rows:
        selected_ids[row["dataset"]][row["method"]][row["group"]].append(row["id"])
    write_json(out_dir / "selected_trace_ids.json", selected_ids)
    write_jsonl(out_dir / "selected_rows.jsonl", selected_rows)
    (out_dir / "cross_dataset_mechanism_summary.md").write_text(markdown_summary(all_stats), encoding="utf-8")
    print(f"[DONE] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

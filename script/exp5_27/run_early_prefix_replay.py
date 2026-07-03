#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lead import load_dataset, format_prompt_from_sample, evaluate_dataset, save_json
from lead.inference import prepare_inputs
from lead.generation_utils import generate_cot


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_dir_for(base: Path, phase: str, dataset: str, run: str) -> Path:
    matches = sorted((base / phase / dataset).glob(f"{run}_gpu*"))
    for path in matches:
        if (path / "results.jsonl").exists():
            return path
    raise FileNotFoundError(f"Missing run dir for {phase}/{dataset}/{run}")


def rows_by_id(run_dir: Path) -> dict[int, dict]:
    return {int(r["id"]): r for r in load_jsonl(run_dir / "results.jsonl")}


def traces_by_id(run_dir: Path) -> dict[int, dict]:
    return {int(r["id"]): r for r in load_jsonl(run_dir / "token_entropy_full.jsonl")}


def extract_choice(text: str | None) -> str | None:
    if not text:
        return None
    tail = text[-1800:]
    patterns = [
        r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, tail)
        if m:
            return m.group(1).upper()
    region = tail.split("</think>")[-1]
    letters = re.findall(r"\b([A-Da-d])\b", region[-400:])
    return letters[-1].upper() if letters else None


def row_is_correct(row: dict | None) -> bool:
    if not row:
        return False
    pred = extract_choice(row.get("model_answer"))
    gold = str(row.get("answer") or "").strip().upper()[:1]
    return bool(pred and gold and pred == gold)


def evaluator_correctness(run_dir: Path, dataset: str) -> dict[int, bool]:
    if dataset == "mmvp":
        rows = load_jsonl(run_dir / "specialized_eval_rows.jsonl")
        if rows:
            return {int(r["id"]): bool(r.get("specialized_is_correct")) for r in rows}
    return {int(i): row_is_correct(r) for i, r in rows_by_id(run_dir).items()}


def token_ids(trace: dict | None, n: int) -> list[int]:
    out = []
    for t in (trace or {}).get("tokens") or []:
        if "token_id" in t:
            out.append(int(t["token_id"]))
        if len(out) >= n:
            break
    return out


def token_text(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def prepare_dataset(root: Path, dataset_key: str) -> tuple[str, list[dict]]:
    if dataset_key == "vstar":
        dataset_path = root / "data/vstar.jsonl"
    elif dataset_key == "mmvp":
        dataset_path = root / "data/mmvp.jsonl"
    else:
        raise ValueError(dataset_key)
    data = load_dataset(str(dataset_path), str(root / "data"))
    return str(dataset_path), data


def build_fixed_ids(base: Path, dataset: str) -> tuple[Path, Path, list[int]]:
    if dataset == "vstar":
        phase = "phase1_vstar_mechanism"
    elif dataset == "mmvp":
        phase = "phase3_cross_dataset_minimal"
    else:
        raise ValueError(dataset)
    cot_dir = run_dir_for(base, phase, dataset, "cot_orign_greedy")
    it_dir = run_dir_for(base, phase, dataset, "initial_transition_only")
    cot_correct = evaluator_correctness(cot_dir, dataset)
    it_correct = evaluator_correctness(it_dir, dataset)
    ids = sorted(i for i in set(cot_correct) & set(it_correct) if (not cot_correct[i]) and it_correct[i])
    return cot_dir, it_dir, ids


def forced_prefix_generate(
    model,
    processor,
    tokenizer,
    sample: dict,
    prefix_ids: list[int],
    args,
) -> str:
    prompt = format_prompt_from_sample(sample, use_cot=(args.cot_prompt_mode == "cot"))
    messages = [{"role": "user", "content": [{"type": "image", "image": sample["image"]}, {"type": "text", "text": prompt}]}]
    model_inputs = prepare_inputs(processor, messages, torch.device(args.device))
    prompt_len = int(model_inputs["input_ids"].shape[1])
    for k, v in list(model_inputs.items()):
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(args.device)
    max_new = max(1, int(args.max_new_tokens))
    with torch.no_grad():
        out = generate_cot(
            model,
            tokenizer,
            **model_inputs,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=max_new,
            do_sample=False,
            forced_prefix_ids=prefix_ids,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def evaluate_prefix_rows(root: Path, dataset_key: str, dataset_path: str, rows: list[dict], out_dir: Path) -> dict:
    results_path = out_dir / "results.jsonl"
    save_jsonl(rows, results_path)
    if dataset_key == "mmvp":
        report = out_dir / "specialized_eval_report.json"
        eval_rows = out_dir / "specialized_eval_rows.jsonl"
        subprocess.run(
            [
                sys.executable,
                "script/evaluate_specialized_results.py",
                "--dataset",
                dataset_path,
                "--results",
                str(results_path),
                "--output_json",
                str(report),
                "--output_results_jsonl",
                str(eval_rows),
            ],
            cwd=root,
            check=True,
        )
        return json.loads(report.read_text(encoding="utf-8"))
    report = evaluate_dataset(rows)
    save_json(report, str(out_dir / "eval_report.json"))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--datasets", nargs="+", default=["vstar", "mmvp"])
    ap.add_argument("--prefix_lengths", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cot_prompt_mode", default="orign")
    ap.add_argument("--limit_ids", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    root = Path.cwd()
    base = Path(args.base_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Loading model: {args.model_name}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    all_summary = []
    for dataset_key in args.datasets:
        dataset_path, dataset = prepare_dataset(root, dataset_key)
        samples = {int(s["id"]): s for s in dataset}
        cot_dir, it_dir, ids = build_fixed_ids(base, dataset_key)
        if args.limit_ids:
            ids = ids[: args.limit_ids]
        cot_rows = rows_by_id(cot_dir)
        it_rows = rows_by_id(it_dir)
        cot_traces = traces_by_id(cot_dir)
        it_traces = traces_by_id(it_dir)
        print(f"{dataset_key}: selected fixed ids={ids}", flush=True)

        for prefix_source, trace_map, source_rows in [
            ("cot_prefix", cot_traces, cot_rows),
            ("initial_transition_prefix", it_traces, it_rows),
        ]:
            for plen in args.prefix_lengths:
                rows = []
                run_dir = out_root / dataset_key / f"{prefix_source}_len{plen:02d}"
                if (run_dir / "results.jsonl").exists():
                    print(f"skip existing {run_dir}", flush=True)
                    rows = load_jsonl(run_dir / "results.jsonl")
                    report_path = run_dir / ("specialized_eval_report.json" if dataset_key == "mmvp" else "eval_report.json")
                    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    for idx, sid in enumerate(ids, 1):
                        sample = samples[sid]
                        prefix = token_ids(trace_map.get(sid), plen)
                        print(f"[{dataset_key} {prefix_source} len={plen}] {idx}/{len(ids)} id={sid}", flush=True)
                        t0 = time.time()
                        try:
                            text = forced_prefix_generate(model, processor, tokenizer, sample, prefix, args)
                            err = None
                        except Exception as exc:
                            text = ""
                            err = type(exc).__name__
                            print(f"ERROR id={sid}: {err}: {exc}", flush=True)
                            torch.cuda.empty_cache()
                        row = dict(sample)
                        row.update({
                            "model_answer": text,
                            "prefix_source": prefix_source,
                            "prefix_len": plen,
                            "prefix_token_ids": prefix,
                            "prefix_text": token_text(tokenizer, prefix),
                            "source_model_answer": (source_rows.get(sid) or {}).get("model_answer"),
                            "cot_model_answer": (cot_rows.get(sid) or {}).get("model_answer"),
                            "initial_transition_model_answer": (it_rows.get(sid) or {}).get("model_answer"),
                            "latency_sec": time.time() - t0,
                            "error_type": err,
                        })
                        row["output_tokens"] = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
                        rows.append(row)
                    report = evaluate_prefix_rows(root, dataset_key, dataset_path, rows, run_dir)
                acc = report.get("accuracy")
                pair_acc = report.get("pair_accuracy")
                all_summary.append({
                    "dataset": dataset_key,
                    "run": f"{prefix_source}_len{plen:02d}",
                    "selected_count": len(ids),
                    "accuracy": acc,
                    "pair_accuracy": pair_acc,
                    "correct": report.get("correct"),
                    "total": report.get("total"),
                    "pair_correct": report.get("pair_correct"),
                    "pair_total": report.get("pair_total"),
                })

    save_jsonl(all_summary, out_root / "summary.jsonl")
    lines = [
        "# Early Prefix Replay Summary",
        "",
        "Causal replay on samples where COT is wrong and initial_transition_only is correct. Prefix tokens are forced from the original trajectory; continuation is plain greedy normal generation.",
        "",
        "| dataset | run | n | acc | pair acc | correct/total | pair correct/total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_summary:
        acc = "NA" if r["accuracy"] is None else f"{r['accuracy']*100:.2f}%"
        pair = "NA" if r["pair_accuracy"] is None else f"{r['pair_accuracy']*100:.2f}%"
        ct = f"{r.get('correct')}/{r.get('total')}"
        pt = "NA" if r.get("pair_total") is None else f"{r.get('pair_correct')}/{r.get('pair_total')}"
        lines.append(f"| {r['dataset']} | {r['run']} | {r['selected_count']} | {acc} | {pair} | {ct} | {pt} |")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_root / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

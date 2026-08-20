#!/usr/bin/env python3
"""Compare frozen TALR with a one-shot visual residual augmentation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageStat
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lead import format_prompt_from_sample, get_math_symbols_ids
from lead.generation_utils import generate_lead
from lead.inference import prepare_inputs


BRANCHES = ("talr", "talr_true_residual", "talr_random_residual")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")


def _atlas_ids(atlas_manifest: Path) -> set[int]:
    return {
        int(str(row["original_id"]).split("::")[-1])
        for row in read_jsonl(atlas_manifest)
    }


def _resolve_image(row: dict[str, Any], dataset_name: str) -> str:
    root = Path("/home/bingxing2/home/scx9ftv/gushuo/datasets")
    if dataset_name == "mmvp":
        path = root / "sources/MMVP__MMVP/MMVP Images" / f"{int(row['id']) + 1}.jpg"
    elif dataset_name == "vstar":
        path = (
            root
            / "sources/craigwu__vstar_bench"
            / str(row["subtopic"])
            / Path(str(row["image"])).name
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def prepare_subset(
    data_path: Path,
    atlas_manifest: Path,
    output_path: Path,
    dataset_name: str,
    pairs: int,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if output_path.is_file():
        return read_jsonl(output_path)
    source = read_jsonl(data_path)
    atlas_ids = _atlas_ids(atlas_manifest)
    if dataset_name != "mmvp":
        eligible = [dict(row) for row in source if int(row["id"]) not in atlas_ids]
        rng = random.Random(seed)
        rng.shuffle(eligible)
        if samples > 0:
            eligible = eligible[:samples]
        rows = sorted(eligible, key=lambda row: int(row["id"]))
        for row in rows:
            row["image"] = _resolve_image(row, dataset_name)
            row["selection"] = "heldout_from_visual_action_atlas"
        write_jsonl(output_path, rows)
        return rows

    excluded_pairs = {sample_id // 2 for sample_id in atlas_ids}
    by_id = {int(row["id"]): dict(row) for row in source}
    eligible = [
        pair_id for pair_id in range((max(by_id) + 1) // 2)
        if pair_id not in excluded_pairs
        and 2 * pair_id in by_id
        and 2 * pair_id + 1 in by_id
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected_pairs = sorted(eligible if pairs <= 0 else eligible[:pairs])
    rows = []
    for pair_id in selected_pairs:
        for sample_id in (2 * pair_id, 2 * pair_id + 1):
            row = by_id[sample_id]
            row["image"] = _resolve_image(row, dataset_name)
            row["pair_id"] = pair_id
            row["selection"] = "heldout_from_visual_action_atlas"
            rows.append(row)
    if len(rows) != len(selected_pairs) * 2:
        raise RuntimeError("Pair-preserving held-out selection failed")
    write_jsonl(output_path, rows)
    return rows


def create_mean_mask(image_path: str, output_path: Path) -> str:
    if output_path.is_file():
        return str(output_path)
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        mean = tuple(int(round(value)) for value in ImageStat.Stat(rgb).mean[:3])
        masked = Image.new("RGB", rgb.size, mean)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked.save(output_path, format="PNG")
    return str(output_path)


def prepare(processor, sample: dict[str, Any], image_path: str, device: torch.device):
    prompt = format_prompt_from_sample(sample, use_cot=False)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = prepare_inputs(processor, messages, device)
    return inputs, int(inputs["input_ids"].shape[1])


def talr_kwargs(math_ids_tensor, max_new_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "alpha_0": 0.4,
        "beta_0": 0.7,
        "max_switch_count": 5,
        "window_size": 128,
        "math_ids_tensor": math_ids_tensor,
        "convergence_words": "</think>",
        "lead_initial_transition_with_refinement": True,
        "lead_refinement_window": 8,
        "lead_refinement_soft_cap": 2,
        "lead_refinement_entropy_threshold": 1.25,
        "lead_refinement_soft_mix_lambda": 0.95,
        "lead_guard_candidate_only": True,
        "lead_disable_answer_zone_lock": True,
    }


def run_talr(
    model,
    processor,
    tokenizer,
    sample: dict[str, Any],
    math_ids_tensor,
    max_new_tokens: int,
    forced_prefix_ids: list[int] | None = None,
    override_step: int = -1,
    external_residual: torch.Tensor | None = None,
    residual_strength: float = 0.75,
) -> tuple[str, list[int], list[dict[str, Any]], float]:
    inputs, prompt_len = prepare(
        processor, sample, str(sample["image"]), next(model.parameters()).device
    )
    trace: list[dict[str, Any]] = []
    kwargs = talr_kwargs(math_ids_tensor, max_new_tokens)
    kwargs["token_trace"] = trace
    kwargs["trace_topk"] = 0
    if forced_prefix_ids is not None:
        kwargs["forced_prefix_ids"] = forced_prefix_ids
    if external_residual is not None:
        kwargs["trace_route_override_step"] = override_step
        kwargs["trace_route_override_kind"] = "method_plus_external_residual"
        kwargs["trace_route_override_mix_lambda"] = residual_strength
        kwargs["trace_external_route_vector"] = external_residual
        kwargs["trace_external_route_source"] = "true_mask_residual"
    started = time.perf_counter()
    with torch.no_grad():
        output = generate_lead(model, tokenizer, **inputs, **kwargs)[0]
    elapsed = time.perf_counter() - started
    token_ids = [int(token) for token in output[prompt_len:].tolist()]
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return text, token_ids, trace, elapsed


def capture_talr_vector(
    model,
    processor,
    tokenizer,
    sample: dict[str, Any],
    image_path: str,
    prefix_ids: list[int],
    event_step: int,
    math_ids_tensor,
) -> dict[str, Any]:
    inputs, _ = prepare(processor, sample, image_path, next(model.parameters()).device)
    collector: list[dict[str, Any]] = []
    kwargs = talr_kwargs(math_ids_tensor, event_step + 1)
    kwargs["forced_prefix_ids"] = prefix_ids
    kwargs["trace_soft_vector_collector"] = collector
    kwargs["trace_capture_soft_vector_step"] = event_step
    with torch.no_grad():
        generate_lead(model, tokenizer, **inputs, **kwargs)
    if len(collector) != 1:
        raise RuntimeError(f"Expected one vector at step {event_step}, got {len(collector)}")
    return collector[0]


def matched_visual_direction(true_capture: dict[str, Any], mask_capture: dict[str, Any]):
    hard = true_capture["hard_embedding"].float()
    true_soft = true_capture["soft_embedding"].float()
    target_norm = float(torch.linalg.vector_norm(true_soft - hard).item())
    mask_delta = (
        mask_capture["soft_embedding"].float()
        - mask_capture["hard_embedding"].float()
    )
    mask_norm = float(torch.linalg.vector_norm(mask_delta).item())
    aligned_mask = hard.clone()
    if mask_norm > 1e-12:
        aligned_mask = hard + mask_delta * (target_norm / mask_norm)
    raw_direction = true_soft - aligned_mask
    raw_norm = float(torch.linalg.vector_norm(raw_direction).item())
    direction = torch.zeros_like(hard)
    if raw_norm > 1e-12:
        direction = raw_direction * (target_norm / raw_norm)
    return direction, target_norm, raw_norm


def random_direction_like(reference: torch.Tensor, norm: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(reference.numel(), generator=generator, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    return direction.reshape_as(reference) * norm


def patch_segmented_vision_sdpa() -> None:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    def segmented_vision_sdpa(module, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        seq_length = hidden_states.shape[0]
        q, k, v = (
            module.qkv(hidden_states)
            .reshape(seq_length, 3, module.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos().float(), emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = modeling_qwen2_5_vl.apply_rotary_pos_emb_vision(q, k, cos, sin)
        boundaries = cu_seqlens.detach().cpu().tolist()
        pieces = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            output = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
            )
            pieces.append(output.squeeze(0).transpose(0, 1))
        return module.proj(torch.cat(pieces, dim=0).reshape(seq_length, -1))

    modeling_qwen2_5_vl.Qwen2_5_VLVisionSdpaAttention.forward = segmented_vision_sdpa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--atlas-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("mmvp", "vstar"), required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--residual-strength", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.residual_strength <= 1.0:
        raise ValueError("Residual strength must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_dir / f"{args.dataset_name}_heldout_subset.jsonl"
    samples = prepare_subset(
        args.data,
        args.atlas_manifest,
        subset_path,
        args.dataset_name,
        args.pairs,
        args.samples,
        args.seed,
    )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    if args.dataset_name == "mmvp":
        pair_ids = sorted({int(row["pair_id"]) for row in samples})
        selected_pair_ids = {
            pair_id for index, pair_id in enumerate(pair_ids)
            if index % args.num_shards == args.shard_index
        }
        samples = [row for row in samples if int(row["pair_id"]) in selected_pair_ids]
    else:
        samples = [
            row for index, row in enumerate(samples)
            if index % args.num_shards == args.shard_index
        ]
    if args.limit is not None:
        samples = samples[:args.limit]

    results_path = args.output_dir / "results.jsonl"
    traces_path = args.output_dir / "trace_summary.jsonl"
    masks_dir = args.output_dir / "masks"
    completed = set()
    if results_path.is_file():
        completed = {
            (str(row["id"]), str(row["branch"]))
            for row in read_jsonl(results_path)
            if row.get("error_type") is None
        }

    patch_segmented_vision_sdpa()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    math_ids = get_math_symbols_ids(tokenizer)
    device = next(model.parameters()).device
    math_ids_tensor = (
        torch.tensor(list(math_ids), device=device) if math_ids else None
    )

    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["id"])
        if all((sample_id, branch) in completed for branch in BRANCHES):
            continue
        print(f"[{index}/{len(samples)}] id={sample_id}", flush=True)
        baseline_text, baseline_tokens, baseline_trace, baseline_time = run_talr(
            model, processor, tokenizer, sample, math_ids_tensor,
            args.max_new_tokens,
        )
        refinement_events = [
            token for token in baseline_trace
            if token.get("lead_refinement_active", False)
        ]
        event_step = int(refinement_events[0]["step"]) if refinement_events else -1
        common = {
            "id": sample["id"],
            "pair_id": sample.get("pair_id"),
            "image": sample["image"],
            "question": sample["question"],
            "options": sample.get("options"),
            "answer": sample.get("answer"),
            "refinement_step": event_step,
            "residual_strength": args.residual_strength,
            "error_type": None,
        }
        append_jsonl(results_path, {
            **common,
            "branch": "talr",
            "model_answer": baseline_text,
            "generated_token_ids": baseline_tokens,
            "latency_seconds": baseline_time,
            "injection_applied": False,
        })
        append_jsonl(traces_path, {
            "id": sample["id"],
            "branch": "talr",
            "refinement_steps": [int(token["step"]) for token in refinement_events],
        })

        if event_step < 0:
            for branch in BRANCHES[1:]:
                append_jsonl(results_path, {
                    **common,
                    "branch": branch,
                    "model_answer": baseline_text,
                    "generated_token_ids": baseline_tokens,
                    "latency_seconds": 0.0,
                    "injection_applied": False,
                })
            continue

        prefix_ids = baseline_tokens[:event_step + 1]
        if args.sanity:
            replay_text, replay_tokens, _, _ = run_talr(
                model, processor, tokenizer, sample, math_ids_tensor,
                args.max_new_tokens,
                forced_prefix_ids=prefix_ids,
            )
            if replay_tokens != baseline_tokens or replay_text != baseline_text:
                raise RuntimeError(
                    f"Forced-prefix TALR replay mismatch for sample {sample_id}"
                )
        mask_path = create_mean_mask(
            str(sample["image"]), masks_dir / f"{sample_id}.png"
        )
        true_capture = capture_talr_vector(
            model, processor, tokenizer, sample, str(sample["image"]),
            prefix_ids, event_step, math_ids_tensor,
        )
        mask_capture = capture_talr_vector(
            model, processor, tokenizer, sample, mask_path,
            prefix_ids, event_step, math_ids_tensor,
        )
        visual_direction, target_norm, raw_residual_norm = matched_visual_direction(
            true_capture, mask_capture
        )
        random_direction = random_direction_like(
            visual_direction, target_norm, stable_seed(sample_id, args.seed)
        )
        for branch, direction in (
            ("talr_true_residual", visual_direction),
            ("talr_random_residual", random_direction),
        ):
            text, token_ids, trace, elapsed = run_talr(
                model, processor, tokenizer, sample, math_ids_tensor,
                args.max_new_tokens,
                forced_prefix_ids=prefix_ids,
                override_step=event_step,
                external_residual=direction,
                residual_strength=args.residual_strength,
            )
            append_jsonl(results_path, {
                **common,
                "branch": branch,
                "model_answer": text,
                "generated_token_ids": token_ids,
                "latency_seconds": elapsed,
                "injection_applied": True,
                "target_soft_hard_norm": target_norm,
                "raw_visual_residual_norm": raw_residual_norm,
            })
            append_jsonl(traces_path, {
                "id": sample["id"],
                "branch": branch,
                "refinement_steps": [
                    int(token["step"]) for token in trace
                    if token.get("lead_refinement_active", False)
                ],
                "override_step": event_step,
                "target_soft_hard_norm": target_norm,
                "raw_visual_residual_norm": raw_residual_norm,
            })
        gc.collect()
        torch.cuda.empty_cache()

    rows = read_jsonl(results_path)
    for branch in BRANCHES:
        branch_rows = [row for row in rows if row.get("branch") == branch]
        write_jsonl(args.output_dir / branch / "results.jsonl", branch_rows)
    config = {
        "model": str(args.model),
        "dataset": f"{args.dataset_name} Atlas-held-out subset",
        "dataset_name": args.dataset_name,
        "pairs": args.pairs,
        "samples": len(samples),
        "excluded_atlas_samples": True,
        "talr": "W8K2-T1.25-lambda0.95-NoGuard",
        "visual_action": "first active TALR refinement + true-minus-mean-mask residual",
        "residual_strength": args.residual_strength,
        "random_control": "matched-norm random direction at the same event",
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

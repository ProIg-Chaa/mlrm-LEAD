#!/usr/bin/env python3
"""Run matched image-source transplants for one-step soft interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageStat
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lead import format_prompt_from_sample
from lead.generation_utils import generate_cot
from lead.inference import prepare_inputs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def create_mean_mask(image_path: str, output_path: Path) -> str:
    if output_path.exists():
        return str(output_path)
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        mean = tuple(int(round(value)) for value in ImageStat.Stat(rgb).mean[:3])
        masked = Image.new("RGB", rgb.size, mean)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked.save(output_path, format="PNG")
    return str(output_path)


def create_gaussian_noise(
    image_path: str,
    output_path: Path,
    seed: int,
    sigma: float = 30.0,
) -> str:
    """Add a fixed pixel-noise dose shared by every dataset."""
    if output_path.exists():
        return str(output_path)
    with Image.open(image_path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(seed)
    noisy = np.clip(array + rng.normal(0.0, sigma, array.shape), 0, 255).astype(
        np.uint8
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy, mode="RGB").save(output_path, format="PNG")
    return str(output_path)


def shuffled_grid(array: np.ndarray, grid_size: int, seed: int) -> np.ndarray:
    """Shuffle equally indexed image cells while preserving their contents."""
    height, width = array.shape[:2]
    ys = np.linspace(0, height, grid_size + 1, dtype=int)
    xs = np.linspace(0, width, grid_size + 1, dtype=int)
    cells = [
        array[ys[row] : ys[row + 1], xs[col] : xs[col + 1]].copy()
        for row in range(grid_size)
        for col in range(grid_size)
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    if np.array_equal(order, np.arange(len(cells))):
        order = np.roll(order, 1)
    output = np.empty_like(array)
    for destination, source_index in enumerate(order):
        row, col = divmod(destination, grid_size)
        target_h = ys[row + 1] - ys[row]
        target_w = xs[col + 1] - xs[col]
        cell = Image.fromarray(cells[int(source_index)], mode="RGB").resize(
            (target_w, target_h), Image.Resampling.BILINEAR
        )
        output[ys[row] : ys[row + 1], xs[col] : xs[col + 1]] = np.asarray(
            cell
        )
    return output


def phase_scramble(array: np.ndarray, seed: int) -> np.ndarray:
    """Destroy spatial phase while approximately preserving color statistics."""
    rng = np.random.default_rng(seed)
    result = np.empty_like(array, dtype=np.float32)
    for channel in range(3):
        values = array[:, :, channel].astype(np.float32)
        spectrum = np.fft.rfft2(values)
        amplitude = np.abs(spectrum)
        phase = rng.uniform(-np.pi, np.pi, spectrum.shape)
        phase[0, 0] = np.angle(spectrum[0, 0])
        scrambled = np.fft.irfft2(
            amplitude * np.exp(1j * phase), s=values.shape
        ).real
        scrambled = (scrambled - scrambled.mean()) / max(scrambled.std(), 1e-6)
        scrambled = scrambled * max(values.std(), 1e-6) + values.mean()
        result[:, :, channel] = scrambled
    return np.clip(result, 0, 255).astype(np.uint8)


def mmvp_pair_image(image_path: str) -> str:
    path = Path(image_path)
    try:
        number = int(path.stem)
    except ValueError as exc:
        raise ValueError(f"MMVP image has no numeric stem: {image_path}") from exc
    paired_number = number + 1 if number % 2 == 1 else number - 1
    paired = path.with_name(f"{paired_number}{path.suffix}")
    if not paired.is_file():
        raise FileNotFoundError(f"MMVP paired image missing: {paired}")
    return str(paired)


def create_dataset_noise(
    dataset: str,
    image_path: str,
    output_path: Path,
    seed: int,
) -> tuple[str, str]:
    """Create a benchmark-specific visual corruption with an explicit rationale."""
    if dataset == "mmvp":
        return mmvp_pair_image(image_path), "paired_contrast_image"
    if output_path.exists():
        policy = {
            "vstar": "spatial_patch_shuffle_4x4",
            "realworldqa": "frequency_phase_scramble",
            "visulogic": "logic_cell_shuffle_3x3",
        }[dataset]
        return str(output_path), policy
    with Image.open(image_path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.uint8)
    if dataset == "vstar":
        transformed = shuffled_grid(array, grid_size=4, seed=seed)
        policy = "spatial_patch_shuffle_4x4"
    elif dataset == "realworldqa":
        transformed = phase_scramble(array, seed=seed)
        policy = "frequency_phase_scramble"
    elif dataset == "visulogic":
        transformed = shuffled_grid(array, grid_size=3, seed=seed)
        policy = "logic_cell_shuffle_3x3"
    else:
        raise ValueError(f"No dataset-specific noise policy for {dataset}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(transformed, mode="RGB").save(output_path, format="PNG")
    return str(output_path), policy


def aligned_to_receiver_hard(
    source_soft: torch.Tensor,
    source_hard: torch.Tensor,
    receiver_hard: torch.Tensor,
    target_delta_norm: float,
) -> torch.Tensor:
    """Move a donor displacement to the receiver hard-token origin."""
    delta = source_soft.float() - source_hard.float()
    norm = float(torch.linalg.vector_norm(delta).item())
    if norm <= 1e-12:
        return receiver_hard.float().clone()
    return receiver_hard.float() + delta * (target_delta_norm / norm)


def random_surrogate(
    receiver_hard: torch.Tensor,
    target_delta_norm: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(
        receiver_hard.numel(), generator=generator, dtype=torch.float32
    )
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    return receiver_hard.float() + target_delta_norm * direction


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().reshape(1, -1), right.float().reshape(1, -1)
        )[0].item()
    )


def prepare(
    processor,
    image_path: str,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    prompt = format_prompt_from_sample(sample, use_cot=False)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = prepare_inputs(processor, messages, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    return inputs, prompt_len


def capture_vector(
    model,
    processor,
    tokenizer,
    image_path: str,
    sample: dict[str, Any],
    prefix_ids: list[int],
    event_step: int,
    args,
) -> dict[str, Any]:
    collector: list[dict[str, Any]] = []
    inputs, _ = prepare(processor, image_path, sample, torch.device(args.device))
    with torch.no_grad():
        generate_cot(
            model,
            tokenizer,
            **inputs,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=event_step + 1,
            do_sample=False,
            forced_prefix_ids=prefix_ids,
            trace_soft_vector_collector=collector,
            trace_capture_soft_vector_step=event_step,
        )
    if len(collector) != 1:
        raise RuntimeError(
            f"Expected one captured vector at step {event_step}, got {len(collector)}"
        )
    return collector[0]


def generate_branch(
    model,
    processor,
    tokenizer,
    sample: dict[str, Any],
    prefix_ids: list[int],
    event_step: int,
    branch: str,
    external_vector: torch.Tensor | None,
    args,
) -> tuple[str, list[int], list[dict[str, Any]], float]:
    inputs, prompt_len = prepare(
        processor, str(sample["true_image"]), sample, torch.device(args.device)
    )
    trace: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {
        **inputs,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "forced_prefix_ids": prefix_ids,
        "token_trace": trace,
        "trace_topk": args.trace_topk,
        "trace_route_override_step": event_step,
        "trace_route_override_kind": (
            "hard" if branch == "hard" else "external_contracted"
        ),
        "trace_route_override_mix_lambda": args.mix_lambda,
        "trace_external_route_source": branch,
    }
    if external_vector is not None:
        kwargs["trace_external_route_vector"] = external_vector
    started = time.perf_counter()
    with torch.no_grad():
        output = generate_cot(model, tokenizer, **kwargs)[0]
    elapsed = time.perf_counter() - started
    token_ids = [int(token) for token in output[prompt_len:].tolist()]
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return text, token_ids, trace, elapsed


def self_test() -> None:
    hard = torch.tensor([1.0, 2.0, 3.0])
    source_hard = torch.tensor([2.0, 1.0, 1.0])
    source_soft = source_hard + torch.tensor([3.0, 4.0, 0.0])
    aligned = aligned_to_receiver_hard(
        source_soft, source_hard, hard, target_delta_norm=2.5
    )
    assert abs(float(torch.linalg.vector_norm(aligned - hard)) - 2.5) < 1e-6
    random_a = random_surrogate(hard, 2.5, 42)
    random_b = random_surrogate(hard, 2.5, 42)
    assert torch.equal(random_a, random_b)
    assert abs(float(torch.linalg.vector_norm(random_a - hard)) - 2.5) < 1e-6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--mix-lambda", type=float, default=0.95)
    parser.add_argument("--generic-noise-sigma", type=float, default=30.0)
    parser.add_argument("--trace-topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--branches",
        default=(
            "hard,true_image,swapped_image,masked_image,random,"
            "generic_noise,dataset_noise"
        ),
    )
    parser.add_argument("--require-reproduction", action="store_true")
    args = parser.parse_args()

    self_test()
    if not 0.0 <= args.mix_lambda <= 1.0:
        raise ValueError("--mix-lambda must be in [0, 1]")
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    branches = [item.strip() for item in args.branches.split(",") if item.strip()]
    allowed = {
        "hard",
        "true_image",
        "swapped_image",
        "masked_image",
        "random",
        "generic_noise",
        "dataset_noise",
    }
    if not branches or not set(branches) <= allowed:
        raise ValueError(f"Unsupported branches: {set(branches) - allowed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    noise_dir = args.output_dir / "noise_images"
    results_path = args.output_dir / "results.jsonl"
    trace_path = args.output_dir / "token_entropy_full.jsonl"
    vectors_path = args.output_dir / "vector_metadata.jsonl"
    for path in (results_path, trace_path, vectors_path):
        if path.exists():
            path.unlink()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    reproduction_failures: list[dict[str, Any]] = []
    branch_count = len(rows) * len(branches)
    completed = 0
    for event_index, event in enumerate(rows, start=1):
        event_id = str(event["event_id"])
        step = int(event["event_step"])
        prefix_ids = [int(value) for value in event["prefix_ids"]]
        if len(prefix_ids) != step + 1:
            raise ValueError(f"{event_id}: prefix length does not match event step")

        sample = {
            "question": event["question"],
            "options": event.get("options"),
            "answer": event.get("answer"),
            "subtopic": event.get("subtopic"),
            "true_image": event["true_image"],
        }
        mask_path = create_mean_mask(
            str(event["true_image"]),
            masks_dir / f"{hashlib.sha1(event_id.encode()).hexdigest()}.png",
        )
        event_seed = stable_seed(event_id, args.seed)
        generic_noise_path = create_gaussian_noise(
            str(event["true_image"]),
            noise_dir / "generic" / f"{hashlib.sha1(event_id.encode()).hexdigest()}.png",
            seed=event_seed,
            sigma=args.generic_noise_sigma,
        )
        dataset_noise_path, dataset_noise_policy = create_dataset_noise(
            str(event["dataset"]),
            str(event["true_image"]),
            noise_dir / "dataset_specific" / f"{hashlib.sha1(event_id.encode()).hexdigest()}.png",
            seed=event_seed,
        )
        capture_images = {"true": str(event["true_image"])}
        if "swapped_image" in branches:
            capture_images["swap"] = str(event["donor_image"])
        if "masked_image" in branches:
            capture_images["mask"] = mask_path
        if "generic_noise" in branches:
            capture_images["generic_noise"] = generic_noise_path
        if "dataset_noise" in branches:
            capture_images["dataset_noise"] = dataset_noise_path
        captures = {
            name: capture_vector(
                model,
                processor,
                tokenizer,
                image_path,
                sample,
                prefix_ids,
                step,
                args,
            )
            for name, image_path in capture_images.items()
        }
        receiver_hard = captures["true"]["hard_embedding"]
        true_soft = captures["true"]["soft_embedding"]
        target_delta_norm = float(
            torch.linalg.vector_norm(true_soft.float() - receiver_hard.float()).item()
        )
        vectors = {"true_image": true_soft}
        if "swap" in captures:
            vectors["swapped_image"] = aligned_to_receiver_hard(
                captures["swap"]["soft_embedding"],
                captures["swap"]["hard_embedding"],
                receiver_hard,
                target_delta_norm,
            )
        if "mask" in captures:
            vectors["masked_image"] = aligned_to_receiver_hard(
                captures["mask"]["soft_embedding"],
                captures["mask"]["hard_embedding"],
                receiver_hard,
                target_delta_norm,
            )
        if "random" in branches:
            vectors["random"] = random_surrogate(
                receiver_hard,
                target_delta_norm,
                event_seed,
            )
        for name in ("generic_noise", "dataset_noise"):
            if name in captures:
                vectors[name] = aligned_to_receiver_hard(
                    captures[name]["soft_embedding"],
                    captures[name]["hard_embedding"],
                    receiver_hard,
                    target_delta_norm,
                )
        vector_geometry = {
            f"{name}_aligned_hard_cosine": cosine(vector, receiver_hard)
            for name, vector in vectors.items()
            if name != "true_image"
        }
        append_jsonl(
            vectors_path,
            {
                "event_id": event_id,
                "event_step": step,
                "receiver_original_id": event["original_id"],
                "donor_original_id": event["donor_original_id"],
                "target_delta_norm": target_delta_norm,
                "true_soft_hard_cosine": cosine(true_soft, receiver_hard),
                **vector_geometry,
                "generic_noise_policy": (
                    f"gaussian_pixel_noise_sigma{args.generic_noise_sigma:g}"
                ),
                "dataset_noise_policy": dataset_noise_policy,
                "dataset_noise_image": dataset_noise_path,
                "true_entropy": captures["true"]["raw_entropy"],
                "true_selected_probability": captures["true"]["selected_token_prob"],
                "capture_statistics": {
                    name: {
                        "entropy": capture["raw_entropy"],
                        "selected_probability": capture["selected_token_prob"],
                    }
                    for name, capture in captures.items()
                },
            },
        )

        for branch in branches:
            vector = None if branch == "hard" else vectors[branch]
            answer, tokens, trace, elapsed = generate_branch(
                model,
                processor,
                tokenizer,
                sample,
                prefix_ids,
                step,
                branch,
                vector,
                args,
            )
            digest = text_hash(answer)
            expected = (
                event.get("expected_hard_text_sha256")
                if branch == "hard"
                else event.get("expected_true_l095_text_sha256")
                if branch == "true_image" and abs(args.mix_lambda - 0.95) < 1e-12
                else None
            )
            reproduced = expected is None or digest == expected
            if expected is not None and not reproduced:
                reproduction_failures.append(
                    {
                        "event_id": event_id,
                        "branch": branch,
                        "expected": expected,
                        "actual": digest,
                    }
                )
            result = {
                **event,
                "branch": branch,
                "model_answer": answer,
                "generated_token_ids": tokens,
                "model_answer_sha256": digest,
                "expected_text_sha256": expected,
                "reproduction_pass": reproduced,
                "latency_seconds": elapsed,
                "error_type": None,
            }
            append_jsonl(results_path, result)
            append_jsonl(
                trace_path,
                {
                    "id": f"{event_id}::{branch}",
                    "event_id": event_id,
                    "branch": branch,
                    "tokens": trace,
                },
            )
            completed += 1
            print(
                f"[{completed}/{branch_count}] event={event_index}/{len(rows)} "
                f"branch={branch} reproduce={reproduced} time={elapsed:.1f}s",
                flush=True,
            )

    summary = {
        "events": len(rows),
        "branches": branches,
        "completed_branches": completed,
        "reproduction_failures": reproduction_failures,
        "mix_lambda": args.mix_lambda,
        "generic_noise_sigma": args.generic_noise_sigma,
        "seed": args.seed,
        "receiver_image_policy": "true image for all decoding branches",
        "donor_usage": "one captured soft-state vector at the event step only",
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.require_reproduction and reproduction_failures:
        print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    (args.output_dir / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

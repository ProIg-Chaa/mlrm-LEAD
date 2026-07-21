#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PATHS = {
    "vstar": Path("/root/gushuo/outputs/experiments/20260716_token_anchored_transition/vstar/original_eot_bridge_step1/token_entropy_full.jsonl"),
    "mmvp": Path("/root/gushuo/outputs/experiments/20260716_token_anchored_transition/mmvp/original_eot_bridge_step1/token_entropy_full.jsonl"),
}


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def main() -> None:
    report = {}
    for name, path in PATHS.items():
        entropy, top1, margin = [], [], []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                step0 = next((token for token in row.get("tokens", []) if token.get("step") == 0), None)
                if step0 is None:
                    continue
                entropy.append(float(step0.get("raw_entropy", step0.get("filtered_entropy", 0.0))))
                top1.append(float(step0.get("raw_top1_prob", 1.0)))
                margin.append(float(step0.get("raw_margin", 1.0)))
        report[name] = {
            "n": len(entropy),
            "entropy_mean": sum(entropy) / len(entropy),
            "entropy_p50": quantile(entropy, 0.5),
            "entropy_p90": quantile(entropy, 0.9),
            "top1_mean": sum(top1) / len(top1),
            "top1_p10": quantile(top1, 0.1),
            "margin_mean": sum(margin) / len(margin),
            "fraction_top1_lt_0_99": sum(x < 0.99 for x in top1) / len(top1),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

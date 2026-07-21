#!/usr/bin/env python3
"""Download and validate OpenVLThinker-7B on NewGpu3."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "ydeng9/OpenVLThinker-7B"
TARGET = Path("/root/autodl-tmp/gushuo/models/OpenVLThinker-7B")
STATUS = TARGET.parent / "OpenVLThinker-7B.download_status.json"
REQUIRED = [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "tokenizer.json",
    "preprocessor_config.json",
]


def write_status(state: str, **extra: object) -> None:
    payload = {
        "repo_id": REPO_ID,
        "target": str(TARGET),
        "state": state,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        **extra,
    }
    STATUS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate() -> dict[str, object]:
    missing = [name for name in REQUIRED if not (TARGET / name).is_file()]
    shards = sorted(TARGET.glob("model-*.safetensors"))
    shard_bytes = sum(path.stat().st_size for path in shards)
    return {
        "missing": missing,
        "shard_count": len(shards),
        "shard_bytes": shard_bytes,
        "valid": not missing and len(shards) == 4 and shard_bytes > 10_000_000_000,
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, 6):
        write_status("downloading", attempt=attempt)
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                local_dir=TARGET,
                max_workers=2,
            )
            validation = validate()
            if not validation["valid"]:
                raise RuntimeError(f"integrity validation failed: {validation}")
            write_status("complete", attempt=attempt, validation=validation)
            print(json.dumps(validation, indent=2), flush=True)
            return 0
        except Exception as exc:
            write_status("retrying", attempt=attempt, error=repr(exc))
            print(f"attempt {attempt} failed: {exc!r}", flush=True)
            if attempt < 5:
                time.sleep(30 * attempt)

    validation = validate()
    write_status("failed", validation=validation)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

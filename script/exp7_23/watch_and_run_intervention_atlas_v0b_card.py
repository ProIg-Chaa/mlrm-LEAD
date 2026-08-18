#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def tmux_session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def wait_for_sessions(names: list[str], poll_seconds: int) -> None:
    while True:
        active = [name for name in names if name and tmux_session_exists(name)]
        if not active:
            return
        print(f"WAIT active sessions: {', '.join(active)}", flush=True)
        time.sleep(poll_seconds)


def ensure_model(source: Path, target: Path) -> None:
    lock = target.with_name(target.name + ".copy.lock")
    if (target / "config.json").exists() and not lock.exists():
        print(f"MODEL ready: {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
        owns_lock = True
    except FileExistsError:
        owns_lock = False
    if owns_lock:
        try:
            command = [
                "rsync", "-a", "--info=progress2", f"{source}/", f"{target}/"
            ]
            print("COPY MODEL", " ".join(command), flush=True)
            subprocess.run(command, check=True)
        finally:
            lock.rmdir()
        return
    while lock.exists() or not (target / "config.json").exists():
        print(f"WAIT model copy: {target}", flush=True)
        time.sleep(30)

def compile_sources(python: str, repo: Path) -> None:
    paths = [
        repo / "main.py",
        repo / "lead/inference.py",
        repo / "lead/generation_utils.py",
        repo / "script/exp7_23/prepare_intervention_atlas_v0b.py",
        repo / "script/exp7_23/run_intervention_atlas_v0b_shard.py",
        repo / "script/exp7_23/summarize_intervention_atlas_v0b.py",
        repo / "script/exp7_23/merge_intervention_atlas_v0b.py",
    ]
    subprocess.run([python, "-m", "py_compile", *map(str, paths)], check=True)


def write_smoke_shard(source: Path, target: Path, count: int = 2) -> None:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= count:
                break
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shard_command(
    args: argparse.Namespace,
    shard_index: int,
    shard_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        args.python,
        str(args.repo / "script/exp7_23/run_intervention_atlas_v0b_shard.py"),
        "--repo",
        str(args.repo),
        "--python",
        args.python,
        "--model",
        str(args.model_target),
        "--shard",
        str(shard_path),
        "--output-dir",
        str(output_dir),
        "--shard-index",
        str(shard_index),
    ]


def run_logged(
    command: list[str], log_path: Path, gpu: str
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return subprocess.Popen(
        command,
        cwd=command[command.index("--repo") + 1],
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def shard_complete(path: Path) -> bool:
    return (path / "SHARD_COMPLETE").exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--model-target", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--shards", type=int, nargs=2, required=True)
    parser.add_argument("--wait-sessions", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.selection_root = args.selection_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    wait_names = [name for name in args.wait_sessions.split(",") if name]
    wait_for_sessions(wait_names, args.poll_seconds)
    ensure_model(args.model_source, args.model_target)
    compile_sources(args.python, args.repo)

    first_shard = args.selection_root / "shards" / f"shard_{args.shards[0]}.jsonl"
    smoke_shard = args.output_root / f"smoke_gpu{args.gpu}.jsonl"
    smoke_output = args.output_root / f"smoke_gpu{args.gpu}"
    if not shard_complete(smoke_output):
        write_smoke_shard(first_shard, smoke_shard)
        smoke = run_logged(
            shard_command(args, -1, smoke_shard, smoke_output),
            smoke_output / "supervisor.log",
            args.gpu,
        )
        if smoke.wait() != 0 or not shard_complete(smoke_output):
            raise RuntimeError(f"Atlas smoke failed on GPU {args.gpu}")
    print(f"SMOKE passed on GPU {args.gpu}", flush=True)

    running: list[tuple[int, subprocess.Popen, Path]] = []
    for shard_index in args.shards:
        shard_path = (
            args.selection_root / "shards" / f"shard_{shard_index}.jsonl"
        )
        shard_output = args.output_root / f"shard_{shard_index}"
        if shard_complete(shard_output):
            continue
        process = run_logged(
            shard_command(args, shard_index, shard_path, shard_output),
            shard_output / "supervisor.log",
            args.gpu,
        )
        running.append((shard_index, process, shard_output))

    failed = []
    for shard_index, process, shard_output in running:
        returncode = process.wait()
        if returncode != 0 or not shard_complete(shard_output):
            failed.append((shard_index, shard_output))

    # Two processes normally fit on an A800. If either fails, retry only the
    # incomplete shard serially after the other process has released memory.
    for shard_index, shard_output in failed:
        if shard_complete(shard_output):
            continue
        shard_path = (
            args.selection_root / "shards" / f"shard_{shard_index}.jsonl"
        )
        process = run_logged(
            shard_command(args, shard_index, shard_path, shard_output),
            shard_output / "serial_retry.log",
            args.gpu,
        )
        if process.wait() != 0 or not shard_complete(shard_output):
            raise RuntimeError(f"Shard {shard_index} failed after serial retry")

    (args.output_root / f"CARD_GPU{args.gpu}_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




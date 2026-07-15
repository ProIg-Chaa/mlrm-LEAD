from io import BytesIO
from pathlib import Path
import base64
import csv
import json
import os
import re
import sys
import zipfile

import pyarrow.parquet as pq
from PIL import Image


ASSET_ROOT = Path("/root/autodl-tmp/gushuo/datasets")
SOURCE_ROOT = ASSET_ROOT / "sources"
IMAGE_ROOT = ASSET_ROOT / "mlrm-LEAD" / "images"
PROJECT_DATA = Path("/root/gushuo/proj/mlrm-LEAD/data")


def read_jsonl(name: str) -> list[dict]:
    with (PROJECT_DATA / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def link_expected_images(name: str, source_root: Path, output: Path) -> None:
    rows = read_jsonl(name)
    source_by_name = {
        path.name: path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = Path(row["image"]).name
        source = source_by_name.get(filename)
        if source is None:
            raise FileNotFoundError(f"No source for {name}: {filename}")
        target = output / filename
        if not target.exists():
            target.symlink_to(source)
    print(f"{name}: linked {len(rows)} images")


def extract_vstar_and_mmvp() -> None:
    link_expected_images(
        "vstar.jsonl",
        SOURCE_ROOT / "craigwu__vstar_bench",
        IMAGE_ROOT / "vstar",
    )
    link_expected_images(
        "mmvp.jsonl",
        SOURCE_ROOT / "MMVP__MMVP" / "MMVP Images",
        IMAGE_ROOT / "mmvp",
    )


def extract_visulogic() -> None:
    rows = read_jsonl("visulogic.jsonl")[:300]
    output = IMAGE_ROOT / "visulogic"
    output.mkdir(parents=True, exist_ok=True)
    archive_path = SOURCE_ROOT / "VisuLogic__VisuLogic" / "images.zip"
    with zipfile.ZipFile(archive_path) as archive:
        names = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        for row in rows:
            filename = Path(row["image"]).name
            target = output / filename
            if not target.exists():
                target.write_bytes(archive.read(names[filename]))
    print(f"VisuLogic300: extracted {len(rows)} images")


INSTRUCTION_RE = re.compile(
    r"\n?Please answer directly with (?:only the letter of the correct option and nothing else|a single word or number)\.?",
    re.I,
)


def clean_realworld_question(text: str) -> str:
    text = INSTRUCTION_RE.sub("", text or "").strip()
    match = re.search(r"(?m)(?:^|\n)\s*A\.\s", text)
    return text[: match.start()].strip() if match else text


def extract_realworldqa() -> None:
    source_rows = []
    for parquet in sorted((SOURCE_ROOT / "xai-org__RealworldQA" / "data").glob("*.parquet")):
        source_rows.extend(pq.read_table(parquet).to_pylist())
    by_question = {clean_realworld_question(row["question"]): row for row in source_rows}
    rows = read_jsonl("realworldqa_fixed_mcq_random200_seed42.jsonl")
    output = IMAGE_ROOT / "realworldqa"
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = by_question[row["question"]]
        target = output / Path(row["image"]).name
        if not target.exists():
            target.write_bytes(source["image"]["bytes"])
    print(f"RealWorldQA fixed200: extracted {len(rows)} images")


def extract_vmcbench() -> None:
    csv.field_size_limit(sys.maxsize)
    rows = read_jsonl("vmcbench_dev.jsonl")
    expected = {str(row["id"]).rsplit("_", 1)[-1]: row for row in rows}
    tsv = SOURCE_ROOT / "suyc21__VMCBench" / "data" / "tsv" / "VMCBench_DEV.tsv"
    output = IMAGE_ROOT / "vmcbench_dev"
    written = 0
    with tsv.open(encoding="utf-8") as stream:
        for source in csv.DictReader(stream, delimiter="\t"):
            row = expected.get(source["index"])
            if row is None:
                continue
            target = output / source["category"] / Path(row["image"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(base64.b64decode(source["image"]))
            written += 1
    if written != len(rows):
        raise RuntimeError(f"VMCBench extracted {written}/{len(rows)}")
    print(f"VMCBench-dev: extracted {written} images")


def validate_remaining_matrix() -> None:
    specs = [
        ("vstar.jsonl", None),
        ("realworldqa_fixed_mcq_random200_seed42.jsonl", None),
        ("mmvp.jsonl", None),
        ("visulogic.jsonl", 300),
        ("vmcbench_dev.jsonl", None),
    ]
    for name, limit in specs:
        rows = read_jsonl(name)
        if limit is not None:
            rows = rows[:limit]
        missing = [row["image"] for row in rows if not Path(row["image"]).is_file()]
        print(f"validate {name}: rows={len(rows)} missing={len(missing)}")
        if missing:
            raise FileNotFoundError(missing[0])


def extract_pope_adversarial() -> None:
    source = SOURCE_ROOT / "lmms-lab__POPE" / "Full" / "adversarial-00000-of-00001.parquet"
    output = IMAGE_ROOT / "pope" / "pope_adversarial"
    output.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(source).to_pylist()
    for index, row in enumerate(rows):
        target = output / f"{index:06d}.jpg"
        if not target.exists():
            Image.open(BytesIO(row["image"]["bytes"])).convert("RGB").save(target)
    print(f"POPE adversarial: {len(rows)} images")


def extract_mmk12_physics() -> None:
    source = SOURCE_ROOT / "FanqingM__MMK12" / "data" / "test-00000-of-00001.parquet"
    output = IMAGE_ROOT / "mmk12" / "physics"
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in pq.read_table(source).to_pylist() if row["subject"].lower() == "physics"]
    for row in rows:
        target = output / f'{row["id"]}.png'
        if not target.exists():
            Image.open(BytesIO(row["image"]["bytes"])).convert("RGB").save(target)
    print(f"MMK12 Physics: {len(rows)} images")


if __name__ == "__main__":
    extract_vstar_and_mmvp()
    extract_visulogic()
    extract_realworldqa()
    extract_vmcbench()
    validate_remaining_matrix()

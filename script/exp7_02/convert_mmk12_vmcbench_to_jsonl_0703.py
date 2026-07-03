import json
import os
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/share/home/wangzixu/liudinghao/gushuo")
DATA_DIR = ROOT / "proj/mlrm-LEAD/data"
IMAGE_ROOT = ROOT / "datasets/mlrm-LEAD/images"


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def image_ext(image_obj, fallback=".png"):
    path = (image_obj or {}).get("path") or ""
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else fallback


def save_image(image_obj, out_path: Path):
    image_obj = image_obj or {}
    data = image_obj.get("bytes")
    if data is None:
        raise ValueError(f"missing image bytes for {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        out_path.write_bytes(data)


def convert_mmk12():
    source = ROOT / "datasets/sources/FanqingM__MMK12/data/test-00000-of-00001.parquet"
    table = pq.read_table(source).to_pylist()
    rows = []
    for i, item in enumerate(table):
        subject = item.get("subject") or "unknown"
        ext = image_ext(item.get("image"), ".png")
        sample_id = item.get("id") or f"mmk12_{i}"
        image_path = IMAGE_ROOT / "mmk12" / subject / f"{sample_id}{ext}"
        save_image(item.get("image"), image_path)
        rows.append(
            {
                "id": f"mmk12_{subject}_{sample_id}",
                "image": str(image_path),
                "question": item.get("question", ""),
                "options": "",
                "answer": (item.get("answer") or "").strip().upper()[:1],
                "subtopic": subject,
                "benchmark": "MMK12",
                "source_id": sample_id,
            }
        )
    write_jsonl(rows, DATA_DIR / "mmk12_all.jsonl")
    for subject in sorted({r["subtopic"] for r in rows}):
        write_jsonl([r for r in rows if r["subtopic"] == subject], DATA_DIR / f"mmk12_{subject}.jsonl")


def convert_vmcbench_dev():
    source = ROOT / "datasets/sources/suyc21__VMCBench/data/dev-00000-of-00001.parquet"
    table = pq.read_table(source).to_pylist()
    rows = []
    for i, item in enumerate(table):
        idx = item.get("index") or i
        category = item.get("category") or "unknown"
        ext = image_ext(item.get("image"), ".jpg")
        image_path = IMAGE_ROOT / "vmcbench_dev" / category / f"{idx}{ext}"
        save_image(item.get("image"), image_path)
        options = "\n".join(
            f"({letter}) {item.get(letter, '')}" for letter in ["A", "B", "C", "D"]
        )
        rows.append(
            {
                "id": f"vmcbench_dev_{idx}",
                "image": str(image_path),
                "question": item.get("question", ""),
                "options": options,
                "answer": (item.get("answer") or "").strip().upper()[:1],
                "subtopic": category,
                "benchmark": "VMCBench-dev",
                "source_index": idx,
            }
        )
    rows = [r for r in rows if r["answer"] in {"A", "B", "C", "D"}]
    write_jsonl(rows, DATA_DIR / "vmcbench_dev.jsonl")


def main():
    convert_mmk12()
    convert_vmcbench_dev()


if __name__ == "__main__":
    main()

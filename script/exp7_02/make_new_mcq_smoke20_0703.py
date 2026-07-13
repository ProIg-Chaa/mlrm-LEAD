import json
from pathlib import Path

DATA_DIR = Path("/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/data")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path.name, len(rows))


def main():
    mcq_names = [
        "vmcbench_dev",
        "mmk12_math",
        "mmk12_physics",
        "mmk12_chemistry",
        "mmk12_biology",
    ]
    for name in mcq_names:
        rows = load_jsonl(DATA_DIR / f"{name}.jsonl")
        rows = [row for row in rows if row.get("options")][:20]
        if len(rows) != 20:
            raise RuntimeError(f"{name}: only {len(rows)} rows with options")
        write_jsonl(rows, DATA_DIR / f"{name}_smoke20.jsonl")

    mathvision = load_jsonl(DATA_DIR / "math_vision.jsonl")[:20]
    write_jsonl(mathvision, DATA_DIR / "math_vision_smoke20.jsonl")


if __name__ == "__main__":
    main()

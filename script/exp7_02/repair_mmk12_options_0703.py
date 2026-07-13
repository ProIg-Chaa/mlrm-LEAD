import json
import re
from pathlib import Path

DATA_DIR = Path("/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/data")
FILES = [
    "mmk12_all.jsonl",
    "mmk12_math.jsonl",
    "mmk12_physics.jsonl",
    "mmk12_chemistry.jsonl",
    "mmk12_biology.jsonl",
]


def split_question_options(text: str):
    matches = list(re.finditer(r"(?<![A-Za-z0-9])([A-D])\.\s*", text))
    labels = [m.group(1) for m in matches]
    if labels[:4] != ["A", "B", "C", "D"]:
        return text, ""

    question = text[: matches[0].start()].strip()
    option_lines = []
    for idx, match in enumerate(matches[:4]):
        label = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < 4 else len(text)
        value = text[start:end].strip()
        option_lines.append(f"({label}) {value}")
    return question, "\n".join(option_lines)


def repair_file(path: Path):
    rows = []
    changed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("options"):
                question, options = split_question_options(row.get("question", ""))
                if options:
                    row["question"] = question
                    row["options"] = options
                    changed += 1
            rows.append(row)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path.name, "rows", len(rows), "changed", changed)


def main():
    for name in FILES:
        repair_file(DATA_DIR / name)


if __name__ == "__main__":
    main()

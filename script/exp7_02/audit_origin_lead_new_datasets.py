#!/usr/bin/env python3
"""Audit new Origin-LEAD reproduction datasets before running experiments.

This script is intentionally read-only with respect to source datasets. It
checks what is already present locally, what has been converted into the
project JSONL format, and which evaluator tier should be used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    tier: str
    group: str
    official_source: str
    access_method: str
    source_dir: str
    converted_jsonl: str
    image_root: str
    answer_type: str
    evaluator: str
    notes: str
    skip_reason: str = ""


DATASETS = [
    DatasetSpec(
        key="vmcbench",
        tier="tier1",
        group="general",
        official_source="HF: suyc21/VMCBench; official repo: yuhui-zh15/autoconverter",
        access_method="HF snapshot; officially supported by VLMEvalKit/lmms-eval",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/suyc21__VMCBench",
        converted_jsonl="data/vmcbench.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/vmcbench",
        answer_type="mcq",
        evaluator="VLMEvalKit preferred; local deterministic MCQ fallback",
        notes="Unified multiple-choice VQA benchmark; use DEV/random300 before full.",
    ),
    DatasetSpec(
        key="pope_random",
        tier="tier1",
        group="hallucination",
        official_source="HF: lmms-lab/POPE or GitHub: RUCAIBox/POPE",
        access_method="HF snapshot or official repo; requires image availability",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE",
        converted_jsonl="data/pope_random.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/pope",
        answer_type="yes_no",
        evaluator="local deterministic yes/no + precision/recall/F1",
        notes="Need confirm whether HF formatted version includes images or references COCO.",
    ),
    DatasetSpec(
        key="pope_popular",
        tier="tier1",
        group="hallucination",
        official_source="HF: lmms-lab/POPE or GitHub: RUCAIBox/POPE",
        access_method="HF snapshot or official repo; requires image availability",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE",
        converted_jsonl="data/pope_popular.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/pope",
        answer_type="yes_no",
        evaluator="local deterministic yes/no + precision/recall/F1",
        notes="Need keep random/popular/adversarial separate in reporting.",
    ),
    DatasetSpec(
        key="pope_adversarial",
        tier="tier1",
        group="hallucination",
        official_source="HF: lmms-lab/POPE or GitHub: RUCAIBox/POPE",
        access_method="HF snapshot or official repo; requires image availability",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE",
        converted_jsonl="data/pope_adversarial.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/pope",
        answer_type="yes_no",
        evaluator="local deterministic yes/no + precision/recall/F1",
        notes="Most important POPE split for robust hallucination stress.",
    ),
    DatasetSpec(
        key="mathvision",
        tier="tier1",
        group="math",
        official_source="HF/GitHub: MathLLMs/MathVision, mathllm/MATH-V",
        access_method="Already converted locally; add official or deterministic evaluator",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/MathLLMs__MathVision",
        converted_jsonl="data/math_vision.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/math_vision",
        answer_type="mcq_or_numeric",
        evaluator="official MathVision preferred; local normalized MCQ/numeric fallback",
        notes="Project has 3040-row JSONL but no Origin LEAD run yet.",
    ),
    DatasetSpec(
        key="mmk12_math",
        tier="tier1",
        group="math",
        official_source="HF: FanqingM/MMK12",
        access_method="HF snapshot",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12",
        converted_jsonl="data/mmk12_math.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/mmk12",
        answer_type="mcq",
        evaluator="local deterministic MCQ exact + by_subject",
        notes="Expected 500-test MCQ subject subset.",
    ),
    DatasetSpec(
        key="mmk12_physics",
        tier="tier1",
        group="science",
        official_source="HF: FanqingM/MMK12",
        access_method="HF snapshot",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12",
        converted_jsonl="data/mmk12_physics.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/mmk12",
        answer_type="mcq",
        evaluator="local deterministic MCQ exact + by_subject",
        notes="Do not treat old PhysUniBench as this benchmark.",
    ),
    DatasetSpec(
        key="mmk12_chemistry",
        tier="tier1",
        group="science",
        official_source="HF: FanqingM/MMK12",
        access_method="HF snapshot",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12",
        converted_jsonl="data/mmk12_chemistry.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/mmk12",
        answer_type="mcq",
        evaluator="local deterministic MCQ exact + by_subject",
        notes="Expected 500-test MCQ subject subset.",
    ),
    DatasetSpec(
        key="mmk12_biology",
        tier="tier1",
        group="science",
        official_source="HF: FanqingM/MMK12",
        access_method="HF snapshot",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12",
        converted_jsonl="data/mmk12_biology.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/mmk12",
        answer_type="mcq",
        evaluator="local deterministic MCQ exact + by_subject",
        notes="Expected 500-test MCQ subject subset.",
    ),
    DatasetSpec(
        key="mmeval_pro",
        tier="tier2",
        group="general",
        official_source="GitHub: chenllliang/MMEvalPro",
        access_method="official repo/data",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/chenllliang__MMEvalPro",
        converted_jsonl="data/mmeval_pro.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/mmeval_pro",
        answer_type="mcq_triplet",
        evaluator="official Genuine Accuracy; sample acc only auxiliary",
        notes="Need preserve triplet_id and report Genuine Accuracy.",
    ),
    DatasetSpec(
        key="mathverse",
        tier="tier2",
        group="math",
        official_source="GitHub: ZrrSkywalker/MathVerse",
        access_method="official repo; evaluation usually needs LLM judge",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/ZrrSkywalker__MathVerse",
        converted_jsonl="data/mathverse_testmini.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/mathverse",
        answer_type="open_math",
        evaluator="official LLM extraction/scoring; quick_match only auxiliary",
        notes="Download metadata/testmini first; do not include in first official main table.",
    ),
    DatasetSpec(
        key="bingo",
        tier="tier2",
        group="hallucination",
        official_source="GitHub: gzcch/Bingo; data via Google Drive",
        access_method="Google Drive manual or gdown; GPT evaluator",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/gzcch__Bingo",
        converted_jsonl="data/bingo.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/bingo",
        answer_type="open_or_mcq",
        evaluator="official GPT eval",
        notes="Second hallucination batch after POPE because data/eval are less stable.",
    ),
    DatasetSpec(
        key="geometry3k",
        tier="tier2",
        group="math",
        official_source="Geometry3K / Inter-GPS resources",
        access_method="needs source confirmation",
        source_dir="/share/home/wangzixu/liudinghao/gushuo/datasets/sources/Geometry3K",
        converted_jsonl="data/geometry3k.jsonl",
        image_root="/share/home/wangzixu/liudinghao/gushuo/datasets/mlrm-LEAD/images/geometry3k",
        answer_type="geometry_structured",
        evaluator="TBD; likely not first-batch deterministic",
        notes="Audit only until image/question/answer format is confirmed.",
    ),
]


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def sample_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def audit_spec(root: Path, spec: DatasetSpec) -> dict:
    converted = root / spec.converted_jsonl
    source_dir = Path(spec.source_dir)
    image_root = Path(spec.image_root)
    row_count = count_jsonl(converted)
    sample = sample_jsonl(converted)
    missing_images = None
    nonempty_answer = None
    options_like = None
    if sample is not None and row_count:
        missing = 0
        answers = 0
        options = 0
        checked = 0
        with converted.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                checked += 1
                image = row.get("image")
                if image and not Path(image).exists():
                    missing += 1
                if str(row.get("answer") or "").strip():
                    answers += 1
                if str(row.get("options") or "").strip():
                    options += 1
        missing_images = missing
        nonempty_answer = answers / checked if checked else None
        options_like = options / checked if checked else None

    if row_count is not None and missing_images == 0:
        status = "converted"
    elif source_dir.exists():
        status = "downloaded"
    else:
        status = "missing"
    if spec.key == "mathvision" and row_count is not None and missing_images == 0:
        status = "ready_smoke"

    return {
        **asdict(spec),
        "source_dir_exists": source_dir.exists(),
        "source_file_count": sum(1 for p in source_dir.rglob("*") if p.is_file()) if source_dir.exists() else 0,
        "converted_exists": converted.exists(),
        "converted_row_count": row_count,
        "image_root_exists": image_root.exists(),
        "missing_images": missing_images,
        "nonempty_answer_rate": nonempty_answer,
        "options_nonempty_rate": options_like,
        "status": status,
        "sample": sample,
    }


def write_markdown(path: Path, records: list[dict]) -> None:
    lines = [
        "# Origin LEAD 新数据集资料搜集与接入审计",
        "",
        "本文件由 `script/exp7_02/audit_origin_lead_new_datasets.py` 生成。已跑过的数据集不列入第一批执行范围。",
        "",
        "## 总览",
        "",
        "| dataset | tier | group | status | rows | image missing | evaluator | 当前动作 |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in records:
        rows = "NA" if r["converted_row_count"] is None else str(r["converted_row_count"])
        miss = "NA" if r["missing_images"] is None else str(r["missing_images"])
        if r["status"] == "ready_smoke":
            action = "可直接 smoke：COT/LEAD 20 条"
        elif r["status"] == "converted":
            action = "补 evaluator smoke 后运行"
        elif r["status"] == "downloaded":
            action = "写/运行转换脚本"
        else:
            action = "等待下载"
        lines.append(
            f"| `{r['key']}` | {r['tier']} | {r['group']} | {r['status']} | {rows} | {miss} | {r['evaluator']} | {action} |"
        )
    lines.extend(["", "## 逐项备注", ""])
    for r in records:
        lines.extend(
            [
                f"### {r['key']}",
                "",
                f"- source: {r['official_source']}",
                f"- access: {r['access_method']}",
                f"- source dir: `{r['source_dir']}`",
                f"- jsonl: `{r['converted_jsonl']}`",
                f"- answer type: `{r['answer_type']}`",
                f"- evaluator: {r['evaluator']}",
                f"- status: `{r['status']}`",
                f"- notes: {r['notes']}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD")
    parser.add_argument("--output_dir", default="result/20260702_origin_lead_dataset_audit")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [audit_spec(root, spec) for spec in DATASETS]

    (out_dir / "dataset_audit.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "dataset_audit.md", records)
    print(json.dumps({"output_dir": str(out_dir), "datasets": len(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

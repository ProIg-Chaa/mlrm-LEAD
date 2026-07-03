# Origin LEAD 新数据集资料搜集与接入审计

本文件由 `script/exp7_02/audit_origin_lead_new_datasets.py` 生成。已跑过的数据集不列入第一批执行范围。

## 总览

| dataset | tier | group | status | rows | image missing | evaluator | 当前动作 |
|---|---|---|---:|---:|---:|---|---|
| `vmcbench` | tier1 | general | downloaded | NA | NA | VLMEvalKit preferred; local deterministic MCQ fallback | 写/运行转换脚本 |
| `pope_random` | tier1 | hallucination | converted | 3000 | 0 | local deterministic yes/no + precision/recall/F1 | 补 evaluator smoke 后运行 |
| `pope_popular` | tier1 | hallucination | converted | 3000 | 0 | local deterministic yes/no + precision/recall/F1 | 补 evaluator smoke 后运行 |
| `pope_adversarial` | tier1 | hallucination | converted | 3000 | 0 | local deterministic yes/no + precision/recall/F1 | 补 evaluator smoke 后运行 |
| `mathvision` | tier1 | math | ready_smoke | 3040 | 0 | official MathVision preferred; local normalized MCQ/numeric fallback | 可直接 smoke：COT/LEAD 20 条 |
| `mmk12_math` | tier1 | math | downloaded | NA | NA | local deterministic MCQ exact + by_subject | 写/运行转换脚本 |
| `mmk12_physics` | tier1 | science | downloaded | NA | NA | local deterministic MCQ exact + by_subject | 写/运行转换脚本 |
| `mmk12_chemistry` | tier1 | science | downloaded | NA | NA | local deterministic MCQ exact + by_subject | 写/运行转换脚本 |
| `mmk12_biology` | tier1 | science | downloaded | NA | NA | local deterministic MCQ exact + by_subject | 写/运行转换脚本 |
| `mmeval_pro` | tier2 | general | missing | NA | NA | official Genuine Accuracy; sample acc only auxiliary | 等待下载 |
| `mathverse` | tier2 | math | missing | NA | NA | official LLM extraction/scoring; quick_match only auxiliary | 等待下载 |
| `bingo` | tier2 | hallucination | missing | NA | NA | official GPT eval | 等待下载 |
| `geometry3k` | tier2 | math | missing | NA | NA | TBD; likely not first-batch deterministic | 等待下载 |

## 逐项备注

### vmcbench

- source: HF: suyc21/VMCBench; official repo: yuhui-zh15/autoconverter
- access: HF snapshot; officially supported by VLMEvalKit/lmms-eval
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/suyc21__VMCBench`
- jsonl: `data/vmcbench.jsonl`
- answer type: `mcq`
- evaluator: VLMEvalKit preferred; local deterministic MCQ fallback
- status: `downloaded`
- notes: Unified multiple-choice VQA benchmark; use DEV/random300 before full.

### pope_random

- source: HF: lmms-lab/POPE or GitHub: RUCAIBox/POPE
- access: HF snapshot or official repo; requires image availability
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE`
- jsonl: `data/pope_random.jsonl`
- answer type: `yes_no`
- evaluator: local deterministic yes/no + precision/recall/F1
- status: `converted`
- notes: Need confirm whether HF formatted version includes images or references COCO.

### pope_popular

- source: HF: lmms-lab/POPE or GitHub: RUCAIBox/POPE
- access: HF snapshot or official repo; requires image availability
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE`
- jsonl: `data/pope_popular.jsonl`
- answer type: `yes_no`
- evaluator: local deterministic yes/no + precision/recall/F1
- status: `converted`
- notes: Need keep random/popular/adversarial separate in reporting.

### pope_adversarial

- source: HF: lmms-lab/POPE or GitHub: RUCAIBox/POPE
- access: HF snapshot or official repo; requires image availability
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE`
- jsonl: `data/pope_adversarial.jsonl`
- answer type: `yes_no`
- evaluator: local deterministic yes/no + precision/recall/F1
- status: `converted`
- notes: Most important POPE split for robust hallucination stress.

### mathvision

- source: HF/GitHub: MathLLMs/MathVision, mathllm/MATH-V
- access: Already converted locally; add official or deterministic evaluator
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/MathLLMs__MathVision`
- jsonl: `data/math_vision.jsonl`
- answer type: `mcq_or_numeric`
- evaluator: official MathVision preferred; local normalized MCQ/numeric fallback
- status: `ready_smoke`
- notes: Project has 3040-row JSONL but no Origin LEAD run yet.

### mmk12_math

- source: HF: FanqingM/MMK12
- access: HF snapshot
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12`
- jsonl: `data/mmk12_math.jsonl`
- answer type: `mcq`
- evaluator: local deterministic MCQ exact + by_subject
- status: `downloaded`
- notes: Expected 500-test MCQ subject subset.

### mmk12_physics

- source: HF: FanqingM/MMK12
- access: HF snapshot
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12`
- jsonl: `data/mmk12_physics.jsonl`
- answer type: `mcq`
- evaluator: local deterministic MCQ exact + by_subject
- status: `downloaded`
- notes: Do not treat old PhysUniBench as this benchmark.

### mmk12_chemistry

- source: HF: FanqingM/MMK12
- access: HF snapshot
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12`
- jsonl: `data/mmk12_chemistry.jsonl`
- answer type: `mcq`
- evaluator: local deterministic MCQ exact + by_subject
- status: `downloaded`
- notes: Expected 500-test MCQ subject subset.

### mmk12_biology

- source: HF: FanqingM/MMK12
- access: HF snapshot
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12`
- jsonl: `data/mmk12_biology.jsonl`
- answer type: `mcq`
- evaluator: local deterministic MCQ exact + by_subject
- status: `downloaded`
- notes: Expected 500-test MCQ subject subset.

### mmeval_pro

- source: GitHub: chenllliang/MMEvalPro
- access: official repo/data
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/chenllliang__MMEvalPro`
- jsonl: `data/mmeval_pro.jsonl`
- answer type: `mcq_triplet`
- evaluator: official Genuine Accuracy; sample acc only auxiliary
- status: `missing`
- notes: Need preserve triplet_id and report Genuine Accuracy.

### mathverse

- source: GitHub: ZrrSkywalker/MathVerse
- access: official repo; evaluation usually needs LLM judge
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/ZrrSkywalker__MathVerse`
- jsonl: `data/mathverse_testmini.jsonl`
- answer type: `open_math`
- evaluator: official LLM extraction/scoring; quick_match only auxiliary
- status: `missing`
- notes: Download metadata/testmini first; do not include in first official main table.

### bingo

- source: GitHub: gzcch/Bingo; data via Google Drive
- access: Google Drive manual or gdown; GPT evaluator
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/gzcch__Bingo`
- jsonl: `data/bingo.jsonl`
- answer type: `open_or_mcq`
- evaluator: official GPT eval
- status: `missing`
- notes: Second hallucination batch after POPE because data/eval are less stable.

### geometry3k

- source: Geometry3K / Inter-GPS resources
- access: needs source confirmation
- source dir: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources/Geometry3K`
- jsonl: `data/geometry3k.jsonl`
- answer type: `geometry_structured`
- evaluator: TBD; likely not first-batch deterministic
- status: `missing`
- notes: Audit only until image/question/answer format is confirmed.


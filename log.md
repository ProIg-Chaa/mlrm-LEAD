# mlrm-LEAD Local Work Log

Updated: 2026-04-26

This file records the important project context so future work can resume without relying on chat history.

## Environment And Paths

- Project root: `/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD`
- Micromamba env: `mlrm-lead`
- Model used in experiments: `/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`
- Dataset root: `/share/home/wangzixu/liudinghao/gushuo/datasets`
- Dataset sources: `/share/home/wangzixu/liudinghao/gushuo/datasets/sources`
- Main GPU used for experiments: GPU 1
- Common proxy used for downloads: `http://127.0.0.1:17991`
- Current shell may have default proxy env pointing to `0.0.0.0:8886`; commands that need 17991 should explicitly set `HTTP_PROXY` and `HTTPS_PROXY`.

## Downloaded Assets

### Model

Downloaded and usable:

- `/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`

### Dataset Sources

Under `/share/home/wangzixu/liudinghao/gushuo/datasets/sources`:

- `PrismaX__PhysUniBench`
- `xai-org__RealworldQA`
- `AI4Math__MathVista`
- `MMVP__MMVP`
- `MathLLMs__MathVision`
- `VisuLogic__VisuLogic`
- `craigwu__vstar_bench`
- `Shengcao1006__MMHal-Bench`

MMHal-Bench was verified complete:

- Repo files: 110
- Local repo files: 110
- Missing: 0
- Images: 97
- Size: about 337M

Prepared project JSONL files in `data/`:

- `demo.jsonl`
- `physunibench.jsonl`
- `math_vision.jsonl`
- `math_vista.jsonl`
- `mmvp.jsonl`
- `realworldqa.jsonl`
- `visulogic.jsonl`
- `vstar.jsonl`
- `mmhal_bench.jsonl`

## Important Code Changes

### Entropy Logging

`main.py` now supports compact token entropy summaries when `--save_token_entropy` is enabled.

Saved file:

- `token_entropy.jsonl`

Each sample stores:

- token count
- `<think>...</think>` detection
- reasoning-token count and ratio
- raw and filtered entropy stats
- LEAD soft-token count and ratio
- relation-token entropy stats
- relation marker counts and category stats

Reasoning span detection was fixed by reconstructing decoded token text, so split pieces like `<th`, `ink`, `>` are handled.

### Relation-Token Entropy

`main.py` now tracks reasoning-relation tokens such as:

- conclusion: `therefore`, `thus`, `hence`, `consequently`, `accordingly`
- contrast: `however`, `but`, `although`, `though`, `instead`, `while`, `whereas`
- causal/condition: `because`, `since`, `so`, `if`, `when`, `as`, `given`, `assuming`, `implies`, `means`
- sequence: `then`, `first`, `second`, `third`, `next`, `finally`, `now`, `also`, `moreover`, `furthermore`
- result: `result`, `results`, `resulting`, `thereby`

Key fields to inspect:

- `avg_reasoning_relation_raw_entropy`
- `avg_reasoning_non_relation_raw_entropy`
- `avg_sample_reasoning_relation_raw_high_gt_1_ratio`
- `relation_marker_counts`
- `relation_category_avg_raw_entropy`

### LEAD Parameters

`main.py` now exposes:

- `--window_size`
- `--save_token_entropy`

`lead/inference.py` passes `window_size` into `generate_lead()`.

`lead/generation_utils.py` entropy clamp changed from `1e-12` to `1e-8` to match the requested numerical-stability setting.

### Summarizer

`script/summarize_token_entropy.py` aggregates compact entropy summaries and supports relation-token fields.

Usage:

```bash
python3 script/summarize_token_entropy.py output/path/token_entropy.jsonl
```

## Scripts Added

### Asset Download

Project-root script:

```bash
bash download_assets.sh
```

Default proxy:

```bash
http://127.0.0.1:17991
```

Override proxy:

```bash
PROXY_URL=http://127.0.0.1:8886 bash download_assets.sh
```

This script downloads/verifies:

- `Fancy-MLLM/R1-Onevision-7B-RL`
- PhysUniBench
- RealWorldQA
- MathVista
- MMVP
- MathVision
- VisuLogic
- VStar
- MMHal-Bench

It also prepares `data/mmhal_bench.jsonl`.

### VStar Paper-Parameter Experiment

Run:

```bash
bash script/run_vstar_lead_paper_params.sh
```

Summarize latest run:

```bash
bash script/summarize_latest_vstar_lead_paper_params.sh
```

Parameters:

- `--method lead`
- `--max_new_tokens 1024`
- `--alpha 0.4`
- `--max_switch_count 5`
- `--window_size 128`
- `--temperature 0.6`
- `--top_p 0.95`
- `--top_k 20`
- `--seed 42`
- `--device cuda`
- `--no-do_sample`
- `--save_token_entropy`

### MMHal-Bench Paper-Parameter Experiment

Run:

```bash
bash script/run_mmhal_lead_paper_params.sh
```

Summarize latest run:

```bash
bash script/summarize_latest_mmhal_lead_paper_params.sh
```

The MMHal script first runs:

```bash
python3 script/prepare_mmhal_bench_jsonl.py
```

The summarizer exports:

- `mmhal_response.json`

MMHal is open-ended, not MCQ. The `main.py` MCQ accuracy is not meaningful for MMHal. Final hallucination scoring should use the official MMHal `eval_gpt4.py` on `mmhal_response.json`.

### Method Comparison And Grid

Added:

- `script/run_method_comparison.py`
- `script/run_entropy_grid.sh`

`run_method_comparison.py` writes markdown reports automatically. Direct `main.py` runs do not, unless a report is generated separately.

## Experiment Directory Policy

From now on, new experiments should use:

```text
output/experiments/YYYYMMDD/experiment_name_HHMMSS/
```

Examples:

```text
output/experiments/20260426/vstar_lead_paper_params_213624/
output/experiments/20260426/mmhal_lead_paper_params_222506/
```

Each experiment directory should contain its own:

- `config.json`
- `results.jsonl`
- `eval_report.json`
- `token_entropy.jsonl`
- `logs/`
- `nohup.log`
- `report.md` if generated

The VStar/MMHal scripts already follow this policy. Older experiments are still in older flat layouts.

## Completed Experiments And Results

### LEAD Parameter Grid On PhysUniBench

Directory family:

- `output/experiments/entropy_grid`

Results:

| alpha | max_switch_count | Accuracy |
|---:|---:|---:|
| 0.4 | 3 | 16.67% |
| 0.4 | 5 | 30.00% |
| 0.6 | 3 | 16.67% |
| 0.6 | 5 | 20.00% |

Best observed local setting:

- `alpha=0.4`
- `max_switch_count=5`

### 30-Sample COT / Greedy COT / LEAD On PhysUniBench

Directory:

- `output/experiments/cot_entropy_probe/compare_methods_20260426_182309`

Results:

| Method | Accuracy | Avg latency | Avg output tokens |
|---|---:|---:|---:|
| `cot` | 23.33% (7/30) | 28.56s | 773.83 |
| `cot_greedy` | 16.67% (5/30) | 34.41s | 835.60 |
| `lead` | 30.00% (9/30) | 36.55s | 788.13 |

Useful points:

- LEAD was best on this small subset.
- Greedy COT had lowest entropy but also lowest accuracy.
- LEAD had more `no_answer_extracted` than normal COT.

### 100-Sample LEAD On PhysUniBench

Directory:

- `output/experiments/medium_best_lead_entropy_compact/compare_methods_20260426_183346`

Result:

- Accuracy: 20.00% (20/100)
- Avg latency: 33.77s
- Avg output tokens: 776.62
- Error types: correct 20, wrong_answer 57, no_answer_extracted 23

Conclusion:

- The 30-sample LEAD gain did not robustly carry over to 100 samples.

### VStar LEAD Paper-Parameter Experiment

Old flat directory:

- `output/experiments/vstar_lead_paper_params_20260426_213624`

Result:

- Accuracy: 72.77% (139/191)
- direct_attributes: 71.30% (82/115)
- relative_position: 75.00% (57/76)
- Failed extraction: 0

Token entropy:

- Total tokens: 23596
- Avg tokens/sample: 123.54
- Reasoning ratio: 85.73%
- Avg reasoning raw entropy: 0.8283
- Reasoning p90 entropy: 2.1945
- Reasoning entropy > 1 ratio: 34.38%
- Reasoning entropy > 2 ratio: 14.07%
- Soft ratio: 1.44%

Relation-token entropy:

- Relation tokens: 507
- Reasoning relation tokens: 491
- Avg reasoning relation raw entropy: 1.3192
- Avg reasoning non-relation raw entropy: 0.8161
- Reasoning relation entropy > 1 ratio: 71.76%

High-frequency relation markers:

- `therefore`: 185
- `as`: 57
- `given`: 57
- `so`: 54
- `since`: 49
- `also`: 25
- `but`: 20

Important conclusion:

- VStar strongly supports the hypothesis that reasoning-relation tokens have higher entropy than ordinary reasoning tokens.

Generated report:

- `output/experiments/vstar_lead_paper_params_20260426_213624/report.md`

### MMHal-Bench LEAD Paper-Parameter Experiment

Current run started by user:

- `output/experiments/20260426/mmhal_lead_paper_params_222506`

Command was launched via:

```bash
bash script/run_mmhal_lead_paper_params.sh
```

This uses the same paper-style LEAD parameters as VStar:

- `alpha=0.4`
- `max_switch_count=5`
- `window_size=128`
- greedy decoding via `--no-do_sample`
- `--save_token_entropy`

Need to check status with:

```bash
tail -f output/experiments/20260426/mmhal_lead_paper_params_222506/nohup.log
```

After completion:

```bash
bash script/summarize_latest_mmhal_lead_paper_params.sh
```

This will export:

- `output/experiments/20260426/mmhal_lead_paper_params_222506/mmhal_response.json`

Then official GPT-based MMHal scoring can be run with the dataset's `eval_gpt4.py`.

## Reports

Project-level experiment report:

- `result/experiment_report.md`

This is a Chinese report summarizing completed experiments before VStar/MMHal scripts were finalized.

Latest VStar report:

- `output/experiments/vstar_lead_paper_params_20260426_213624/report.md`

## Git / README Context

README has been updated with:

- local reproducibility scripts
- `download_assets.sh`
- VStar and MMHal scripts
- new experiment directory policy
- `--window_size`
- `--save_token_entropy`
- entropy logging details
- MMHal-Bench dataset preparation

`.gitignore` was added to ignore:

- `__pycache__/`
- `*.py[cod]`
- `*.egg-info/`
- `.pytest_cache/`
- `output/`
- `logs/`
- `*.nohup.log`
- `*.part`

Suggested git add command from previous turn:

```bash
git add \
  .gitignore \
  README.md \
  main.py \
  lead/inference.py \
  lead/generation_utils.py \
  download_assets.sh \
  data/*.jsonl \
  script/prepare_mmhal_bench_jsonl.py \
  script/run_entropy_grid.sh \
  script/run_method_comparison.py \
  script/run_mmhal_lead_paper_params.sh \
  script/run_vstar_lead_paper_params.sh \
  script/summarize_latest_mmhal_lead_paper_params.sh \
  script/summarize_latest_vstar_lead_paper_params.sh \
  script/summarize_token_entropy.py \
  result/experiment_report.md \
  log.md
```

Remote desired by user:

```bash
git remote set-url origin https://github.com/ProIg-Chaa/mlrm-LEAD
git push -u origin HEAD
```

If branch needs explicit main:

```bash
git push -u origin HEAD:main
```

## Cautions

- User requested no dangerous deletion operations. Avoid `rm`, `git reset --hard`, or destructive cleanup unless explicitly authorized.
- Do not treat MMHal-Bench `main.py` MCQ accuracy as meaningful; MMHal requires GPT scoring.
- The current `python3` base environment may have broken/incomplete torch/transformers. For project runs use `micromamba run -n mlrm-lead ...` or the env python.
- Shared filesystem may not preserve executable bit; use `bash script/name.sh`.
- GPU 0 often has unrelated memory use. Experiments normally target GPU 1.
- Direct `main.py` runs do not auto-generate `report.md`; `run_method_comparison.py` does. For direct runs, generate report separately or add report generation later.

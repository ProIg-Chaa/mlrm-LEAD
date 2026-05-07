# mlrm-LEAD Local Work Log

Updated: 2026-04-29

This file records the current high-value context so future work can resume without chat history.

## Environment And Paths

- Project root: `/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD`
- Micromamba env: `mlrm-lead`
- Main model path: `/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`
- Dataset root: `/share/home/wangzixu/liudinghao/gushuo/datasets`
- Common proxy for downloads: `http://127.0.0.1:17991`
- Experiments are stored under `output/experiments/YYYYMMDD/experiment_name_HHMMSS/`

## Datasets Available In `data/`

- `physunibench.jsonl` 3304
- `math_vision.jsonl` 3040
- `math_vista.jsonl` 1000
- `mmvp.jsonl` 300
- `realworldqa.jsonl` 765
- `visulogic.jsonl` 1000
- `vstar.jsonl` 191
- `mmhal_bench.jsonl` 96
- diagnostic subsets:
  - `vstar_anchor_diagnostic_union.jsonl`
  - `vstar_anchor_regressions_7.jsonl`
  - `mmhal_bench_balanced_2pertype.jsonl`

## Important Code Changes

### Token Entropy And Reasoning Annotation

- `main.py` supports `--save_token_entropy` and `--save_full_token_entropy`
- `token_entropy.jsonl` stores compact summaries
- `token_entropy_full.jsonl` stores per-token traces
- per-token traces include:
  - `raw_entropy`
  - `filtered_entropy`
  - `selected_prob`
  - `raw_selected_prob`
  - `confidence`
  - `token_text`
  - `is_reasoning_token`
  - `is_relation_token`
  - `relation_category`

### Relation Token Statistics

`main.py` annotates reasoning relation markers such as:

- conclusion: `therefore`, `thus`, `hence`
- contrast: `however`, `but`, `although`
- causal/condition: `because`, `since`, `so`, `if`, `when`, `as`
- sequence: `then`, `first`, `next`, `finally`, `also`
- result: `result`, `results`, `resulting`, `thereby`

### Pure-Soft Method

Added `generate_pure_soft()` in [generation_utils.py](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/lead/generation_utils.py).

Behavior:

- first step uses prompt `input_ids`
- later steps use `inputs_embeds` only
- next input embedding is `probs_original @ embedding_matrix`
- output text still uses discrete chosen tokens for evaluation
- trace logs both:
  - `selected_prob`
  - `raw_selected_prob`
- later confidence analysis has mainly used `raw_selected_prob`

### LEAD / Anchor Work

`lead_attenachor` exists and has been tested on VStar.

Important settings exposed in `main.py`:

- `--visual_anchor_top_m`
- `--visual_anchor_attn_last_k`
- `--visual_anchor_lambda_scale`
- `--visual_anchor_entropy_upper`
- `--visual_anchor_skip_nonword`
- `--visual_anchor_single_use`
- `--soft_trigger_mode legacy|dual_delta2`
- `--soft_warning_margin`
- `--soft_confirm_margin`
- `--soft_delta2_threshold`
- `--soft_repeat_warning_boost`
- `--soft_repeat_confirm_boost`
- `--soft_repeat_delta2_boost`
- `--soft_repeat_cooldown`
- `--soft_post_reset_ref_margin`
- `--soft_post_reset_cooldown`

Known conclusion so far:

- `lead_attenachor` has not beaten the old `lead` baseline on tested VStar subsets
- `top_m=32` was too aggressive
- `dual_delta2 + larger window` made switching behavior more reasonable but did not reverse the final ranking

## New Scripts Added

- `script/prepare_uniform_subset.py`
  - evenly samples a fixed-size subset from a JSONL dataset
- `script/run_pure_soft_physunibench300_mmvp_parallel.sh`
  - PhysUniBench uniform 300 on GPU 0
  - MMVP full on GPU 1
- `script/run_cot_physunibench300_mmvp_parallel.sh`
  - same as above, method changed to `cot`
- `script/evaluate_specialized_results.py`
  - specialized re-evaluation for datasets whose answer format is not plain `A/B/C/D`
  - supports `mmvp` and `physunibench`
- `script/plot_pure_soft_correct_wrong_curves.py`
  - supports:
    - `--results_format default`
    - `--results_format specialized`

## Important Evaluation Caveats

### MMVP

- default repository evaluator is wrong for MMVP
- reason: MMVP uses `"(a) ..."` / `"(b) ..."` answers
- repository default evaluator only compares `A/B/C/D`
- use `script/evaluate_specialized_results.py --mode mmvp`

### PhysUniBench

- default evaluator underestimates format failures
- many outputs do not end in clean `A/B/C/D` letters
- specialized evaluator can recover some cases by option-text matching
- but low accuracy on PhysUniBench is still mostly a model/output behavior issue, not only an evaluator issue

## Most Important Recent Experiments

### 1. VStar Pure-Soft 50

Directory:

- `output/experiments/20260429/vstar_pure_soft_50_203818_setsid`

Report:

- `result/vstar_pure_soft_confidence_vs_correctness_report_zh.md`
- `result/vstar_pure_soft_correct_wrong_curves.png`
- `result/vstar_pure_soft_correct_wrong_summary.json`

Key result:

- official-style accuracy: `29/50 = 58.0%`
- wrong samples had:
  - higher `mean_raw_conf`
  - higher `last10_raw_conf`
  - higher `last20_raw_conf`
  - lower `mean_raw_entropy`
  - longer outputs
  - longer latency

Interpretation:

- wrong answers are often not hesitant
- they are often lower-entropy, higher-confidence, longer wrong trajectories

### 2. MMVP Pure-Soft Full 300

Directory:

- `output/experiments/20260429/pure_soft_phys300_mmvp_parallel_212553/mmvp_full_gpu1`

Important files:

- `specialized_eval_report.json`
- `specialized_results.jsonl`
- `result/mmvp_pure_soft_correct_wrong_curves.png`
- `result/mmvp_pure_soft_correct_wrong_summary.json`

Key result after specialized re-evaluation:

- `191/300 = 63.67%`
- `failed_extraction = 14`

Confidence findings:

- correct `mean_raw_conf = 0.7559`
- wrong `mean_raw_conf = 0.7838`
- correct `last20_raw_conf = 0.8560`
- wrong `last20_raw_conf = 0.8737`
- correct `mean_raw_entropy = 0.9155`
- wrong `mean_raw_entropy = 0.8312`
- correct output length `149.0`
- wrong output length `284.7`

High-confidence wrong-sample signal:

- top 5 by `mean_raw_conf`: `100%` wrong
- top 5 by `last10_raw_conf`: `80%` wrong
- top 5 by `last20_raw_conf`: `80%` wrong
- `last20_raw_conf >= 0.95` group had accuracy only `27.6%`

Interpretation:

- MMVP reproduces the same phenomenon as VStar
- wrong samples are often more confident, lower-entropy, longer, and slower

### 3. PhysUniBench Pure-Soft Uniform 300

Directory:

- `output/experiments/20260429/pure_soft_phys300_mmvp_parallel_212553/physunibench_uniform300_gpu0`

Important files:

- `specialized_eval_report.json`
- `specialized_results.jsonl`

Key result after specialized re-evaluation:

- `14/300 = 4.67%`
- `failed_extraction = 196`

Interpretation:

- this is not mainly an evaluator bug
- pure-soft often fails to converge to stable MCQ answers on PhysUniBench
- many outputs are long, malformed, or do not cleanly map back to options

### 4. COT PhysUniBench 300 + MMVP 300

Directory:

- `output/experiments/20260429/cot_phys300_mmvp_parallel_232652/physunibench_uniform300_gpu0`
- `output/experiments/20260429/cot_phys300_mmvp_parallel_232652/mmvp_full_gpu1`

Status:

- both completed with `300/300` results and full token traces
- no related `main.py` process remained running when checked

Specialized re-evaluation:

- MMVP COT: `202/300 = 67.33%`, `failed_extraction = 0`
- PhysUniBench COT: `31/300 = 10.33%`, `failed_extraction = 164`

Important script fix:

- `script/plot_pure_soft_correct_wrong_curves.py` now falls back to `selected_prob` when `raw_selected_prob` is absent.
- reason: COT traces store `selected_prob`, not `raw_selected_prob`; before this fix COT confidence summaries were incorrectly all zero.

MMVP COT confidence-vs-correctness:

- correct `mean_raw_conf = 0.8809`
- wrong `mean_raw_conf = 0.8787`
- correct `last20_raw_conf = 0.9455`
- wrong `last20_raw_conf = 0.9406`
- correct `mean_raw_entropy = 0.7455`
- wrong `mean_raw_entropy = 0.7648`
- correct output length `121.9`
- wrong output length `137.1`
- top 5 by `last20_raw_conf`: `80%` wrong
- `last20_raw_conf >= 0.95` group accuracy: `68.0%`

Interpretation:

- MMVP COT does not clearly reproduce the pure-soft pattern.
- wrong answers are slightly longer/slower, but not more confident or lower-entropy on average.
- only the extreme top-5 `last20_raw_conf` ranking is wrong-heavy.

PhysUniBench COT confidence-vs-correctness:

- correct `mean_raw_conf = 0.9169`
- wrong `mean_raw_conf = 0.9276`
- correct `last20_raw_conf = 0.9459`
- wrong `last20_raw_conf = 0.9650`
- correct `mean_raw_entropy = 0.4932`
- wrong `mean_raw_entropy = 0.4303`
- correct output length `605.5`
- wrong output length `676.2`
- top 20 by `mean_raw_conf`: `100%` wrong
- `last20_raw_conf >= 0.95` group accuracy: `7.7%`

Interpretation:

- PhysUniBench COT does reproduce the previous phenomenon.
- wrong answers are more confident, lower-entropy, longer, and slower.
- however, PhysUniBench still has many extraction failures, so separate clean-answer vs malformed-answer analysis is recommended.

### MMVP Official Evaluation Note

- MMVP official repository: `tsb0601/MMVP`
- official metric is `pair accuracy`, not single-question accuracy
- official repository does not provide a fixed local `(a)/(b)` string extractor as the final scorer
- official workflow is:
  - generate an `answer.jsonl` file with question, gold answer, and model response
  - then use `scripts/gpt_grader.py` to ask GPT whether each response is correct
  - count a pair as correct only if both questions in that pair are judged correct

Local support added:

- `script/evaluate_mmvp_official.py`
  - converts project `results.jsonl` into MMVP official `answer_file` format
  - mirrors the official GPT-judge prompt and pair-accuracy aggregation
  - can run in `--convert_only` mode without API access

Current blocker:

- the shared environment did not have `OPENAI_API_KEY` set at the time of checking
- therefore the official GPT-judge stage could not be executed yet
- converted official-format answer file for the latest MMVP COT run:
  - `output/experiments/20260429/cot_phys300_mmvp_parallel_232652/mmvp_full_gpu1/official_mmvp_answer.jsonl`

### 5. LEAD MMVP + VStar Parallel

Directory:

- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0`
- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/vstar_full_gpu1`

Key results:

- VStar LEAD default eval: `139/191 = 72.77%`
- MMVP LEAD default eval: still `0/300` with the repository evaluator and should be ignored

MMVP evaluator fix:

- `script/evaluate_specialized_results.py` now uses a stricter MMVP extractor:
  - searches only the tail answer region
  - accepts explicit `\boxed{a}` / `\boxed{b}` and clear `Answer: (a)/(b)` style outputs
  - maps explicit option-text answers back to `(a)/(b)` when needed
  - removes the unsafe `tail_ab` fallback
  - reports `pair_accuracy` directly for MMVP

MMVP LEAD after specialized re-evaluation:

- sample accuracy: `211/300 = 70.33%`
- pair accuracy: `63/150 = 42.0%`
- method breakdown:
  - `direct_ab`: `272` samples
  - `option_label_match`: `28` samples

Important files:

- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0/specialized_eval_report.json`
- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0/specialized_results.jsonl`
- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0/eval_report.json`
  - corrected and replaced with specialized MMVP results
- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0/eval_report_default_incorrect.json`
  - backup of the original wrong default MMVP report (`0/300`)

Confidence-vs-correctness summary:

- MMVP LEAD:
  - correct `mean_raw_conf = 0.8797`
  - wrong `mean_raw_conf = 0.8851`
  - correct `last20_raw_conf = 0.9437`
  - wrong `last20_raw_conf = 0.9429`
  - correct `mean_raw_entropy = 0.7545`
  - wrong `mean_raw_entropy = 0.7382`
  - correct output length `108.5`
  - wrong output length `114.7`
  - interpretation:
    - wrong answers are slightly more confident on average and slightly lower-entropy
    - but the effect is weak
    - high-confidence tails are not strongly wrong-dominated

- VStar LEAD:
  - correct `mean_raw_conf = 0.8797`
  - wrong `mean_raw_conf = 0.8739`
  - correct `last20_raw_conf = 0.9567`
  - wrong `last20_raw_conf = 0.9484`
  - correct `mean_raw_entropy = 0.7291`
  - wrong `mean_raw_entropy = 0.7642`
  - correct output length `109.8`
  - wrong output length `156.6`
  - interpretation:
    - VStar LEAD does not show the previous strong "wrong is more confident, lower-entropy" pattern
    - wrong answers are longer and slower, but also higher-entropy on average

Plotting caveat:

- `script/plot_pure_soft_correct_wrong_curves.py` could not be used in the current `mlrm-lead` environment because `matplotlib` requires `numpy>=1.23` while the environment currently has `numpy==1.22.0`
- numerical summaries were computed directly without the plotting dependency

Revised interpretation across MMVP + VStar:

- MMVP LEAD shows only a weak average trend toward wrong answers being slightly more confident and lower-entropy.
- VStar LEAD does not support the same pattern on average; wrong answers are longer and slower, but also higher-entropy.
- Therefore, these two datasets do not support a strong claim that LEAD still exhibits the earlier "wrong is more confident and lower-entropy" behavior.

## Recommended Next Steps

- For MMVP and VStar: continue confidence-vs-correctness analysis, because the signal is already stable.
- For PhysUniBench: do not trust default `eval_report.json`; use specialized re-evaluation first.
- For future comparisons between `pure_soft` and `cot`, always preserve:
  - `results.jsonl`
  - `token_entropy_full.jsonl`
  - specialized re-evaluation report when dataset format requires it

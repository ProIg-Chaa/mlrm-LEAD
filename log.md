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

## Current Project State (2026-05-08)

### Main active conclusions

1. `lead_attenachor` on `VStar` has not beaten the original `lead` baseline.
2. `pure_soft` and some `cot` settings can show a strong
   "wrong is more confident, lower-entropy, longer, and slower" pattern.
3. `lead` weakens that pattern on average on several datasets, but does not
   eliminate the dangerous high-confidence error tail.
4. `MMVP` and `PhysUniBench` both require corrected evaluation:
   - `MMVP`: use specialized evaluator and track `pair_accuracy`
   - `PhysUniBench`: use specialized re-evaluation before drawing conclusions

### Datasets with usable recent LEAD traces

- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0`
- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/vstar_full_gpu1`
- `output/experiments/20260501/lead_phys300_visulogic_parallel_155552/physunibench_uniform300_gpu0`
- `output/experiments/20260501/lead_phys300_visulogic_parallel_155552/visulogic_full_gpu1`

These all contain:

- `results.jsonl`
- `token_entropy_full.jsonl`
- sample-level `output_tokens`
- sample-level `latency_sec`

### VStar pure-soft status

- only a `50`-sample subset run is currently present:
  - `output/experiments/20260429/vstar_pure_soft_50_203818_setsid`
- no full `VStar pure_soft` experiment directory has been found yet

### Attention-analysis motivation

The next mechanism question is:

- in pure `cot`, when token entropy is high, is the model's attention to visual
  tokens abnormally weak or abnormally diffuse?

Current code already provides a good reuse point:

- `lead/generation_utils.py`
  - `_compute_dynamic_visual_anchor(...)`

That code already knows how to aggregate current-token attention over visual
tokens. The likely next clean implementation is to log visual-attention summary
statistics for each token during `cot`, rather than storing full attention
matrices.

## Experiment Launch Chain

### Standard runtime outputs

For all recent experiments, the minimum files to preserve are:

- `config.json`
- `results.jsonl`
- `eval_report.json`
- `token_entropy.jsonl`
- `token_entropy_full.jsonl` when available
- `nohup.log`
- `run_command.sh`

### Typical workflow

1. Launch a dataset-specific script under `script/`
2. Wait for `results.jsonl` and `token_entropy_full.jsonl`
3. If dataset is `MMVP` or `PhysUniBench`, run specialized re-evaluation
4. Run correct/wrong confidence summaries and optional plots
5. Update `log.md` and write summary notes into `result/`

### Stable launch pattern

Recent stable scripts use the environment Python path directly:

- `/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python`

This is preferred over `micromamba run -n mlrm-lead ...` because recent
parallel launches hit `mamba` proc-lock waiting.

### Existing useful launch scripts

- `script/run_lead_mmvp_vstar_parallel.sh`
- `script/run_lead_phys300_visulogic_parallel.sh`
- `script/run_cot_physunibench300_mmvp_parallel.sh`
- `script/run_pure_soft_physunibench300_mmvp_parallel.sh`
- `script/run_vstar_lead_paper_params.sh`

### Evaluation chain by dataset

#### MMVP

1. Run inference
2. Do **not** trust the default repository `eval_report.json`
3. Run:
   - `script/evaluate_specialized_results.py --mode mmvp`
4. Use `pair_accuracy` when reporting benchmark-style results

#### PhysUniBench

1. Run inference
2. Do **not** trust default `eval_report.json` alone
3. Run:
   - `script/evaluate_specialized_results.py --mode physunibench`
4. Inspect `failed_extraction` before interpreting accuracy

#### VStar / VisuLogic

1. Run inference
2. Default evaluator is usually usable
3. For confidence analysis, use:
   - `script/plot_pure_soft_correct_wrong_curves.py`

### Current immediate next task

- run full `VStar pure_soft`
- then use the resulting traces to compare against:
  - `VStar lead`
  - earlier `VStar pure_soft 50`
- after that, add token-level visual-attention summary logging for `cot`

## 2026-05-08: COT visual-attention summary logging completed

The COT path now supports token-level visual-attention summary logging, aimed at
checking whether high-entropy generation steps are associated with weak visual
grounding.

### Code changes

- `lead/generation_utils.py`
  - `generate_cot(...)` now optionally records per-token visual attention
    summaries.
  - added `_summarize_visual_attention(...)`
- `lead/inference.py`
  - passes visual-attention logging flags into `generate_cot(...)`
- `main.py`
  - added:
    - `--save_visual_attn_summary`
    - `--visual_attn_summary_last_k`

### Logged per-token fields

When `--save_visual_attn_summary` is enabled, each token in
`token_entropy_full.jsonl` can now include:

- `visual_attn_available`
- `visual_attn_mass`
- `visual_attn_top1`
- `visual_attn_top4_sum`
- `visual_attn_entropy`
- `visual_attn_token_count`

These are computed from the current generated token's decoder attention over
prompt visual tokens only, aggregated over the last `k` layers
(`--visual_attn_summary_last_k`, default `4`).

### Analysis script

Added:

- `script/analyze_cot_visual_attention_vs_entropy.py`

This script compares visual-attention statistics for:

- all eligible tokens
- high-entropy tokens by absolute threshold
- high-entropy tokens by per-sample top quantile

and splits results into:

- overall
- correct samples
- wrong samples

Useful flags:

- `--reasoning_only`
- `--exclude_nonword`

### Launch script

Added:

- `script/run_vstar_cot_visual_attn_full.sh`

This runs full `VStar` with:

- `method=cot`
- `--no-do_sample`
- `--save_token_entropy`
- `--save_full_token_entropy`
- `--save_visual_attn_summary`

### Recommended first target

Use `VStar` first for this mechanism analysis. It is cleaner than
`PhysUniBench` and better for asking whether high-entropy reasoning tokens are
looking at the image weakly, diffusely, or normally.

## 2026-05-12: VStar clean COT / visual reanchor / mean-anchor control

本阶段围绕 VStar 上的 COT 错题纠错做了几件事：先修正 clean COT 基线，再在 clean wrong subset 上比较 no-op、dynamic visual anchor 和 simple mean anchor。

### 1. Clean VStar COT 基线

此前带 attention logging 的 COT 路径会改变推理行为，因此不能作为“干净 COT”基线。后来补跑了不记录 attention 的 clean COT full：

- 目录：
  - `output/experiments/20260511_205609/vstar_cot_clean_full_gpu0`
- 结果：
  - `137/191 = 71.73%`
  - failed extraction: `0`

这个结果也解释了为什么论文中 VStar COT 可以到 60% 以上；之前 20% 多的结果来自污染过的 attention logging 路径，不应作为 clean baseline。

### 2. Clean wrong subset

基于 clean COT full 的错题构造了新子集：

- 文件：
  - `data/vstar_wrong_subset_from_cot_clean.jsonl`
- 大小：
  - `54`
- 定义：
  - clean COT full 中评估错误的样本
- 因此该子集上 clean COT baseline 为：
  - `0/54`

旧的 `data/vstar_wrong_subset_from_cot_visual_attn_rerun.jsonl` 有 `124` 条，但它来自带 attention logging 的污染基线，后续只能作为参考，不能作为主结论依据。

### 3. Visual reanchor 核心代码

主要代码入口：

- `lead/generation_utils.py`
  - `generate_cot_visual_reanchor(...)`
  - `_compute_dynamic_visual_anchor(...)`
  - `_summarize_visual_attention(...)`
- `lead/inference.py`
  - 当 `--method cot_visual_reanchor` 时路由到 `generate_cot_visual_reanchor`
- `main.py`
  - 增加 `cot_visual_reanchor` 方法和相关 CLI 参数

当前 low-visual reanchor 触发逻辑：

- `raw_entropy >= reanchor_entropy_threshold`
- `visual_attn_mass <= reanchor_visual_attn_threshold`
- step 在 `reanchor_min_step` 到 `reanchor_max_step` 内
- 未超过 `reanchor_max_trigger_count`
- cooldown 已结束

当前默认/常用参数：

- `reanchor_entropy_threshold = 1.0`
- `reanchor_visual_attn_threshold = 0.12`
- `reanchor_lambda = 0.15`
- `reanchor_top_m = 4`
- `reanchor_attn_last_k = 4`
- `reanchor_max_trigger_count = 1`
- `reanchor_cooldown = 32`

### 4. Dynamic visual anchor 的定义

在触发 token 上：

1. 取 decoder attention 的最后 `reanchor_attn_last_k` 层。
2. 对 head 平均。
3. 对层平均。
4. 只保留 prompt 中视觉 token 位置。
5. 从视觉 token 中按 attention 分数选 top-m。
6. 取这些视觉 token 在 prompt prefill 后的最后层 hidden states。
7. 用当前 raw probability 得到 soft embedding：
   - `soft_emb = raw_probs @ embedding_matrix`
8. 用 `selected_visual_states @ soft_emb / sqrt(hidden_size)` 计算 latent 权重。
9. 对 top-m 视觉 hidden states 做 softmax 加权求和，得到 dynamic anchor。
10. 将 anchor 混入下一步输入 embedding：
    - `next_emb = (1 - lambda) * next_emb + lambda * anchor`

注意：这里的视觉 hidden states 不是初始 token embedding，而是 prompt prefill 经过整个模型后的最后层 hidden states。

### 5. No-op / dynamic early 对照

目录：

- `output/experiments/20260511_214958/vstar_clean_wrong_subset_reanchor_noop_early_parallel`

结果：

| 设置 | 正确率 | 触发样本 | 触发样本修正 | 未触发样本修正 |
|---|---:|---:|---:|---:|
| no-op, `reanchor_max_trigger_count=0` | `13/54 = 24.07%` | `0/54` | `0` | `13` |
| dynamic early, `step <= 10` | `15/54 = 27.78%` | `25/54` | `9` | `6` |

重要 caveat：

- no-op 已经能把 `13/54` 改对。
- 说明 `cot_visual_reanchor` 路径本身和 clean COT 不完全等价。
- 可能来自：
  - 强制 eager attention
  - 使用 `inputs_embeds` 路径继续解码
  - cache / hidden-state 路径差异

因此，不能把 dynamic early 相对 clean COT 的全部收益都归因于 visual anchor。更合理的因果比较是 dynamic early vs no-op。

dynamic early 相对 no-op：

- 净增 `+2/54`
- dynamic-only 修正样本：
  - `[57, 60, 132]`
- no-op-only 修正样本：
  - `[9]`

### 6. Mean-anchor control

为验证当前 dynamic anchor 是否比简单平均合理，新增了参数：

- `--reanchor_anchor_mode`
  - `dynamic`: 原方法，top-m 后 latent soft embedding 加权
  - `mean`: 同样 top-m 视觉 token，但直接简单平均

新增脚本：

- `script/vstar_reanchor/run_vstar_clean_wrong_subset_mean_anchor_early.sh`

实验目录：

- `output/experiments/20260512_131349/vstar_clean_wrong_subset_mean_anchor_early_gpu0`

结果：

| 设置 | anchor 聚合 | 正确率 | 触发样本 |
|---|---|---:|---:|
| no-op | 不触发 | `13/54 = 24.07%` | `0/54` |
| dynamic early | top-m + latent 加权 | `15/54 = 27.78%` | `25/54` |
| mean early | top-m 简单平均 | `11/54 = 20.37%` | `25/54` |

样本重叠：

- `dynamic early ∩ mean early = 11`
- dynamic early 独有修正：
  - `[54, 60, 129, 167]`
- mean early 独有修正：
  - 无

结论：

- 在同样触发样本数和同样 early window 下，dynamic anchor 明显优于 simple mean anchor。
- 这支持当前“top-m 后再用 soft embedding latent 权重聚合”的设计比 naive mean 更合理。

### 7. 当前主要报告

详细报告写在：

- `result/vstar_wrong_subset_cot_visual_reanchor_report_zh.md`

报告中同时保留了旧 `124` 条 polluted wrong subset 的实验结果和新 `54` 条 clean wrong subset 的对照结果。后续写论文/总结时，应优先引用 clean wrong subset 的 no-op / dynamic / mean 对照。

### 8. 脚本目录整理

VStar / reanchor 相关脚本已移动到：

- `script/vstar_reanchor/`

当前包括：

- `run_vstar_cot_clean_full.sh`
- `run_vstar_pure_soft_full.sh`
- `run_vstar_cot_visual_attn_full.sh`
- `run_vstar_cot_visual_attn_full_rerun.sh`
- `prepare_vstar_wrong_subset.py`
- `analyze_cot_visual_attention_vs_entropy.py`
- `run_vstar_wrong_subset_cot_visual_reanchor.sh`
- `run_vstar_wrong_subset_cot_visual_reanchor_timing_parallel.sh`
- `run_vstar_clean_wrong_subset_noop_early_parallel.sh`
- `run_vstar_clean_wrong_subset_mean_anchor_early.sh`

这些脚本内部仍然显式 `cd` 到项目根目录，所以从新目录执行不影响运行。

### 9. 后续方向

较有价值的下一步：

1. 更干净地拆分实现路径因素：
   - 分别控制 eager attention、`inputs_embeds` continuation、cache 路径。
   - 目标是构造一个更接近 clean COT 的 no-op baseline。

2. high-entropy + high-visual-attention route：
   - low-visual route 当前是 visual reanchor。
   - 对 high entropy 且 visual attention 已经高的 token，可以尝试 soft embedding 隐式推理：
     - `next_emb = (1 - beta) * hard_emb + beta * soft_emb`
   - 这一路不应再注入视觉 anchor，因为模型此时已经在看图，问题可能更像语义/推理分叉。

3. 早期触发继续细分：
   - `step <= 5`
   - `step <= 10`
   - `step <= 20`
   - 或基于 entropy 突升、visual attention 下降趋势做自适应触发。

## 2026-05-14: Online sidecar attention 观测验证

本阶段解决了一个关键方法问题：直接在主推理路径中打开 attention 会污染推理，因此改成“在线旁路观测”。

### 1. COT 口径修正

当前项目已新增：

- `--cot_prompt_mode orign|step`

默认：

- `orign`

含义：

- `orign`: 对齐原项目，不额外追加 `Please think step by step...`
- `step`: 复现之前显式 step-by-step prompt 的实验口径

对齐原项目的 VStar COT full：

- 目录：
  - `output/experiments/20260513_210625/vstar_cot_orign_aligned_full_gpu0`
- 结果：
  - `120/191 = 62.83%`
- 与 `/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-orign` 原项目 COT full 完全一致：
  - 总正确数一致
  - correct set 一致
  - extracted answer 一致
  - `model_answer` 文本逐样本完全一致

因此后续若说“原版/论文口径 COT baseline”，应使用：

- `cot_prompt_mode=orign`
- `do_sample=True`
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`

之前的 `137/191 = 71.73%` 是：

- `cot_prompt_mode=step`
- `--no-do_sample`

不是原项目 COT 口径。

### 2. 在线旁路 attention 设计

目标：

- 主路径保持 clean COT
- 主 forward 不开 `output_attentions`
- 主路径不切 eager
- 当某个已生成 token 的 `raw_entropy` 超过阈值时，临时开一个旁路 replay：
  - 使用同一个模型
  - 临时切到 eager
  - replay 已固定的 `prompt + generated prefix`
  - 只记录该 token 对 prompt visual tokens 的 attention
  - 立刻恢复 attention implementation
  - 不改主路径 KV cache
  - 不参与下一 token 选择

新增参数：

- `--sidecar_attn_on_entropy`
- `--sidecar_attn_entropy_threshold`
- `--sidecar_attn_last_k`

代码位置：

- `lead/generation_utils.py`
  - `generate_cot(...)`
  - `_observe_sidecar_visual_attention(...)`
- `lead/inference.py`
  - 将 sidecar 参数传入 `generate_cot`
- `main.py`
  - CLI 和 config 记录

### 3. 在线旁路验证实验

实验：

- `output/experiments/20260514_151652/vstar_cot_orign_online_sidecar_attn_h2_gpu0`

配置：

- `method=cot`
- `cot_prompt_mode=orign`
- `sidecar_attn_on_entropy=True`
- `sidecar_attn_entropy_threshold=2.0`
- `save_token_entropy=True`
- `save_full_token_entropy=True`
- `save_visual_attn_summary=False`

结果：

- Accuracy: `120/191 = 62.83%`
- 与 clean aligned COT 完全一致
- `model_answer` 逐样本完全一致：`191/191`

旁路观测统计：

- 总生成 token: `40575`
- sidecar observed token: `4681`
- 覆盖样本: `174/191`
- sidecar error: `0`
- visual attention mass:
  - mean `0.1952`
  - median `0.1719`
  - p10 `0.0420`
  - p90 `0.3965`

结论：

> 在线 sidecar attention replay 可以记录高熵 token 的视觉注意力，并且在当前 VStar full 实验中没有污染主推理输出。

### 4. 后续重做实验优先级

现在应优先重做依赖 attention 的机制分析，使用在线 sidecar 而不是主路径 attention logging。

实验 1：

- 对齐原项目 COT full
- 在线 sidecar attention
- 记录高熵 token 的视觉 attention
- 分析：
  - `H >= 1.0`
  - `H >= 1.5`
  - `H >= 2.0`
  - correct vs wrong
  - direct_attributes vs relative_position
  - reasoning/content/boilerplate token
  - early/mid/late

实验 2：

- 基于 aligned COT `120/191` 构造 wrong subset，共 `71` 条
- 在 wrong subset 上做 sidecar trace 和机制分析

实验 3：

- 在 sidecar 观测不污染的前提下，重新设计 intervention：
  - 不再在开场模板 token 上触发
  - 优先用 entropy spike + semantic filter
  - anchor 注入要尽量小，并尽量避免长期 `inputs_embeds` continuation

## 2026-05-15: sidecar 新增两个视觉 grounding 指标

用户提出在在线 sidecar attention 里加入两个新观察量：

1. visual attention concentration
   - 先只在 image/visual tokens 内部归一化 attention。
   - 计算视觉注意力熵 `H_vis = -sum alpha_j log alpha_j`。
   - 用 `log(|V_img|)` 归一化后得到 `H_vis_norm`。
   - 集中度定义为 `C_vis = 1 - H_vis_norm`。
   - 含义：高值表示注意力集中在较少视觉 token 上，低值表示视觉 attention 分散。

2. hidden-state visual alignment
   - 取当前生成 token 的最后层 hidden state。
   - 取 prompt prefill 后视觉 token 的最后层 hidden states。
   - 计算当前 hidden state 与所有视觉 token hidden states 的 cosine similarity。
   - 记录 max cosine 和 top-4 mean cosine。
   - 含义：高值表示当前语言状态贴近某些视觉 token 表示，低值表示语言状态和视觉表征对齐弱。

已完成代码接入：

- `lead/generation_utils.py`
  - `_observe_sidecar_visual_attention(...)` 现在在 sidecar replay 时同时请求 `output_hidden_states=True`。
  - `_summarize_visual_attention(...)` 新增：
    - `sidecar_visual_attn_entropy_norm`
    - `sidecar_visual_attn_concentration`
  - 新增 `_summarize_hidden_visual_alignment(...)`，输出：
    - `sidecar_hidden_visual_align_max`
    - `sidecar_hidden_visual_align_top4_mean`
    - `sidecar_hidden_visual_align_token_count`
- `script/vstar_reanchor/analyze_online_sidecar_attention.py`
  - 分析表新增：
    - `conc_mean`
    - `align_max_mean`
    - `align_top4_mean`

验证：

- `python -m py_compile lead/generation_utils.py lead/inference.py main.py script/vstar_reanchor/analyze_online_sidecar_attention.py` 通过。
- 用小张量检查过 concentration 和 hidden alignment 的数值范围。
- 旧的 sidecar trace 不包含这些字段，需要重跑 sidecar 实验后才能分析新指标。

新增全量脚本：

- `script/vstar_reanchor/run_vstar_cot_online_sidecar_metrics_h1_full.sh`
  - 默认 `GPU_ID=1`
  - VStar full
  - `method=cot`
  - `cot_prompt_mode=orign`
  - `sidecar_attn_entropy_threshold=1.0`
  - 输出目录名：`vstar_cot_orign_online_sidecar_metrics_h1_gpu${GPU_ID}`
  - 运行目录内会生成 `analyze_after_done.sh`，用于跑完后产出 `sidecar_attention_metrics_analysis.md`

## 2026-05-20: 补充实验接手文档

用户指出当前实验脉络可以理解，但具体启动和路由实现主要由助手操作，存在离线后难以复现实验的问题。

已新增两份接手文档：

- `EXPERIMENT_RUNBOOK_zh.md`
  - 说明项目路径、模型路径、数据路径、脚本目录、输出目录。
  - 说明一次实验如何启动、如何查 pid/log/GPU、如何跑 `compare_after_done.sh`。
  - 记录代码入口：
    - `main.py`：命令行参数定义。
    - `lead/inference.py`：根据 `method` 把参数传给 generation 函数。
    - `lead/generation_utils.py`：真正实现 `generate_pure_soft`、`generate_lead`、`generate_lead_attenachor`。
  - 解释当前路由的共同机制：在 pure-soft 中通过 `route_mask` 决定下一步输入用 `normal_emb = E[next_token]` 还是 `soft_emb = probs_original @ E`。
  - 逐个说明：
    - low-confidence diffuse collapse
    - format cooldown
    - cooldown2 + late64_repeat_gate
    - answer_zone_discrete
    - LEAD soft veto
  - 给出新增路由的推荐修改路径和脚本复制模板。

- `script/exp5_16/README.md`
  - 说明当前脚本目录下每个脚本的用途。
  - 记录常用启动、检查、比较命令。
  - 说明不同类型新实验应该复制哪个脚本。

当前正在跑的两个实验也已写入 runbook：

- `cooldown2 + late64_repeat_gate`
  - `output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0`
  - PID `2825516`
- `answer_zone_discrete`
  - `output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full/answer_zone_discrete_gpu1`
  - PID `2828661`

## 2026-05-30: LEAD 开头 transition 消融与跨数据集验证

最新检查发现，标准 LEAD 在 VStar 上平均每个样本实际触发 soft 介入次数很少，约为 `1.7` 次/样本；同时 LEAD 固定在每个样本第 0 步先走一次 soft / transition 机制。因此本轮开始验证一个新的关键假设：

> LEAD 的主要收益可能并不来自后续稀疏动态 soft trigger，而来自 generation 开头 latent / soft transition 对整条 reasoning trajectory 的初始化影响。

新增和使用的主要实验目录：

- `output/experiments/20260529_163618/vstar_lead_cot_sanity_matrix`
- `output/experiments/20260529_225807/vstar_lead_soft_quota_sweep`
- `output/experiments/20260530_013153/cross_dataset_lead_transition_quota`

VStar sanity matrix：

- `cot_orign_greedy`: `131/191 = 68.59%`
- `lead_force_normal`: `131/191 = 68.59%`
- `initial_soft_only`: `132/191 = 69.11%`
- `initial_transition_only`: `138/191 = 72.25%`
- `initial_transition_only_no_anchor`: `138/191 = 72.25%`
- `lead`: `139/191 = 72.77%`

关键结论：

- `lead_force_normal` 与 COT 同分，说明去掉 soft / transition 后 LEAD 路径本身没有额外收益。
- `initial_soft_only` 只有 `132/191`，说明“只第 0 步 soft”不是主要收益来源。
- `initial_transition_only` 只比 full LEAD 少 1 题，且 `no_anchor` 版本同分，说明主要收益来自开头 transition，而不是 simple visual anchor。

跨数据集结果：

- RealWorldQA fixed:
  - `lead`: `129/200 = 64.50%`
  - `initial_transition_only`: `127/200 = 63.50%`
  - `quota20`: `126/200 = 63.00%`
  - `quota05_guard`: `134/200 = 67.00%`
- VisuLogic300:
  - `lead`: `74/300 = 24.67%`
  - `initial_transition_only`: `85/300 = 28.33%`
  - `quota20`: `69/300 = 23.00%`
  - `quota05_guard`: `67/300 = 22.33%`
- MMVP:
  - `lead`: `211/300 = 70.33%`, pair acc `42.00%`
  - `initial_transition_only`: `211/300 = 70.33%`, pair acc `42.00%`
  - `quota20`: `205/300 = 68.33%`, pair acc `42.67%`
  - `quota05_guard`: `211/300 = 70.33%`, pair acc `42.67%`

逐样本翻转结论：

- MMVP 上 `initial_transition_only` 与 full LEAD item-level 和 pair-level 都完全一致，fixed/damaged 都是 `0/0`。
- RealWorldQA fixed 上 `initial_transition_only` 相对 full LEAD 为 fixed `1`、damaged `3`、net `-2`；`quota05_guard` 为 fixed `11`、damaged `6`、net `+5`。
- VisuLogic300 上 `initial_transition_only` 相对 full LEAD 明显正向，默认评估多 11 题；轻量逐样本复核约 fixed `40`、damaged `27`、net `+13`。收益主要来自 Attribute / Other，Spatial Reasoning 有下降。

当前方法含义：

```text
不是：
高熵 / 不确定 token 上频繁动态 soft 介入 -> 性能提升

更像是：
generation 起始阶段的一次 latent transition -> 改变后续整条 reasoning trajectory
```

后续建议：

1. 优先拆解 `initial_transition` 本身，例如 `k1/k2/k4/k8`、`no_anchor`、`random/mean embedding control`、是否需要 entropy 条件。
2. 后续 soft quota 只保留非常保守的版本观察，例如 `initial_transition + quota05_guard`。
3. 不建议继续优先扫大 quota；`quota20` 在 RealWorldQA、MMVP item、VisuLogic 上都不稳定。
4. 当前发现已经整理成独立报告：`result/5-27/lead_initial_transition_cross_dataset_20260530.md`。
## 2026-06-02 Early Trajectory Commitment 机制优先重跑

本轮按新的主线重跑计划落地：验证多模态推理轨迹是否在生成极早期被锁定，以及 LEAD 的主要收益是否来自开头 `soft -> normal` transition，而不是中后段 entropy-gated 动态触发。

代码改动：
- 新增 CLI/config/生成透传参数：`--lead_initial_transition_delay_steps N`。
- `N=0` 等价现有 `initial_transition_only`。
- `N>0` 时先正常 hard decoding N 个生成 step，在第 N 步插入一次 soft latent 扰动，并在第 N+1 步执行对应的 `soft -> normal` transition，之后继续 normal。
- `token_entropy_full.jsonl` 的 trace 增加 `lead_initial_transition_delay_steps`、`lead_delayed_transition_entry`、`lead_delayed_transition_exit`、`to_normal`、`to_soft` 字段，便于后续 early-token divergence 分析。

新增脚本：
- `script/exp5_27/run_rerun_early_path_dependence_mechanism.sh`
- `script/exp5_27/summarize_rerun_early_path_dependence.py`
- `script/exp5_27/analyze_early_token_divergence.py`

实验矩阵：
- Phase 1：VStar clean component controls，包含 COT、LEAD force normal、full LEAD、initial soft、initial transition、no_to_normal、no_linebreak、no_anchor、no_linebreak_no_to_normal。
- Phase 2：VStar timing curve，`transition_step0/1/2/4/8/16/32`。
- Phase 2 cross projection：把 `step0/4/16/32` 外推到 MMVP、VisuLogic300、RealWorldQA fixed200。
- Phase 3：VStar/MMVP/VisuLogic300/RealWorldQA fixed200 的最小跨数据集矩阵，包含 COT、force normal、full LEAD、initial soft、initial transition、no_to_normal、no_anchor、quota05_guard。

输出根目录：
`output/experiments/20260602_220321/rerun_early_path_dependence_mechanism`

启动状态：
- GPU0 queue PID: `2950179`
- GPU1 queue PID: `2950180`
- 当前首批 run 已启动：GPU0 `cot_orign_greedy`，GPU1 `lead_force_normal`。
- 每个 run 启动前都会执行 `python -m py_compile main.py lead/inference.py lead/generation_utils.py`。
- 汇总命令：`bash output/experiments/20260602_220321/rerun_early_path_dependence_mechanism/compare_after_done.sh`

预期输出：
- `summary.json`
- `summary.md`
- `pairwise_deltas.json`
- MMVP 每个 run 的 `specialized_eval_report.json`
- RealWorldQA fixed200 每个 run 的 `realworldqa_mcq_eval.json`
- `early_token_divergence.md`


## 2026-06-04 格式稳定与置信度扩散 Guard 重跑

目标：把此前的 ormat_cooldown2 与 confidence-diffusion / late64 repeat veto 按 early trajectory commitment 同样的控制变量风格重跑，判断它们是主机制还是退化修复 guard。

计划文档：
esult/5-27/format_confidence_diffusion_rerun_plan_20260604.md

新增 runner：script/exp5_27/run_rerun_format_confidence_diffusion_guard.sh

正式输出目录：output/experiments/20260604_131704/rerun_format_confidence_diffusion_guard

矩阵：
- Phase 1 cross-dataset guard：VStar / MMVP / VisuLogic300 / RealWorldQA fixed200，每个数据集 11 个 run：COT、force normal、full LEAD、initial_transition_only、lead_format2、lead_diffuse_veto、lead_guard、quota05、quota05_format2、quota05_diffuse_veto、quota05_guard。
- Phase 2 VStar pure-soft guard：pure_soft、pure_soft_format2、pure_soft_diffuse_collapse、pure_soft_guard。

关键口径：
- 不保存 --save_full_token_entropy，避免长 trace 写入触发 OSError: [Errno 7] Argument list too long。
- 保留 --save_token_entropy --trace_topk 20，主表和 mean soft ratio 仍可汇总。
- 汇总命令：ash output/experiments/20260604_131704/rerun_format_confidence_diffusion_guard/compare_after_done.sh

启动状态：
- GPU0 queue PID: 3082450
- GPU1 queue PID: 3082451
- 首批 run：GPU0 star/cot_orign_greedy，GPU1 star/lead_force_normal。
- 当前仍处于模型加载/共享存储 I/O 阶段，尚未写出结果。

## 2026-06-06 Guard 补跑与两条调参 sweep 启动

用户要求：先把 20260604 guard 重跑中未完成的实验补完，然后启动两个调参方向：
1. early transition delay refine；
2. quota ratio / quota+format2 sweep。

Guard 补跑目录：`output/experiments/20260604_131704/rerun_format_confidence_diffusion_guard`

补跑策略：
- 清理 26 个 missing run 的 partial `results/eval/token_entropy` 文件。
- 先并行补 RealWorldQA fixed200 与 VStar pure-soft guard。
- 再单进程串行补 VisuLogic300，避免两张卡同时跑 VisuLogic 再次触发系统 `Killed`。
- Guard 补完后自动执行：`bash output/experiments/20260604_131704/rerun_format_confidence_diffusion_guard/compare_after_done.sh`

接力脚本：`output/experiments/20260604_131704/rerun_format_confidence_diffusion_guard/continue_guard_then_start_tuning.sh`

当前后台状态：
- master PID: `3216578`
- nonvis GPU0 queue PID: `3216584`
- nonvis GPU1 queue PID: `3216585`
- 当前首批补跑：RealWorldQA fixed200 `cot_orign_greedy_gpu0` 与 `lead_force_normal_gpu1`，已进入 CUDA。

调参脚本：`script/exp5_27/run_tune_transition_delay_quota.sh`

调参矩阵：
- Direction 1：transition delay refine，补 MMVP/RealWorldQA/VisuLogic 的 `transition_step1/step2`。
- Direction 2：quota ratio sweep，在 VStar/MMVP/RealWorldQA 上跑 `0.02/0.03/0.05/0.08` 及对应 `+format2`。
- 调参输出目录将在 guard 补跑完成后由脚本自动生成，形如 `output/experiments/<STAMP>/tune_transition_delay_quota_format`。

## 2026-06-15 Early Prefix Replay 因果实验

- 在 `gpu11` 完成 early prefix replay 实验，输出目录：`output/experiments/20260615_early_prefix_replay/early_prefix_replay`。
- 实验对象是 COT 错、`initial_transition_only` 对的样本：VStar 18 个，MMVP 12 个。
- 方法：强制生成前 `8/16/32/64` 个 token 分别来自 COT 错误轨迹或 initial-transition 修复轨迹，然后切回普通 greedy COT 继续生成。
- 结果：COT prefix 基本锁死错误轨迹，VStar 四个长度均为 `1/18`，MMVP 四个长度均为 `0/12`；initial-transition prefix 随长度增加明显恢复正确率，VStar 到 64 token 为 `14/18`，MMVP 到 32/64 token 分别为 `10/12`、`11/12`。
- 结论：这比 early token divergence 更接近因果证据，说明 early transition 不是单纯格式修复，而是在前 32-64 token 内重定向了可延续的视觉推理轨迹；后续普通 greedy 会沿着早期轨迹继续展开。
- 详细报告：`result/5-27/early_prefix_replay_report_20260615.md`。

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

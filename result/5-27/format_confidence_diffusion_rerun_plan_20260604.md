# 格式稳定与置信度扩散 Guard 重跑计划

## 目标

验证此前的两个工程性发现：

1. `format_cooldown2` 主要是在格式边界附近稳定输出，降低 pure-soft/soft 退化风险。
2. `late64_repeat_gate` / confidence-diffusion veto 主要是在低置信、margin 小、局部 entropy spike 且出现重复退化时，把 soft 退回 hard embedding。

这轮要判断它们是否是新的主机制，还是只是在 early transition 主机制之外的“退化修复 guard”。

## 固定口径

- 模型：`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`
- 输出根目录：`output/experiments/<STAMP>/rerun_format_confidence_diffusion_guard`
- 生成参数：`--no-do_sample --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --save_token_entropy --trace_topk 20`
- 不保存 `--save_full_token_entropy`，避免长 trace 在部分样本上触发 `OSError: [Errno 7] Argument list too long`。
- 数据集：
  - VStar full: `data/vstar.jsonl`
  - MMVP full: `data/mmvp.jsonl`
  - VisuLogic300: `data/visulogic.jsonl --limit 300`
  - RealWorldQA fixed200: `data/realworldqa_fixed_mcq_random200_seed42.jsonl`

## 方法定义

- `format2`: `--lead_format_cooldown --format_cooldown_steps 2`
- `diffuse_veto`: `--lead_soft_veto_on_diffuse --lead_veto_entropy_window 16 --lead_veto_entropy_alpha 2.0 --lead_veto_min_history 4 --lead_veto_min_entropy 1.0 --lead_veto_low_conf_tau 0.20 --lead_veto_low_margin_tau 0.05 --lead_veto_min_step 64 --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 32 --lead_veto_recent_repeat_tau 0.35`
- `guard`: `format2 + diffuse_veto`
- `quota05_*`: 在 `--lead_soft_quota_ratio 0.05` 上叠加对应 guard。

## Phase 1：VStar guard component 复核

目的：先确认 guard 的贡献是否来自 dynamic LEAD/后续 soft，而不是 early transition。

Runs:

- `cot_orign_greedy`
- `lead_force_normal`
- `lead`
- `initial_transition_only`
- `lead_format2`
- `lead_diffuse_veto`
- `lead_guard`
- `quota05`
- `quota05_format2`
- `quota05_diffuse_veto`
- `quota05_guard`
- `pure_soft`
- `pure_soft_format2`
- `pure_soft_diffuse_collapse`
- `pure_soft_guard`

预期：

- `initial_transition_only` 仍应解释 VStar 大部分收益。
- `lead_format2/diffuse/guard` 若收益小于 `initial_transition_only`，说明 guard 不是主机制。
- `quota05_guard` 若 fixed > damaged 且长度/failed 不恶化，可作为辅助路线。
- pure-soft guard 若显著降低 maxed/long/failed，则说明它是在修退化，而不是解释 LEAD 收益。

## Phase 2：跨数据集 guard 最小矩阵

每个数据集跑：

- `cot_orign_greedy`
- `lead_force_normal`
- `lead`
- `initial_transition_only`
- `lead_format2`
- `lead_diffuse_veto`
- `lead_guard`
- `quota05`
- `quota05_format2`
- `quota05_diffuse_veto`
- `quota05_guard`

评估：

- VStar/VisuLogic：default evaluator + by_subtopic
- MMVP：specialized evaluator + sample acc + pair acc
- RealWorldQA：fixed MCQ evaluator

主表报告：

- accuracy
- fixed/damaged against COT
- fixed/damaged against full LEAD
- output length mean
- long>=256
- maxed 1024
- failed extraction
- mean soft ratio
- MMVP pair accuracy
- VStar/VisuLogic by_subtopic

## 判断标准

- 如果 `lead_guard` 不能稳定超过 `initial_transition_only`，guard 不应被写成主机制。
- 如果 `quota05_guard` 在 RealWorldQA/VStar 上 fixed > damaged，但在 VisuLogic/MMVP 不稳定，则它是 dataset-specific conservative auxiliary。
- 如果 `diffuse_veto` 单独几乎无效，但和 quota/format 组合有效，则说明 diffusion 方法主要是“限制后续 soft 的坏尾部”。
- 如果 `format2` 只改善长度、maxed、failed，而 accuracy 不稳定，则它属于格式稳定 guard。


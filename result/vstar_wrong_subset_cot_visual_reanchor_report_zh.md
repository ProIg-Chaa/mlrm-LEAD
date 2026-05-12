# VStar Wrong Subset: COT Visual Reanchor 阶段报告

日期：2026-05-11

## 1. 问题背景

前面对 `VStar COT` 的 token 级分析显示：

- 高熵 reasoning tokens 整体上并不是更少看图，而是更强、更集中地关注视觉 token。
- 但错题样本中的高熵 tokens，其 `visual_attn_mass` 明显低于对题样本。

因此，本阶段实验验证一个具体假设：

> 当模型处于高熵且视觉注意力偏低的局部状态时，注入少量视觉 anchor，是否能把原本做错的样本拉回正确答案。

## 2. 数据与基线

错误子集来自：

- 基线结果：
  - `output/experiments/20260509_110659/vstar_cot_visual_attn_full_rerun_gpu0/results.jsonl`
- 子集文件：
  - `data/vstar_wrong_subset_from_cot_visual_attn_rerun.jsonl`

子集定义：

- 只保留基线 `cot` 中 **非 OOM 且评估为 wrong** 的样本
- 不包含 `error_type` 样本

子集大小：

- `124`

因此，该子集上的原始 `cot` 基线为：

- `0 / 124`

注意：这是按子集定义得到的纠错基线，不是全量 VStar COT accuracy。

## 3. 方法配置

新增方法：

- `cot_visual_reanchor`

核心逻辑：

- 在生成过程中记录当前 token 的 `raw_entropy` 和视觉注意力摘要。
- 当满足 `高熵 + 低视觉注意力` 条件时，构造动态 visual anchor。
- 将 visual anchor 以小系数混入下一步输入 embedding。

默认触发条件：

- `raw_entropy >= 1.0`
- `visual_attn_mass <= 0.12`

默认干预参数：

- `reanchor_lambda = 0.15`
- `reanchor_top_m = 4`
- `reanchor_attn_last_k = 4`
- `reanchor_max_trigger_count = 1`
- `reanchor_cooldown = 32`

生成参数：

- `max_new_tokens = 1024`
- `temperature = 0.6`
- `top_p = 0.95`
- `top_k = 20`
- `seed = 42`
- `--no-do_sample`

## 4. 实验目录

不限触发时机：

- `output/experiments/20260510_115145/vstar_wrong_subset_cot_visual_reanchor_gpu0`

触发时机对照：

- `early`: `output/experiments/20260510_155236/vstar_wrong_subset_cot_visual_reanchor_timing_parallel/early_step_le_10_gpu0_rerun`
- `mid`: `output/experiments/20260510_155236/vstar_wrong_subset_cot_visual_reanchor_timing_parallel/mid_step_11_30_gpu1`
- `late`: `output/experiments/20260510_155236/vstar_wrong_subset_cot_visual_reanchor_timing_parallel/late_step_ge_31_gpu0_rerun`

说明：

- 原始 `early_step_le_10_gpu0` 和 `late_step_ge_31_gpu0` 曾因 GPU 0 并发导致 OOM 污染，后续分析只使用 `*_rerun` 目录。

## 5. 主结果

| 设置 | 触发窗口 | 正确率 | 触发样本 | 触发样本修正 | 未触发样本修正 |
|---|---:|---:|---:|---:|---:|
| 原始 COT wrong subset | 无干预 | `0/124` | - | - | - |
| all_window | 不限时机 | `40/124 = 32.26%` | `123` | `40` | `0` |
| early | `step <= 10` | `43/124 = 34.68%` | `64` | `25` | `18` |
| mid | `11 <= step <= 30` | `41/124 = 33.06%` | `100` | `35` | `6` |
| late | `step >= 31` | `38/124 = 30.65%` | `113` | `37` | `1` |

主结论：

- `early` 最好，修正 `43` 个样本。
- `late` 最差，修正 `38` 个样本。
- `early` 的触发数明显少于 `all_window`，但正确率更高。

这说明当前结果不是“触发越多越好”，而是早期窗口更可能命中可修复阶段。

## 6. 触发效率

触发样本内的修正率：

- `all_window`: `40 / 123 = 32.52%`
- `early`: `25 / 64 = 39.06%`
- `mid`: `35 / 100 = 35.00%`
- `late`: `37 / 113 = 32.74%`

`early` 的触发效率最高。

第一触发点统计：

| 设置 | mean step | median step | mean entropy | mean visual_attn_mass |
|---|---:|---:|---:|---:|
| all_window | `15.42` | `8` | `2.18` | `0.0827` |
| early | `4.58` | `4` | `2.67` | `0.0819` |
| mid | `20.81` | `22` | `1.75` | `0.0855` |
| late | `50.69` | `35` | `2.06` | `0.0813` |

解释：

- `early` 触发点更早，同时 entropy 更高。
- `late` 触发虽然仍然符合低视觉注意力条件，但修复率下降。
- 这支持“错误轨迹早期更可逆，后期更难救回”的解释。

## 7. 修复类型

原始 wrong subset 中，错误主要分两类：

- `baseline_failed_extraction`: 原始 COT 没有可抽取答案，共 `95`
- `baseline_wrong_letter`: 原始 COT 给了错误字母，共 `29`

各设置修正数量：

| 设置 | failed extraction 修正 | wrong letter 修正 |
|---|---:|---:|
| all_window | `35/95` | `5/29` |
| early | `37/95` | `6/29` |
| mid | `37/95` | `4/29` |
| late | `34/95` | `4/29` |

判断：

- 收益主要来自修复 failed extraction 或坏轨迹收束问题。
- 也有少量明确的答案翻转。
- `early` 在两类错误上都略优。

## 8. 子类表现

按 VStar 子类：

| 设置 | direct_attributes | relative_position |
|---|---:|---:|
| all_window | `20/63` | `20/61` |
| early | `21/63` | `22/61` |
| mid | `21/63` | `20/61` |
| late | `18/63` | `20/61` |

`early` 对两个子类都不差，其中 `relative_position` 提升更明显。

## 9. 样本重叠

不同设置修正的样本并不完全相同。

重叠数量：

- `all_window ∩ early = 32`
- `all_window ∩ mid = 29`
- `all_window ∩ late = 23`
- `early ∩ mid = 22`
- `early ∩ late = 27`
- `mid ∩ late = 22`

额外现象：

- `early` 相比 `all_window` 多修正 `11` 个样本。
- `all_window` 相比 `early` 多修正 `8` 个样本。
- 四种设置合并后，共有 `67` 个不同样本曾被至少一种设置修正。

这说明不同触发窗口修复的是部分不同的错误模式。后续可以考虑自适应时机，而不是只选固定窗口。

## 10. Caveat: 需要 no-op 对照

一个重要 caveat 是：

- `early` 中有 `18` 个未触发样本也变对了。
- `mid` 中有 `6` 个未触发样本也变对了。
- `late` 中有 `1` 个未触发样本也变对了。

这说明当前 `cot_visual_reanchor` 与原始 `generate_cot` 并非严格只在 `reanchor_triggered=True` 时才改变行为。

可能来源包括：

- `cot_visual_reanchor` 使用 `inputs_embeds` 路径继续解码
- 强制 eager attention
- cache / hidden-state 路径细节与原始 COT 不完全一致

因此，目前不能把全部收益都归因于 visual anchor 注入本身。

下一步必须补一个 no-op reanchor baseline：

- 使用同一个 `cot_visual_reanchor` 代码路径
- 但禁止触发，例如：
  - `--reanchor_max_trigger_count 0`
  - 或 `--reanchor_entropy_threshold 999`

如果 no-op 仍能修很多题，说明收益里有实现路径因素；如果 no-op 接近 `0/124`，则 early reanchor 的因果证据会更强。

## 11. 当前阶段判断

现有结果支持以下较稳妥结论：

1. `cot_visual_reanchor` 在 VStar wrong subset 上确实能修正一批原始 COT 错题。
2. 主要收益来自修复 failed extraction / 坏轨迹收束。
3. 少量样本存在真实答案翻转。
4. 早期触发窗口 `step <= 10` 当前表现最好。
5. 晚期触发 `step >= 31` 表现最弱，说明等轨迹跑远后再补视觉，修复能力下降。
6. 当前仍需 no-op baseline 来排除实现路径差异。

最保守的一句话总结：

> 在 VStar COT 错题子集上，高熵低视觉注意力条件下的视觉 reanchor 有纠错信号；其中早期触发更有效，但收益归因还需要 no-op 对照进一步确认。

## 12. 下一步实验建议

优先级最高：

1. `cot_visual_reanchor_noop`
   - `reanchor_max_trigger_count = 0`
   - 同样跑 `124` 个 wrong subset

2. early 窗口细分
   - `step <= 5`
   - `step <= 10`
   - `step <= 20`

3. 门控收紧
   - `raw_entropy >= 1.5`
   - `visual_attn_mass <= 0.10`

4. 趋势触发
   - 熵突升，而不是绝对高熵
   - 视觉注意力下降，而不是绝对低视觉注意力

如果 no-op 对照确认收益主要来自真实 anchor 注入，后续应重点做 early-window 内的精细门控。

## 13. Clean COT 基线后的 no-op / dynamic / mean 对照

日期：2026-05-12

前面 `124` 条 wrong subset 来自带 attention logging 的 COT 路径，后来确认该路径会污染原始 COT accuracy。因此补了一轮 clean COT 全量基线：

- clean VStar COT full:
  - `output/experiments/20260511_205609/vstar_cot_clean_full_gpu0`
  - `137/191 = 71.73%`
- clean COT wrong subset:
  - `data/vstar_wrong_subset_from_cot_clean.jsonl`
  - 共 `54` 条
  - 该子集上 clean COT 由定义为 `0/54`

在这 `54` 条 clean wrong subset 上，使用相同 `cot_visual_reanchor` 解码路径做三组对照：

| 设置 | anchor 聚合 | 触发窗口 | 正确率 | official failed_extraction | 触发样本 | 触发样本修正 | 未触发样本修正 |
|---|---|---:|---:|---:|---:|---:|---:|
| no-op | 不触发 | `0-1000000` | `13/54 = 24.07%` | `2` | `0/54` | `0` | `13` |
| dynamic early | top-m + latent 加权 | `0-10` | `15/54 = 27.78%` | `2` | `25/54` | `9` | `6` |
| mean early | top-m 简单平均 | `0-10` | `11/54 = 20.37%` | `2` | `25/54` | `5` | `6` |

实验目录：

- no-op / dynamic early:
  - `output/experiments/20260511_214958/vstar_clean_wrong_subset_reanchor_noop_early_parallel`
- mean early:
  - `output/experiments/20260512_131349/vstar_clean_wrong_subset_mean_anchor_early_gpu0`

样本重叠：

- `dynamic early ∩ no-op = 12`
- `mean early ∩ no-op = 9`
- `dynamic early ∩ mean early = 11`
- dynamic early 独有修正：`[54, 60, 129, 167]`
- mean early 独有修正：无

相对 no-op：

- dynamic early 净增 `+2/54`，其中新增修正 `[57, 60, 132]`，但丢失 `[9]`
- mean early 净减 `-2/54`，新增修正 `[57, 132]`，但丢失 `[9, 54, 129, 167]`

判断：

1. no-op 已经能把 `13/54` 改对，说明 `cot_visual_reanchor` 解码路径本身仍然不是完全干净的“只注入才改变”对照。
2. 在同一触发集合上，dynamic anchor 明显优于 mean anchor：
   - dynamic early: `15/54`
   - mean early: `11/54`
3. 三组 official failed_extraction 都是 `2`，所以本轮主要差异来自答案是否正确，而不是报错/OOM 数量。
4. 因此，当前 top-m 后再用 latent soft embedding 做加权的设计是有价值的；简单平均并没有验证出更稳的效果，反而破坏更多输出轨迹。

更保守的一句话结论：

> 在 clean VStar COT 错题子集上，视觉 reanchor 的总收益仍受解码路径差异影响；但在相同触发条件下，dynamic visual anchor 优于简单均值 anchor，支持当前 anchor 构造方式比 naive mean 更合理。

## 14. 脚本整理

本阶段 VStar / reanchor 相关脚本已收进：

- `script/vstar_reanchor/`

包括：

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

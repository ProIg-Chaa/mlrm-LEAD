# Format Cooldown 过度干预验证实验

日期：2026-05-21

实验目录：

`output/experiments/20260521_191535/format_overintervention_gates_mmvp_visulogic`

## 实验目的

在跨数据集实验中，`bestcombo = format_cooldown2 + late64_repeat_gate` 在 MMVP 和 VisuLogic 上接近但没有超过 LEAD。此前分析发现，尤其在 VisuLogic 上，`bestcombo` 的 format cooldown 触发非常频繁：

- MMVP：平均 `fmt_active = 41.4`
- VisuLogic-300：平均 `fmt_active = 251.6`

因此需要验证一个假设：

> `bestcombo` 在 MMVP / VisuLogic 上不如 LEAD，是否是因为 format cooldown 对正常推理 token 过度干预？

本实验通过给 format cooldown 增加不确定性 gate，减少高置信格式 token 上的离散化干预。

## 实验设置

基线：

- `pure_soft`
- `LEAD`
- `bestcombo`

测试 gate：

| 名称 | 触发条件 |
|---|---|
| `gate_entropy10` | format token 只有 `raw_entropy >= 1.0` 才触发 cooldown |
| `gate_top080_margin040` | format token 只有 `raw_top1 <= 0.80` 或 `margin <= 0.40` 才触发 |
| `gate_strict` | format token 只有 `raw_entropy >= 1.5` 或 `top1 <= 0.60` 或 `margin <= 0.25` 才触发 |

其他设置保持 bestcombo 不变：

- `--pure_soft_format_cooldown --format_cooldown_steps 2`
- `--pure_soft_collapse_on_diffuse`
- `--collapse_min_step 64`
- `--collapse_require_repeat_degen`
- `--collapse_repeat_ngram 3`
- `--collapse_recent_repeat_tau 0.35`

## MMVP 结果

| 方法 | sample acc | pair acc | fmt active/样本 | fmt token/样本 | maxed | failed |
|---|---:|---:|---:|---:|---:|---:|
| pure_soft | 183/300 = 61.00% | 48/150 = 32.00% | 0.0 | 0.0 | 29 | 28 |
| LEAD | 211/300 = 70.33% | 63/150 = 42.00% | 0.0 | 0.0 | 0 | 0 |
| bestcombo | 201/300 = 67.00% | 60/150 = 40.00% | 41.4 | 23.7 | 4 | 1 |
| gate_entropy10 | 194/300 = 64.67% | 56/150 = 37.33% | 12.2 | 6.2 | 7 | 5 |
| gate_top080_margin040 | 192/300 = 64.00% | 55/150 = 36.67% | 18.2 | 9.3 | 9 | 6 |
| gate_strict | 195/300 = 65.00% | 58/150 = 38.67% | 12.0 | 6.1 | 6 | 4 |

结论：MMVP 上减少 format cooldown 干预后，准确率和 pair accuracy 都下降。`bestcombo` 的强 format cooldown 是有效的稳定器，不是主要负担。

## VisuLogic-300 结果

| 方法 | acc | fmt active/样本 | fmt token/样本 | maxed | failed_real |
|---|---:|---:|---:|---:|---:|
| pure_soft | 53/300 = 17.67% | 0.0 | 0.0 | 100 | 91 |
| LEAD | 74/300 = 24.67% | 0.0 | 0.0 | 7 | 29 |
| bestcombo | 73/300 = 24.33% | 251.6 | 155.8 | 26 | 18 |
| gate_entropy10 | 53/300 = 17.67% | 40.2 | 20.5 | 33 | 23 |
| gate_top080_margin040 | 66/300 = 22.00% | 63.7 | 32.9 | 30 | 26 |
| gate_strict | 67/300 = 22.33% | 36.2 | 18.5 | 32 | 30 |

结论：VisuLogic 上虽然 bestcombo 的 format 干预非常频繁，但简单削弱干预并没有提升，反而从 `24.33%` 降到 `17.67%-22.33%`。

这说明大量 format cooldown 并非单纯噪声。它显著减少了 pure-soft 的长输出和答案抽取失败：

- pure_soft：`maxed = 100`，`failed_real = 91`
- bestcombo：`maxed = 26`，`failed_real = 18`
- gate 后：`maxed = 30-33`，`failed_real = 23-30`

## 与 VStar 既有结果的一致性

这个结论和 VStar 上此前的消融现象一致。

VStar 中：

- `cooldown2` 单独已经很强；
- `cooldown2 + late64_repeat_gate` 达到当前 bestcombo；
- 削弱 format cooldown 的变体，例如 highrisk-only、min_step32、normal/highrisk 分步、mixed lambda、budget cap 等，都没有超过 bestcombo。

因此三个数据集共同显示：

> 当前 format cooldown 不是“过度干预导致性能下降”的主要原因；相反，它是 pure-soft 路由中最关键的稳定化机制之一。

## 当前解释

`format_cooldown2` 的作用并不只是修最终答案格式。它在 soft latent reasoning 中承担了更广泛的稳定功能：

1. 抑制 soft embedding 的长输出退化；
2. 降低答案抽取失败；
3. 在标点、括号、answer/option 等结构 token 附近把推理状态拉回离散轨道；
4. 与 `late64_repeat_gate` 配合，减少后期重复或低置信扩散。

因此，不能简单按“format active 太多”来判断它有害。对于 MMVP 和 VisuLogic，减少 format active 后反而损失性能。

## 下一步含义

这组实验把一个方向基本排除掉：

> 简单削弱 format cooldown 不是提升 bestcombo 跨数据集表现的好方向。

下一步更合理的方向是保留强 format cooldown，并补充 LEAD 中缺失的视觉先验：

- 使用类似原始 LEAD 的轻量 `<|image_pad|>` bias；
- 避免强 dynamic visual anchor；
- 将视觉 bias 作为小强度、低成本、非 attention 的补充；
- 重点比较 `bestcombo` 与 `bestcombo + lightweight image_pad bias`。

当前结论：

> bestcombo 和 LEAD 的差距，不主要来自 format cooldown 过度干预，而更可能来自 LEAD 的简单视觉 anchor 和 soft/normal 动态切换机制。format cooldown 应继续保留为核心稳定器。

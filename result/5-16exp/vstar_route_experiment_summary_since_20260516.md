# 2026-05-16 以来 VStar 路由实验阶段总结

本文整理 2026-05-16 以来围绕 VStar、pure-soft、LEAD、format cooldown、低置信扩散坍缩、多信号混合等实验的主要脉络。主体只总结已经完成并分析过的实验；正在运行的 budget cap 实验不计入最终结论。

## 1. 总体问题

本轮实验最初的问题是：

> 在 VStar / MMVP 等视觉推理数据集上，模型出错时是否更自信？高熵 token 是否和视觉关注不足有关？能否在高熵或不稳定时加入视觉/离散信息提升性能？

经过多轮实验后，主线逐渐从“只解释高熵和视觉 attention”转向：

> pure-soft 在 VStar 上的主要问题不是单纯视觉 grounding 不足，而是生成路径在格式 token、长输出、重复退化和答案边界处不稳定。有效方法是对特定 token 区域进行路由，把下一步输入从 soft embedding 临时切回 discrete token embedding。

## 2. 基础方法和路由动作

### 2.1 pure-soft

pure-soft 每一步不是把采样/argmax 得到的 token embedding 作为下一步输入，而是用完整词表概率分布加权：

```python
soft_emb = probs_original @ E
```

其中 `E` 是词表 embedding 矩阵。

优点是保留分布信息；问题是容易在不稳定区域引起长输出、重复、格式漂移。

### 2.2 discrete / normal embedding

普通 COT 或 greedy decoding 的下一步输入是：

```python
normal_emb = E[next_token]
```

本轮多数路由的共同动作是：

```python
last_emb = normal_emb
```

也就是在某些 token 上把 pure-soft 临时“坍缩”回离散 token 路径。

### 2.3 hard discrete 与 mixed embedding

hard discrete：

```python
last_emb = normal_emb
```

mixed embedding：

```python
format_emb = lambda * normal_emb + (1 - lambda) * soft_emb
```

实验显示，VStar 上 format 路由的收益依赖 hard discrete；mixed lambda 明显弱于 hard discrete。

## 3. 路由信号解释

### 3.1 low-confidence diffuse collapse

含义：

- 当前 token 是局部 entropy spike；
- raw top1 概率低，或 top1-top2 margin 小；
- 可选要求最近输出已经有重复退化；
- 命中后下一步输入从 `soft_emb` 切回 `normal_emb`。

它对应的是：

> 模型当前分布很散，且可能正在进入重复/长输出退化，此时用离散 token 做一次止损。

### 3.2 late64

含义：

- 低置信扩散坍缩只允许在生成步数大于等于 64 后触发。

直觉：

> 早期推理需要保留 soft 的探索性；后期才主要防止重复和长输出。

### 3.3 repeat_gate

含义：

- 低置信扩散坍缩不仅要求 entropy / confidence 条件，还要求最近输出出现重复退化迹象。

直觉：

> 只在模型真的开始重复或发散时才止损，减少误伤。

### 3.4 format cooldown

含义：

- 当前 token 是格式 token，例如换行、标点、括号、`answer`、`think` 等；
- 命中后接下来若干步使用 `normal_emb`。

直觉：

> pure-soft 在格式边界、模板边界和答案结构附近容易漂移。短暂离散化可以稳定模板和后续内容。

### 3.5 answer_zone_discrete

含义：

- 检测到 `</think` 或 `answer` 后，认为进入答案区；
- 答案区后续 token 用 `normal_emb`。

直觉：

> 最终答案区更需要稳定格式，不一定需要 soft embedding。

实验显示该信号能修格式，但介入太晚，不能解决 reasoning 阶段的退化。

### 3.6 highrisk_only

含义：

- format cooldown 只对高危结构 token 生效，例如 `answer`、`think`、括号、冒号等；
- 普通标点、换行不触发。

实验显示该信号过窄，说明普通格式 token 也对稳定 pure-soft 有贡献。

### 3.7 min_step32

含义：

- 前 32 个生成步不允许 format cooldown，之后才允许。

实验显示早期 format cooldown 也有贡献，完全跳过早期会掉分。

### 3.8 normal1_highrisk2

含义：

- 普通格式 token cooldown 1 步；
- 高危格式 token cooldown 2 步。

实验显示普通格式 token 只给 1 步不够，说明 hard cooldown2 的强度是重要因素。

### 3.9 mixed lambda

含义：

- format token 命中后不直接 hard discrete，而是使用：

```python
lambda * normal_emb + (1 - lambda) * soft_emb
```

实验显示 `lambda=0.75` 和 `lambda=0.50` 都不如 hard discrete。

## 4. 实验展开过程

## 4.1 第一阶段：COT / LEAD / pure-soft 三方法对比

实验目录：

```text
output/experiments/20260516_183300/exp1_vstar_spike_type_parallel
```

结果：

| 方法 | Acc | direct_attributes | relative_position |
|---|---:|---:|---:|
| COT | 131/191 = 68.59% | 84/115 = 73.04% | 47/76 = 61.84% |
| LEAD | 139/191 = 72.77% | 82/115 = 71.30% | 57/76 = 75.00% |
| pure-soft | 112/191 = 58.64% | 70/115 = 60.87% | 42/76 = 55.26% |

主要观察：

- LEAD 相比 COT 的优势主要来自 `relative_position`。
- pure-soft 明显弱于 COT 和 LEAD。
- pure-soft 错误中有较多长输出、重复和格式异常。
- 高熵 token 类型多样，不只是视觉不确定，也包含 format / relation / diffuse 等。

阶段结论：

> 后续应把高熵 token 分类，不同信号对应不同路由动作。

## 4.2 第二阶段：low-confidence diffuse collapse

最早在错题并集上测试：

```text
output/experiments/20260517_181331/pure_soft_collapse_wrong_union_parallel
```

结果：

| 方法 | Acc |
|---|---:|
| baseline | 23/102 = 22.55% |
| collapse | 41/102 = 40.20% |

在全量 VStar 上：

```text
output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel
```

结果：

| 方法 | Acc |
|---|---:|
| pure-soft baseline | 112/191 = 58.64% |
| collapse | 114/191 = 59.69% |

后续 precision 消融：

```text
output/experiments/20260518_173645/pure_soft_collapse_precision_vstar_full
```

结果：

| 方法 | Acc |
|---|---:|
| strict_threshold | 116/191 = 60.73% |
| patience2 | 111/191 = 58.12% |
| late64 | 119/191 = 62.30% |
| repeat_gate | 119/191 = 62.30% |

阶段结论：

> 低置信扩散坍缩是有效止损路由，尤其能减少长输出和重复，但单独使用不是主增益来源。

## 4.3 第三阶段：format cooldown 成为核心突破

实验目录：

```text
output/experiments/20260519_181159/pure_soft_format_cooldown_vstar_full
output/experiments/20260519_234017/pure_soft_format_cooldown_ablation_vstar_full
```

结果：

| 方法 | Acc | mean len | long>=256 | max1024 | missing_answer |
|---|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 237.29 | 33 | 18 | 40 |
| format_cooldown8 | 136/191 = 71.20% | 121.28 | 12 | 0 | 17 |
| format_cooldown4 | 138/191 = 72.25% | 123.72 | 12 | 1 | - |
| format_cooldown2 | 142/191 = 74.35% | 131.08 | 9 | 4 | 16 |

其中 `format_cooldown2` 相比 baseline：

- fixed：`40`
- damaged：`10`
- net：`+30`

阶段结论：

> pure-soft 在格式边界附近非常不稳定。短暂 hard discrete 能显著稳定推理和答案输出。cooldown2 是当前最强单路由。

## 4.4 第四阶段：cooldown2 + late64 repeat gate

实验目录：

```text
output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full
```

结果：

| 方法 | Acc | direct_attributes | relative_position |
|---|---:|---:|---:|
| cooldown2 + late64_repeat_gate | 143/191 = 74.87% | 86/115 = 74.78% | 57/76 = 75.00% |

相比 pure-soft baseline：

- fixed：`40`
- damaged：`9`
- net：`+31`

相比单独 cooldown2：

- fixed：`2`
- damaged：`1`
- net：`+1`

阶段结论：

> 主力是 format cooldown2；late64 repeat gate 是小幅正向补强。

## 4.5 第五阶段：answer-zone 路由与多信号混合

实验目录：

```text
output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full
output/experiments/20260520_133545/pure_soft_multisignal_mix_vstar_full
```

结果：

| 方法 | Acc | 相比 baseline | long>=256 | max1024 |
|---|---:|---:|---:|---:|
| answer_zone | 126/191 = 65.97% | net +14 | 33 | 18 |
| cooldown2 + answer_zone | 141/191 = 73.82% | net +29 | 8 | 4 |
| cooldown2 + late64_repeat + answer_zone | 143/191 = 74.87% | net +31 | 8 | 3 |

阶段结论：

> answer-zone 能修答案格式，但介入太晚。它不能解决 reasoning 阶段的长输出和重复退化，与 best combo 简单叠加没有新增收益。

## 4.6 第六阶段：format cooldown 细化失败

实验目录：

```text
output/experiments/20260520_185703/pure_soft_format_refine_vstar_full
```

结果：

| 方法 | Acc | direct | relative | long>=256 | max1024 |
|---|---:|---:|---:|---:|---:|
| highrisk_only | 132/191 = 69.11% | 77/115 | 55/76 | 23 | 12 |
| min_step32 | 139/191 = 72.77% | 81/115 | 58/76 | 20 | 8 |

相对 cooldown2：

- highrisk_only：fixed 12 / damaged 22 / net -10
- min_step32：fixed 15 / damaged 18 / net -3

阶段结论：

> format cooldown 的收益不只来自 answer/think/括号等高危 token；普通标点、换行、早期格式 token 也有稳定作用。

## 4.7 第七阶段：variable cooldown 与 mixed embedding

实验目录：

```text
output/experiments/20260520_194525/pure_soft_format_variable_and_mixed_vstar_full
```

结果：

| 方法 | Acc | direct | relative | long>=256 | max1024 |
|---|---:|---:|---:|---:|---:|
| normal1_highrisk2 | 131/191 = 68.59% | 77/115 | 54/76 | 16 | 11 |
| normal1_highrisk2 + late64 | 129/191 = 67.54% | 75/115 | 54/76 | 15 | 9 |
| mix lambda=0.75 | 135/191 = 70.68% | 80/115 | 55/76 | 15 | 6 |
| mix lambda=0.50 | 135/191 = 70.68% | 79/115 | 56/76 | 13 | 7 |

阶段结论：

> 全局弱化 discrete 强度会掉分。VStar 上 format 路由需要 hard discrete，不适合用 mixed embedding 简单替代。

## 5. Damaged 样本分析

当前 best combo 相比 pure-soft baseline 的 damaged 样本为：

```text
[34, 51, 75, 81, 120, 126, 135, 150, 175]
```

数量：`9`

关键发现：

- 这 9 个样本在 `cooldown2` 中已经错误，因此主要 damage 来源是 format cooldown，而不是 late64 repeat gate。
- 单独 `answer_zone` 在这 9 个样本上全部正确，说明损坏发生在 reasoning 阶段，不是最终答案区。
- damaged 样本 format cooldown 触发量均值为 `54.11`，明显高于 baseline 正确且未 damaged 样本的 `32.57`。

分布：

| 类型 | 数量 |
|---|---:|
| direct_attributes | 4 |
| relative_position | 5 |

阶段结论：

> 下一步不应继续全局削弱 cooldown2，而应保留 hard cooldown2，并增加 damaged-aware 保护门控。

## 6. 当前正在验证的方向

当前正在运行：

```text
output/experiments/20260520_204403/pure_soft_format_budget_cap_vstar_full
```

配置：

- `cooldown2_cap60`
- `cooldown2_cap50`
- `cooldown2_cap60_late64_repeat`
- `cooldown2_cap50_late64_repeat`

目标：

> 每个样本最多允许 N 个 token 进入 format cooldown，从而限制过量 format 介入，尝试恢复 damaged 样本，同时保留 hard cooldown2 的稳定收益。

该实验尚未纳入本文结论。

## 7. 阶段总结果

截至目前，已完成实验的最强配置仍是：

```text
cooldown2 + late64_repeat_gate
```

准确率：

```text
143/191 = 74.87%
```

核心结论：

1. pure-soft 的 VStar 退化主要表现为格式漂移、长输出、重复退化和答案边界不稳定。
2. low-confidence diffuse collapse 有止损作用，但不是主增益来源。
3. format cooldown2 是核心突破，将 pure-soft 从 `58.64%` 提升到 `74.35%`。
4. late64 repeat gate 在 cooldown2 上小幅补强，形成当前 best `74.87%`。
5. answer-zone 介入太晚，适合修格式但不能解决 reasoning 退化。
6. highrisk-only、min-step、normal1_highrisk2、mixed lambda 都说明：不能简单削弱 hard discrete。
7. 下一阶段应做 damaged-aware 保护门控，而不是继续降低整体路由强度。
8. 视觉信息可以作为后续诊断方向，但当前主线仍然是生成路径稳定化。


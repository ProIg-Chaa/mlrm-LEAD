# 2026-05-16 以来 VStar 路由实验阶段总结

本文整理 2026-05-16 以来围绕 VStar、pure-soft、LEAD、format cooldown、低置信扩散坍缩、多信号混合、轻量视觉 bias 与统一路由框架等实验的主要脉络。主体总结已经完成并分析过的实验，并更新到 2026-05-25 当前进度。

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

### 3.10 image_pad visual bias

含义：

- 在 pure-soft 的 `soft_emb` 中混入 `<|image_pad|>` token 的 embedding：

```python
biased_soft_emb = (1 - lambda) * soft_emb + lambda * E[<|image_pad|>]
```

直觉：

> 用非常轻量的视觉 anchor 给 soft 推理补一点视觉信息。

实验显示该信号有明显阶段依赖：

- full / early bias 会明显伤 VStar；
- mid bias 相对安全，并在 VStar / VisuLogic 上有小幅收益；
- late bias 最安全，但基本没有新增收益。

### 3.11 phase gate

当前阶段划分：

| 阶段 | step 范围 | 当前观察 |
|---|---|---|
| early | `0-128` | VStar 上危险，MMVP 上有收益 |
| mid | `129-512` | 当前最像通用安全收益区 |
| late | `513+` | 安全但收益弱 |

阶段实验说明，视觉信息不是不能用，而是不能在错误阶段粗暴注入。

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

## 4.8 第八阶段：跨数据集 base / LEAD / bestcombo 对照

实验目录：

```text
output/experiments/20260520_231938/cross_dataset_base_lead_bestcombo
```

代表性结果：

| 数据集 | pure-soft | LEAD | bestcombo |
|---|---:|---:|---:|
| MMVP sample | - | 211/300 = 70.33% | 201/300 = 67.00% |
| MMVP pair | - | 63/150 = 42.00% | 60/150 = 40.00% |
| VisuLogic300 | 53/300 = 17.67% | 约 74-76/300 = 24.67%-25.33% | 73/300 = 24.33% |
| VStar | 112/191 = 58.64% | 139/191 = 72.77% | 143/191 = 74.87% |

注：

- VStar bestcombo 在不同轻量抽取脚本中有时计为 `144/191`，但本文沿用项目 eval_report 中的 `143/191 = 74.87%` 作为阶段主表结果。
- MMVP 使用修正后的 sample / pair 评估；pair 正确要求同一个 pair 两题都答对。

阶段结论：

> bestcombo 在 VStar 上明显有效，但在 MMVP 上不如 LEAD。这说明 format/collapse 路由解决的是 soft 推理退化，不等价于 LEAD 的视觉 anchor 能力；不同数据集需要不同 route profile。

## 4.9 第九阶段：LEAD simple visual anchor 消融

实验目录：

```text
output/experiments/20260521_152817/lead_simple_anchor_ablation_mmvp_visulogic_vstar
```

结果：

| 数据集 | 原 LEAD | 关闭 simple anchor |
|---|---:|---:|
| MMVP sample | 211/300 = 70.33% | 209/300 = 69.67% |
| MMVP pair | 63/150 = 42.00% | 61/150 = 40.67% |
| VisuLogic300 | 74/300 = 24.67% | 65/300 = 21.67% |
| VStar | 139/191 = 72.77% | 137/191 = 71.73% |

阶段结论：

> 原始 LEAD 确实包含一个轻量 `<|image_pad|>` anchor。它不是后续实验中的动态 attention anchor，但它有稳定收益，尤其在 VisuLogic 上更明显。这也支持“轻量视觉信息有用，但不能强注入”的判断。

## 4.10 第十阶段：format 过度干预问题

实验目录：

```text
output/experiments/20260521_191535/format_overintervention_gates_mmvp_visulogic
```

核心结果：

| 数据集 | bestcombo | gate_entropy10 | gate_top080_margin040 | gate_strict |
|---|---:|---:|---:|---:|
| MMVP sample | 201/300 = 67.00% | 194/300 = 64.67% | 192/300 = 64.00% | 195/300 = 65.00% |
| MMVP pair | 60/150 = 40.00% | 56/150 = 37.33% | 55/150 = 36.67% | 58/150 = 38.67% |
| VisuLogic300 | 73/300 = 24.33% | 53/300 = 17.67% | 66/300 = 22.00% | 67/300 = 22.33% |

阶段结论：

> 减少 format cooldown 触发并没有减少主要损伤，反而明显掉分。format cooldown 不是简单的过度干预，它是 pure-soft 稳定化的核心动作之一。

## 4.11 第十一阶段：image_pad bias 全程 / entropy gate / phase gate

### 4.11.1 full image_pad bias

实验目录：

```text
output/experiments/20260522_125332/bestcombo_image_pad_bias_vstar_mmvp_visulogic
```

代表性结果：

| 数据集 | bestcombo | full bias λ=0.05 |
|---|---:|---:|
| VStar | 143/191 = 74.87% | 135/191 = 70.68% |
| MMVP sample | 201/300 = 67.00% | 207/300 = 69.00% |
| MMVP pair | 60/150 = 40.00% | 63/150 = 42.00% |
| VisuLogic300 | 73/300 = 24.33% | 约 74/300 = 24.67% |

结论：

> full visual bias 对 MMVP 有收益，但明显伤 VStar。视觉信息有用，但全程注入不稳定。

### 4.11.2 entropy-gated image_pad bias

实验目录：

```text
output/experiments/20260522_184028/bestcombo_image_pad_bias_entropy_gate
```

核心结果：

| 数据集 | bestcombo | entropy>=1.0 | entropy>=1.5 | entropy>=2.0 |
|---|---:|---:|---:|---:|
| VStar | 143/191 = 74.87% | 133/191 = 69.63% | 142/191 = 74.35% | 132/191 = 69.11% |
| MMVP sample | 201/300 = 67.00% | 208/300 = 69.33% | 199/300 = 66.33% | 198/300 = 66.00% |
| VisuLogic300 | 73/300 = 24.33% | 67/300 = 22.33% | 69/300 = 23.00% | 62/300 = 20.67% |

结论：

> “高熵就加视觉”不成立。高熵 token 类型复杂，entropy gate 不能区分视觉不确定、format 不稳定、关系推理或低置信扩散。

### 4.11.3 phase-gated image_pad bias

实验目录：

```text
output/experiments/20260523_121058/bestcombo_image_pad_bias_phase_gate
```

结果：

| 数据集 | bestcombo | full bias 0.05 | early | mid | late |
|---|---:|---:|---:|---:|---:|
| VStar | 143/191 = 74.87% | 135/191 = 70.68% | 135/191 = 70.68% | 约 144-145/191 = 75%+ | 143/191 = 74.87% |
| MMVP sample | 201/300 = 67.00% | 207/300 = 69.00% | 207/300 = 69.00% | 201/300 = 67.00% | 201/300 = 67.00% |
| MMVP pair | 60/150 = 40.00% | 63/150 = 42.00% | 63/150 = 42.00% | 60/150 = 40.00% | 60/150 = 40.00% |
| VisuLogic300 | 73/300 = 24.33% | 约 74/300 = 24.67% | 73-74/300 | 76-77/300 = 25%+ | 71-72/300 |

阶段结论：

> early 视觉 bias 是 VStar 的主要伤害来源；mid 是当前最有价值的视觉注入窗口；late 安全但新增收益弱。MMVP 更吃 early visual bias，而 VStar 不能 early 加。

## 4.12 第十二阶段：VStar early damage 集 lambda sweep

实验目录：

```text
output/experiments/20260524_131252/vstar_damage_image_pad_lambda_sweep
```

破坏集定义：

```text
bestcombo 原本答对，但 early image_pad_bias λ=0.05 答错的 VStar 样本
```

样本数：

```text
18
```

结果：

| 方法 | 破坏集正确率 |
|---|---:|
| no_bias bestcombo | 18/18 = 100.00% |
| full λ=0.01 | 10/18 = 55.56% |
| full λ=0.02 | 10/18 = 55.56% |
| full λ=0.03 | 9/18 = 50.00% |
| full λ=0.05 | 1/18 = 5.56% |
| early λ=0.01 | 10/18 = 55.56% |
| early λ=0.02 | 9/18 = 50.00% |
| early λ=0.03 | 9/18 = 50.00% |
| early λ=0.05 | 0/18 = 0.00% |
| mid λ=0.01 | 16/18 = 88.89% |
| mid λ=0.02 | 17/18 = 94.44% |
| mid λ=0.03 | 17/18 = 94.44% |
| mid λ=0.05 | 17/18 = 94.44% |
| late λ=0.01 | 18/18 = 100.00% |
| late λ=0.02 | 18/18 = 100.00% |
| late λ=0.03 | 18/18 = 100.00% |
| late λ=0.05 | 18/18 = 100.00% |

阶段结论：

> 破坏不是单纯因为 λ=0.05 太大，而是 early 视觉注入本身危险。即使 λ=0.01，early 仍会破坏 8/18 个原本正确样本。full bias 的主要伤害也来自 early 阶段。

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

## 6. 统一路由框架准备

文档：

```text
result/5-16exp/unified_routing_framework_direction1_20260524.md
```

代码准备：

- `lead/generation_utils.py`：在 `generate_pure_soft(...)` 中新增 route annotation 字段；
- `script/exp5_16/analyze_route_summary.py`：新增统一 route summary 脚本。

新增 trace 字段包括：

- `generation_phase`
- `route_signal`
- `route_action`
- `route_priority`
- `route_suppressed_by`
- `is_highrisk_format_token`
- `visual_bias_candidate`
- `visual_bias_effective`
- `entropy_spike_mask`
- `diffuse_mask`
- `repeat_degen_detected`

这些字段只用于记录，不改变当前生成行为。

统一路由框架当前抽象为：

```text
token / sample state -> uncertainty type -> intervention action
```

第一版优先级：

```text
answer_zone / collapse hard discrete
> format cooldown
> mid image_pad visual bias
> pure_soft
```

smoke test：

```text
/tmp/mlrm_route_smoke
```

验证结果：

- `py_compile` 通过；
- 小样本 pure-soft 能正常跑完；
- `token_entropy_full.jsonl` 中 route 字段正常写出；
- `analyze_route_summary.py` 能正常生成 route summary。

阶段结论：

> 当前已经具备“无行为变化地记录路由状态”的能力。后续每个 run 都能统计哪个 route 生效、哪个 route 被覆盖、fixed/damaged 样本分别由哪些 route 主导。

## 7. 当前阶段总结果

截至目前，已完成实验的 VStar 主线最强配置仍是：

```text
cooldown2 + late64_repeat_gate
```

准确率：

```text
143/191 = 74.87%（项目 eval_report 口径）
```

核心结论：

1. pure-soft 的 VStar 退化主要表现为格式漂移、长输出、重复退化和答案边界不稳定。
2. low-confidence diffuse collapse 有止损作用，但不是主增益来源。
3. format cooldown2 是核心突破，将 pure-soft 从 `58.64%` 提升到 `74.35%`。
4. late64 repeat gate 在 cooldown2 上小幅补强，形成当前 best `74.87%`。
5. answer-zone 介入太晚，适合修格式但不能解决 reasoning 退化。
6. highrisk-only、min-step、normal1_highrisk2、mixed lambda 都说明：不能简单削弱 hard discrete。
7. LEAD 的 simple `<|image_pad|>` anchor 有稳定收益，说明轻量视觉信息有用。
8. 但 image_pad bias 不能全程或 early 粗暴注入；VStar damage 集显示 early λ=0.01 已经会破坏 8/18 个原本正确样本。
9. mid 阶段 visual bias 是当前最有希望的视觉注入窗口；late 安全但收益弱。
10. 高熵不等于视觉不足，entropy-gated visual bias 不稳定；必须把高熵 token 继续区分为 format / relation / visual / diffuse low-conf / answer 等类型。
11. 下一阶段重点应从继续调单个阈值，转向统一路由框架：显式记录 `route_signal -> route_action`，分析 fixed/damaged 来源，再设计 Router v0。

## 8. 下一步建议

当前最合理的下一步不是继续扫 λ，而是先用新增 route annotation 做一轮“无行为变化复跑”和分析：

1. 重跑 VStar bestcombo，确认新增 trace 不改变结果。
2. 对 bestcombo 生成 route summary，统计 format/collapse/default 的分布。
3. 重跑或选取 mid image_pad bias 配置，分析 visual bias 实际命中哪些 token、被 format/collapse 覆盖多少。
4. 对 fixed/damaged 样本比较 route 分布，确认 damage 来自哪个 route。
5. 在 VStar / MMVP / VisuLogic 上比较 route profile，决定是否需要数据集级 route profile。

建议 Router v0 候选：

```text
format cooldown2
+ late64 repeat-gated collapse
+ mid-only image_pad_bias lambda=0.02 或 0.03
+ answer-zone hard discrete（可选）
```

但 Router v0 正式实验应建立在 route summary 结果之后。

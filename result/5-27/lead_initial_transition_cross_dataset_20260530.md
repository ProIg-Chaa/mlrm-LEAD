# LEAD 开头 transition 消融与跨数据集结果分析

时间：2026-05-30  
项目：`/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD`

## 1. 背景

最新检查发现，标准 LEAD 在 VStar 上平均每个样本实际触发 soft 介入次数很少，约为 `1.7` 次/样本；同时 LEAD 固定在每个样本的第 0 步先走一次 soft / transition 机制。

这引出一个关键怀疑：

> LEAD 的主要收益可能并不来自后续稀疏的动态 soft trigger，而来自 generation 开头的 latent / soft transition 对整条推理轨迹的初始化影响。

为验证这个假设，新增了开头消融：

- `lead_force_normal`：强制正常 COT 路径，不走 LEAD soft。
- `initial_soft_only`：只保留第 0 步 soft。
- `initial_transition_only`：只保留 LEAD 开头 transition，之后正常 COT 推理。
- `initial_transition_only_no_anchor`：去掉 simple visual anchor 后，只保留开头 transition。
- 后续又扩展到跨数据集：
  - `lead`
  - `initial_transition_only`
  - `quota20`
  - `quota05_guard`

主要实验目录：

```text
output/experiments/20260529_163618/vstar_lead_cot_sanity_matrix
output/experiments/20260529_225807/vstar_lead_soft_quota_sweep
output/experiments/20260530_013153/cross_dataset_lead_transition_quota
```

## 2. 几种方法的区别

这一轮实验里最容易混淆的是 `initial_soft_only` 和 `initial_transition_only`。它们都只让 LEAD 在开头附近发生作用，但保留的机制不同。

LEAD 的生成循环里，每一步都会先得到当前 token 的 logits、采样/贪心得到 `next_token`，然后构造两种下一步输入：

```text
normal_emb = E[next_token]
soft_emb   = probs_original @ E
```

最终下一步喂给模型的是 `normal_emb` 还是 `soft_emb`，由 LEAD 的 mode 和额外路由决定。

### 2.1 `cot_orign_greedy`

普通 COT / hard decoding baseline。

- 不使用 LEAD 的 soft embedding。
- 下一步输入始终是当前生成 token 的离散 embedding：

```text
last_emb = normal_emb
```

它是判断 LEAD 是否真的带来收益的 hard baseline。

### 2.2 `lead_force_normal`

代码仍走 `method=lead` 的生成函数，但强制关闭 soft 路径。

关键效果：

```text
mode 初始为 normal
is_soft 全程为 False
last_emb = normal_emb
```

因此它等价于“LEAD 框架下的 hard COT”。如果它和 `cot_orign_greedy` 同分，说明 LEAD 函数本身的 wrapper、prompt、评估路径没有额外贡献。

### 2.3 full `lead`

标准 LEAD。

核心机制有三层：

1. 初始 mode 为 soft。
2. 第 0 步 soft embedding 会额外混入一个很弱的换行 embedding：

```text
if step == 0:
    soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb
```

3. 后续根据 entropy 相对参考熵的变化，在 soft / normal 之间切换：

```text
to_normal = mode == soft 且当前熵下降到阈值以下
to_soft   = mode == normal 且当前熵上升且满足 window 条件
```

当发生 `to_normal` 时，LEAD 不是直接使用普通 token embedding，而是做一次 transition 混合：

```text
normal_emb = beta * soft_emb + (1 - beta) * end_thinking_emb
```

所以 full LEAD 不只是“某些高熵 token 用 soft”，还包含：

- 开头 soft 状态；
- 第 0 步弱换行混合；
- 从 soft 切回 normal 时的 `end_thinking_emb` transition；
- 后续可能再次从 normal 切回 soft。

这次发现的关键点是：full LEAD 在 VStar 上后续 soft 触发平均只有约 `1.7` 次/样本，因此后续动态触发很难解释大部分收益。

### 2.4 `initial_soft_only`

只保留第 0 步 soft 输入，不保留后续 transition。

代码效果：

```text
is_soft = is_soft & (step == 0)
```

同时因为 `lead_initial_soft_only=True`，代码会跳过 step>0 的 `to_normal` transition 混合：

```text
if step > 0 and not lead_initial_soft_only:
    normal_emb = beta * soft_emb + (1 - beta) * end_thinking_emb
```

因此 `initial_soft_only` 的实际含义是：

```text
第 0 步：last_emb = 0.9 * soft_emb + 0.1 * line_break_emb
第 1 步以后：last_emb = normal_emb
不使用 end_thinking transition
不允许后续 to_soft
```

它用于回答：

> 是否只要第 0 步用一次 soft embedding 就能解释 LEAD 收益？

VStar 结果显示不能：`initial_soft_only` 只有 `132/191`，明显低于 full LEAD 的 `139/191`。

### 2.5 `initial_transition_only`

只保留 LEAD 开头的 soft-to-normal transition，之后正常 COT 推理。

它和 `initial_soft_only` 的关键区别是：`initial_transition_only` 虽然同样只允许 step 0 处于 soft，但不会跳过 step>0 的 transition 混合。

实际效果可以理解为：

```text
第 0 步：
    使用 soft 输入，并混入弱 line_break_emb

第 1 步附近：
    mode 已经从初始 soft 切到 normal
    但 normal_emb 会被替换成一次 mixed transition：
        normal_emb = beta * soft_emb + (1 - beta) * end_thinking_emb

之后：
    禁止后续 to_soft
    基本回到正常 COT / hard decoding
```

也就是说，`initial_transition_only` 保留的是“开头 latent 状态如何落回 normal 推理轨道”的过渡过程，而不是单纯第 0 步 soft。

这正是它比 `initial_soft_only` 强很多的原因之一：

```text
initial_soft_only       = 只给开头一个 soft perturbation
initial_transition_only = soft 开头 + soft-to-normal transition
```

### 2.6 `initial_transition_only_no_anchor`

在 `initial_transition_only` 基础上关闭 simple visual anchor。

标准 LEAD 默认会把 `<think>` anchor 替换为 `<|image_pad|>` embedding：

```text
if not lead_disable_simple_visual_anchor:
    thinking_token_id = image_pad_id
```

`no_anchor` 版本关闭这个替换，目的是判断开头 transition 的收益是否来自视觉 anchor。

VStar 上：

```text
initial_transition_only           = 138/191
initial_transition_only_no_anchor = 138/191
```

因此当前证据表明，VStar 上这部分收益主要不是 simple visual anchor 带来的。

### 2.7 `quota20`

在 full LEAD 基础上额外加入 soft quota。

含义：

```text
目标 soft 次数 ≈ lead_soft_quota_ratio * 当前已生成步数
quota20 即 lead_soft_quota_ratio = 0.20
```

如果当前样本累计 soft 次数低于 quota，且还没有进入锁定 normal 的状态，就强行让当前步走 soft：

```text
is_soft = is_soft | lead_soft_quota_mask
```

它用于测试：

> 如果 full LEAD 后续触发太少，人为增加 soft 比例是否更好？

结果显示 `quota20` 跨数据集不稳，尤其 MMVP item accuracy 和 VisuLogic 都下降，说明“更多 soft”不是通用解。

### 2.8 `quota05_guard`

保守 quota + guard 版本。

它的目标是只增加少量后续 soft，并用 guard 抑制明显危险状态。当前实验里它对应较低的 soft quota，平均 soft ratio 大约 `0.04-0.05`，同时打开保护逻辑，例如低置信扩散 veto / format guard 一类规则，使 soft 不覆盖明显应该 hard discrete 的位置。

可以把它理解为：

```text
full LEAD
+ 少量额外 soft quota
+ 遇到危险 token/state 时 veto soft，回到 normal
```

它和 `quota20` 的区别是：

- `quota20` 更像“强行提高 soft 覆盖率”；
- `quota05_guard` 更像“只补一点 soft，并尽量避免格式、低置信、退化状态上的 damage”。

跨数据集结果显示：

- RealWorldQA fixed 上 `quota05_guard` 最好，`134/200 = 67.00%`；
- MMVP item acc 不变，pair acc 小幅 +1 pair；
- VisuLogic 上明显变差。

因此它不是通用正收益，但说明“后续 soft 并非完全无用”，只是必须非常少、非常保守、强依赖数据集。

### 2.9 小结

这一轮各方法的关系可以压缩成：

| 方法 | 第 0 步 soft | soft-to-normal transition | 后续动态 soft | simple visual anchor | 额外 quota | guard |
|---|---:|---:|---:|---:|---:|---:|
| `cot_orign_greedy` | 否 | 否 | 否 | 否 | 否 | 否 |
| `lead_force_normal` | 否 | 否 | 否 | 可在代码中存在但不生效 | 否 | 否 |
| `initial_soft_only` | 是 | 否 | 否 | 默认有 | 否 | 否 |
| `initial_transition_only` | 是 | 是 | 否 | 默认有 | 否 | 否 |
| `initial_transition_only_no_anchor` | 是 | 是 | 否 | 否 | 否 | 否 |
| full `lead` | 是 | 是 | 是 | 默认有 | 否 | 否 |
| `quota20` | 是 | 是 | 是 | 默认有 | 是，较强 | 否 |
| `quota05_guard` | 是 | 是 | 是 | 默认有 | 是，较弱 | 是 |

核心发现也因此更明确：

> 真正接近 full LEAD 的不是 `initial_soft_only`，而是 `initial_transition_only`。也就是说，收益不是单纯来自第 0 步 soft，而是来自开头 latent 状态以及它回落到 normal 推理轨道时的 transition。

## 3. VStar sanity matrix

VStar full 191 题结果如下：

| run | correct | accuracy |
|---|---:|---:|
| `cot_orign_greedy` | 131/191 | 68.59% |
| `cot_step_greedy` | 137/191 | 71.73% |
| `lead_force_normal` | 131/191 | 68.59% |
| `initial_soft_only` | 132/191 | 69.11% |
| `initial_soft_only_no_anchor` | 132/191 | 69.11% |
| `initial_transition_only` | 138/191 | 72.25% |
| `initial_transition_only_no_anchor` | 138/191 | 72.25% |
| `lead_no_anchor_rerun_no_fulltrace` | 137/191 | 71.73% |
| `lead` | 139/191 | 72.77% |

关键观察：

1. `lead_force_normal` 与 `cot_orign_greedy` 完全同分，说明如果去掉 LEAD 的 soft / transition，LEAD 路径本身不提供额外收益。
2. `initial_soft_only` 只有 `132/191`，说明“只做第 0 步 soft”不是主要收益来源。
3. `initial_transition_only` 达到 `138/191`，只比 full LEAD 少 1 题。
4. `initial_transition_only_no_anchor` 同样是 `138/191`，说明这个收益不依赖 simple visual anchor。
5. full LEAD 后续平均约 `1.7` 次/样本的动态触发，在 VStar 上只带来非常小的边际收益。

阶段结论：

> VStar 上，LEAD 的主要收益几乎可以由开头 transition 复现；后续稀疏 soft trigger 不是主要贡献项。

## 4. 跨数据集主结果

跨数据集实验目录：

```text
output/experiments/20260530_013153/cross_dataset_lead_transition_quota
```

汇总结果：

| dataset | run | correct | accuracy | avg len | maxed | mean soft ratio |
|---|---|---:|---:|---:|---:|---:|
| RealWorldQA fixed | `lead` | 129/200 | 64.50% | 139.39 | 1 | 0.0142 |
| RealWorldQA fixed | `initial_transition_only` | 127/200 | 63.50% | 140.88 | 1 | 0.0097 |
| RealWorldQA fixed | `quota20` | 126/200 | 63.00% | 138.72 | 0 | 0.1691 |
| RealWorldQA fixed | `quota05_guard` | 134/200 | 67.00% | 135.02 | 0 | 0.0467 |
| VisuLogic300 | `lead` | 74/300 | 24.67% | 492.17 | 4 | 0.0247 |
| VisuLogic300 | `initial_transition_only` | 85/300 | 28.33% | 514.44 | 10 | 0.0022 |
| VisuLogic300 | `quota20` | 69/300 | 23.00% | 502.72 | 6 | 0.1700 |
| VisuLogic300 | `quota05_guard` | 67/300 | 22.33% | 484.25 | 3 | 0.0450 |
| MMVP | `lead` | 211/300 | 70.33% | 110.38 | 0 | 0.0118 |
| MMVP | `initial_transition_only` | 211/300 | 70.33% | 109.83 | 0 | 0.0108 |
| MMVP | `quota20` | 205/300 | 68.33% | 108.54 | 0 | 0.1460 |
| MMVP | `quota05_guard` | 211/300 | 70.33% | 110.20 | 0 | 0.0378 |

MMVP specialized pair accuracy：

| run | item acc | pair acc |
|---|---:|---:|
| `lead` | 70.33% | 42.00% |
| `initial_transition_only` | 70.33% | 42.00% |
| `quota20` | 68.33% | 42.67% |
| `quota05_guard` | 70.33% | 42.67% |

## 5. 相对 full LEAD 的逐样本翻转

### 5.1 RealWorldQA fixed

| run | fixed | damaged | net |
|---|---:|---:|---:|
| `initial_transition_only` | 1 | 3 | -2 |
| `quota20` | 9 | 12 | -3 |
| `quota05_guard` | 11 | 6 | +5 |

解释：

- `initial_transition_only` 基本逼近 full LEAD，只净损 2 题。
- `quota20` 触发太多，fixed 和 damaged 都明显增加，但净值为负。
- `quota05_guard` 是当前 RealWorldQA fixed 上最好的配置，说明该数据集可能确实需要少量后续 soft，但必须被 guard 严格限制。

### 5.2 MMVP

Item-level：

| run | fixed | damaged | net |
|---|---:|---:|---:|
| `initial_transition_only` | 0 | 0 | 0 |
| `quota20` | 9 | 15 | -6 |
| `quota05_guard` | 5 | 5 | 0 |

Pair-level：

| run | fixed pairs | damaged pairs | net |
|---|---:|---:|---:|
| `initial_transition_only` | 0 | 0 | 0 |
| `quota20` | 8 | 7 | +1 |
| `quota05_guard` | 4 | 3 | +1 |

解释：

- `initial_transition_only` 与 full LEAD 在 MMVP 上逐样本完全一致，不只是总分相同。
- 后续 quota soft 对 item accuracy 有风险；`quota20` item-level 净损 6 题。
- 但 pair accuracy 有轻微改善，说明 quota 可能修复一些 pair 一致性样本，同时伤害更多单题。
- `quota05_guard` 保持 item acc 不变，并让 pair acc +1 pair，是一个弱正信号，但幅度很小。

### 5.3 VisuLogic300

用默认评估口径看，`initial_transition_only` 比 full LEAD 多 11 题。用轻量逐样本抽取复核时，趋势一致，约为：

| run | fixed | damaged | net |
|---|---:|---:|---:|
| `initial_transition_only` | 40 | 27 | +13 |
| `quota20` | 41 | 45 | -4 |
| `quota05_guard` | 36 | 44 | -8 |

按 subtopic：

| subtopic | lead | initial_transition_only | change |
|---|---:|---:|---:|
| Attribute Reasoning | 5/29 | 12/29 | +7 |
| Other | 7/31 | 12/31 | +5 |
| Positional Reasoning | 8/37 | 10/37 | +2 |
| Quantitative Reasoning | 28/100 | 29/100 | +1 |
| Spatial Reasoning | 21/78 | 16/78 | -5 |
| Stylistic Reasoning | 5/25 | 6/25 | +1 |

解释：

- VisuLogic 上，full LEAD 的后续触发可能不是正贡献，甚至可能伤害一部分样本。
- `initial_transition_only` 主要提升 Attribute / Other，对 Spatial Reasoning 有损伤。
- `quota20` 和 `quota05_guard` 都低于 full LEAD，说明在 VisuLogic 上继续增加后续 soft 不稳定。

## 6. 方法含义

这一轮结果把 LEAD 的解释重心明显推向了开头 transition：

```text
不是：
高熵 / 不确定 token 上频繁动态 soft 介入 -> 性能提升

更像是：
generation 起始阶段的一次 latent transition -> 改变后续整条 reasoning trajectory
```

这对当前研究叙事很重要：

1. LEAD 的平均触发次数太少，后续触发难以解释大部分收益。
2. VStar 和 MMVP 上，开头 transition 几乎完全复现 full LEAD。
3. VisuLogic 上，去掉后续触发反而更好。
4. RealWorldQA fixed 上，full LEAD 和 initial transition 很接近，但 `quota05_guard` 有额外收益，提示后续 soft 不是完全无用，而是必须非常稀疏、保守、有 guard。

因此，后续不应继续只围绕“高熵时是否触发 LEAD”调参，而应把问题拆成两个部分：

1. `initial transition` 为什么有效？
2. 后续 soft intervention 在什么条件下才值得保留？

## 7. 建议下一步

### 7.1 先拆开 initial transition

建议做一个最小矩阵：

| run | 目的 |
|---|---|
| `cot_orign_greedy` | 原始 hard baseline |
| `initial_soft_only` | 只测第 0 步 soft |
| `initial_transition_k1/k2/k4/k8` | 测 transition 长度 |
| `initial_transition_no_anchor` | 确认是否依赖 visual anchor |
| `initial_transition_no_entropy_condition` | 看是否只是固定扰动 |
| `initial_transition_random_or_mean_emb_control` | 排除任意 embedding perturbation 都有效的可能 |

核心指标：

- accuracy
- fixed / damaged against COT
- fixed / damaged against full LEAD
- early generated tokens 是否发生系统性变化
- 输出长度、maxed、failed extraction

### 7.2 后续 soft 只保留保守版本

当前证据不支持继续扩大后续 soft：

- `quota20` 在 RealWorldQA、MMVP item、VisuLogic 上都不稳。
- `quota05_guard` 在 RealWorldQA 和 MMVP pair 上有弱正信号，但在 VisuLogic 上明显负。

建议下一轮只保留：

```text
initial_transition_only
initial_transition + quota05_guard
```

不要继续优先扫大 quota。

### 7.3 报告叙事建议

可以把当前发现写成：

> LEAD 的收益并非主要来自大量在线动态介入。跨数据集消融显示，一个只发生在生成起点的 latent transition 就能复现 VStar 和 MMVP 上几乎全部 full LEAD 收益，并在 VisuLogic 上超过 full LEAD。后续 soft intervention 的收益高度依赖数据集和 guard 设计，过量 quota 会带来明显 damage。因此，真正值得解释的是 early latent transition 对推理轨迹的初始化效应，而不是平均仅约 1.7 次/样本的后续触发本身。

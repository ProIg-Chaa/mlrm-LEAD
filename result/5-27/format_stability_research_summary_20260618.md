# 格式稳定路线研究总结：Format Stability / Confidence Diffusion Guard

## 目录

- [1. 背景与问题](#1-背景与问题)
- [2. 术语与方法速查](#2-术语与方法速查)
- [3. 方法机制](#3-方法机制)
- [4. 早期 confidence 现象](#4-早期-confidence-现象)
- [5. VStar pure-soft format 系列](#5-vstar-pure-soft-format-系列)
- [6. confidence diffusion / collapse 系列](#6-confidence-diffusion--collapse-系列)
- [7. 多信号组合与 answer-zone 实验](#7-多信号组合与-answer-zone-实验)
- [8. 跨数据集 format gate 与 LEAD guard 复核](#8-跨数据集-format-gate-与-lead-guard-复核)
- [9. quota + format 调参](#9-quota--format-调参)
- [10. 这条路线的最终定位](#10-这条路线的最终定位)
- [11. 可保留结论](#11-可保留结论)

## 1. 背景与问题

这一阶段研究的核心问题是：

> soft / pure-soft 生成为什么会出现格式退化、长输出、重复、答案漂移和抽取失败？能否通过格式稳定与置信度扩散控制，把 soft 方法救回来？

这条路线是从 pure-soft 的失败形态出发的。早期实验发现，很多错题不是“低置信、不确定、短输出”，而是 **低熵、高置信、长输出**。模型经常很自信地沿着错误输出继续展开，最后产生长解释、重复、格式边界不清或答案无法抽取。

因此，format 稳定路线关注的是生成过程中的 **输出稳定性**：

- 格式边界是否稳定；
- 答案区域是否能 cleanly map 回选项；
- soft embedding 是否导致输出边界变糊；
- 长输出、重复、maxed 是否能减少；
- pure-soft 是否能从严重退化恢复到可用状态。

本报告只整理“格式稳定 / confidence diffusion guard”这条路线本身，不展开其它后续机制研究。

## 2. 术语与方法速查

### 2.1 基础生成词条

| 词条 | 含义 | 为什么重要 |
|---|---|---|
| `token` | 模型每一步生成的离散文本单位。 | 所有实验都在逐 token 控制下一步输入。 |
| `logits` | 模型对词表每个 token 的未归一化分数。 | 后续转为概率分布。 |
| `probs_original` | 原始 logits softmax 后的概率分布。 | pure-soft 用它加权平均 embedding。 |
| `entropy` | 概率分布的不确定性。分布越散，entropy 越高。 | 用来识别扩散或 spike。 |
| `top1_prob` | 最高概率 token 的概率。 | 低 top1 表示当前选择不集中。 |
| `margin` | top1 和 top2 token 的概率差。 | 小 margin 表示候选很接近，下一步不稳定。 |
| `selected_prob` / `raw_conf` | 被选中 token 的概率，raw 通常指未过滤前概率。 | 早期发现 wrong samples 的 raw_conf 反而更高。 |
| `output_tokens` / `len mean` | 输出 token 数。 | 衡量长输出与退化。 |
| `failed_extraction` | evaluator 无法抽取合法答案。 | 格式稳定路线重点要降低的失败。 |
| `maxed1024` | 输出达到 `max_new_tokens=1024`。 | 通常表示没有自然停止，可能重复或跑飞。 |
| `long>=256` | 输出长度不少于 256 token 的样本数。 | 衡量长输出比例。 |
| `soft ratio` | 实际使用 soft embedding 的比例。 | 判断方法到底干预了多少 token。 |

### 2.2 hard / soft embedding

普通 hard generation：

```text
normal_emb = E[next_token]
last_emb = normal_emb
```

也就是选中一个离散 token，把这个 token 的 embedding 喂回模型。

pure-soft generation：

```text
soft_emb = probs_original @ E
last_emb = soft_emb
```

也就是不只使用 top-1 token，而是把整个词表 embedding 按概率加权平均。  
它保留分布信息，但也会削弱离散 token 边界，容易带来格式模糊、重复和答案漂移。

format 稳定路线的共同本质是：

```text
默认使用 soft_emb；
在格式边界、答案边界或低置信扩散时，
临时切回 normal_emb 或 hard-biased embedding。
```

### 2.3 数据集与评估词条

| 词条 | 含义 |
|---|---|
| VStar | 本阶段最核心的数据集，最早发现 `format_cooldown2` 很强。 |
| MMVP | 有 pair 结构；除了 sample accuracy，还看 pair accuracy。 |
| VisuLogic300 | 输出长、推理复杂，容易暴露 long/maxed 问题。 |
| RealWorldQA fixed200 | 修正图文错配后的 RealWorldQA 200 样本 MCQ 子集。旧错配版本不纳入主结论。 |
| `acc` | 主 evaluator 的 sample-level accuracy。 |
| `pair acc` | MMVP 专用；一对样本都答对才算 pair correct。 |
| `fixed/damaged` | 某方法相对 baseline 修对了多少错题、损坏了多少对题。这里只作辅助分析。 |

### 2.4 方法名速查

| 方法 | 定义 | 定位 |
|---|---|---|
| `cot_orign_greedy` | 普通 hard COT greedy decoding。 | hard baseline。 |
| `pure_soft` | 每一步都用 `soft_emb = probs_original @ E`。 | 暴露 soft 退化的主要对象。 |
| full `lead` | 标准 LEAD 生成。 | 对照方法，用来观察 format/guard 是否能改善已有 soft 路由。 |
| `lead_force_normal` | 走 LEAD wrapper，但强制 hard/normal。 | 检查 wrapper 本身影响。 |
| `format_cooldown2` | 命中格式 token 后，短暂退回 hard/normal，持续 2 步。 | 最强的 pure-soft format 修复器。 |
| `format_cooldown4/8` | cooldown 步数为 4 或 8。 | 检查 cooldown 太长是否过度离散化。 |
| `format_cooldown2_min_step32` | 第 32 步后才允许 format cooldown。 | 检查早期格式边界是否重要。 |
| `highrisk_only_cooldown2` | 只对高风险格式 token 触发。 | 缩窄触发范围的控制。 |
| `format_mix_lambda050/075` | format token 附近不完全 hard，而是 hard/soft 混合。 | 检查是否需要完全离散化。 |
| `normal1_highrisk2` | 普通格式 token cooldown 1 步，高风险 token cooldown 2 步。 | 变长 cooldown 控制。 |
| `cooldown2_cap50/60` | 限制 format cooldown 总触发次数。 | 检查是否过度触发。 |
| `diffuse_collapse` | entropy spike + 低 top1/小 margin 时退回 hard。 | 低置信扩散控制。 |
| `strict_threshold` | 更严格的 diffuse collapse 阈值。 | 提高触发精度的控制。 |
| `patience2` | diffuse 信号需要重复出现才触发。 | 降低误触发。 |
| `late64` | 第 64 步后才允许 collapse。 | 避免过早打断。 |
| `repeat_gate` | 只有检测到重复退化才 collapse。 | 针对重复坏尾部。 |
| `late64_repeat_gate` | `late64 + repeat_gate`。 | 后期重复退化保护。 |
| `guard` | `format_cooldown2 + diffuse/late64 repeat gate`。 | 组合稳定器。 |
| `answer_zone_discrete` | 进入 `</think>` 或 answer 区域后强制 hard。 | 更窄的答案区格式稳定。 |
| `cooldown2_answer_zone` | `format_cooldown2 + answer_zone_discrete`。 | 多信号组合。 |
| `lead_soft_veto_on_diffuse` | LEAD 本来要 soft 时，若出现 diffuse/repeat 信号则 veto soft。 | LEAD 后续 soft 的退化保护。 |
| `quota05` | 允许少量后续 soft，比例约 5%。 | soft 比例调参。 |
| `quota05_format2` | `quota05 + format_cooldown2`。 | quota 的格式稳定版本。 |
| `quota05_guard` | `quota05 + format_cooldown2 + diffuse veto`。 | quota 的组合 guard 版本。 |
| `format gate` | 只在满足 entropy/top1/margin 条件时触发 format cooldown。 | 防止 format 过度干预。 |

### 2.5 容易混淆的区别

`format_cooldown2` 和 `answer_zone_discrete`：

- `format_cooldown2` 管广义格式边界：换行、标点、括号、`answer`、`think` 等。
- `answer_zone_discrete` 只管答案区域，更窄。
- 实验上，单独 answer-zone 远弱于 `format_cooldown2`。

`format_cooldown2` 和 `diffuse_collapse`：

- `format_cooldown2` 根据 token 类型触发，重点是格式边界。
- `diffuse_collapse` 根据概率分布形态触发，重点是低置信扩散和重复。
- 前者是主要贡献，后者更像后期坏尾部保护。

`format_mix_lambda` 和 `format_cooldown2`：

- `format_cooldown2` 通常相当于 format 区域完全 hard。
- `format_mix_lambda050/075` 只部分 hard，保留一部分 soft。
- 实验显示混合不如完全 hard，说明格式边界更需要离散化。

`quota` 和 pure-soft：

- pure-soft 是几乎每一步都 soft。
- quota 是只允许少量位置 soft。
- quota 过多会引入新的不稳定，因此需要 format guard 或更小比例。

## 3. 方法机制

### 3.1 format cooldown

format cooldown 的流程是：

1. 当前步先正常选出 `next_token`。
2. 把 `next_token` decode 成文本。
3. 判断它是不是格式 token。
4. 如果是，就启动一个短 cooldown。
5. cooldown 期间下一步输入从 soft 改成 hard 或 hard-biased。

核心动作可以概括为：

```text
format_emb = lambda * normal_emb + (1 - lambda) * soft_emb
last_emb = format_emb
```

主配置中 `lambda=1.0`，所以 format 区域基本就是：

```text
last_emb = normal_emb
```

这能恢复离散 token 边界，减少格式漂移和答案边界模糊。

### 3.2 confidence diffusion collapse

confidence diffusion collapse 的触发逻辑是：

```text
entropy spike
+ top1_prob 低
+ margin 小
+ 可选后期/重复条件
```

触发后：

```text
last_emb = normal_emb
```

它不关心 token 是否是格式 token，而是关心概率分布是否进入“扩散、不集中、容易重复”的状态。

### 3.3 repeat gate 与 late gate

repeat gate 检测近期生成是否有重复退化，例如最近 n-gram 是否出现重复，或最近窗口 duplicate ratio 是否过高。

late gate 则规定只有后期才允许 collapse，例如：

```text
collapse_min_step = 64
```

组合起来就是 `late64_repeat_gate`：

```text
第 64 步以后
+ 出现低置信扩散
+ 出现重复退化
=> 临时 hard collapse
```

它的目的不是提高早期推理能力，而是防止后半段坏尾部继续扩散。

### 3.4 format gate

format gate 是对 `format_cooldown2` 的过干预控制。  
它不是见到格式 token 就触发，而是额外要求当前分布满足某些不稳定条件，例如：

- entropy 高于阈值；
- top1_prob 低于阈值；
- margin 小于阈值；
- 或更严格的组合条件。

它的目标是减少 format cooldown 的触发次数，但后续结果显示：触发太少会漏掉很多需要稳定的边界。

## 4. 早期 confidence 现象

### 4.1 VStar pure-soft 50

| setting | correct/total | acc |
|---|---:|---:|
| VStar pure-soft 50 | 29/50 | 58.00% |

错误样本表现出更高 raw confidence、更低 entropy、更长输出和更长 latency。  
这说明错题不是简单“不确定”，而是可能进入高置信错误展开。

### 4.2 MMVP pure-soft full

| setting | correct/total | acc | failed extraction |
|---|---:|---:|---:|
| MMVP pure-soft full | 191/300 | 63.67% | 14 |

| metric | correct | wrong |
|---|---:|---:|
| mean_raw_conf | 0.7559 | 0.7838 |
| last20_raw_conf | 0.8560 | 0.8737 |
| mean_raw_entropy | 0.9155 | 0.8312 |
| output length | 149.0 | 284.7 |

高置信尾部明显偏错：

- top 5 by `mean_raw_conf`：100% wrong
- top 5 by `last10_raw_conf`：80% wrong
- top 5 by `last20_raw_conf`：80% wrong
- `last20_raw_conf >= 0.95` 的 accuracy 只有 27.6%

### 4.3 PhysUniBench pure-soft uniform300

| setting | correct/total | acc | failed extraction |
|---|---:|---:|---:|
| PhysUniBench pure-soft uniform300 | 14/300 | 4.67% | 196 |

这组结果说明 pure-soft 的失败不只是 evaluator 问题，而是经常无法收敛到稳定 MCQ 格式。

## 5. VStar pure-soft format 系列

VStar full 的关键 baseline：

| run | correct/total | acc |
|---|---:|---:|
| COT baseline | 131/191 | 68.59% |
| LEAD baseline | 139/191 | 72.77% |
| pure-soft baseline | 112/191 | 58.64% |

### 5.1 cooldown 步数

| run | correct/total | acc | 结论 |
|---|---:|---:|---|
| `format_cooldown8` | 136/191 | 71.20% | 有效，但 cooldown 偏长 |
| `format_cooldown2` | 142/191 | 74.35% | 最强配置 |
| `format_cooldown4` | 138/191 | 72.25% | 有效但弱于 2 步 |

`format_cooldown2` 是这个阶段最重要的发现：它把 pure-soft 从 58.64% 拉到 74.35%。

### 5.2 缩窄触发范围

| run | correct/total | acc | 结论 |
|---|---:|---:|---|
| `format_cooldown2_min_step32` | 139/191 | 72.77% | 延迟到 32 步后触发会下降 |
| `highrisk_only_cooldown2` | 132/191 | 69.11% | 只管 high-risk token 太窄 |
| `normal1_highrisk2` | 131/191 | 68.59% | 普通格式 1 步、高风险 2 步不够 |
| `normal1_highrisk2_late64_repeat` | 129/191 | 67.54% | 再叠 late64 repeat 后更低 |

结论：有效信号不只在少数 high-risk token 上。普通换行、标点、括号等广义格式边界也需要稳定。

### 5.3 hard/soft 混合强度

| run | correct/total | acc | 结论 |
|---|---:|---:|---|
| `format_mix_lambda050` | 135/191 | 70.68% | 半 hard 半 soft，不如完全 hard |
| `format_mix_lambda075` | 135/191 | 70.68% | 75% hard 仍不如 cooldown2 |

这说明格式边界附近更需要明确的离散 token embedding，而不是继续保留较多 soft 成分。

### 5.4 触发次数 cap

| run | correct/total | acc | 结论 |
|---|---:|---:|---|
| `cooldown2_cap50` | 140/191 | 73.30% | 限制触发次数后仍有效，但弱于原始 cooldown2 |
| `cooldown2_cap60` | 140/191 | 73.30% | 与 cap50 接近 |
| `cooldown2_cap50_late64_repeat` | 139/191 | 72.77% | 组合后没有超过 cooldown2 |
| `cooldown2_cap60_late64_repeat` | 140/191 | 73.30% | 没有超过 cooldown2 |

cap 实验说明：`format_cooldown2` 不是靠无限触发取胜，但过早限制触发次数会损失一部分收益。

## 6. confidence diffusion / collapse 系列

### 6.1 collapse 精度消融

| run | correct/total | acc | 结论 |
|---|---:|---:|---|
| `pure_soft_baseline` | 112/191 | 58.64% | 原始 pure-soft |
| `pure_soft_collapse_diffuse` | 114/191 | 59.69% | 普通 diffuse collapse 提升很小 |
| `strict_threshold` | 116/191 | 60.73% | 严格阈值略好 |
| `patience2` | 111/191 | 58.12% | patience 过保守，反而下降 |
| `late64` | 119/191 | 62.30% | 后期触发更稳 |
| `repeat_gate` | 119/191 | 62.30% | 重复 gate 更稳 |
| `late64_repeat_gate` | 119/191 | 62.30% | 与 late64/repeat gate 接近 |

diffuse/collapse 系列的 accuracy 提升有限，但它确认了一个方向：不要太早、太频繁 collapse；更合理的是后期、重复退化时再触发。

### 6.2 与 format2 组合

| run | correct/total | acc |
|---|---:|---:|
| `format_cooldown2` | 142/191 | 74.35% |
| `cooldown2_late64_repeat_gate` | 143/191 | 74.87% |

组合后比 `format_cooldown2` 多 1 题，说明 late64 repeat gate 可以补一点后期坏尾部，但主贡献仍然来自 format cooldown。

## 7. 多信号组合与 answer-zone 实验

### 7.1 answer zone

| run | correct/total | acc | 结论 |
|---|---:|---:|---|
| `answer_zone_discrete` | 126/191 | 65.97% | 单独 answer-zone 太窄 |
| `cooldown2_answer_zone` | 141/191 | 73.82% | 接近 cooldown2，但没有超过 |
| `cooldown2_late64_repeat_answer_zone` | 143/191 | 74.87% | 与 cooldown2+late64 repeat 持平 |

answer-zone 的结论是：答案区 hard 化确实合理，但单独只管答案区不够。真正有效的是更宽的 format cooldown。

### 7.2 多信号组合定位

多信号组合基本可以分成三层：

1. format cooldown：负责格式边界，是主贡献。
2. late64 repeat gate：负责后期重复坏尾部，是辅助保护。
3. answer-zone discrete：负责答案区边界，是窄范围补丁。

组合收益最高的是：

```text
cooldown2 + late64_repeat_gate
cooldown2 + late64_repeat_gate + answer_zone
```

两者都是 143/191 = 74.87%。这说明 answer-zone 没有在已有组合上继续提供明显增益。

## 8. 跨数据集 format gate 与 LEAD guard 复核

### 8.1 format overintervention gates

这组实验试图解决一个问题：

> `format_cooldown2` 是否过度干预？如果只在 entropy/top1/margin 显示不稳定时触发，会不会更好？

MMVP：

| run | acc | pair acc | failed | avg tokens | maxed |
|---|---:|---:|---:|---:|---:|
| `pure_soft_ref` | 61.00% | 32.00% | 28 | 200.4 | 29 |
| `lead_ref` | 70.33% | 42.00% | 0 | 111.4 | 0 |
| `bestcombo_ref` | 67.00% | 40.00% | 1 | 124.2 | 4 |
| `gate_entropy10` | 64.67% | 37.33% | 5 | 131.5 | 7 |
| `gate_top080_margin040` | 64.00% | 36.67% | 6 | 135.5 | 9 |
| `gate_strict` | 65.00% | 38.67% | 4 | 129.4 | 6 |

VisuLogic300：

| run | acc | failed real | avg tokens | maxed |
|---|---:|---:|---:|---:|
| `pure_soft_ref` | 17.67% | 91 | 656.6 | 100 |
| `lead_ref` | 24.67% | 29 | 493.3 | 7 |
| `bestcombo_ref` | 24.33% | 18 | 555.4 | 26 |
| `gate_entropy10` | 17.67% | 23 | 548.6 | 33 |
| `gate_top080_margin040` | 22.00% | 26 | 576.0 | 30 |
| `gate_strict` | 22.33% | 30 | 553.3 | 32 |

结论：format gate 能减少部分触发，但 accuracy 不稳定。过度缩窄触发条件会漏掉需要稳定的格式边界。

### 8.2 LEAD guard 复核

这组实验把 format/diffuse guard 接到 LEAD 与 quota 上，检查它是否能稳定改善标准 LEAD 或后续 soft quota。

VStar：

| run | acc | len mean | maxed1024 | soft ratio |
|---|---:|---:|---:|---:|
| COT | 68.59% | 116.4 | 0 | 0.00% |
| full LEAD | 72.77% | 122.5 | 1 | 1.40% |
| `lead_format2` | 71.20% | 122.2 | 1 | 1.31% |
| `lead_guard` | 71.20% | 122.2 | 1 | 1.31% |
| `quota05` | 70.68% | 120.1 | 1 | 4.91% |
| `quota05_format2` | 73.82% | 117.9 | 1 | 4.58% |
| `quota05_guard` | 73.82% | 117.8 | 1 | 4.58% |

MMVP：

| run | acc | pair acc | len mean | soft ratio |
|---|---:|---:|---:|---:|
| COT | 68.00% | 39.33% | 110.2 | 0.00% |
| full LEAD | 70.33% | 42.00% | 110.4 | 1.18% |
| `lead_format2` | 70.33% | 42.00% | 110.3 | 1.15% |
| `lead_guard` | 70.33% | 42.00% | 110.3 | 1.15% |
| `quota05` | 70.67% | 43.33% | 110.3 | 4.05% |
| `quota05_format2` | 70.33% | 42.67% | 110.2 | 3.78% |
| `quota05_guard` | 70.33% | 42.67% | 110.2 | 3.78% |

VisuLogic300：

| run | acc | len mean | maxed1024 | soft ratio |
|---|---:|---:|---:|---:|
| COT | 21.00% | 527.0 | 12 | 0.00% |
| full LEAD | 24.67% | 492.2 | 4 | 2.47% |
| `lead_format2` | 27.67% | 495.3 | 1 | 1.64% |
| `lead_guard` | 27.67% | 498.0 | 1 | 1.58% |
| `quota05` | 23.00% | 493.9 | 1 | 5.14% |
| `quota05_guard` | 22.33% | 484.2 | 3 | 4.50% |

RealWorldQA fixed200：

| run | acc | len mean | failed | soft ratio |
|---|---:|---:|---:|---:|
| COT | 66.00% | 140.0 | 1 | 0.00% |
| full LEAD | 64.50% | 139.4 | 2 | 1.42% |
| `lead_format2` | 64.00% | 139.7 | 1 | 1.29% |
| `lead_guard` | 64.00% | 139.7 | 1 | 1.29% |
| `quota05` | 65.00% | 132.5 | 0 | 4.84% |
| `quota05_format2` | 66.50% | 135.1 | 1 | 4.68% |
| `quota05_guard` | 67.00% | 135.0 | 1 | 4.67% |

跨数据集结论：

- `lead_format2/lead_guard` 对标准 LEAD 不稳定，不是通用提升。
- `quota05_format2/guard` 在 VStar 和 RealWorldQA 有帮助。
- MMVP 上 `quota05` 本身略好，format2 不增加。
- VisuLogic 上 guard 可降低 maxed，但不保证 accuracy 最高。

## 9. quota + format 调参

### 9.1 VStar

| run | acc | soft ratio |
|---|---:|---:|
| `quota002` | 72.25% | 2.47% |
| `quota003` | 71.73% | 3.10% |
| `quota005` | 70.68% | 4.91% |
| `quota005_format2` | 73.82% | 4.58% |
| `quota008` | 72.25% | 7.27% |
| `quota008_format2` | 72.77% | 6.96% |

VStar 上，quota05 单独会伤，但叠加 format2 后最好。

### 9.2 MMVP

| run | acc | pair acc | soft ratio |
|---|---:|---:|---:|
| `quota002` | 69.33% | 40.67% | 2.06% |
| `quota003` | 68.67% | 41.33% | 2.76% |
| `quota005` | 70.67% | 43.33% | 4.05% |
| `quota005_format2` | 70.33% | 42.67% | 3.78% |
| `quota008` | 67.33% | 37.33% | 6.13% |

MMVP 上，最优是 `quota005` 本身。format2 没有带来额外提升。

### 9.3 RealWorldQA fixed200

| run | acc | soft ratio |
|---|---:|---:|
| `quota002` | 66.00% | 2.61% |
| `quota003` | 67.50% | 3.32% |
| `quota003_format2` | 67.00% | 3.10% |
| `quota005` | 65.00% | 4.84% |
| `quota005_format2` | 66.50% | 4.68% |
| `quota008` | 65.50% | 7.22% |
| `quota008_format2` | 64.00% | 7.14% |

RealWorldQA fixed200 上，最优是 `quota003`，不是 format2 版本。

## 10. 这条路线的最终定位

format 稳定路线的可靠结论是：

1. pure-soft 的退化真实存在，不只是 evaluator 问题。
2. 退化形态包括低熵高置信错题、长输出、重复、答案漂移、抽取失败和 maxed。
3. `format_cooldown2` 是最强的 pure-soft 修复器。
4. `diffuse_collapse / late64_repeat_gate` 单独 accuracy 提升有限，但能作为后期重复坏尾部保护。
5. `answer_zone_discrete` 单独太窄，不能替代广义 format cooldown。
6. `format gate` 缩窄触发范围后跨数据集不稳定，说明很多普通格式边界也需要稳定。
7. 在标准 LEAD 上，format/diffuse guard 的增益不稳定；在 quota 或 pure-soft 上更有价值。

最准确的定位是：

> format stability 是 soft / pure-soft 的输出稳定 guardrail，主要修复格式边界和后期坏尾部；它不是一个单独稳定提高所有数据集 accuracy 的通用 reasoning 方法。

## 11. 可保留结论

### 11.1 方法层面

可保留的核心方法：

- `format_cooldown2`
- `cooldown2 + late64_repeat_gate`
- `pure_soft_guard`
- 在部分 quota 设置中叠加 `format2`

不应作为主配置的尝试：

- `highrisk_only_cooldown2`
- `format_mix_lambda050/075`
- `normal1_highrisk2`
- 单独 `answer_zone_discrete`
- 过窄的 `format gate`
- 过大的 quota 比例，例如 `quota008`

### 11.2 结果层面

最关键的正结果：

| setting | 结果 |
|---|---:|
| VStar pure-soft baseline | 112/191 = 58.64% |
| VStar `format_cooldown2` | 142/191 = 74.35% |
| VStar `cooldown2_late64_repeat_gate` | 143/191 = 74.87% |
| VStar formal `pure_soft_guard` | 74.87%，len mean 127.3，maxed1024 3 |
| VStar `quota005_format2/guard` | 73.82% |
| RealWorldQA fixed200 `quota05_guard` | 67.00% |
| RealWorldQA fixed200 `quota003` | 67.50% |

最关键的负结果：

| setting | 结果 |
|---|---:|
| `pure_soft_collapse_diffuse` | 114/191 = 59.69%，提升很小 |
| `answer_zone_discrete` | 126/191 = 65.97%，单独不足 |
| MMVP `gate_strict` | 65.00%，低于 LEAD ref |
| VisuLogic `gate_strict` | 22.33%，低于 LEAD ref |
| MMVP `quota008` | 67.33%，明显不稳 |
| RealWorldQA `quota008_format2` | 64.00%，明显下降 |

### 11.3 报告结论

这条路线最终可以总结为：

> 格式稳定方法显著修复 pure-soft 的输出退化，尤其能降低长输出、重复、maxed 和格式边界漂移；其中 `format_cooldown2` 是最有效的单一组件，`late64_repeat_gate` 是后期坏尾部保护。跨数据集结果显示，这些方法更适合作为 soft/pure-soft 或 quota 方法的稳定器，而不是单独的通用 accuracy 提升机制。

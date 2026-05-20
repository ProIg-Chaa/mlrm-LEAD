# VStar pure-soft 低置信扩散 collapse 精确路由实验记录

日期：2026-05-18  
数据集：`data/vstar.jsonl`，全量 191 题  
模型：`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`  
基础推理方式：`pure_soft`，`cot_prompt_mode=orign`，greedy 解码  
核心实验目录：`output/experiments/20260518_173645/pure_soft_collapse_precision_vstar_full`

## 1. 背景

前序实验发现，pure-soft 在 VStar 上存在明显的长输出退化问题，尤其是高熵、低置信、top1-top2 margin 很低的扩散型 token。针对这一现象，先做了一个简单策略：

```text
如果当前 token 是低置信扩散型 entropy spike，则把下一步输入从 soft embedding 坍缩为当前离散 token embedding。
```

这个策略在错题并集上效果明显，但在全量 VStar 上收益较小，因为它同时修复了一批错误样本，也破坏了一批原本正确的样本。

因此本轮实验的目标是：让 collapse 触发更精确，减少对原本正确样本的破坏。

## 2. 相关前序结果

### 2.1 错题并集实验

实验目录：

`output/experiments/20260517_181331/pure_soft_collapse_wrong_union_parallel`

错题并集来自 exp1 中 CoT、LEAD、pure-soft 三种方法错误样本的并集，共 102 题。

| 方法 | 正确率 | 平均长度 | p90 长度 | 打满 1024 |
|---|---:|---:|---:|---:|
| pure-soft baseline | 23/102 = 22.55% | 360.98 | 1024 | 18 |
| diffuse-collapse | 41/102 = 40.20% | 200.14 | 284 | 6 |

变化：

- fixed：23 题
- damaged：5 题
- 净提升：+18 题
- collapse 总触发：264 次
- 有触发样本：84/102

结论：在困难/错误样本上，低置信扩散 collapse 明显缓解 pure-soft 退化。

### 2.2 全量普通 collapse

实验目录：

`output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel`

| 方法 | 正确率 | 平均长度 | p90 长度 | 长输出 >=256 | 打满 1024 |
|---|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 237.29 | 1023 | 33 | 18 |
| diffuse-collapse | 114/191 = 59.69% | 157.86 | 251 | 18 | 7 |

变化：

- fixed：23 题
- damaged：21 题
- 净提升：+2 题
- changed outputs：141/191
- collapse 总触发：428 次
- 有触发样本：146/191

结论：普通 collapse 能明显减少长输出，但全量上触发太宽，破坏了不少本来正确的样本。

## 3. 本轮精确路由实验设计

本轮保留低置信扩散作为基础候选条件：

```text
entropy spike
AND
(raw_top1_prob < low_conf_tau OR raw_top1 - raw_top2 < low_margin_tau)
```

在此基础上加入更精确的 gate。

### 3.1 strict_threshold

更严格的低置信扩散阈值：

```text
collapse_entropy_alpha = 2.5
collapse_low_conf_tau = 0.12
collapse_low_margin_tau = 0.03
```

目标：只在更明显危险的低置信扩散上触发。

### 3.2 patience2

近邻窗口中第 2 次出现候选扩散时才 collapse：

```text
collapse_patience = 2
collapse_patience_window = 16
```

目标：避免单个偶然 spike 触发。

### 3.3 late64

前 64 个生成 token 不允许 collapse：

```text
collapse_min_step = 64
```

目标：保护早期视觉理解和推理展开阶段。

### 3.4 repeat_gate

只有出现重复退化迹象时才允许 collapse：

```text
collapse_require_repeat_degen = true
collapse_repeat_ngram = 3
collapse_recent_repeat_window = 32
collapse_recent_repeat_tau = 0.35
```

目标：把 collapse 从“看到扩散就介入”变成“看到扩散且已有退化苗头才介入”。

## 4. 本轮全量结果

对比命令：

```bash
bash output/experiments/20260518_173645/pure_soft_collapse_precision_vstar_full/compare_after_done.sh
```

| 方法 | 正确率 | 相比 baseline | fixed / damaged | changed outputs | 平均长度 | p90 | 长输出 >=256 | 打满 1024 | 触发样本 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | - | - | - | 237.29 | 1023 | 33 | 18 | - |
| 普通 collapse | 114/191 = 59.69% | +2 | 23 / 21 | 141 | 157.86 | 251 | 18 | 7 | 146 |
| strict_threshold | 116/191 = 60.73% | +4 | 15 / 11 | 86 | 200.32 | 383 | 26 | 17 | 90 |
| patience2 | 111/191 = 58.12% | -1 | 15 / 16 | 84 | 194.30 | 324 | 24 | 14 | 86 |
| late64 | 119/191 = 62.30% | +7 | 12 / 5 | 55 | 181.09 | 260 | 21 | 10 | 58 |
| repeat_gate | 119/191 = 62.30% | +7 | 10 / 3 | 43 | 193.94 | 322 | 25 | 12 | 45 |

collapse 触发统计：

| 方法 | collapse 总次数 | 有触发样本 | 平均触发 | median | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| 普通 collapse | 428 | 146/191 | 2.24 | 2 | 5 | 21 |
| strict_threshold | 157 | 90/191 | 0.82 | 0 | 2 | 8 |
| patience2 | 168 | 86/191 | 0.88 | 0 | 2 | 10 |
| late64 | 144 | 58/191 | 0.75 | 0 | 3 | 8 |
| repeat_gate | 86 | 45/191 | 0.45 | 0 | 1 | 12 |

## 5. 主要结论

### 5.1 低置信扩散确实是有用信号

从错题并集和全量实验看，低置信扩散与 pure-soft 退化高度相关。对这类 token 继续使用 soft embedding 容易导致输出变长、重复、格式漂移，甚至打满 1024。

collapse 后，长输出数量整体下降：

- baseline：长输出 >=256 有 33 题，打满 1024 有 18 题
- 普通 collapse：长输出 >=256 降到 18 题，打满 1024 降到 7 题
- late64：长输出 >=256 为 21 题，打满 1024 为 10 题
- repeat_gate：长输出 >=256 为 25 题，打满 1024 为 12 题

### 5.2 但普通 collapse 触发太宽

普通 collapse 全量上 changed outputs 达到 141/191，虽然 fixed 23 题，但 damaged 21 题，净收益只有 +2。

这说明：

```text
低置信扩散是危险信号，但不是充分条件。
```

需要加入阶段、重复退化等上下文 gate。

### 5.3 early token 不宜轻易介入

`late64` 是本轮最佳之一：

- 正确率：119/191 = 62.30%
- fixed / damaged：12 / 5
- changed outputs：55
- 触发样本：58/191

它说明早期 high entropy / diffuse 不一定是坏事。模型在前几十个 token 中可能正在做视觉描述、目标定位、问题理解或正常推理展开。此时强制 collapse 会破坏本来正确的推理轨迹。

### 5.4 重复退化 gate 是最精确的信号

`repeat_gate` 同样达到 119/191 = 62.30%，但更干净：

- fixed：10
- damaged：3
- changed outputs：43
- 触发样本：45/191

这说明如果把 collapse 限制在“低置信扩散 + 已出现重复退化苗头”的场景，能显著降低对正确样本的破坏。

这个结果很符合当前假设：

```text
不是所有高熵扩散都需要 collapse；
真正危险的是正在把 pure-soft 带向重复/长输出退化的扩散。
```

### 5.5 patience2 不成立

`patience2` 没有提升：

- 正确率：111/191 = 58.12%
- fixed / damaged：15 / 16
- 净变化：-1

这说明简单地“等第二次扩散再介入”不是好判断。扩散次数本身不够区分正常推理和退化推理。

## 6. 当前方法理解

现在可以把 pure-soft 的风险粗略分成四类：

```text
低置信扩散 -> discrete / collapse，不要继续 soft
格式不确定 -> discrete / cooldown
关系不确定 -> 可能短 soft，但要强约束
视觉不确定 -> 可能弱视觉 anchor
```

其中证据最强的是低置信扩散。它现在已经有两层结论：

1. 低置信扩散会影响 pure-soft 的稳定性。
2. 低置信扩散必须结合上下文 gate，否则会破坏正确样本。

目前最有效的上下文 gate 是：

```text
late-only：不要太早介入
repeat-gated：出现重复退化苗头再介入
```

## 7. 代码改动位置

核心实现：

- `lead/generation_utils.py`
  - `generate_pure_soft(...)` 中加入 collapse 路由。
  - 新增低置信扩散判断、entropy spike 判断、late gate、patience gate、repeat degeneration gate。
  - trace 中记录 `collapse_on_diffuse`、`collapse_candidate`、`raw_top1_prob`、`raw_margin` 等字段。

参数入口：

- `main.py`
  - 新增 `--pure_soft_collapse_on_diffuse`
  - 新增 `--collapse_entropy_window`
  - 新增 `--collapse_entropy_alpha`
  - 新增 `--collapse_min_history`
  - 新增 `--collapse_min_entropy`
  - 新增 `--collapse_low_conf_tau`
  - 新增 `--collapse_low_margin_tau`
  - 新增 `--collapse_min_step`
  - 新增 `--collapse_patience`
  - 新增 `--collapse_patience_window`
  - 新增 `--collapse_require_repeat_degen`
  - 新增 `--collapse_repeat_ngram`
  - 新增 `--collapse_recent_repeat_window`
  - 新增 `--collapse_recent_repeat_tau`

推理传参：

- `lead/inference.py`
  - 在 `method == "pure_soft"` 时把上述参数传入 `generate_pure_soft(...)`。

脚本：

- `script/exp5_16/run_pure_soft_collapse_precision_vstar_full.sh`

## 8. 下一步建议

优先做组合实验：

```text
late64 + repeat_gate
```

预期：

- damaged 可能进一步下降。
- fixed 也可能下降。
- 关键看净收益是否超过 +7，以及 changed outputs 是否继续减少。

其次可以做：

```text
late32 + repeat_gate
late96 + repeat_gate
strict_threshold + late64
strict_threshold + repeat_gate
```

目标是找到最稳的触发边界，而不是单纯最大化 collapse 次数。

目前阶段最值得写进核心结论的是：

```text
低置信扩散是 pure-soft 退化的重要信号；
但只有结合生成阶段或退化迹象进行精确路由，才能在提升正确率的同时减少对正确样本的破坏。
```

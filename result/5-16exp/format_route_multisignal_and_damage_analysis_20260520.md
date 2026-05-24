# VStar pure-soft format 路由、多信号混合与 damaged 分析

时间：2026-05-20

## 1. 背景

本阶段目标是继续提升 VStar full 上 pure-soft 路由方法的成绩。

此前最强配置为：

- `cooldown2 + late64_repeat_gate`
- 目录：`output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0`
- 准确率：`143/191 = 74.87%`

核心问题变成：

- 如何减少这个 best combo 相比 pure-soft baseline 造成的 `9` 个 damaged 样本；
- format cooldown 是否可以进一步精细化；
- answer-zone、多信号融合、mixed embedding 是否能带来收益。

## 2. 当前核心实现

代码位置：

- 参数定义：`main.py`
- 参数传递：`lead/inference.py`
- 路由实现：`lead/generation_utils.py` 的 `generate_pure_soft(...)`

pure-soft 中每步构造两类 embedding：

```python
normal_emb = E[next_tokens]
soft_emb = probs_original @ E
```

原始 hard format cooldown 动作为：

```python
last_emb = torch.where(route_mask[:, None], normal_emb, soft_emb)
```

近期新增参数：

- `--format_cooldown_min_step`
- `--format_cooldown_highrisk_only`
- `--format_cooldown_normal_steps`
- `--format_cooldown_highrisk_steps`
- `--format_cooldown_mix_lambda`

其中 mixed embedding 版本为：

```python
format_emb = lambda * normal_emb + (1 - lambda) * soft_emb
```

collapse / answer-zone 仍使用 hard discrete。

## 3. 主要结果汇总

### 3.1 基线与当前 best

| 方法 | Acc | direct | relative | long>=256 | max1024 | missing_answer |
|---|---:|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 70/115 | 42/76 | 33 | 18 | 40 |
| cooldown2 | 142/191 = 74.35% | 86/115 | 56/76 | 9 | 4 | 16 |
| cooldown2 + late64_repeat_gate | 143/191 = 74.87% | 86/115 | 57/76 | 8 | 3 | 15 |

当前 best 相比 pure-soft baseline：

- fixed：`40`
- damaged：`9`
- net：`+31`

### 3.2 answer-zone 与多信号混合

实验目录：

- `output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full`
- `output/experiments/20260520_133545/pure_soft_multisignal_mix_vstar_full`

| 方法 | Acc | 相比 baseline | long>=256 | max1024 |
|---|---:|---:|---:|---:|
| answer_zone | 126/191 = 65.97% | fixed 16 / damaged 2 / net +14 | 33 | 18 |
| cooldown2 + answer_zone | 141/191 = 73.82% | fixed 39 / damaged 10 / net +29 | 8 | 4 |
| cooldown2 + late64_repeat + answer_zone | 143/191 = 74.87% | fixed 40 / damaged 9 / net +31 | 8 | 3 |

结论：

- answer-zone 单独有格式修复能力，尤其能消除 `empty_paren_answer`。
- 但它太晚，不能解决 reasoning 阶段退化和长输出。
- 与 cooldown2 / best combo 叠加后没有新增收益。

### 3.3 format cooldown 细化

实验目录：

- `output/experiments/20260520_185703/pure_soft_format_refine_vstar_full`

| 方法 | Acc | direct | relative | long>=256 | max1024 | missing_answer |
|---|---:|---:|---:|---:|---:|---:|
| highrisk_only | 132/191 = 69.11% | 77/115 | 55/76 | 23 | 12 | 30 |
| min_step32 | 139/191 = 72.77% | 81/115 | 58/76 | 20 | 8 | 17 |

相对 cooldown2：

- `highrisk_only`: fixed 12 / damaged 22 / net -10
- `min_step32`: fixed 15 / damaged 18 / net -3

触发量：

- `highrisk_only`: format cooldown total `2339`
- `min_step32`: format cooldown total `8434`
- 原 `cooldown2`: 约 `8587`

结论：

- 只保留高危结构 token 不够，普通标点/换行也参与稳定 pure-soft。
- 跳过前 32 步也会损失性能，说明 early reasoning 阶段的 format cooldown 有实际贡献。

### 3.4 variable cooldown 与 mixed embedding

实验目录：

- `output/experiments/20260520_194525/pure_soft_format_variable_and_mixed_vstar_full`

| 方法 | Acc | direct | relative | long>=256 | max1024 | missing_answer |
|---|---:|---:|---:|---:|---:|---:|
| normal1_highrisk2 | 131/191 = 68.59% | 77/115 | 54/76 | 16 | 11 | 24 |
| normal1_highrisk2 + late64 | 129/191 = 67.54% | 75/115 | 54/76 | 15 | 9 | 22 |
| mix lambda=0.75 | 135/191 = 70.68% | 80/115 | 55/76 | 15 | 6 | 18 |
| mix lambda=0.50 | 135/191 = 70.68% | 79/115 | 56/76 | 13 | 7 | 18 |

相对 cooldown2：

- `normal1_highrisk2`: fixed 10 / damaged 21 / net -11
- `mix lambda=0.75`: fixed 10 / damaged 17 / net -7
- `mix lambda=0.50`: fixed 6 / damaged 13 / net -7

结论：

- 把普通格式 token 从 cooldown2 降为 cooldown1 会明显损失。
- mixed embedding 强度不足，不能替代 hard discrete。
- 当前 VStar 上，format 路由需要足够强的 hard discrete 才能抑制 pure-soft 退化。

## 4. Damaged 样本分析

当前 best combo 相比 pure-soft baseline 的 damaged 样本：

```text
[34, 51, 75, 81, 120, 126, 135, 150, 175]
```

数量：`9`

其中：

- direct_attributes：`4`
- relative_position：`5`

一个关键发现：

> 这 9 个 damaged 全部在 `cooldown2` 中已经错误，因此主要 damage 来源是 format cooldown，而不是 late64 repeat gate。

另一个关键发现：

> 单独 `answer_zone` 在这 9 个样本上全部正确，说明这些样本不是最终答案区需要离散化，而是在 reasoning 阶段被强 format cooldown 改坏。

### 4.1 damaged 样本明细

| id | subtopic | GT | best 提取 | best fmt 次数 | collapse 次数 | 备注 |
|---:|---|---|---|---:|---:|---|
| 34 | direct_attributes | A | B | 72 | 0 | helmet color，baseline 长输出但抽取正确 |
| 51 | direct_attributes | A | C | 62 | 0 | helmet color |
| 75 | direct_attributes | A | D | 49 | 0 | backpack color |
| 81 | direct_attributes | A | None | 51 | 0 | flag color，best 丢答案 |
| 120 | relative_position | B | A | 40 | 1 | stroller/person relation |
| 126 | relative_position | A | B | 36 | 0 | trash can / baby carriage |
| 135 | relative_position | B | A | 58 | 2 | white trousers / person in blue |
| 150 | relative_position | B | None | 39 | 0 | suitcase / river，best 丢答案 |
| 175 | relative_position | B | A | 80 | 0 | surfboard / umbrella |

### 4.2 damaged 与其它组的触发差异

| 组 | n | best fmt mean | best fmt median | best collapse sum | best len mean |
|---|---:|---:|---:|---:|---:|
| damaged | 9 | 54.11 | 51 | 3 | 160.1 |
| baseline correct 且未 damaged | 103 | 32.57 | 26 | 9 | 98.5 |
| fixed | 40 | 42.52 | 40 | 11 | 148.9 |
| remaining wrong | 39 | 74.85 | 42 | 12 | 173.5 |

解释：

- damaged 样本的 format cooldown 触发量明显高于“baseline 正确且未损坏”的样本。
- 但 fixed 样本也有较高触发量，因此不能简单用触发次数低阈值屏蔽，否则会损失收益。
- remaining wrong 的触发量更高，说明高触发量同时对应“难题/退化题”，不是单独的 damage 判据。

## 5. 如何减少这 9 个 damaged

### 5.1 不建议继续做的方向

不建议继续：

- 全局降低 format cooldown 强度；
- 普通格式 cooldown1 + 高危 cooldown2；
- mixed lambda 替代 hard discrete；
- 高危 token only；
- answer-zone 与 cooldown2 简单 OR 融合。

这些实验都已经显示会明显损失整体准确率。

### 5.2 更有希望的方向一：damaged-aware 保护门控

核心思想：

> 保留 hard cooldown2 的强稳定能力，只在“疑似本来已经稳定”的样本上减少干预。

候选门控：

1. 低熵稳定门控

```text
如果当前 raw_entropy 很低且 raw_top1_prob 很高，则不触发 format cooldown
```

直觉：

- damaged 样本里不少是 baseline 原本能短答案正确的题。
- 在模型已经高置信时，强行 hard discrete 可能把视觉判断路径改偏。

可跑参数：

- `format_skip_if_entropy_lt=0.5`
- `format_skip_if_entropy_lt=0.8`
- 或 `format_skip_if_top1_gt=0.85`

2. format budget cap

```text
每个样本最多允许 N 次 format cooldown
```

候选：

- cap 40
- cap 50
- cap 60

直觉：

- damaged 的 fmt median 是 `51`，未损坏 baseline-correct 的 median 是 `26`。
- 但 fixed median 是 `40`，所以 cap 不能太低。

建议先试：

- `cap=60`，较保守；
- `cap=50`，更强；
- 不建议先试 `cap=40`，可能伤 fixed。

3. relative-position 保护不建议作为硬规则

damaged 里 relative_position 有 5 个，但 fixed 里也有大量 relative_position 收益。因此不能简单按 subtopic 关掉路由。

### 5.3 更有希望的方向二：late64 repeat gate 不碰低熵关系题

best combo 的 9 个 damaged 中只有 3 次 collapse，分布在：

- id 120：collapse 1
- id 135：collapse 2

这些样本在 cooldown2 中已经错，因此 collapse 不是根因。但它可能进一步加深 relation 错误。

可以尝试：

```text
collapse 只在 output length >= 128 后触发
```

或：

```text
collapse 只在 max entropy spike >= 4.0 且重复率更高时触发
```

但预期收益小，因为主要 damage 来自 cooldown2。

### 5.4 更有希望的方向三：错题子集上测试视觉 anchor

视觉信息现在可以开始做，但建议作为“剩余错误/损坏保护”的旁线，而不是全量主干。

优先级：

1. 在 best combo remaining wrong + damaged 子集上做 sidecar 视觉诊断。
2. 看 damaged 是否视觉 grounding 本来正确但被文本路由改坏。
3. 只在 relation/direct 中低视觉 grounding 的 token 上测试轻量视觉 anchor。

不建议直接全量加入视觉 anchor，因为此前 attention/eager 和主路径污染问题已经很明显。

## 6. 下一步最推荐实验

优先做两类：

### 实验 A：format cooldown budget cap

目标：

- 减少 format cooldown 过量介入的 damaged；
- 保留 hard cooldown2 强稳定性。

建议并行：

- `cooldown2_cap60`
- `cooldown2_cap50`
- `cooldown2_cap60 + late64_repeat`
- `cooldown2_cap50 + late64_repeat`

### 实验 B：低熵/高置信 skip gate

目标：

- 在模型已经稳定时不强制离散；
- 避免把原本 baseline 正确的短答案样本改坏。

建议并行：

- `cooldown2_skip_entropy_lt_0.5`
- `cooldown2_skip_entropy_lt_0.8`
- `cooldown2_skip_top1_gt_0.85`
- 与 `late64_repeat` 组合一个最佳版本。

判断标准：

- 是否超过 `143/191`；
- damaged 是否少于 `9`；
- fixed 是否保持接近 `40`；
- `long>=256` 和 `max1024` 是否仍接近 best combo。

## 7. 当前结论

当前最稳结论：

1. `cooldown2 + late64_repeat_gate` 仍是当前最佳：`143/191 = 74.87%`。
2. format cooldown 的收益依赖 hard discrete，不宜全局弱化。
3. damage 主要来自 cooldown2 的 reasoning 阶段强干预，而不是 answer-zone 或 late64。
4. 下一步要做“保护门控”，不是继续降低整体路由强度。
5. 视觉信息可以开始做诊断，但不应直接替代当前主线。


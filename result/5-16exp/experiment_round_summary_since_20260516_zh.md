# 2026-05-16 以来实验总报告：pure-soft 路由、视觉注入与跨数据集验证

更新时间：2026-05-28  
项目：`/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD`  
模型：`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`

本文汇总 2026-05-16 以来这一轮主要实验。重点是：从高熵 token 观察出发，逐步定位 pure-soft 的退化来源，设计 format / collapse / visual bias 路由，并在 VStar、MMVP、VisuLogic、RealWorldQA 上验证边界。

## 1. 核心问题与实验主线

最初的问题是：

> 模型在错题上是否更自信？高熵 token 是否意味着视觉 grounding 不足？能否在高熵或视觉注意力不足时注入视觉信息提升性能？

经过本轮实验，问题被重新表述为：

> pure-soft 的主要退化不是单纯视觉信息不足，而是生成轨迹在格式 token、长输出、重复退化和答案边界附近不稳定。有效方法不是全程加视觉，而是在特定 token / 阶段上做保守路由。

当前最重要的技术抽象是：

```text
token / sample state -> route signal -> route action
```

主要动作：

| action | 含义 |
|---|---|
| `pure_soft` | 下一步输入使用 `probs @ embedding_matrix` |
| `hard_discrete` | 下一步输入切回 `embedding[next_token]` |
| `format_cooldown` | 格式 token 后若干步 hard discrete |
| `image_pad_bias` | 在 soft embedding 中混入小比例 `<|image_pad|>` embedding |

## 2. 起点：VStar 三方法与高熵 token 分类

实验目录：

```text
output/experiments/20260516_183300/exp1_vstar_spike_type_parallel
```

VStar 全量 191 题：

| 方法 | Acc | direct_attributes | relative_position |
|---|---:|---:|---:|
| COT | 131/191 = 68.59% | 84/115 = 73.04% | 47/76 = 61.84% |
| LEAD | 139/191 = 72.77% | 82/115 = 71.30% | 57/76 = 75.00% |
| pure-soft | 112/191 = 58.64% | 70/115 = 60.87% | 42/76 = 55.26% |

高熵 spike 规则：

```text
H_t > local_mean(16) + 2.0 * local_std(16)
min_history = 4
min_entropy = 1.0
```

高熵 token 类型分布：

| 方法 | spikes | visual | relation | format | answer | diffuse_low_conf | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| COT | 1188 | 11.7% | 3.4% | 4.9% | 4.6% | 29.1% | 46.3% |
| LEAD | 1166 | 13.0% | 3.8% | 5.0% | 4.2% | 28.5% | 45.6% |
| pure-soft | 1390 | 11.9% | 3.2% | 5.8% | 3.6% | 34.5% | 41.0% |

关键观察：

- 高熵 token 并不等于视觉 token。
- pure-soft 的 `diffuse_low_conf` 比例更高，且错题中更明显。
- pure-soft 错题长度显著更长：正确样本均长 113.9，错误样本均长 412.2。
- pure-soft 长输出 >=256 有 33 题，很多接近或打满 1024。

阶段结论：

> 高熵要继续分型。视觉不确定、格式不稳定、低置信扩散、关系推理不能用同一种干预处理。

## 3. 低置信扩散 collapse：有效但必须保守

核心设想：

```text
entropy spike
AND (raw_top1_prob 低 OR top1-top2 margin 小)
=> 下一步输入从 soft embedding 坍缩为 discrete embedding
```

### 3.1 错题并集实验

目录：

```text
output/experiments/20260517_181331/pure_soft_collapse_wrong_union_parallel
```

| 方法 | Acc | 平均长度 | p90 | 打满1024 |
|---|---:|---:|---:|---:|
| pure-soft baseline | 23/102 = 22.55% | 360.98 | 1024 | 18 |
| diffuse-collapse | 41/102 = 40.20% | 200.14 | 284 | 6 |

在困难/错题并集上，collapse 明显有效：fixed 23，damaged 5，net +18。

### 3.2 VStar 全量与精确 gate

目录：

```text
output/experiments/20260518_173645/pure_soft_collapse_precision_vstar_full
```

| 方法 | Acc | fixed / damaged | changed | 平均长度 | 长>=256 | 打满1024 | 触发样本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | - | - | 237.29 | 33 | 18 | - |
| 普通 collapse | 114/191 = 59.69% | 23 / 21 | 141 | 157.86 | 18 | 7 | 146 |
| strict_threshold | 116/191 = 60.73% | 15 / 11 | 86 | 200.32 | 26 | 17 | 90 |
| patience2 | 111/191 = 58.12% | 15 / 16 | 84 | 194.30 | 24 | 14 | 86 |
| late64 | 119/191 = 62.30% | 12 / 5 | 55 | 181.09 | 21 | 10 | 58 |
| repeat_gate | 119/191 = 62.30% | 10 / 3 | 43 | 193.94 | 25 | 12 | 45 |

组合实验：

```text
output/experiments/20260518_200744/pure_soft_collapse_late64_repeat_gate_vstar_full
```

| 方法 | Acc | changed | fixed | damaged | net |
|---|---:|---:|---:|---:|---:|
| late64 + repeat_gate | 119/191 = 62.30% | 34 | 8 | 1 | +7 |

阶段结论：

> low-confidence diffuse 是有效危险信号，但不是充分条件。加入 step>=64 和 repeat degeneration 后，damage 显著下降，说明早期高熵/扩散可能是正常推理展开，不宜轻易干预。

## 4. Format cooldown：本轮最关键突破

核心设想：

```text
如果当前 token 是换行、标点、括号、think、answer、option 等格式/结构 token：
    接下来若干步使用 discrete embedding
```

目录：

```text
output/experiments/20260519_181159/pure_soft_format_cooldown_vstar_full
output/experiments/20260519_234017/pure_soft_format_cooldown_ablation_vstar_full
```

| 方法 | Acc | mean len | long>=256 | max1024 | missing_answer |
|---|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 237.29 | 33 | 18 | 40 |
| format_cooldown8 | 136/191 = 71.20% | 121.28 | 12 | 0 | 17 |
| format_cooldown4 | 138/191 = 72.25% | 123.72 | 12 | 1 | - |
| format_cooldown2 | 142/191 = 74.35% | 131.08 | 9 | 4 | 16 |

`format_cooldown2` 相对 pure-soft baseline：

```text
fixed = 40
damaged = 10
net = +30
```

阶段结论：

> pure-soft 在格式边界附近非常不稳定。短暂 hard discrete 能显著稳定推理和答案输出，是当前最重要的正收益来源。

## 5. Bestcombo：format cooldown2 + late64 repeat gate

目录：

```text
output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full
```

| 方法 | Acc | direct_attributes | relative_position | long>=256 | max1024 |
|---|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 70/115 | 42/76 | 33 | 18 |
| cooldown2 | 142/191 = 74.35% | 86/115 | 56/76 | 9 | 4 |
| cooldown2 + late64_repeat_gate | 143/191 = 74.87% | 86/115 | 57/76 | 8 | 3 |

相对 pure-soft baseline：

```text
fixed = 40
damaged = 9
net = +31
```

解释：

- 主收益来自 `format_cooldown2`。
- `late64_repeat_gate` 在此基础上小幅补强，同时保持低 damage。
- 这个组合后来简称 `bestcombo`。

## 6. Answer-zone、多信号与削弱 format 的消融

目录：

```text
output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full
output/experiments/20260520_133545/pure_soft_multisignal_mix_vstar_full
output/experiments/20260520_185703/pure_soft_format_refine_vstar_full
output/experiments/20260520_194525/pure_soft_format_variable_and_mixed_vstar_full
```

### 6.1 Answer-zone

| 方法 | Acc | 相比 baseline | long>=256 | max1024 |
|---|---:|---:|---:|---:|
| answer_zone | 126/191 = 65.97% | net +14 | 33 | 18 |
| cooldown2 + answer_zone | 141/191 = 73.82% | net +29 | 8 | 4 |
| cooldown2 + late64_repeat + answer_zone | 143/191 = 74.87% | net +31 | 8 | 3 |

结论：answer-zone 能修答案格式，但介入太晚，不能解决 reasoning 阶段的长输出/重复退化。

### 6.2 削弱 format cooldown 的实验

| 方法 | Acc | long>=256 | max1024 | 结论 |
|---|---:|---:|---:|---|
| highrisk_only | 132/191 = 69.11% | 23 | 12 | 只保留 answer/think/括号等高危 token 不够 |
| min_step32 | 139/191 = 72.77% | 20 | 8 | 跳过早期 format 会掉分 |
| normal1_highrisk2 | 131/191 = 68.59% | 16 | 11 | 普通格式只 cooldown1 不够 |
| mix lambda=0.75 | 135/191 = 70.68% | 15 | 6 | mixed embedding 不如 hard discrete |
| mix lambda=0.50 | 135/191 = 70.68% | 13 | 7 | 同上 |

阶段结论：

> 不能简单弱化 format 路由。VStar 上 format cooldown 的收益依赖 hard discrete，且普通标点/换行也有稳定作用。

## 7. Damaged 样本分析

bestcombo 相比 pure-soft baseline 的 damaged 样本：

```text
[34, 51, 75, 81, 120, 126, 135, 150, 175]
```

数量：9。

关键发现：

- 这 9 个样本在 `cooldown2` 中已经错误，damage 主要来自 format cooldown，不是 late64 repeat gate。
- 单独 `answer_zone` 在这 9 个样本上全部正确，说明损坏发生在 reasoning 阶段，而不是最终答案区。
- damaged 样本 format cooldown 触发量均值为 54.11，高于 baseline 正确且未 damaged 样本的 32.57。
- 但 fixed 样本触发量也较高，不能简单按触发次数屏蔽。

阶段结论：

> format cooldown 是核心稳定器，但确实会破坏少量本来正确样本。下一步应该做 damaged-aware 保护，而不是全局削弱 format。

## 8. 跨数据集：bestcombo 的能力边界

跨数据集代表结果：

| 数据集 | pure-soft | LEAD | bestcombo |
|---|---:|---:|---:|
| VStar | 112/191 = 58.64% | 139/191 = 72.77% | 143/191 = 74.87% |
| MMVP sample | 183/300 = 61.00% | 211/300 = 70.33% | 201/300 = 67.00% |
| MMVP pair | 48/150 = 32.00% | 63/150 = 42.00% | 60/150 = 40.00% |
| VisuLogic300 | 53/300 = 17.67% | 74/300 = 24.67% | 73/300 = 24.33% |

结论：

- bestcombo 在 VStar 上明显强。
- MMVP 上 bestcombo 接近但低于 LEAD。
- VisuLogic 上 bestcombo 接近 LEAD，但没有超过。
- 这说明 bestcombo 解决的是 pure-soft 的生成退化，不等价于 LEAD 的视觉 anchor 能力。

## 9. Format 过度干预假设被基本否定

目录：

```text
output/experiments/20260521_191535/format_overintervention_gates_mmvp_visulogic
```

设计：给 format cooldown 加不确定性 gate，减少触发。

MMVP：

| 方法 | sample acc | pair acc | fmt active/样本 | maxed | failed |
|---|---:|---:|---:|---:|---:|
| pure_soft | 183/300 = 61.00% | 48/150 = 32.00% | 0.0 | 29 | 28 |
| LEAD | 211/300 = 70.33% | 63/150 = 42.00% | 0.0 | 0 | 0 |
| bestcombo | 201/300 = 67.00% | 60/150 = 40.00% | 41.4 | 4 | 1 |
| gate_entropy10 | 194/300 = 64.67% | 56/150 = 37.33% | 12.2 | 7 | 5 |
| gate_top080_margin040 | 192/300 = 64.00% | 55/150 = 36.67% | 18.2 | 9 | 6 |
| gate_strict | 195/300 = 65.00% | 58/150 = 38.67% | 12.0 | 6 | 4 |

VisuLogic300：

| 方法 | Acc | fmt active/样本 | maxed | failed_real |
|---|---:|---:|---:|---:|
| pure_soft | 53/300 = 17.67% | 0.0 | 100 | 91 |
| LEAD | 74/300 = 24.67% | 0.0 | 7 | 29 |
| bestcombo | 73/300 = 24.33% | 251.6 | 26 | 18 |
| gate_entropy10 | 53/300 = 17.67% | 40.2 | 33 | 23 |
| gate_top080_margin040 | 66/300 = 22.00% | 63.7 | 30 | 26 |
| gate_strict | 67/300 = 22.33% | 36.2 | 32 | 30 |

阶段结论：

> 简单削弱 format cooldown 不是好方向。format cooldown 虽然触发多，但显著减少 maxed 和 failed，是 pure-soft 稳定化的核心。

## 10. LEAD simple visual anchor 消融

目录：

```text
output/experiments/20260521_152817/lead_simple_anchor_ablation_mmvp_visulogic_vstar
```

| 数据集 | 原 LEAD | 关闭 simple anchor |
|---|---:|---:|
| MMVP sample | 211/300 = 70.33% | 209/300 = 69.67% |
| MMVP pair | 63/150 = 42.00% | 61/150 = 40.67% |
| VisuLogic300 | 74/300 = 24.67% | 65/300 = 21.67% |
| VStar | 139/191 = 72.77% | 137/191 = 71.73% |

阶段结论：

> 原始 LEAD 的 `<|image_pad|>` simple anchor 是轻量但有效的视觉先验，尤其对 VisuLogic 贡献明显。这支持“视觉信息有用，但应轻量注入”的判断。

## 11. 视觉 image_pad bias：有用但阶段敏感

### 11.1 Full visual bias

目录：

```text
output/experiments/20260522_125332/bestcombo_image_pad_bias_vstar_mmvp_visulogic
```

| 数据集 | bestcombo | full bias λ=0.05 |
|---|---:|---:|
| VStar | 143/191 = 74.87% | 135/191 = 70.68% |
| MMVP sample | 201/300 = 67.00% | 207/300 = 69.00% |
| MMVP pair | 60/150 = 40.00% | 63/150 = 42.00% |
| VisuLogic300 | 73/300 = 24.33% | 约 74/300 = 24.67% |

结论：full bias 对 MMVP 有收益，但明显伤 VStar。

### 11.2 Entropy-gated visual bias

目录：

```text
output/experiments/20260522_184028/bestcombo_image_pad_bias_entropy_gate
```

| 数据集 | bestcombo | entropy>=1.0 | entropy>=1.5 | entropy>=2.0 |
|---|---:|---:|---:|---:|
| VStar | 143/191 = 74.87% | 133/191 = 69.63% | 142/191 = 74.35% | 132/191 = 69.11% |
| MMVP sample | 201/300 = 67.00% | 208/300 = 69.33% | 199/300 = 66.33% | 198/300 = 66.00% |
| VisuLogic300 | 73/300 = 24.33% | 67/300 = 22.33% | 69/300 = 23.00% | 62/300 = 20.67% |

结论：

> 高熵不等于视觉不足。entropy gate 不能区分 visual / format / relation / diffuse，因此不稳定。

### 11.3 Phase-gated visual bias

目录：

```text
output/experiments/20260523_121058/bestcombo_image_pad_bias_phase_gate
```

| 数据集 | bestcombo | full bias 0.05 | early | mid | late |
|---|---:|---:|---:|---:|---:|
| VStar | 143/191 = 74.87% | 135/191 = 70.68% | 135/191 = 70.68% | 约 144-145/191 | 143/191 = 74.87% |
| MMVP sample | 201/300 = 67.00% | 207/300 = 69.00% | 207/300 = 69.00% | 201/300 = 67.00% | 201/300 = 67.00% |
| MMVP pair | 60/150 = 40.00% | 63/150 = 42.00% | 63/150 = 42.00% | 60/150 = 40.00% | 60/150 = 40.00% |
| VisuLogic300 | 73/300 = 24.33% | 约 74/300 | 73-74/300 | 76-77/300 | 71-72/300 |

阶段结论：

- VStar 的主要伤害来自 early visual bias。
- MMVP 更吃 early visual bias。
- mid 是当前最像通用安全窗口。
- late 安全但新增收益弱。

### 11.4 VStar early damage 集 lambda sweep

目录：

```text
output/experiments/20260524_131252/vstar_damage_image_pad_lambda_sweep
```

破坏集：bestcombo 原本答对，但 early image_pad_bias λ=0.05 答错的 VStar 样本，共 18 题。

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
| late λ=0.01/0.02/0.03/0.05 | 18/18 = 100.00% |

阶段结论：

> early 视觉注入本身危险，不只是 λ 太大。即使 λ=0.01，early 仍会破坏 8/18 个原本正确样本。

## 12. 统一路由框架与 route annotation

文档：

```text
result/5-16exp/unified_routing_framework_direction1_20260524.md
```

代码位置：

| 功能 | 文件 |
|---|---|
| 参数定义 | `main.py` |
| 参数传递 | `lead/inference.py` |
| 路由核心 | `lead/generation_utils.py` 的 `generate_pure_soft(...)` |
| route summary | `script/exp5_16/analyze_route_summary.py` |

新增 trace 字段：

```text
generation_phase
route_signal
route_action
route_priority
route_suppressed_by
is_highrisk_format_token
visual_bias_candidate
visual_bias_effective
entropy_spike_mask
diffuse_mask
repeat_degen_detected
```

第一版路由优先级：

```text
answer_zone / collapse hard discrete
> format cooldown
> mid image_pad visual bias
> pure_soft
```

VStar route annotation 复跑目录：

```text
output/experiments/20260525_164737/vstar_route_annotated_full
```

结果：

| 方法 | Acc |
|---|---:|
| bestcombo_route_annotated | 143/191 = 74.87% |
| router_v0_midbias002 | 142/191 = 74.35% |
| router_v0_midbias002_answerzone | 142/191 = 74.35% |
| router_v0_midbias003 | 144/191 = 75.39% |

解释：

- route annotation 不改变 bestcombo 行为，结果与原 bestcombo 对齐。
- `midbias003` 在 VStar 上小幅超过 bestcombo，但幅度很小，需要跨数据集验证。
- answer-zone 继续没有稳定新增收益。

## 13. RealWorldQA：发现数据错配并修复

原始 RealWorldQA200 实验目录：

```text
output/experiments/20260526_143510/realworldqa200_route_methods
```

最初结果很低：

| 方法 | 原 RealWorldQA200 Acc |
|---|---:|
| COT | 24.5% |
| LEAD | 29.0% |
| pure-soft | 21.5% |
| bestcombo | 20.5% |
| router_midbias003 | 19.5% |

审计后发现两个问题：

1. 抽取脚本旧版存在风险：任意 `(A)` 匹配、answer region 太窄、option-text fallback 容易误判/漏判。
2. 更严重的是数据图文错配：`data/realworldqa_mcq_random200_seed42.jsonl` 的问题/答案和本地 HuggingFace 源不一致，但图片路径按源序号排列。

典型例子：

```text
旧 id=8：题目问 dog facing，但图片 000008.webp 是夜间街景，没有狗。
源 parquet id=8：题目问 road slope，与该图片匹配。
fixed id=9：题目问 dog facing，图片 000009.webp 确实有狗。
```

修复脚本与数据：

| 类型 | 路径 |
|---|---|
| 构建脚本 | `script/build_realworldqa_from_source.py` |
| strict evaluator | `script/evaluate_realworldqa_mcq_strict.py` |
| fixed 全量 | `data/realworldqa_fixed_from_source.jsonl` |
| fixed MCQ 200 | `data/realworldqa_fixed_mcq_random200_seed42.jsonl` |
| 审计报告 | `result/realworldqa200_eval_audit_20260526.md` |

fixed 数据统计：

| 项 | 数值 |
|---|---:|
| total | 765 |
| MCQ total | 428 |
| MCQ selected | 200 |
| missing images | 0 |
| answer dist | A=65, B=64, C=70, D=1 |

## 14. fixed RealWorldQA200 重跑结果

目录：

```text
output/experiments/20260526_235349/realworldqa200_fixed_route_methods
```

结果：

| 方法 | 内置评估 | MCQ 抽取 | strict 下界 | 平均长度 | 1024截断 |
|---|---:|---:|---:|---:|---:|
| COT | 66.0% | 132/200 = 66.0% | 43.5% | 140.0 | 1 |
| LEAD | 64.0% | 128/200 = 64.0% | 44.5% | 141.3 | 1 |
| pure-soft | 56.0% | 112/200 = 56.0% | 44.5% | 230.5 | 13 |
| bestcombo | 63.5% | 128/200 = 64.0% | 45.0% | 141.7 | 1 |
| router_midbias003 | 63.5% | 126/200 = 63.0% | 44.5% | 141.5 | 1 |

相对 pure-soft：

| 方法 | fixed | damaged | net |
|---|---:|---:|---:|
| COT | 38 | 18 | +20 |
| LEAD | 35 | 19 | +16 |
| bestcombo | 30 | 14 | +16 |
| router_midbias003 | 29 | 15 | +14 |

相对 COT：

| 方法 | fixed | damaged | net |
|---|---:|---:|---:|
| LEAD | 7 | 11 | -4 |
| pure-soft | 18 | 38 | -20 |
| bestcombo | 22 | 26 | -4 |
| router_midbias003 | 21 | 27 | -6 |

阶段结论：

- fixed 后 RealWorldQA 恢复正常量级，COT 66%。
- pure-soft 明显退化：56%，平均长度 230.5，13 条截断。
- bestcombo 能把 pure-soft 拉回 64%，同时将长度压回 141.7、截断降到 1。
- 但 bestcombo 没超过 COT，只是接近 COT/LEAD。
- 当前 `router_midbias003` 没有带来收益，略低于 bestcombo。

## 15. 当前总体结论

### 15.1 已经比较稳的结论

1. pure-soft 在多个数据集上会引入生成退化，典型表现为长输出、重复、答案边界不稳定和抽取失败。
2. 低置信扩散是危险信号，但必须结合阶段和退化迹象；`late64 + repeat_gate` 是保守有效的止损路由。
3. `format_cooldown2` 是当前最强稳定化机制，在 VStar 上把 pure-soft 从 58.64% 提到 74.35%。
4. `bestcombo = format_cooldown2 + late64_repeat_gate` 在 VStar 上达到 74.87%，并在 RealWorldQA fixed 上把 pure-soft 从 56% 拉回 64%。
5. 简单削弱 format cooldown 不是好方向，MMVP / VisuLogic / VStar 都显示它是核心稳定器。
6. LEAD simple visual anchor 有稳定贡献，说明轻量视觉信息有用。
7. 视觉注入强依赖阶段。VStar 上 early visual bias 危险，mid 更安全，late 安全但收益弱。
8. “高熵就加视觉”不成立。高熵 token 类型复杂，必须分 route signal。

### 15.2 仍不稳或被否定的方向

| 方向 | 目前判断 |
|---|---|
| 全程 image_pad bias | 不稳，伤 VStar |
| entropy-gated visual bias | 不稳，高熵类型太混杂 |
| early visual bias | VStar 上危险，即使 λ=0.01 也会伤 |
| answer-zone 单独路由 | 有格式修复，但介入太晚 |
| highrisk-only format | 过窄，掉分 |
| mixed format embedding | 强度不够，不如 hard discrete |
| 简单减少 format 触发 | 跨数据集均不理想 |

## 16. 对下一步的建议

我建议下一阶段不要继续无目的扫阈值，而是做两条主线。

### 16.1 主线一：COT/LEAD 主路径 + soft 局部路由

目前 pure-soft 默认路径不够稳。更合理的目标是：

```text
以 COT 或 LEAD 为主路径；
只在局部高风险 token / 局部阶段启用 soft 或 route action；
目标是 fixed > damaged，而不是让 pure-soft 全程接管。
```

RealWorldQA fixed 的结果说明：

```text
COT = 66%
pure-soft = 56%
bestcombo = 64%
```

这意味着 bestcombo 能救 pure-soft，但还没有超过干净 COT。下一步应尝试：

```text
COT 主路径
+ repeat_degen / format_uncertain / long-output early signal 局部路由
```

目标：

```text
保住 COT 66%，同时修复一部分 COT 错题。
```

### 16.2 主线二：视觉路由只在更精确子集上做

不要全量粗暴视觉注入。更合理的实验子集：

```text
COT 错、bestcombo 也错、LEAD 对
或
LEAD 修复 COT 的样本
```

这些样本更可能是真正需要视觉先验的题。视觉 route 可以先测试：

```text
mid-only image_pad_bias
visual-state gated image_pad_bias
弱 λ，不进入 early
```

### 16.3 主线三：route profile 跨数据集分析

固定方法：

```text
COT
LEAD
pure-soft
bestcombo
router_midbias003
```

固定输出：

```text
accuracy
length / maxed
failed extraction
fixed / damaged
route_signal 分布
route_action 分布
correct/wrong route profile
```

目标是判断每个数据集适合哪种 route profile，而不是假设一个规则通吃所有数据集。

## 17. 关键文件索引

| 内容 | 路径 |
|---|---|
| VStar 总结旧版 | `result/5-16exp/vstar_route_experiment_summary_since_20260516.md` |
| 统一路由框架 | `result/5-16exp/unified_routing_framework_direction1_20260524.md` |
| RealWorldQA 审计 | `result/realworldqa200_eval_audit_20260526.md` |
| strict RealWorldQA evaluator | `script/evaluate_realworldqa_mcq_strict.py` |
| RealWorldQA fixed 构建 | `script/build_realworldqa_from_source.py` |
| route summary 脚本 | `script/exp5_16/analyze_route_summary.py` |
| VStar route annotated | `output/experiments/20260525_164737/vstar_route_annotated_full` |
| RealWorldQA fixed 重跑 | `output/experiments/20260526_235349/realworldqa200_fixed_route_methods` |


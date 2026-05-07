# VStar 上 `lead_attenachor` 详细实验说明

## 1. 研究目标

本轮实验的目标是验证新方法 `lead_attenachor` 在 VStar 上是否能够优于原始 `lead`，并重点观察：

1. 动态视觉 anchor 是否会在高熵推理点被有效触发；
2. 触发后是否能降低推理 token 的局部熵；
3. 是否能减少答案抽取失败或错误推理；
4. 参数 `top_m`、anchor 强度和若干 guard 机制是否能带来稳定收益。

## 2. 本轮改动参数说明

本轮围绕 `lead_attenachor` 主要调整了以下参数：

- `visual_anchor_top_m`
  - 含义：当前 token 对视觉 token 的 attention 排序后，取前 `m` 个视觉 token 参与动态 anchor 构造。
  - 作用：控制视觉候选范围。
  - 直觉：值越大，anchor 看到的视觉信息越宽；值越小，anchor 更聚焦，但也可能过窄。

- `visual_anchor_lambda_scale`
  - 含义：对视觉 anchor 融合强度 `lambda_t` 的额外缩放系数。
  - 作用：控制动态 anchor 对 `soft_emb` 的改写幅度。
  - 直觉：值越大，视觉 anchor 影响越强；值越小，注入越保守。

- `visual_anchor_entropy_upper`
  - 含义：当前 token 的原始熵若高于该阈值，则跳过 anchor 注入。
  - 作用：避免在极端高熵、明显失稳的点上继续强行注入。
  - 直觉：这是一个“过热保护”。

- `visual_anchor_single_use`
  - 含义：每个样本最多只允许一次 anchor 注入。
  - 作用：限制重复触发，避免后段多次进入 soft 时不断重写轨迹。
  - 直觉：适合防止像长退化样本那样反复触发。

- `visual_anchor_skip_nonword`
  - 含义：如果当前 token 解码后主要是空白、标点或非正常词面，则跳过 anchor 注入。
  - 作用：避免在明显不适合作为“语义锚点”的 token 上注入。
  - 直觉：这是一个词面过滤器。

这些参数里，实验里最核心的是：

1. `visual_anchor_top_m`
2. `visual_anchor_lambda_scale`
3. `visual_anchor_entropy_upper`

而 `single_use` 和 `skip_nonword` 更像是附加 guard。

## 3. 方法概述

`lead_attenachor` 相对原始 `lead` 的主要变化是：

- 只在 `normal -> soft` 的第一个 token 上尝试注入视觉 anchor；
- 不使用固定视觉 anchor；
- 动态 anchor 的构造流程为：
  1. 读取当前 token 对视觉 token 的 attention；
  2. 在视觉 token 中取 top-m；
  3. 用当前 latent embedding 对这 top-m 视觉 token 做 attention pooling；
  4. 得到当前 step 的动态视觉 anchor；
  5. 使用 `(1 - lambda_t) * soft_emb + lambda_t * anchor_t` 作为下一步输入 embedding。

当前实现只使用最后四层 decoder attention。

## 4. 实验设置

### 3.1 数据集

- 数据集：`data/vstar.jsonl`
- 样本数：`191`

### 3.2 对照方法

- 旧 `lead`
- 初版 `lead_attenachor`
- 后续两版参数调整：
  - `top4_guarded`
  - `top4_midguard`

### 3.3 共同设置

- `alpha = 0.4`
- `max_switch_count = 5`
- `window_size = 128`
- `max_new_tokens = 1024`
- `temperature = 0.6`
- `top_p = 0.95`
- `top_k = 20`
- greedy decoding

## 5. 主要实验结果

### 5.1 全量 VStar 对比

| 方法 | 实验目录 | 正确 / 总数 | 准确率 |
|---|---|---:|---:|
| `lead` | `output/experiments/vstar_lead_paper_params_20260426_213624` | 139 / 191 | 72.77% |
| 初版 `lead_attenachor` | `output/experiments/20260428/vstar_lead_attenachor_entropy_140553` | 137 / 191 | 71.73% |
| `top4_guarded` | `output/experiments/20260428/vstar_lead_attenachor_top4_guarded_151421` | 136 / 191 | 71.20% |
| `top4_midguard` | `output/experiments/20260428/vstar_lead_attenachor_top4_midguard_173005` | 133 / 191 | 69.63% |

整体结论：

- 旧 `lead` 仍然最好；
- 初版 `lead_attenachor` 比旧版少 2 题；
- `top4_guarded` 没有提分，只是略微止损；
- `top4_midguard` 明显变差。

### 5.2 子任务结果

#### 旧 `lead`

- `direct_attributes`: `82 / 115`
- `relative_position`: `57 / 76`

#### 初版 `lead_attenachor`

- `direct_attributes`: `81 / 115`
- `relative_position`: `56 / 76`

#### `top4_guarded`

- `direct_attributes`: `81 / 115`
- `relative_position`: `55 / 76`

#### `top4_midguard`

- `direct_attributes`: `79 / 115`
- `relative_position`: `54 / 76`

可以看到，两类子任务都没有出现新方法稳定优于旧 `lead` 的情况。

## 6. 初版 `lead_attenachor` 的细分析

### 6.1 熵统计

初版实验目录：

- `output/experiments/20260428/vstar_lead_attenachor_entropy_140553`

主要统计：

- 平均输出 token：`135.09`
- `soft_ratio`：`3.29%`
- `reasoning_ratio`：`87.08%`
- `avg_reasoning_raw_entropy`：`0.867`
- reasoning `p90`：`2.282`
- `avg_reasoning_relation_raw_entropy`：`1.323`
- `avg_reasoning_non_relation_raw_entropy`：`0.857`

与旧 `lead` 对比：

- 输出更长；
- soft 介入更多；
- reasoning 平均熵略高；
- relation token 依然明显高于普通 reasoning token。

### 6.2 Anchor 触发情况

- anchor 触发样本：`42`
- anchor 事件：`48`
- reasoning token 上的 anchor：`48`
- relation token 上的 anchor：`5`

说明：

- anchor 触发很稀疏；
- 大部分样本根本不触发；
- 触发位置更多落在属性词、位置词和判断词附近，而不是大量落在 relation token 本体上。

### 6.3 局部熵变化

anchor 前后窗口统计：

- anchor token 自身平均熵：`2.024`
- 前 5 token 平均熵：`0.619`
- 后 5 token 平均熵：`2.977`

相对位置曲线大致表现为：

- 注入前：熵较低
- 注入点：熵明显升高
- 注入后第 1 个 token：熵进一步暴涨
- 后续几步才逐渐回落

这意味着：

- 当前 anchor 不是在“高熵峰后稳定化”；
- 更像是在“高熵段入口”发生；
- 而且注入后并没有马上把轨迹拉回稳定区。

## 7. 初版回退样本分析

与旧 `lead` 相比，初版 `lead_attenachor`：

- 修好：`86, 113, 128, 162, 172`
- 拉坏：`19, 49, 74, 137, 140, 154, 178`

净效果：`+5 / -7 = -2`

### 7.1 回退样本的主要类型

#### A. 输出退化 / 无法抽取答案

代表样本：

- `19`
- `74`
- `137`
- `140`
- `178`

共同现象：

- anchor token 已处于高熵状态；
- 后续 1-5 个 token 熵继续暴涨；
- 输出出现重复、标点堆积、空白、语言混杂或异常延长；
- 最终无法稳定输出 `A/B/C/D`，评测得到 `None`。

#### B. 语义漂移 / 选项翻转

代表样本：

- `49`
- `154`

这类样本没有完全崩格式，但答案判断方向被拉偏。

### 7.2 结论

当前新方法的主要失败模式不是“完全没看图”，而是：

1. 在失稳边缘触发；
2. 注入后进入更长的高熵 soft 段；
3. 输出退化；
4. 最终要么抽不出答案，要么被局部视觉证据带偏。

## 8. `top4_guarded` 实验

实验目录：

- `output/experiments/20260428/vstar_lead_attenachor_top4_guarded_151421`

参数：

- `visual_anchor_top_m = 4`
- `visual_anchor_lambda_scale = 0.35`
- `visual_anchor_single_use = true`
- `visual_anchor_skip_nonword = true`
- `visual_anchor_entropy_upper = 2.0`

### 8.1 结果

- 准确率：`136 / 191 = 71.20%`
- 平均输出 token：`121.45`
- `soft_ratio`：`2.88%`
- `avg_reasoning_raw_entropy`：`0.905`

anchor 统计：

- 触发样本：`23`
- 触发事件：`23`
- relation anchor：`3`
- anchor 前 5 token 平均熵：`0.691`
- anchor 后 5 token 平均熵：`1.907`

### 8.2 解释

这版的工程效果是明显的：

- 输出更短；
- 触发更少；
- 局部失稳程度也比初版轻。

但准确率没有上升，说明：

- guard 虽然减少了失稳；
- 也同时减少了原本少量有正作用的触发。

与初版 `lead_attenachor` 相比：

- 修好：`19, 74, 137, 140, 154`
- 拉坏：`86, 113, 123, 128, 162, 172`

净变化：`+5 / -6 = -1`

## 9. `top4_midguard` 实验

实验目录：

- `output/experiments/20260428/vstar_lead_attenachor_top4_midguard_173005`

参数：

- `visual_anchor_top_m = 4`
- `visual_anchor_lambda_scale = 0.5`
- `visual_anchor_entropy_upper = 2.5`

### 9.1 结果

- 准确率：`133 / 191 = 69.63%`
- 平均输出 token：`120.83`
- `soft_ratio`：`2.76%`
- `avg_reasoning_raw_entropy`：`0.899`

anchor 统计：

- 触发样本：`31`
- 触发事件：`32`
- relation anchor：`5`
- anchor 前 5 token 平均熵：`0.629`
- anchor 后 5 token 平均熵：`2.289`

### 9.2 解释

这版比 `top4_guarded` 更常触发，但准确率更差。

它不是“没触发所以没效果”，而是：

- 触发回来了；
- 但不稳定触发也回来了。

与 `top4_guarded` 相比：

- 修好：`152`
- 拉坏：`19, 74, 140, 175`

净变化：`+1 / -4 = -3`

## 10. 诊断子集实验

为了更系统地比较参数组合，把几轮实验中所有“发生变化”的关键样本合并成一个 15 条诊断子集：

- `19, 49, 74, 86, 113, 123, 128, 137, 140, 152, 154, 162, 172, 175, 178`

子集文件：

- `data/vstar_anchor_diagnostic_union.jsonl`

### 10.1 旧方法和几版新方法在该子集上的结果

| 配置 | 正确 / 15 |
|---|---:|
| 旧 `lead` | `9/15` |
| 初版 `lead_attenachor` | `7/15` |
| `top4_guarded` | `6/15` |
| `top4_midguard` | `3/15` |

这个子集比最早的 7 条回退子集更严格，也更接近真实问题面。

### 10.2 参数网格结果

网格目录：

- `output/experiments/20260428/anchor_union_grid_181534`

结果：

| 配置 | 正确 / 15 | 正确样本 |
|---|---:|---|
| `top4_only` | `6/15` | `19, 128, 137, 140, 175, 178` |
| `top4_upper25` | `5/15` | `19, 137, 140, 154, 175` |
| `top4_single_upper25_l035` | `5/15` | `74, 137, 140, 154, 175` |
| `top4_single_skip_upper25_l035` | `5/15` | `74, 137, 140, 154, 175` |
| `top4_single_upper25_l040` | `5/15` | `74, 137, 140, 154, 175` |
| `top6_single_upper25_l035` | `6/15` | `123, 137, 140, 154, 162, 175` |
| `top8_only` | `6/15` | `123, 137, 140, 154, 172, 175` |
| `top8_upper25` | `6/15` | `123, 137, 140, 154, 172, 175` |

### 10.3 网格实验结论

1. `top_m=32` 过大，这一点已经比较明确。
2. 将 `top_m` 收缩到 `4/6/8` 后，的确能减少一部分退化。
3. 但没有任何一组新参数组合超过初版 `lead_attenachor` 的 `7/15`。
4. `single_use / skip_nonword / entropy_upper / lambda_scale` 这些 guard 没有形成稳定收益。
5. 更准确地说，目前不同参数只是“修好了不同的一批样本”，没有出现明显的全局最优。

## 11. 综合结论

截至目前，可以较稳定地得出这些判断：

### 11.1 已经明确成立的结论

1. `lead_attenachor` 工程实现已经跑通；
2. relation token 在 VStar 上依然整体高熵；
3. 动态 anchor 确实会触发；
4. 当前主要失败模式不是“没看图”，而是：
   - 在高熵失稳边缘触发；
   - 注入后进入更长的高熵段；
   - 输出退化；
   - 答案抽取失败；
   - 少量样本出现语义漂移；
5. `top_m=32` 明显过大，是当前最明确暴露出来的问题。

### 11.2 尚未得到支持的结论

1. 目前没有任何一版 `lead_attenachor` 在 VStar 全量上超过旧 `lead`；
2. 目前也没有任何后续参数组合稳定超过初版 `lead_attenachor`；
3. 额外增加 guard 并没有带来稳定的正收益；
4. “触发更少”本身也不等于“结果更好”。

## 12. 当前阶段的判断

这轮研究目前最重要的价值不是“找到最优参数”，而是已经比较清楚地定位了问题：

- 动态视觉 anchor 的想法在机制上是可执行的；
- 但当前触发位置、视觉候选选择和注入强度都还不够稳；
- `top_m` 过大已经被反复验证；
- 其余 guard 目前更像是局部止损手段，而不是稳定增益来源。

如果后续继续做，最合理的方向是：

1. 继续围绕更小的 `top_m` 做更细粒度测试；
2. 不要再继续叠加过多复杂 guard；
3. 优先研究“哪些样本会触发、为什么触发、触发后为什么进入失稳段”，而不是继续做大范围参数堆叠。

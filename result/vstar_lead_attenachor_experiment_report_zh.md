# VStar 上 `lead_attenachor` 实验结论

## 1. 目标

验证新方法 `lead_attenachor` 是否能在 VStar 上优于原始 `lead`，并观察动态视觉 anchor 对推理 token 熵的影响。

## 2. 主要实验

### 基线

| 方法 | 目录 | 准确率 |
|---|---|---:|
| `lead` | `output/experiments/vstar_lead_paper_params_20260426_213624` | `72.77%` |
| 初版 `lead_attenachor` | `output/experiments/20260428/vstar_lead_attenachor_entropy_140553` | `71.73%` |

### 后续参数实验

| 配置 | 目录 | 准确率 |
|---|---|---:|
| `top4_guarded` | `output/experiments/20260428/vstar_lead_attenachor_top4_guarded_151421` | `71.20%` |
| `top4_midguard` | `output/experiments/20260428/vstar_lead_attenachor_top4_midguard_173005` | `69.63%` |

## 3. 关键观察

### 3.1 初版 `lead_attenachor`

- 相比旧 `lead` 少 `2` 题。
- anchor 触发样本：`42`
- anchor 事件：`48`
- relation token 上的 anchor：`5`
- 局部熵现象：
  - anchor 前 5 token 平均熵：`0.619`
  - anchor 后 5 token 平均熵：`2.977`

结论：

- anchor 不是把高熵峰压下去；
- 更像是在高熵段入口触发，之后熵继续上升；
- 主要问题是触发后输出退化、答案抽取失败，少数样本出现语义漂移。

### 3.2 `top4_guarded`

参数特征：

- `top_m=4`
- 更小 `lambda`
- `single_use`
- `skip_nonword`
- `entropy_upper=2.0`

结果：

- 比初版更短、更少触发；
- anchor 触发样本降到 `23`；
- 但准确率仍低于初版和旧 `lead`。

结论：

- 这版主要是在止损；
- 减少了失稳，但也削弱了原本少量有效触发。

### 3.3 `top4_midguard`

参数特征：

- `top_m=4`
- `lambda_scale=0.5`
- `entropy_upper=2.5`
- 不再加 `single_use / skip_nonword`

结果：

- anchor 触发样本回升到 `31`；
- 但准确率进一步掉到 `69.63%`。

结论：

- 触发更多，但无效触发也更多；
- 说明当前放松 guard 会重新引入失稳。

## 4. 诊断子集实验

为了快速比较参数组合，把多轮实验里发生变化的关键样本合并成 15 条诊断子集：

- `19, 49, 74, 86, 113, 123, 128, 137, 140, 152, 154, 162, 172, 175, 178`

已有方法在该子集上的结果：

| 方法/配置 | 正确 / 15 |
|---|---:|
| 旧 `lead` | `9/15` |
| 初版 `lead_attenachor` | `7/15` |
| `top4_guarded` | `6/15` |
| `top4_midguard` | `3/15` |

后续网格测试结果：

| 配置 | 正确 / 15 |
|---|---:|
| `top4_only` | `6/15` |
| `top4_upper25` | `5/15` |
| `top4_single_upper25_l035` | `5/15` |
| `top4_single_skip_upper25_l035` | `5/15` |
| `top4_single_upper25_l040` | `5/15` |
| `top6_single_upper25_l035` | `6/15` |
| `top8_only` | `6/15` |
| `top8_upper25` | `6/15` |

结论：

- `top_m=32` 过大，这一点已经比较明确；
- 但把 `top_m` 收到 `4/6/8` 后，仍没有任何一组超过初版 `lead_attenachor` 的 `7/15`；
- `single_use / skip_nonword / entropy_upper / lambda_scale` 这些 guard 没有形成稳定收益。

## 5. 当前结论

到目前为止，可以较确定地说：

1. `lead_attenachor` 工程上已经跑通；
2. relation token 仍整体高熵；
3. 动态 anchor 确实会触发，但当前触发质量不稳定；
4. `top_m=32` 明显过大；
5. 当前主要失败模式不是“完全没看图”，而是：
   - 触发后进入高熵失稳段；
   - 输出退化；
   - 答案抽取失败；
   - 少量样本语义被拉偏。

## 6. 最终判断

- 目前没有任何一版 `lead_attenachor` 在 VStar 全量上超过旧 `lead`。
- 目前也没有任何后续参数组合稳定超过初版 `lead_attenachor`。
- 这阶段最有价值的收获不是“找到最优参数”，而是明确定位了问题：
  - `top_m` 过大；
  - anchor 常在失稳边缘触发；
  - 额外 guard 暂时没有带来稳定增益。

如果后续继续试，最值得保留的方向是：

- 继续围绕更小的 `top_m` 做实验；
- 不要再继续叠加复杂 guard。

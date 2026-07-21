# TALR 性能线与 Early Transition 机制线：完整实验设计与当前状态

**日期：** 2026-07-16  
**用途：** 统一记录两条互补实验线的研究问题、方法定义、控制变量、已完成结果、运行修复与论文解释边界。

## 摘要

本研究同时推进两条实验线，但它们不是竞争的论文主线。

1. **性能线（TALR）**检验一个可部署的方法能否在严格匹配的 COT/LEAD 基线上带来稳定收益。
2. **机制线（Early Transition）**检验 latent intervention 的作用是否集中在生成初期，并测量该影响何时外化为离散 reasoning prefix。

它们共享 COT、full LEAD、Initial Transition 基线，却回答不同问题：性能线回答“方法是否值得作为主方法”；机制线回答“为什么 early transition 会改变最终答案”。论文中应采用 `ETC -> LTI -> TALR` 的单链结构：Early Trajectory Commitment（ETC）是发现，Latent Trajectory Initialization（LTI）是设计原则，TALR 是由该原则导出的实现。

截至本报告，机制线的 VStar/MMVP 控制与 47 条 same-prefix replay 已完成；性能线的旧 TALR 结果被发现存在**定义不一致**，且 Vision-R1 VStar 的 Initial Transition/TALR 有 176/191 条 CUDA OOM，不能纳入结论。已启动严格修复队列：先在独占显存下重跑 Vision-R1 Initial Transition，再用新实现重跑真正的 transition-preserving TALR 两模型四数据集矩阵。

## 1. 共同研究问题

LEAD 类方法通常依赖 token entropy 在 soft/latent embedding 与 normal/discrete embedding 之间动态路由。它隐含两个假设：

- token-level uncertainty 能指出何时 latent reasoning 更有价值；
- 在整个生成过程中持续或动态地切换 route 会带来累积收益。

已有 probe 对这两个假设提出挑战：pure-soft 中高 token confidence 并不稳定代表最终正确；错误路径会出现低熵、高置信、长输出；增加 soft 使用量并不单调提高 accuracy；format guard 能显著降低格式退化，却不能稳定改善 reasoning accuracy。

因此，本轮实验将问题改写为：

> Latent intervention 是否主要在 early trajectory commitment 之前有效？其影响是否会在少数离散 token 中外化，从而使后续普通 COT 沿不同 prefix 演化？

## 2. 统一生成与评测口径

除专门说明外，两条线均采用：

| 项目 | 固定设置 |
|---|---|
| 解码 | greedy，`do_sample=False` |
| 随机性 | seed 42 |
| prompt | origin COT prompt (`cot_prompt_mode=orign`) |
| 最大长度 | 1024 new tokens |
| LEAD 参数 | `alpha=0.4`，`max_switch_count=5`，`window_size=128` |
| trace | 主表保存轻量 route summary；机制 replay 保存所需完整 token trace |
| 模型 | 主模型 R1-Onevision-7B-RL；外部模型 Vision-R1-7B |

核心数据集为 VStar（191）、MMVP（300）、RealWorldQA fixed200（200）和 VisuLogic300（300）。MMVP 只使用 specialized sample/pair evaluator；RealWorldQA 只使用专用 MCQ evaluator；VStar/VisuLogic 使用 corrected last-answer extractor。运行时 `runtime error` 与 `failed extraction` 单独计数，绝不混入普通错误。

## 3. 性能线：真正的 TALR 主实验

### 3.1 四种比较方法

| 方法 | 路由定义 | 要回答的问题 |
|---|---|---|
| COT | 全程离散 top-1 embedding | 离散推理基线 |
| Full LEAD | 原始 entropy-routed soft/normal 机制 | 现有方法基线 |
| Initial Transition | step 0 soft，首次 `soft -> normal` bridge，之后 normal | 分离 early transition 的作用 |
| True TALR | early transition + quota 0.05 refinement + format/repeat guard | ETC/LTI 导出的受约束方法 |

### 3.2 True TALR 的精确定义

此前使用的 `initial_transition_only` 语义是“transition 后永久 normal”，代码会显式禁止 quota soft。因此，旧的 `transition_preserving_quota05_guard_min2` 实际上走的是 **标准 LEAD + quota+guard**：它会继承标准 LEAD 的 step-0 soft 与通常发生在 step-1 的首次转 normal，但仍允许后续 entropy-driven `to_soft`。它不是“只保留早期 transition、后续仅由 quota refinement”的新版 TALR，故不能与新版 TALR 严格混表；但它是一个有价值的 `LEAD + quota + guard` 历史对照。

为保证方法定义与论文一致，新增 `--lead_initial_transition_with_refinement`：

1. 第 0 步使用 soft initializer，并保留原始弱 linebreak mix；
2. 第 1 步执行 `soft -> normal` bridge；
3. 禁止后续 entropy-driven `to_soft`，避免重新引入 full LEAD 的动态变量；
4. 仅通过 quota 0.05 允许少量后续 soft refinement；
5. 从 step 2 起，format cooldown2 和 late diffuse/repeat veto 可将风险 token 强制回到 normal embedding。

若第 \(t\) 步 soft 标记为 \(m_t\)，TALR 的后续 latent budget 满足：

\[
\frac{1}{T}\sum_{t>1}m_t \leq 0.05.
\]

这里的 quota 是有限修正机会，不是“soft 越多越好”的假设；guard 的职责是减少输出退化，不应单独被称为 reasoning gain。

### 3.3 性能指标与判定

每个模型、数据集和方法记录：

- accuracy / MMVP sample 与 pair accuracy；
- failed extraction、runtime errors；
- fixed、damaged 与净收益（相对 COT 和 full LEAD）；
- 平均输出长度、`long>=256`、`maxed1024`；
- soft ratio、switch count、format cooldown 次数、veto 次数。

预注册判定规则：

- TALR 在至少 3/4 核心数据集不低于 COT，且四集平均 delta 为正：可作为主方法。
- TALR 稳定优于 full LEAD、但不稳定优于 COT：定位为机制导出的稳定化简策略。
- TALR 多数数据集低于 COT：不强行作为主性能贡献，论文转为机制审计，TALR 仅作实践性附加结果。

### 3.4 当前性能状态与修复

旧阶段表中 R1-RL 的 quota+guard 结果看似在四个核心数据集均不低于 COT。它包含标准 LEAD 自带的早期 transition，但没有关闭后续 entropy dynamic routing，因此不能作为“仅 early transition + quota refinement”的 true TALR 结论。

Vision-R1 VStar 的旧 Initial Transition 和旧 quota+guard run 也不有效：二者都有 176/191 条 `OutOfMemoryError`。异常表面上表现为 6.81% accuracy、约 7.7 token 平均长度和 176 个“failed extraction”，但根因是运行时显存竞争，而不是模型突然拒绝 transition。第一条样本及其余 15 条未出错样本可以正常生成完整 COT，进一步说明这不是方法输出模式。

修复队列遵循以下顺序：

1. 在独占 GPU、`expandable_segments:True` 下重跑 Vision-R1 VStar Initial Transition；
2. 对新 True TALR 做 R1-RL VStar 两条 smoke，确认配置记录了 `lead_initial_transition_with_refinement=true`；
3. 顺序运行 R1-RL 与 Vision-R1 的四数据集 true TALR；
4. 重新运行 MMVP/RealWorldQA 专用 evaluator，生成新版统一主表；
5. 仅当结果行数完整、runtime error 为 0 且 config 审计通过时，替换论文中 TALR 的数值。

## 4. 机制线：Early Transition 的因果解耦

### 4.1 假设与可证伪表述

机制线不再使用“长期 hidden-state basin”这种过强表述。待检验的精确假设是：

> Early soft/bridge route 先形成短暂 latent-state difference；该差异在前几个生成 token 内外化为不同的离散 prefix；之后普通 autoregressive decoding 沿该 prefix 延续。

该假设允许两种反例：如果 hard boundary-only 与完整 transition 相同，收益可能只是格式边界 steering；如果相同 prefix 后两 route 不再分叉，差异可能只是首个离散 token 的文本分叉。

### 4.2 机制控制矩阵

| 控制 | 保留什么 | 移除什么 | 判别目标 |
|---|---|---|---|
| Initial Transition | step0 soft、bridge、后续 normal | - | 机制参照 |
| cache rebuild prefix=1 | 自然生成第 1 个 token | 之后的 soft/mixed KV history | 一个 token 后 latent history 是否仍必要 |
| cache rebuild prefix=2 | 自然生成前两个 token | 之后的 soft/mixed KV history | 两 token 后 effect 是否已外化 |
| hard boundary-only | linebreak 与 `</think>` bridge | probability-weighted soft semantic embedding | 区分 boundary steering 与 soft semantic effect |
| same-prefix replay 1/2/4 | COT 的相同可见 prefix | 两 route 的自由首 token 差异 | 相同文本前缀后是否仍存在 route-induced divergence |
| full LEAD vs IT | 相同 early transition | full LEAD 的 late routing | late routing 的平均净效用 |

所有 forced-prefix hard replay 必须逐 token 复现 COT；否则该样本标记 `force_sanity_failed`，不参与机制推断。本轮 47 条 replay 均通过该 sanity check。

### 4.3 已完成的 cache rebuild 与 boundary 结果

| 数据集 | 控制 | Accuracy | 相对 COT fixed/damaged | 与完整 IT 的预测一致率 |
|---|---|---:|---:|---:|
| VStar | Initial Transition | 72.25% | 17 / 9 | 100.00% |
| VStar | cache rebuild prefix=1 | 71.20% | 15 / 9 | 90.05% |
| VStar | cache rebuild prefix=2 | 72.77% | 20 / 11 | 87.96% |
| VStar | hard boundary-only | 69.63% | 15 / 12 | 85.34% |
| MMVP | Initial Transition | 70.33% sample, 42.00% pair | 12 / 5 | 100.00% |
| MMVP | cache rebuild prefix=1 | 68.67% sample, 38.67% pair | 10 / 8 | 95.00% |
| MMVP | cache rebuild prefix=2 | 70.67% sample, 44.00% pair | 12 / 4 | 95.00% |

当前最保守、最有信息量的读法是：

- prefix=1 重建在 VStar 和 MMVP 都损失了 Initial Transition 的一部分优势；
- prefix=2 重建恢复或超过 Initial Transition，尤其 MMVP pair 由 38.67% 回到 44.00%；
- hard boundary-only 在 VStar 仅 69.63%，显著弱于 72.25% 的完整 transition；
- 因而现有证据更符合“early soft/mixed effect 在约两个 token 内外化”，而不符合“纯 newline/`</think>` boundary 已足够”的解释。

但由于 VStar 的 prefix=2 不是严格等于 IT，且统计置信区间尚未写入主文，不能把这表述成已经证明的精确 token 边界；应写成跨 VStar/MMVP 一致的、支持性的机制证据。

### 4.4 Same-prefix replay

从 VStar COT 与 Initial Transition 的四象限中选择 47 条样本：18 条 `COT wrong -> IT correct`、9 条 `COT correct -> IT wrong`、16 条 both-correct、4 条 both-wrong。每条强制使用 COT 的前 1、2、4 个真实 token，并分别在 hard route 与 transition route 下继续生成。

| 强制共享 prefix | 有效 replay | 后续分叉 | 答案不一致 | transition/hard 正确数 |
|---|---:|---:|---:|---:|
| 1 token | 47 | 42 | 28 | 34 / 25 |
| 2 tokens | 47 | 42 | 28 | 34 / 25 |
| 4 tokens | 47 | 40 | 25 | 31 / 25 |

分组结果尤其清楚：在 18 条 fixed 样本中，prefix 1/2 时全部 18 条后续分叉，transition continuation 全部正确而 hard continuation 全部错误；在 9 条 damaged 样本中也全部分叉，但方向相反。这说明 early route 的影响不是单向“修复器”，而是能选择不同轨迹；它既可能 fixed，也可能 damaged。both-correct 样本则可以分叉但答案保持相同，说明 token-level divergence 不必然导致 answer flip。

这里可以做出的结论是：**在共享可见 prefix 后，soft/transition route 仍会系统性影响后续生成与最终答案。** 这里不能做出的结论是：它已经证明长期 latent memory 一直保存到完整推理结束；cache rebuild 反而说明长期保存不是必要条件。

### 4.5 Late routing utility

full LEAD 与 Initial Transition 的逐样本对比刻画 late dynamic routing 的边际作用：

| 数据集 | fixed | damaged | 净收益 | Agreement | McNemar p |
|---|---:|---:|---:|---:|---:|
| VStar | 2 | 1 | +1 | 97.38% | 1.0000 |
| MMVP | 0 | 0 | 0 | 100.00% | 1.0000 |
| RealWorldQA fixed200 | 3 | 2 | +1 | 96.50% | 1.0000 |
| VisuLogic300 | 31 | 39 | -8 | 56.67% | 0.4030 |

这些结果支持的只是“late routing 的平均净贡献有限且依数据集而变”，而不是“entropy routing 总是无效”。VStar/MMVP 上几乎没有额外样本被 late routing 改变；VisuLogic 的变动很多但 fixed/damaged 相抵且偏负，提示其可能主要扰动已经成形的长推理轨迹。

## 5. 两条线如何共同支撑论文

两条线在论文中的分工如下：

```text
Probe：confidence mismatch / more-soft failure / format degeneration
                     ↓
Mechanism line：ETC 与短暂 latent effect 的 prefix externalization
                     ↓
Design principle：LTI（早期初始化，随后离散推理）
                     ↓
Performance line：True TALR 的跨模型、跨数据集收益与稳定性
```

性能线通过时，TALR 是主方法，机制线解释为什么它采用 early initializer 而不是全程 dynamic routing。性能线若未通过但机制控制稳定，文章仍可保留 ETC/LTI 发现，不过应转为“对 latent routing 的机制审计与重设计”，不能声称 TALR 普遍提高 reasoning accuracy。

## 6. 证据边界与风险清单

1. confidence mismatch 说明 token confidence 不是通用 correctness oracle，不说明所有高 confidence 都错误。
2. format guard 可靠改善长度、重复、格式和抽取失败，不自动证明 reasoning improvement。
3. LEAD 的 early transition 是已有代码成分；我们的贡献是将其与 late routing 解耦、提出 ETC/LTI，并据此构造 True TALR。
4. same-prefix replay 是在可控 prefix 下的强轨迹证据；仍需避免将其扩张为完整模型内部因果图。
5. 旧 quota+guard run 与真正的 transition-preserving TALR 不是同一个方法，不能混入同一主表。
6. Vision-R1 的 OOM run 已明确隔离；它们不计作方法失败或 failed extraction。
7. 实际投稿数值只使用修复队列完成后的 config-audited 结果。

## 7. 产物与当前运行状态

机制线产物：

- `transition_externalization/externalization_curve.md/json`
- `transition_externalization/late_routing_utility.md/json`
- `transition_externalization/vstar/same_prefix_replay/summary.json`
- `transition_externalization/transition_mechanism_report.md`

性能修复产物：

- `true_talr_core_runs/<model>/<dataset>/talr_early_quota05_guard_min2/`
- `main_summary_true_talr/talr_core_main_table.md/json`
- `main_summary_true_talr/pairwise_fixed_damaged.json`

修复 worker 运行顺序为：Vision-R1 VStar Initial Transition -> True TALR smoke -> R1-RL 四数据集 True TALR -> Vision-R1 四数据集 True TALR -> specialized evaluation -> unified summary。每个完成 run 都要求完整行数、`eval_report.json`、config 审计和 0 runtime error。

## 8. 一句话版本

> 我们不再把 latent reasoning 看作应当持续开启的替代 COT，而把它看作一种在最早期选择推理方向的初始化机制；TALR 的目标是在这一短暂机会之后，尽量让离散 COT 稳定地完成推理。

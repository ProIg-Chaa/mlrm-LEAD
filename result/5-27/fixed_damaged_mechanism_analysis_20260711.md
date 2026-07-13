# Fixed/Damaged 跨数据集机制分析实验

> 本报告主表只使用 checkpoint、prompt、greedy decoding、seed=42、max_new_tokens=1024 完全匹配的逐样本结果。sampled 论文复现不进入因果主表。

## 目录

1. [为什么做这次分析](#1-为什么做这次分析)
2. [比较对象与样本分组](#2-比较对象与样本分组)
3. [分析了哪些数据](#3-分析了哪些数据)
4. [跨数据集主结果](#4-跨数据集主结果)
5. [格式稳定路线的收益与损坏](#5-格式稳定路线的收益与损坏)
6. [早期 transition 路线的收益与损坏](#6-早期-transition-路线的收益与损坏)
7. [事件级 trace 与反事实 replay](#7-事件级-trace-与反事实-replay)
8. [组合方法](#8-组合方法)
9. [结论与证据边界](#9-结论与证据边界)

## 1. 为什么做这次分析

单看 accuracy 只能知道方法整体涨跌，不能回答收益来自哪里，也不能区分真正的视觉/推理修复、输出退化、答案抽取差异和随机采样噪声。本实验把 COT 与每个方法按同一 sample id 配对，专门研究 `COT 错→方法对` 与 `COT 对→方法错` 两类翻转，并通过事件级 replay 检验单次 intervention 是否真的造成后续轨迹变化。

## 2. 比较对象与样本分组

- `pure_soft_format2`：全程以 soft embedding 推进；命中格式 token 后执行 2 步 hard/normal cooldown。它代表格式稳定主线。
- `initial_transition_only`：开头使用第 0 步 soft 路由，随后执行一次 soft→normal transition，之后保持 normal COT。它代表早期轨迹主线。
- `transition_preserving_quota05_guard_min2`：保留开头 transition，之后只允许约 5% soft quota；format cooldown 从 step 2 才允许触发，并加入 late diffuse/repeat veto。
- `fixed`：COT 错、方法对；`damaged`：COT 对、方法错；另保留 `both_correct`、`both_wrong` 作为控制组。
- `extraction_only_flip`：宽松语义答案与 gold 一致，但严格答案区抽取失败。该类不应被解释成推理能力变化。

## 3. 分析了哪些数据

核心四个数据集同时分析 format 与 transition：VStar、MMVP、VisuLogic300、RealWorldQA fixed200。format 路线额外覆盖 VMCBench-dev、POPE Random/Popular/Adversarial，以及 MMK12 Math/Physics/Chemistry/Biology。每个数据集×方法最多选 40 个 fixed、40 个 damaged、20 个 both-correct、20 个 both-wrong 生成 sample cards；核心数据集另人工审计 5 个 fixed 和 5 个 damaged，共 40 条代表样本。

逐样本数据包括：gold/pred、完整输出、前 1/2/4/8/16/32 token、首次分叉位置、输出长度、3-gram 重复、答案 marker/反转、soft ratio、switch/format trigger、早期 entropy/top1，以及 subtopic/subject。MMVP 使用 specialized sample/pair evaluator，POPE 另外保留二分类混淆矩阵口径。

## 4. 跨数据集主结果

| 数据集 | 方法 | COT | 方法 | fixed | damaged | 净值 | McNemar p | 抽取翻转 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vstar | Pure-soft + Format2 | 68.06% | 73.82% | 23 | 12 | +11 | 0.0895 | 6 |
| mmvp | Pure-soft + Format2 | 68.00% | 67.33% | 26 | 28 | -2 | 0.8919 | 0 |
| visulogic300 | Pure-soft + Format2 | 21.33% | 22.00% | 42 | 40 | +2 | 0.9122 | 0 |
| realworldqa_fixed200 | Pure-soft + Format2 | 66.00% | 64.00% | 22 | 26 | -4 | 0.6655 | 0 |
| vmcbench_dev | Pure-soft + Format2 | 75.70% | 73.50% | 56 | 78 | -22 | 0.0693 | 10 |
| pope_random | Pure-soft + Format2 | 85.97% | 82.50% | 69 | 173 | -104 | 0.0000 | 49 |
| pope_popular | Pure-soft + Format2 | 84.63% | 81.47% | 84 | 179 | -95 | 0.0000 | 49 |
| pope_adversarial | Pure-soft + Format2 | 83.53% | 80.37% | 94 | 189 | -95 | 0.0000 | 48 |
| mmk12_math | Pure-soft + Format2 | 47.60% | 49.20% | 80 | 72 | +8 | 0.5703 | 10 |
| mmk12_physics | Pure-soft + Format2 | 41.00% | 39.20% | 76 | 85 | -9 | 0.5285 | 5 |
| mmk12_chemistry | Pure-soft + Format2 | 45.00% | 42.40% | 62 | 75 | -13 | 0.3052 | 7 |
| mmk12_biology | Pure-soft + Format2 | 47.60% | 47.60% | 78 | 78 | +0 | 1.0000 | 3 |
| vstar | Initial transition only | 68.06% | 71.73% | 17 | 10 | +7 | 0.2478 | 6 |
| mmvp | Initial transition only | 68.00% | 70.33% | 12 | 5 | +7 | 0.1435 | 0 |
| visulogic300 | Initial transition only | 21.33% | 29.67% | 49 | 24 | +25 | 0.0046 | 0 |
| realworldqa_fixed200 | Initial transition only | 66.00% | 64.00% | 6 | 10 | -4 | 0.4545 | 0 |
| vstar | Transition-preserving quota05 guard (min2) | 68.06% | 73.82% | 22 | 11 | +11 | 0.0801 | 5 |
| mmvp | Transition-preserving quota05 guard (min2) | 68.00% | 70.33% | 11 | 4 | +7 | 0.1185 | 0 |
| visulogic300 | Transition-preserving quota05 guard (min2) | 21.33% | 23.00% | 43 | 38 | +5 | 0.6570 | 0 |
| realworldqa_fixed200 | Transition-preserving quota05 guard (min2) | 66.00% | 67.00% | 13 | 11 | +2 | 0.8388 | 0 |

### MMVP pair consistency

| 方法 | COT pair acc | 方法 pair acc | fixed pairs | damaged pairs | 净值 |
|---|---:|---:|---:|---:|---:|
| Pure-soft + Format2 | 39.33% | 40.67% | 16 | 14 | +2 |
| Initial transition only | 39.33% | 42.00% | 9 | 5 | +4 |
| Transition-preserving quota05 guard (min2) | 39.33% | 42.67% | 9 | 4 | +5 |

## 5. 格式稳定路线的收益与损坏

### 5.1 总体结果

Format2 不是一个跨数据集稳定超过 COT 的主方法。它在 VStar（+5.76pp）和 MMK12-Math（+1.60pp）有净收益，在 VisuLogic300 基本持平；但在 VMCBench、MMVP、RealWorldQA、MMK12 Physics/Chemistry 下降，在 POPE 三组均显著下降约 3.2–3.5pp。

### 5.2 触发很多，不等于带来收益

| 数据集 | fixed 方法长度 | damaged 方法长度 | fixed format 次数 | damaged format 次数 | fixed 首次分叉 | damaged 首次分叉 |
|---|---:|---:|---:|---:|---:|---:|
| vstar | 140.3 | 226.8 | 42.5 | 66.4 | 24.2 | 22.6 |
| mmvp | 156.4 | 266.9 | 50.4 | 85.1 | 13.8 | 12.8 |
| visulogic300 | 559.8 | 526.1 | 255.5 | 226.2 | 25.5 | 29.1 |
| realworldqa_fixed200 | 129.0 | 204.0 | 40.1 | 59.1 | 18.8 | 21.2 |
| vmcbench_dev | 344.5 | 416.4 | 156.5 | 186.9 | 25.8 | 26.4 |
| pope_random | 130.2 | 705.5 | 62.3 | 391.0 | 15.1 | 15.5 |
| pope_popular | 124.4 | 638.3 | 58.1 | 351.2 | 15.8 | 15.5 |
| pope_adversarial | 122.9 | 628.5 | 56.1 | 343.1 | 16.0 | 15.4 |
| mmk12_math | 476.0 | 512.6 | 284.4 | 288.6 | 28.5 | 36.8 |
| mmk12_physics | 559.3 | 598.5 | 248.4 | 243.2 | 36.6 | 39.2 |
| mmk12_chemistry | 531.8 | 604.2 | 223.0 | 256.7 | 30.0 | 35.1 |
| mmk12_biology | 495.9 | 544.6 | 186.2 | 216.5 | 29.8 | 31.2 |

最清楚的反例是 POPE：damaged 样本平均输出约 628–706 tokens，而对应 COT 只有约 84–87 tokens；damaged 样本平均触发 343–391 次 format cooldown。也就是说，失败不是因为 guard 没触发，而是高频局部 hard 切换没有把轨迹真正带回稳定的 COT 状态，反而伴随长输出/重复退化。VStar、MMVP、RealWorldQA 和 VMCBench 也普遍表现为 damaged 组比 fixed 组更长、format 触发更多。

这支持一个更精确的定位：format cooldown 能修复部分 pure-soft 的格式边界，但它只是局部下一步 embedding 路由，并不重建由 soft history 形成的 KV/history state。因此它能减少某些表面退化，却不能保证恢复 COT reasoning manifold，也不能稳定创造超越 COT 的能力。

### 5.3 代表样本

VStar 的 format fixed 多集中在空间关系，例如电话相对台灯、红车相对警车、鼓相对黄气球；但同一方法也会损坏瓶盖颜色和垃圾桶相对位置。MMVP 同时出现相机视角、蝴蝶足部可见性的 fixed 与 damaged，说明收益不是简单由任务类别决定。人工审计的 40 条代表样本中，format/transition 两条路线都同时出现语义答案轨迹翻转和生成退化，不能把所有 fixed/damaged 都归为抽取问题。

## 6. 早期 transition 路线的收益与损坏

Initial transition 在 VStar（+3.67pp）、MMVP（+2.33pp）和 VisuLogic300（+8.34pp）为正，在 RealWorldQA fixed200（-2.00pp）为负。VisuLogic 的 49 fixed / 24 damaged 达到 McNemar p=0.0046，是当前最强的配对证据；VStar 和 MMVP 的净值均为 +7，但样本量下尚未达到显著。

VStar fixed 以空间关系为主；MMVP 既能修复视角和细粒度部件可见性，也会损坏时钟读数、背鳍可见性等样本。旧 VisuLogic/RealWorldQA transition run 缺少完整 token trace，因此这两个数据集只保留配对结果，不伪造 early-divergence/soft-ratio 结论；它们已进入选中样本补 trace 队列。

## 7. 事件级 trace 与反事实 replay

Replay 从原 prompt 确定性重跑到目标事件，断言事件前 generated prefix 与原 trace 完全一致，然后只替换该事件的一步 route embedding。事件后恢复原方法 policy。这样可以把相关性的 fixed/damaged 分析提升为单次 intervention 的局部因果检验。

2 样本 smoke 状态：`通过`；检查行数 4，失败项 0。smoke 已验证 prefix match、事件 embedding geometry 和 forced-answer probe。forced-answer probe 是在隔离 cache 副本后追加固定 answer marker 得到的诊断量，不代表自然生成置信度。

Instrumentation actual-branch smoke：`通过`；它额外要求开启 trace/probe 后的完整 token 序列与旧 run 逐 token 一致。

完整 replay 输出包括 step0/to-normal、首次/最高风险 format trigger 的 actual、hard、raw-soft、method-soft 分支；报告下一 token、8/16/32 token edit distance、最终答案、forced-answer gold margin，以及 top20+residual bucket 的近似 JS/KL。视觉指标来自隔离 cache 的事件 route probe：聚合 decoder 最后四层视觉 attention，并计算当前 hidden 与 prompt visual hidden 的 alignment；若后端不暴露 tensor，则逐事件显式标记 unavailable，route-to-anchor cosine 不冒充 attention。
Actual smoke 事件视觉诊断：visual attention available 0/4，hidden-visual alignment available 4/4。
完整 replay 当前已汇总 2344 个样本分支，其中 prefix mismatch=0、下一 token 改变=124。

### 7.1 反事实分支结果

下表的 fixed/damaged 是“分支相对 actual 方法轨迹”的变化，只适用于分层选中的分析样本，不能当作全数据集 accuracy。

| 数据集 | 方法/事件 | 分支 | n | 最终答案改变 | fixed/damaged | 32-token 距离 | next-token JS |
|---|---|---|---:|---:|---:|---:|---:|
| mmvp | Initial transition only/step0 | hard | 57 | 12 | 2/4 | 0.121 | 0.0000 |
| mmvp | Initial transition only/step0 | raw_soft | 57 | 13 | 3/6 | 0.133 | 0.0000 |
| mmvp | Initial transition only/to_normal | hard | 57 | 12 | 4/6 | 0.123 | 0.0000 |
| mmvp | Pure-soft + Format2/format_first | method_soft | 94 | 34 | 15/10 | 0.211 | 0.0003 |
| mmvp | Pure-soft + Format2/format_maxrisk | method_soft | 94 | 35 | 8/15 | 0.454 | 0.1666 |
| vstar | Initial transition only/step0 | hard | 67 | 21 | 8/11 | 0.127 | 0.0000 |
| vstar | Initial transition only/step0 | raw_soft | 67 | 25 | 7/11 | 0.152 | 0.0000 |
| vstar | Initial transition only/to_normal | hard | 67 | 20 | 3/11 | 0.138 | 0.0000 |
| vstar | Pure-soft + Format2/format_maxrisk | method_soft | 75 | 22 | 6/9 | 0.464 | 0.1980 |

- **step0 的影响并不要求下一 token 立刻改变。** VStar/MMVP 的 hard 或 raw-soft 分支在 step0 后常保持相同下一 token、近似相同 top-20 logits，但 32-token 轨迹和最终答案随后分叉。这是连续 embedding 写入 KV/history 后产生路径依赖的直接证据。
- **去掉实际 soft→normal mixed transition 会造成持续分叉。** to-normal 事件改成普通 hard/raw-soft 后，VStar 67 条中 20 条最终答案改变，MMVP 57 条中 12 条改变；在这组选中样本上 damaged 多于 fixed。
- **最高风险 format cooldown 具有局部保护作用。** 在 max-risk 事件把 actual hard cooldown 改回 soft，VStar 75 条中 22 条、MMVP 94 条中 35 条最终答案改变，且分别为 6/9、8/15 fixed/damaged。它说明这一事件的 hard guard 能避免部分损坏，但不等于整个 Format2 方法能稳定超过 COT。
- **第一处 format trigger 不是统一关键点。** VStar 在该事件改回 soft 没有改变最终答案；MMVP 则改变 34/94，并呈 15 fixed / 10 damaged。format intervention 的作用高度依赖事件时机与数据集。

完整结构校验：2344 个分支，prefix mismatch=0，actual trajectory mismatch=0，forced probe unavailable=0。hidden-visual alignment 可用 2344/2344；当前 attention backend 未暴露可用 attention tensors，因此 visual-attention mass 为 unavailable。

MMVP 使用小写 `(a)/(b)` 选项，而首轮 forced probe 使用了大写 `A-E` token；因此已有 MMVP forced gold margin 不进入结论。代码已改为按 options 自动选择标签大小写，后续 probe 补跑采用正确 token。
修正后的 MMVP smoke：`通过`，lowercase choice token 4/4，gold margin 范围 0.410–0.732。这只验证 probe 口径，不替代完整旧分支的 margin。

## 8. 组合方法

VStar min2 gate：`通过`。判据为 fixed>damaged，且 failed/long/maxed 不劣于 min0。
但 VStar 的 min0、min2、min4 三个 timing control 均为 141/191（73.82%），且 min0/min2 的 fixed、damaged、failed、long、maxed 完全相同。因此 gate 的通过说明组合整体可用，不构成 `min_step=2` 本身有效的证据；VStar 前 4 步内实际上没有产生能区分三者的 format intervention。

## 9. 结论与证据边界

1. **Format2 的可靠价值是 guardrail，不是通用能力增益。** 它能在部分数据集修复 pure-soft，但高触发频率并不保证收益；POPE 的损坏反而与极长输出和数百次 cooldown 同时出现。
2. **Early transition 的跨数据集趋势更有机制价值，但并非普适提升。** 它在 VStar/MMVP/VisuLogic 为正，在 RealWorldQA 为负，说明早期轨迹干预会同时产生修复与损坏。
3. **极早期连续状态可以在离散 token 不变时改写远期轨迹。** step0 replay 中下一 token 与 top-20 logits 几乎不变，但 hard/raw-soft 分支在之后 32 token 和最终答案上明显分叉；这是 early path-dependence 的局部因果证据。
4. **抽取差异不是主结果的全部解释。** 主表单列 extraction-only flips；人工代表样本还显示大量明确的语义答案翻转和生成退化。
5. **证据层级必须分开。** 2344 个 replay 分支已通过 prefix/actual-trajectory assertion，可以对选中事件作局部因果表述；离线全数据集 fixed/damaged 仍是配对相关性，不能把选中子集的分支比例外推成总体 accuracy。

人工代表样本客观翻转类型计数：{'semantic_answer_trajectory_flip': 26, 'generation_degeneration': 14}。这些标签基于问题语义、gold/pred、输出长度、重复和 trace 信号；视觉正确性沿用 benchmark gold，不额外声称人工重新标注图片真值。

### 当前产物

已生成配置校验 manifest、逐样本四组结果、统计表、40 条人工语义审计、sample cards、selected trace ids、smoke event traces 和 counterfactual branch schema。完整 replay 与组合实验由 GPU 队列续跑，结束后同一报告生成器会自动刷新最终数字。

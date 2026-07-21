# 论文主线重整报告：从 LEAD 审计到早期初始化

**日期：** 2026-07-17  
**状态：** 投稿前主线定稿草案；等待 step-0 2x2 与 exact-original LEAD 审计后冻结。

## 摘要

本工作的核心不应再表述为“让 soft reasoning 稳定超过 COT”，也不应把原 LEAD 已有的 transition 写成我们的模块创新。经过源码审计、机制控制和跨数据集复核后，最可信的研究问题是：

> entropy-routed latent decoding 的有效干预窗口究竟在哪里，以及一个极早期连续状态如何影响后续的离散多模态推理？

论文应由两条严格分离、但因果衔接的线组成：

1. **原始 LEAD 审计线**：在 exact-original 实现上，分离固定 early path 与中后期 entropy routing，审计各自的实际贡献；
2. **机制导出的设计线**：在不把原 LEAD 的 early path 误归为己有的前提下，验证受约束 handoff / refinement 是否构成可复现的改进。其最小形式和命名必须由当前 step-0 2x2 决定。

TALR 保留为由该机制导出的受约束工程扩展，而不是当前论文唯一或主要的科学主张。

## 1. 问题重述

多模态 latent decoding 通常用下一 token 分布的 embedding 期望代替离散 token embedding：

\[
s_t=\sum_{v\in\mathcal V}p_t(v)e(v).
\]

LEAD 进一步以 entropy 为信号，在 soft/latent 与 normal/discrete route 之间切换。其隐含假设是：token uncertainty 可以指示何时需要 latent reasoning，且更多或更及时的 soft routing 会改善最终正确性。

我们的研究发现这套假设缺少稳定的实证支持：token confidence 并不是跨数据集可泛化的 correctness oracle；持续 pure-soft 容易出现重复、超长、格式与答案漂移；后期动态 routing 的 fixed 与 damaged 经常接近抵消。因此，真正的问题不是“何时多做 soft”，而是：

> 连续干预应当在何时出现、持续多久，以及如何被安全地交接给离散 COT？

## 2. 关键去混杂：原始 LEAD 的 early path 与后续扩展必须分开

此前被统称为“initial transition”的实现实际包含三个应被拆开的原始 LEAD 因素：

1. step-0 的 soft state；
2. step-0 的 `0.9 soft + 0.1 newline` 混合；
3. step-1 的 EOT / `</think>`-related bridge。

源码溯源显示，第 2 项已存在于仓库的 2026-03-08 原始 LEAD 提交，且 4 月历史版本仍保留它；它**不是我们后来加入的修改**。因此，此前在代码等价的 original path 上完成的 initial-transition 组件实验，确实可以用于支持“该固定 early path 恢复了 full LEAD 的大部分收益”这一审计观察。

真正需要避免的是另一种混淆：不能把 step-0 soft/newline、step-1 bridge 和中后期 entropy routing 合并称为单一的“dynamic routing gain”。它们需要在同一评测口径下分别测量。

### 2.1 原始 LEAD 审计线

该线应冻结 original-code commit，并至少复核：COT、original LEAD、force-normal、initial-soft-only、initial-transition-only 以及 late-routing fixed/damaged。已有的大量历史组件实验是这条结论的主要证据；补做 exact-original 最小复核的作用是把 checkpoint、代码 hash、prompt、evaluator 和样本顺序固定到可公开复现的版本，而不是推翻历史发现。

最终论文必须明确实现版本和 git commit，并将之后加入的 quota、cooldown、veto 和任何实验开关与 original path 分开报告。

### 2.2 我们的方法线

我们的候选设计是在原始 early path 审计的基础上，测试更受约束的 handoff / refinement 策略，例如 step-1 direct-hard handoff、有限 quota 和稳定 guard。它不宣称发明 LEAD 的 transition 或 soft+newline 初始混合；贡献只能是组件级机制识别、对早期连续到离散交接的可检验解释，以及由此导出的受约束设计。

## 3. 机制主张：短暂连续影响的快速离散外化

当前最稳健的机制表述是：

> An early continuous intervention creates a transient state difference that is rapidly externalized into the first few discrete reasoning tokens; ordinary autoregressive decoding subsequently preserves and amplifies the induced prefix.

它刻意不宣称长期 hidden-state basin，也不把一切归因为首 token 文本差异。

### 已有支撑

- **Timing / component controls**：早期干预比中后期触发更有信息量；移除 `to_normal` 后，性能倾向退回 initial-soft-only 附近。
- **Same-token replay**：共享首 token 的有效样本中，42/47 仍在后续分叉，说明首 token 不同不是全部解释。
- **Cache rebuild**：保留 transition 产生的前两个离散 token 后，重建纯离散 KV cache 不明显损失已有准确率，说明长期保留 soft KV 并非必要。
- **Token-Anchored negative control**：step-1 的 soft 分布接近 one-hot；用其与实际 token embedding 线性混合会坍缩为 direct-hard control。因此它不能作为独立方法，也排除了“任意 smooth handoff 都有效”的简单解释。

### 证据边界

same-prefix / replay 的分叉本身仍是机制关联证据。只有 exact prefix replay、分支一致性和相同 evaluator 下的对照，才能支持局部因果描述。论文不应使用“永久 trajectory basin”或“所有推理在第 0 步锁定”等强表述。

## 4. 分解原始 LEAD early path 的 step-0 2x2

定义 hard token route 为 \(h_0=e(y_0)\)，soft route 为 \(s_0\)。令 \(n=e(\mathrm{newline})\)。四个条件统一在 step 1 直接回到 hard route，后续为 normal greedy COT：

\[
z_1=e(y_1),\qquad z_t=e(y_t),\quad t\ge2.
\]

| 条件 | Step-0 state | 问题 |
|---|---|---|
| hard / no-newline | \(h_0\) | wrapper/COT 等价控制 |
| hard / newline | \(0.9h_0+0.1n\) | newline 单独作用 |
| soft / no-newline | \(s_0\) | continuous state 单独作用 |
| soft / newline | \(0.9s_0+0.1n\) | 当前完整 initializer |

交互项为：

\[
\Delta_{\mathrm{int}}=A_{S,+}-A_{S,-}-A_{H,+}+A_{H,-}.
\]

当前在 R1-Onevision-7B-RL 的 VStar 191 与 MMVP 300 上运行，使用 greedy、seed 42、1024 token；MMVP 额外报告 pair accuracy。报告 fixed/damaged、failed extraction、长度与配对统计。

结果首先决定原始 LEAD 的 fixed early path 究竟由什么成分驱动；只有在此基础上，才决定受约束 handoff 变体应如何命名和贡献强度：

- **soft-only 主效应**：原始 early path 的主要作用来自 latent state；受约束变体可被称为 latent-initialization-based；
- **soft × newline 正交互**：原始设计是 structure-guided latent initialization；
- **newline 主效应**：原始设计主要是 early structural steering，不再声称 latent reasoning；
- **无稳定主效应**：只保留为 audit-derived observation，不承担独立方法贡献。

## 5. 关于 EOT / anchor 的最小结论

最小 anchor identity control 已在 VStar 与 MMVP 完成。它比较 `</think>`-related first subtoken、`<think>`-related first subtoken、newline 与 direct-hard；所有条件共享 step-0 initializer、forced step-1 handoff 和 beta=0.7。

| Dataset | direct hard | `</think>` | `<think>` | newline |
|---|---:|---:|---:|---:|
| VStar | **73.82%** | 71.73% | 68.59% | 71.73% |
| MMVP sample | 67.67% | **70.33%** | 68.33% | 67.00% |
| MMVP pair | 39.33% | **42.00%** | 40.00% | 37.33% |

结论是 anchor 身份具有明显数据集依赖性：`</think>` 不是可由任意格式 token 替换的无关细节，但也没有跨任务稳定支配 direct-hard。由于继承实现使用的是字符串编码后的首个 subtoken embedding，该实验不是完整 lexical tag 语义测试。它作为负控和边界条件已足够，不再扩展特殊 anchor 搜索。

## 6. TALR 的正确位置

True TALR 的定义为：

1. Early transition / initializer；
2. 后续 soft quota 5% 的 budgeted refinement；
3. `format cooldown=2, min-step=2` 与 late diffuse/repeat veto 的 discrete stability guard。

它的后两项主要管理生成退化风险，不应被写成直接提升视觉推理能力。

| Model | VStar | MMVP sample/pair | RealWorldQA fixed200 | VisuLogic300 | Mean delta vs COT |
|---|---:|---:|---:|---:|---:|
| R1-Onevision-7B-RL | 72.25 vs 68.06 | 70.67/42.67 vs 68.00/39.33 | 65.50 vs 66.00 | 23.67 vs 21.33 | +2.17 pp |
| Vision-R1-7B | 82.20 vs 77.49 | 73.33/48.00 vs 73.67/48.67 | 67.50 vs 67.00 | 20.67 vs 23.00 | +0.64 pp |

R1-RL 上 True TALR 为 3/4 不低于 COT，满足本轮预先设定的主方法门槛；Vision-R1 仅 2/4 不低于 COT，说明跨模型稳定性不足。故 TALR 可作为机制导出的实用扩展和性能结果，但不应成为压倒机制审计的标题主角。

## 7. 支撑动机的 probe 结果

- **Confidence--correctness mismatch**：在 pure-soft 的 2,291 样本分析中，mean-confidence AUROC 跨数据集约为 0.369--0.485；去除抽取因素后也不能把结论夸大成“高置信总是错误”。准确表述是：token confidence 不是可泛化的 correctness oracle。
- **Pure-soft 退化与 format 稳定**：VStar 上 pure-soft 58.64%，而 format cooldown2 / guard 约 74--75%，同时明显减少长输出、maxed 和重复。它证明 guard 能修复 decoding degeneration，不证明 guard 提升 reasoning ability。
- **More-soft 不是单调策略**：quota 的最佳值随数据集变化，较高 quota 会失稳；这反对“提高 soft ratio 即提高推理能力”的简单假设。

## 8. 建议的论文叙事与章节结构

### 单一中心主张

> In multimodal reasoning, latent interventions are most informative in a narrow early initialization window, where their transient effect is rapidly externalized into a discrete reasoning prefix rather than requiring persistent latent decoding.

### 章节逻辑

1. **Introduction**：现有 entropy routing 假定 uncertainty 指示何时 soft；提出“何时干预”而不是“做多少 soft”的问题。
2. **Audit observations**：confidence mismatch、pure-soft failure、early/late routing 的差异；严格区分 exact-original 与改进实现。
3. **Mechanism analysis**：timing、same-prefix replay、cache rebuild、negative controls，提出短暂状态到离散 prefix 的外化解释。
4. **Method**：根据 2x2 的实际结果确定最小 early initializer；TALR 作为可选的受约束扩展。
5. **Experiments**：两模型四核心数据集主表、稳定性指标、fixed/damaged、机制对照和诚实的负例。
6. **Limitations**：任务依赖、模型依赖、greedy setting、并非通用 uncertainty router。

## 9. 贡献表述：可写与不可写

### 在 2x2 和 original audit 完成后可写

1. 对 entropy-routed latent decoding 做 component-level audit，分离早期固定路径与后期动态 routing；
2. 提出并验证“早期连续影响快速外化为离散 prefix”的机制解释；
3. 基于审计发现提出或评估受约束的 early handoff / TALR 扩展；任何独立方法主张均须与原始 LEAD early path 明确区分；
4. 用准确率、输出稳定性、fixed/damaged 与 replay 控制共同评估收益和损伤。

### 不能写

- TALR 或 soft reasoning 在所有数据集/模型上优于 COT；
- confidence 与 correctness 总是负相关；
- format guard 提升了视觉 reasoning 能力；
- LEAD transition 是我们的原创模块；
- `</think>` 是普适 reasoning boundary prior；
- late entropy routing 完全无效，或所有 reasoning 在 step 0 就永久锁定。

## 10. 投稿前优先级

1. 完成 step-0 2x2，并据其冻结方法最小形式；
2. 冻结 exact-original LEAD 版本，完成最小审计矩阵；
3. 输出两模型四数据集的统一 corrected/specialized 主表，并只保留可追溯、配置匹配的 run；
4. 对核心 pairwise 比较报告 fixed/damaged、McNemar、bootstrap CI 和 agreement；
5. 增加低成本 \(H_0\) 与 \(1-\cos(s_0,e(y_0))\) 分层分析，区分分布性 soft effect 与 newline steering；
6. 停止新的 anchor / dense beta 搜索，避免把论文拖入缺乏判别力的调参。

## 最终判断

当前最有希望的论文不是“一种普适地超过 COT 的 soft decoding 方法”，而是一篇机制更严谨的工作：审计动态 latent routing 的假设，定位其有效窗口，并把连续状态如何交接为离散推理前缀变成一个可验证的设计问题。若 2x2 给出清晰的 soft 或交互效应，该工作将同时拥有审计发现、机制解释和最小方法；若它只显示 newline 或强任务依赖，论文仍可诚实地收束为对 early decoding intervention 的条件性机制研究。

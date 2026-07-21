# 从性能竞争到归因审计：多模态 Latent Decoding 的论文主线建议

## 执行摘要

当前最稳妥、也最有辨识度的论文不应宣称“我们提出了一个普适地优于 COT 的 latent routing 方法”。现有结果不支持这个更强的说法：TALR 在 R1-Onevision-7B-RL 上总体有正收益，但在 Vision-R1 上并不稳定；不同数据集的最优 early transition 组件也不相同。

更准确的主线是一个**机制归因审计（mechanistic attribution audit）**：在已审计的 LEAD 实现、模型与多模态推理任务中，通常归因于 entropy-driven latent routing 的整体收益，主要集中在一个由多个组件构成的早期 transition；该影响会很快通过最初的离散 reasoning prefix 外化，而不是以可测的持久 latent/KV 状态继续主导后续生成。

建议在论文中使用如下英文中心句：

> **In the audited multimodal reasoning settings, the gains commonly attributed to entropy-driven latent routing are concentrated in a composite early transition. Its effect is not maintained as a persistent latent state, but is rapidly mediated by the first discrete reasoning prefix.**

这句话的限定十分重要：

- `in the audited settings`：不以两个模型和若干 benchmark 代表整个 latent reasoning 领域；
- `commonly attributed`：质疑的是收益归因，不是宣判后期 entropy routing 在任何任务都无用；
- `rapidly mediated`：不否认 soft 状态在产生最初 token 时的短暂作用，只陈述匹配 prefix 后未观察到残余效应。

论文的价值不在于发现“早期 token 很重要”，而在于证明：**full-method vs. COT 的分数差，不能自动归因于 latent semantic routing。** 一个表面统一的 latent transition 实际混合了 soft source、结构性 cue、boundary bridge 和后期 routing；不做逐组件匹配控制时，机制归因并不可识别。

## 1. 研究问题：现有解释中的 Attribution Gap

entropy-routed latent decoding 的直觉通常是：当 token 分布的熵较高时，用 soft embedding 保留多个候选，而非立刻坍缩为 top-1 token，从而改善推理。LEAD 也以 entropy 作为软/硬路由的核心信号。

但是实际实现不是单纯的“高熵时使用 soft state”。以当前审计到的 LEAD 早期路径为例，生成开始处同时包含：

1. 来自 token 分布的 soft embedding；
2. step-0 的 newline 结构性混入；
3. 与 `</think>` 相关的 early bridge；
4. 随后的 entropy-gated late routing 与其他稳定规则。

因此，下面的推断并不成立：

\[
\text{Full LEAD gain over COT}
\;\Rightarrow\;
\text{gain from latent semantic routing}.
\]

这就是论文要回答的 attribution gap：**当一个 latent decoding 方法提升准确率时，如何区分收益来自 distributional latent semantics，还是来自隐藏在 transition 中的固定结构 cue、phase-boundary prior 或早期 prefix steering？**

这不是针对 LEAD 的“找茬式消融”。它提出了一个对 training-free latent decoding 更一般的评测要求：若方法含有复合 transition，就需要 component-level matched controls，而不能只报告 full method 与 COT 的总分差。

## 2. 当前证据链：三次去混杂

### 2.1 早期路径与后期 entropy routing 分离

历史上已经完成的大量 matched controls 一致指向一个现象：只保留早期 `soft -> normal` 路径的 `initial_transition_only`，在 VStar 与 MMVP 等任务上可以恢复 full LEAD 的大部分收益；而 full LEAD 相对 early-only 的后期 routing，常同时带来 fixed 与 damaged，净收益较小且依赖任务。

这支持的结论是：**aggregate gain 的主要部分位于被忽略的 early path；late routing 的平均净效用有限、并且条件化。**

它不支持以下更强说法：

- “后期 entropy routing 从来没有用”；
- “所有模型和数据集都只由第 0 步决定”；
- “initial transition 已经是一个普适更强的方法”。

这部分历史结果可以作为主文的起点，但在投稿前必须完成一次 exact-original audit：冻结原始 commit、模型 checkpoint、prompt、采样、evaluator 与 route 日志，并统一复跑/复评 COT、full original LEAD、early-only 和 late-only（或关闭 early path）的最小集合。目的不是推翻已有发现，而是把它变成可追溯、可复核的论文证据。

### 2.2 Transition Cube：early transition 不是单一 latent operator

为拆开 early transition，最新受控实验将其写成三个因素：

\[
T=(S,N,B),
\]

其中 (S\) 为 soft source，(N\) 为 step-0 newline structural cue，(B\) 为 EOT/`</think>`-related bridge。实验固定 step-1 handoff、随后关闭后期 routing，因此它是对 early transition 的受控 component analysis，而不是 natural full LEAD 的替代跑法。

下表是 R1-Onevision-7B-RL 在统一 corrected/specialized 评测下的八格结果；MMVP 括号内为 pair accuracy。

| Dataset | hard/no-NL/direct | hard/NL/direct | soft/no-NL/direct | soft/NL/direct | hard/no-NL/EOT | hard/NL/EOT | soft/no-NL/EOT | soft/NL/EOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VStar accuracy | 68.06 | 73.82 | 68.59 | 73.82 | 69.11 | 69.63 | 71.73 | 72.25 |
| MMVP sample accuracy | 68.00 | 69.00 | 69.33 | 67.67 | 68.00 | 68.33 | 68.33 | 70.33 |
| MMVP pair accuracy | 39.33 | 42.00 | 41.33 | 39.33 | 40.00 | 40.00 | 40.67 | 42.00 |

VStar 的 factorial bootstrap 表明：newline 主效应为 `+3.01 pp`，95% CI `[+0.39, +5.76]`；soft 主效应为 `+1.44 pp`，CI `[-1.05, +3.93]`；EOT 主效应为 `-0.39 pp`，CI `[-2.75, +1.96]`。因此，在该受控设定中，VStar 的显著主效应来自 newline，而不是可分离的 soft semantic contribution。

MMVP 上的所有单因子 CI 均跨零；但组合关系与 VStar 不同，例如 direct handoff 下 soft 与 newline 呈现负向交互，而 EOT bridge 改变了观察到的交互格局。对此最严谨的结论是：

> **The transition components are neither universally additive nor uniformly beneficial. Their main effects and interactions vary by task.**

尤其不能把 MMVP 的小幅点估计写成稳定增益，也不能将 EOT 描述为“修复”了某种冲突；它只是调制了被观察到的组件交互。

这组实验是论文最有辨识度的部分之一：它直接展示了**composite latent transition 可以提高准确率，但其收益未必可以归因于 latent semantics。** newline 是否只是 prompt/format trick 并不是反驳，恰恰是核心审计发现：一个简单结构 cue 足以解释显著 aggregate gain 时，full transition 的增益就不能被自动归为 soft reasoning。

### 2.3 持久 latent state 与离散 prefix 中介分离

此前的 cache-rebuild externalization 控制显示：保留 transition 自然产生的前缀并用离散 token 重建 cache 后，VStar 与 MMVP 的表现大致在 prefix 长度 2 处恢复；而 prefix 1 的恢复不完整。这提示 early effect 会在极早 token 内外化。

更干净的新 Same-Prefix Logit Probe 进一步控制了前两个离散 token：它比较 `hard + newline + EOT` 与 `soft + newline + EOT` 两条路径，强制使用完全相同的前两个 COT token，随后测量 step-2 distribution。结果为：

| Dataset | samples | mean JS divergence | top-1 different | top-20 overlap | first free token different |
|---|---:|---:|---:|---:|---:|
| VStar | 80 | `3.22e-9` | 0 | 0.9788 | 0 |
| MMVP | 100 | `1.10e-8` | 0 | 0.9830 | 0 |

在这些测试路径与样本上，匹配前两个 token 后没有测到 soft/hard cache 的残余差异。结合 cache rebuild，当前最合适的机制链是：

\[
\text{transient early state perturbation}
\rightarrow
\text{selection of first discrete prefix}
\rightarrow
\text{standard autoregressive continuation}.
\]

而不是：

\[
\text{same prefix} + \text{different latent cache}
\rightarrow
\text{persistent trajectory difference}.
\]

这里必须保持边界：我们没有证明 KV cache 在所有模型、所有路径中都无关；只是在受测 transition 路径中，前两 token 一旦匹配，未发现可测的非 token 中介残余影响。旧的自然 replay 出现过更长的分叉，但它混入了不同 route/control 条件，只能作为探索性轨迹证据，不能覆盖这项更严格的 matched-prefix probe。

## 3. TALR 的正确定位：机制导出的受约束工程策略

TALR 的现定义为：保留 early transition，限制后续 soft quota，并使用 cooldown/min-step 和 late diffuse/repeat veto 抑制退化。它来自上述审计逻辑：若主要有效干预集中在早期，而持续 soft 会带来长输出、重复、格式边界与答案漂移风险，则应保留 early path、约束后期 soft、在必要时回归离散状态。

当前统一结果如下：

| Model | VStar | MMVP sample / pair | RealWorldQA fixed200 | VisuLogic300 | 对 COT 的整体定位 |
|---|---:|---:|---:|---:|---|
| R1-Onevision-7B-RL TALR | 72.25 vs 68.06 | 70.67 / 42.67 vs 68.00 / 39.33 | 65.50 vs 66.00 | 23.67 vs 21.33 | 四集平均约 `+2.17 pp`，3/4 不低于 COT |
| Vision-R1 TALR | 82.20 vs 77.49 | 73.33 / 48.00 vs 73.67 / 48.67 | 67.50 vs 67.00 | 20.67 vs 23.00 | 四集平均约 `+0.64 pp`，2/4 不低于 COT |

因此 TALR 可以作为一项有意义的 practical extension：机制洞察能导出一个可部署的受约束策略，并在部分模型/任务上改善稳定性或准确率。它不应承担论文 headline method，也不应写为 universal COT replacement。诚实展示其跨模型不稳定性，反而与审计主线一致：不存在一个统一有效的 latent policy。

此前 pure-soft 与 format-stability 系列仍然有价值，但其位置应是补充证据：`format_cooldown2` 与 guard 能显著修复 pure-soft 的长输出、重复与 maxed 退化；这说明它们是输出稳定 guardrail，而非 latent reasoning 普遍增强器。

## 4. 建议的论文故事结构

### 第一幕：潜力与不可识别的归因

介绍连续 soft state 能在解码中保留候选不确定性，entropy-routed methods 因而被视为 training-free latent reasoning 的路径。指出 full method 的提升不能识别收益来源，因为实现同时包含 fixed early path、结构 cue、phase bridge 与 late routing。

### 第二幕：收益首先集中在被忽略的 early path

通过 exact-original matched controls 展示 COT、force-normal、initial-soft-only、initial-transition-only 与 full LEAD。主图左侧放 accuracy，右侧放 early-only 对 full LEAD 的 fixed/damaged matrix：后期 routing 并非“零作用”，但其正负改变会抵消，因而平均净效用有限且 task-dependent。

### 第三幕：Transition Cube 推翻单一机制归因

将 early transition 分解成 (S,N,B)，用 cube/interactions 展示 VStar 与 MMVP 的不同主效应与交互。这里的中心发现不是 “early transition works”，而是 “transition 是任务依赖的复合干预，aggregate improvement 不能直接归因于 latent semantics”。

### 第四幕：效应由离散 prefix 中介

以 cache rebuild 和 same-prefix logits 收束因果链：transition 改变早期 token；一旦前两个 token 对齐，后续分布与首个自由 token 也对齐。由此把“大而模糊的 hidden-state basin”收紧为可检验的 early prefix mediation。

### 第五幕：设计启示而非万能方法

将 TALR 作为根据审计结果得到的 constrained routing extension。报告收益和负结果，表明它证明可部署性而不是普适优越性。

## 5. 主贡献的写法

建议只保留三条主要贡献，避免把 TALR 包装成与机制发现同等强度的第四贡献：

1. **Attribution audit.** We conduct a component-level audit of LEAD and separate its fixed early transition from entropy-triggered late routing, showing that the observed aggregate gains are concentrated in the former while the latter exhibits limited and task-dependent net utility.
2. **Composite transition.** We decompose the early transition into distributional soft input, structural newline steering, and an EOT-related bridge. A factorial analysis reveals task-dependent main effects and interactions, demonstrating that aggregate improvements cannot be directly attributed to latent semantics.
3. **Prefix mediation.** Through cache reconstruction and matched-prefix logit probes, we show that the early intervention acts transiently by changing the first discrete reasoning prefix; after matching the first two tokens, the tested soft and hard paths become distributionally indistinguishable.

随后可用一句补充 TALR：

> We additionally evaluate a constrained routing policy derived from these findings and document both its gains and limitations across models.

## 6. 图表与实验呈现

建议正文只保留四张关键图：

1. **Figure 1: Attribution gap overview.** 三列：claimed view `entropy -> dynamic soft routing -> better reasoning`；audited implementation `soft source + newline + EOT bridge + late routing`；observed mediation `composite early transition -> first discrete prefix -> hard autoregressive continuation`。
2. **Figure 2: Early path versus late routing.** accuracy 与 paired fixed/damaged 并排；不要只展示平均分。
3. **Figure 3: Transition Cube.** 两个数据集分别绘制 interaction plot：横轴 hard/soft，两条线为 newline off/on，分面为 direct/EOT。让任务依赖的交互一眼可见。
4. **Figure 4: Prefix mediation.** 左侧为 cache rebuild 的 prefix-length curve，右侧为 same-prefix 的 JS/top-1/free-token agreement。

主表保留 COT、original LEAD、initial transition、TALR，并诚实展示模型间差异。pure-soft、format guard、beta、更多 anchor 的全量表格放附录，作为排除替代解释与稳定性分析。

## 7. 当前可以写与不能写的结论

### 可以写

- LEAD 的受审计 early path 是复合而非单一操作；
- 在当前模型和任务中，early path 承担了 full-method aggregate gain 的重要部分，late routing 的净效用有限且依赖任务；
- VStar 的受控 cube 中，newline 是显著主效应，soft 的可分离主效应没有达到可靠水平；
- 组件贡献和交互跨 VStar/MMVP 不一致；
- 在 matched-prefix probe 的 tested paths 上，前两 token 匹配后未观察到可测 residual cache effect；
- TALR 是从审计得到的实用受约束策略，而非普适最优方法。

### 不能写

- “entropy routing 无效”或“late routing 没有价值”；
- “soft reasoning 不含语义信息”；
- “newline 一定是更深层 reasoning mechanism”；
- “KV cache 完全不重要”；
- “TALR 普适优于 COT/LEAD”；
- “LEAD 论文的全部收益都来自 prompt trick”。

## 8. 投稿前的最小高优先级工作

时间有限时，不应继续掉入 dense beta sweep、更多 special anchor、KV 层级 patching、新 router、更多 quota 或大量新 benchmark 的泥潭。最缺的是证据闭环，而不是更多现象。

### P0：Exact-original provenance audit

冻结 original LEAD commit、模型 checkpoint、prompt/template、generation 参数、evaluator 与 newline/EOT 的代码来源。最小比较为 COT、full original LEAD、initial-transition-only、late-only/early-off，并输出 route count、route position 与 full-vs-initial paired fixed/damaged。它把历史发现变成可复查的主文证据。

### P0：Cube 的配对统计完善

无需增加新的 cube 配置；补齐每个单因子比较的 fixed/damaged、McNemar exact test、bootstrap CI，以及 MMVP sample 与 pair 两级统计。MMVP 的小幅单因子差异应明确标记不确定，论文重点放在交互格局与可重复的设计结论上。

### P1：第二模型最小机制复现

在 Vision-R1 上只跑 VStar/MMVP 的四个判别条件：hard-noNL-direct/COT、exact original initial transition、移除 soft source、移除 bridge 或 structural component。即使第二模型的模式不同，也支持“component utility is model-dependent”，并显著降低 case-study-only 的审稿风险。

## 9. 标题与最终判断

推荐标题：

> **What Drives Latent Decoding Gains? Auditing Early Transitions in Multimodal Reasoning**

更稳妥的备选：

> **Rethinking Latent Decoding Gains: A Component-Level Audit of Early Multimodal Reasoning**

当前论文的最大优势不再是“找到一个很高的数字”，而是完成了三层去混杂：early path 对 late routing、transition 组件之间、以及 persistent latent state 对 discrete-prefix mediation。只要 exact-original provenance 与配对统计补齐，并将结论严格收束在审计范围内，这是一篇可以自洽的机制审计论文。

最大的风险也已经很清楚：不是故事不够大，而是来源审计不够可追溯，或者为了强调结论把 task/model-specific 结果过度泛化。最好的论文姿态不是制造一个更宏大的 universal claim，而是让每次主张收缩都由更严格的控制实验支撑。

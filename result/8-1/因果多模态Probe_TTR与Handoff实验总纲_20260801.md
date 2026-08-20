# 因果多模态 Probe、Trajectory Trust Region 与 Handoff 实验总纲

> 文档状态：后续实验约束总纲，版本 1.0  
> 日期：2026-08-01  
> 项目：Multimodal Latent Intervention / TALR / Latent-to-Discrete Handoff  
> 主模型：R1-Onevision-7B-RL  
> 文档用途：约束后续实验立项、变量、数据、评测、升级与停止决策  
> 适用原则：任何新增实验必须映射到本总纲的研究问题；无法映射时先修订总纲，不直接扩充队列  

## 摘要

本阶段不再以继续扫描 TALR 的固定超参数为主要任务，而围绕一个更明确的问题展开：**如何识别真正由视觉证据支持的 latent intervention，并在接受干预后约束其轨迹风险，最终将有效影响安全外化为离散 prefix。** 为此，总纲将后续研究拆分为三个可独立证伪的模块：Causally Supervised Multimodal Utility Probe 负责判断是否值得干预；Trajectory Trust Region（TTR）负责决定允许偏移多远；Discrete Checkpoint and Causal Reset 负责决定何时清除 soft history。三个模块必须分别通过预注册门槛，才允许构成联合方法。

现有证据为该路线提供动机，但不保证成功。Intervention Atlas 证明存在较大 oracle gap 和明显的时机、强度异质性；ProbeV3 证明 actionability 可学习而 correctness direction 难以学习；图像来源 transplant 证明 soft state 对来源敏感，却尚未证明 true-image correctness alignment；TTR-Z 在 RealWorldQA64 正向而在 VStar64 负向；corrected handoff 则表明 prefix=2 的离散 checkpoint 具有积极信号。因此，后续实验必须优先回答因果来源、方向判断和风险边界，而不是直接宣称已经形成新方法。

**一句话总目标：** 建立一个由因果标签监督、由轨迹信赖域约束、并通过离散 checkpoint 完成安全退出的多模态 latent intervention 框架。

**关键词：** causal utility；multimodal intervention；trajectory trust region；discrete checkpoint；causal reset；externalization

## 目录

PDF 转换脚本自动生成目录。

---

## 1. 总研究问题与贡献边界

### 1.1 总研究问题

> 在多模态推理过程中，如何判断一次 image-conditioned latent intervention 是否值得执行；若值得执行，应允许其偏离当前离散轨迹多远；其影响何时已被离散 prefix 充分承载，从而可以安全退出 soft reasoning？

它被拆分为三个严格不同的问题：

| 模块 | 研究问题 | 不负责解决的问题 |
|---|---|---|
| Causal Multimodal Probe | 是否值得执行；变化是否依赖真实图片 | 不直接控制干预强度 |
| Trajectory Trust Region | 接受干预后允许偏移多远 | 不判断答案方向是否正确 |
| Checkpoint and Reset | 何时 latent effect 已外化，可以删除 soft history | 不创造新的正确视觉信息 |

### 1.2 当前可以使用的主张

- Latent action 具有真实但高度异质的轨迹影响；
- Actionability 比 directional correctness utility 更容易预测；
- 固定 $\lambda$ 不对应固定的真实轨迹扰动；
- soft state 对图片来源敏感，但 true-image correctness alignment 尚未成立；
- 短离散 prefix 能够承载部分 latent influence；
- 单纯 embedding-distance TTR 不具备跨任务稳定性。

### 1.3 当前禁止使用的主张

- “我们已经学会预测 latent intervention utility”；
- “True-image soft state 比随机方向更正确”；
- “TTR 已经稳定提高 TALR”；
- “prefix=2 是跨模型通用最优长度”；
- “Probe 是因果模型”；
- “加入视觉 hidden state 就等于方法具有多模态创新”。

---

## 2. 已有证据与实验状态

| 证据批次 | 状态 | 核心结果 | 对后续的约束 |
|---|---|---|---|
| Full Intervention Atlas | 完成 | 991 样本、7,928 events、15,856 branches；oracle gap 明显 | 证明候选空间，不作为在线成绩 |
| ProbeV3 | 完成 | Actionability AUROC 0.905；Fix/Damage AUROC 约 0.52–0.55 | 不再只扩同类标签和 MLP 容量 |
| Structured CF analysis | 完成 | 32 token 内仅 28.55% correctness-changing events 可见 | 短文本分叉不能直接作为 utility 标签 |
| Corrected Handoff | 完成 | prefix=2：VStar 72.77%，MMVP 70.67% / pair 44.00% | 支持 checkpoint/reset，需跨模型验证 |
| Image-source transplant $\lambda=0.95$ | 完成 | True image 未优于 hard/random/noise | 必须先证明 visual specificity |
| Image-source transplant $\lambda=0.80$ | 运行中 | true/swapped 两分支 | 完成前不锁定视觉结论 |
| TTR-Z | 完成 | RW64 +2 correct；VStar64 -2 至 -3 correct | embedding L2 不能单独作为主控制信号 |

---

## 3. 统一符号与因果对象

### 3.1 基本状态

离散 hard embedding：

$$
h_t=E(y_t).
$$

真实图片条件下的 soft embedding：

$$
s_t^{\mathrm{true}}
=
\sum_v p_t(v\mid x^{\mathrm{true}},y_{<t})E(v).
$$

控制图片来源 $c$ 的 soft embedding：

$$
s_t^c,
\qquad
c\in\{\mathrm{swapped},\mathrm{masked},\mathrm{noise}\}.
$$

强度为 $\eta$ 的 action：

$$
z_t^c(\eta)
=
h_t+\eta(s_t^c-h_t),
\qquad \eta\in[0,1].
$$

### 3.2 样本级最终 utility

令 $C(\cdot)\in\{0,1\}$ 表示最终答案正确性：

$$
U_t^{c,\eta}
=
C\left(Y(\operatorname{do}(z_t^c(\eta)))\right)
-
C\left(Y(\operatorname{do}(h_t))\right).
$$

其取值为：

- $+1$：fixed；
- $0$：correctness neutral；
- $-1$：damaged。

### 3.3 Visual specificity

定义真实图片相对控制来源的效应优势：

$$
V_t^\eta
=
U_t^{\mathrm{true},\eta}
-
\frac{1}{|\mathcal C|}
\sum_{c\in\mathcal C}U_t^{c,\eta}.
$$

其中：

$$
\mathcal C
=
\{\mathrm{swapped},\mathrm{masked},\mathrm{noise}\}.
$$

$V_t^\eta>0$ 才表示 true-image action 比一般扰动更有价值。仅仅 true-image 分支改变输出，不足以说明 visual specificity。

### 3.4 稠密但非最终的辅助标签

最终 correctness 标签稀疏，可同时记录 forced-answer margin：

$$
m_t^c
=
\ell_t^c(y_{\mathrm{gold}})
-
\max_{y\neq y_{\mathrm{gold}}}\ell_t^c(y).
$$

定义：

$$
\Delta m_t^c=m_t^c-m_t^{\mathrm{hard}}.
$$

该标签仅用于训练和机制分析，因为它使用 gold option；线上特征不得包含 gold identity 或未来答案。

---

## 4. 数据、模型与开发边界

### 4.1 数据角色

| 角色 | 数据集 | 用途 | 是否允许调参 |
|---|---|---|---|
| 历史开发域 | VStar、VisuLogic、MMK12、VMCBench | 标签分析、模型结构开发 | 允许，但记录查看次数 |
| 内部验证域 | RealWorldQA、MMVP | 复核方向，不宣称严格 unseen | 只允许一次阶段性选择 |
| 锁定外部域 | Vision-R1 / OpenVL 上的 VMCBench、POPE-Adversarial、MMK12 子科 | 最终跨模型、跨任务验证 | 不允许根据结果回调参数 |

VStar、MMVP、RealWorldQA、VisuLogic 都已被反复查看，论文中不能称为严格未见测试集。最终工作必须保留至少一个新模型与一组未用于方法选择的 benchmark cells。

### 4.2 模型角色

| 模型 | 角色 |
|---|---|
| R1-Onevision-7B-RL | 完整机制开发与消融 |
| Vision-R1-7B | 第一外部验证模型 |
| OpenVLThinker-7B | 不同模板/训练路线的第二外部验证 |

### 4.3 固定生成口径

```text
greedy / no sampling
seed = 42
max_new_tokens = 1024
cot_prompt_mode = orign
same checkpoint / processor / evaluator
```

机制开发阶段不使用 sampling 噪声。锁参后再补多 seed 或 paper-style sampling robustness。

---

## 5. 研究问题与预注册假设

### RQ1：视觉来源是否具有 correctness-aligned 因果效应

**假设 H1：** 在相同 prefix、位置和 action strength 下，true-image soft action 的净 utility 高于 swapped/masked/noise control。

**反证条件：** true image 与控制来源的 fixed-damaged、gold margin 和 source-specific utility 无稳定差异。

### RQ2：视觉特异性是否可被在线特征预测

**假设 H2：** true-vs-hard 或 true-vs-masked 的即时分布差异，比单独 entropy 更能预测 $V_t$ 和 directional utility。

**反证条件：** leave-one-dataset-out AUROC 与简单 entropy/margin baseline 无实质差异。

### RQ3：输出空间 trust region 能否减少 damage

**假设 H3：** one-step JS/logit divergence 比 embedding L2 更能识别高风险 action，并在保留部分 fixed 的同时减少 damaged。

**反证条件：** TTR-P 仅降低 coverage/fixed，不能减少 damaged，或退化为固定 $\eta$。

### RQ4：latent influence 何时已完成离散外化

**假设 H4：** 存在较短 checkpoint，使 reset 后结果保留 intervention 的有效影响，同时删除长期 soft-history 依赖。

**反证条件：** reset 总是消除收益，或 checkpoint 长度与稳定性没有可重复关系。

### RQ5：Probe 与 TTR 是否互补

**假设 H5：** Probe 负责减少错误方向 action，TTR 负责限制剩余 action 的幅度，联合策略优于任一单模块。

**反证条件：** 联合结果不优于最强单模块，或额外复杂度只降低 coverage。

---

## 6. Phase A：完成因果标签与视觉来源审计

### A0：完成现有 $\lambda=0.80$ source-strength

当前运行：true-image 与 swapped-image，共 2,048 branches。不得中途根据 partial accuracy 修改分支或 $\lambda$。

### A1：统一现有标签

对每个 event 写出：

```text
hard outcome
true-image outcome
swapped-image outcome
masked-image outcome
random/noise outcome
fixed / damaged / neutral
answer changed
forced-answer margin delta
first divergence
8/16/32-token mismatch
event position and type
```

### A2：分析矩阵

必须按以下维度报告：

- dataset；
- event step：1/2/4/8/16/32/entropy/random；
- action strength：0.80/0.95；
- hard baseline correctness；
- true-image vs each control；
- visual specificity $V_t$；
- bootstrap 95% CI 与 paired test。

### A3：进入 Probe 阶段的门槛

至少满足一个条件：

1. True-image 在两个数据集上的 pooled `fixed-damaged` 高于 controls；
2. 某个预定义 early window 内 $V_t$ 的 bootstrap CI 不跨 0；
3. true-vs-control gold-margin contrast 对最终 fix/damage 有稳定排序能力。

若三个条件全部不满足：

- 停止“正确视觉 utility probe”主张；
- 只将图像来源实验作为负结果；
- 后续 TTR/Handoff 仍可独立进行。

---

## 7. Phase B：Causally Supervised Multimodal Utility Probe

### B1：Probe 的定位

Probe 是用因果标签监督的预测器，不是因果模型。其线上目标分三层：

1. $\hat A_t$：actionability；
2. $\hat V_t$：visual specificity；
3. $\hat U_t$：directional utility。

### B2：允许使用的线上特征

#### Tier 1：单路径、低成本

- entropy、top-1、margin、top-k mass；
- entropy trend；
- soft-hard relative L2 / cosine；
- token position、early-window offset；
- token type、answer-zone、format boundary；
- 已缓存的 question-conditioned visual-hidden summary；
- soft state 与 visual summary 的 cosine/norm。

#### Tier 2：成对即时反事实，允许一个额外 forward

- true-soft vs hard one-step JS；
- true-soft vs masked-soft one-step JS；
- top-1 switch；
- top-k overlap；
- logit margin contraction/expansion；
- true-vs-masked soft direction cosine。

#### 禁止作为线上特征

- gold answer identity；
- 最终 correctness；
- 未来输出长度；
- 最终 first-divergence position；
- 使用完整反事实 rollout 后才得到的字段；
- dataset name 的直接 one-hot shortcut。

### B3：模型与基线

只比较：

| 模型 | 作用 |
|---|---|
| Entropy threshold | 现有路由信号基线 |
| Logistic regression | 可解释线性基线 |
| Small MLP | 检查有限非线性 |
| Multi-task Probe | Actionability + specificity + utility |

不在本阶段引入 Transformer、RNN 或大 hidden-state classifier。若低维特征不支持方向判断，应修改问题或特征，而不是先扩大模型。

### B4：训练约束

- 按 original sample 分组切分，禁止同一样本不同 event 泄漏；
- 主要报告 leave-one-dataset-out；
- 类别不平衡使用 class weight，不复制测试样本；
- 超参数只在开发域选择；
- 固定随机种子并保存 split manifest；
- 同时报告 AUROC、AUPRC、coverage、fixed、damaged 和 calibration。

### B5：成功门槛

Probe 进入在线实验必须同时满足：

- 外部 conditional utility AUROC ≥ 0.60；
- 在 10%–30% coverage 区间，至少两个 held-out dataset 的 `fixed-damaged > 0`；
- 相对 entropy baseline，damage rate 降低至少 20%；
- 不依赖 dataset identity 才能达到结果。

若未通过，Probe 不进入联合主方法。

---

## 8. Phase C：Trajectory Trust Region

### C1：TTR-Z 已完成

结论：embedding relative L2 能改变实际 $\eta$，但 RW64 正向、VStar64 负向；所有 candidate 均被接受。TTR-Z 只作为几何 baseline，不继续调半径全量扫描。

### C2：TTR-P

对同一 prefix 计算 hard 与候选 action 的下一步分布：

$$
D_p(\eta)
=
D_{\mathrm{JS}}
\left(
p_{t+1}^{\eta}
\parallel
p_{t+1}^{\mathrm{hard}}
\right).
$$

从离散集合中选择最大允许强度：

$$
\eta_t^*
=
\max_{\eta\in\{0,0.5,0.8,0.9,0.95,1.0\}}
\eta
\quad
\text{s.t.}
D_p(\eta)\le\epsilon_p.
$$

半径只能由开发事件的 JS 分位数校准，不得根据 64 样本 accuracy 反复修改。

### C3：首轮矩阵

数据：VStar64、RealWorldQA64。

| 方法 | 目的 |
|---|---|
| Fixed $\lambda=0.95$ | 当前 TALR action baseline |
| TTR-Z r=0.67 | 几何 baseline |
| TTR-P q25 | 保守输出半径 |
| TTR-P q50 | 中等输出半径 |

不同时扫描 window、cap、entropy threshold、Probe 和 reset。

### C4：成功门槛

相对 fixed $\lambda=0.95$：

- pooled damaged 至少减少 25%；
- pooled fixed 至少保留 60%；
- VStar 与 RealWorldQA 的 net 均不为负；
- 平均额外 forward 次数 ≤ 2/event；
- failed/long/maxed 不增加。

未通过则停止在线 TTR，不进入更多数据集。

---

## 9. Phase D：Discrete Checkpoint and Causal Reset

### D1：研究对象

接受一次 action 后生成长度为 $H$ 的离散 prefix：

$$
\widehat B_t
=
(\widehat y_{t+1},\ldots,\widehat y_{t+H}).
$$

随后用原始图像、文本 prompt 和全部离散 token 重建 KV：

$$
(K,V)_{\mathrm{reset}}
=
F(x,y_{\le t},\widehat B_t).
$$

### D2：矩阵

只测试：

| 配置 | 作用 |
|---|---|
| No reset | 保留 soft history |
| Reset $H=1$ | 极短外化 |
| Reset $H=2$ | 现有最佳候选 |
| Reset $H=4$ | 检查更长前缀 |

prefix=8 已在 VStar 明显退化，不进入首轮主矩阵。

### D3：指标

- accuracy、fixed/damaged；
- reset vs no-reset prediction agreement；
- first free-token JS；
- prefix reproduction；
- latency、重算 token 数；
- repeated/long/maxed；
- intervention effect retention。

### D4：成功门槛

- 至少两个数据集上 reset 保留 no-reset 的 80% 以上 fixed；
- damaged 不增加；
- reset 后相同 prefix 的 route-specific JS 接近 0；
- checkpoint 长度的结论可在第二模型复现。

---

## 10. Phase E：联合方法

只有 Phase B 与 Phase C 至少一项通过，且 Phase D 通过，才允许进入联合实验。

### E1：联合策略

```text
Early transition / hard COT
        ↓
early candidate proposal
        ↓
multimodal causal Probe：是否值得干预
        ↓
TTR：允许偏移多远
        ↓
执行一次受约束 latent action
        ↓
生成短离散 checkpoint
        ↓
causal reset
        ↓
hard COT
```

形式化为：

$$
a_t
=
\mathbf 1[widehat U_t>\tau_u]
\mathbf 1[widehat V_t>\tau_v],
$$

$$
z_t
=
\begin{cases}
h_t+\eta_t^*(s_t-h_t), & a_t=1,\\
h_t, & a_t=0.
\end{cases}
$$

### E2：最小联合消融

| 方法 | Probe | TTR | Reset |
|---|---:|---:|---:|
| Initial Transition | - | - | - |
| Fixed TALR | - | - | - |
| Probe only | ✓ | - | - |
| TTR only | - | ✓ | - |
| Probe + TTR | ✓ | ✓ | - |
| Full joint | ✓ | ✓ | ✓ |

禁止在联合实验中重新调 Probe threshold、TTR radius 和 checkpoint horizon。必须使用各模块阶段已经锁定的配置。

### E3：联合成功门槛

- 相对 Fixed TALR，两个内部验证数据集 pooled net 为正；
- 至少 3/4 核心 cells 不低于 Fixed TALR 超过 1 pp；
- damage rate 降低；
- 外部模型上方向一致；
- 延迟与额外 forward 成本完整报告。

---

## 11. Phase F：跨模型与完整主表

### F1：进入条件

只有联合方法在内部验证通过后才扩展。若 Probe 或 TTR 单模块未通过，则只验证通过的 Handoff 或保留固定 TALR，不强行组成完整方法。

### F2：主表建议

模型：

- R1-Onevision-7B-RL；
- Vision-R1-7B；
- OpenVLThinker-7B。

数据集：

- VStar；
- MMVP；
- RealWorldQA；
- VisuLogic；
- VMCBench；
- POPE-Adversarial；
- MMK12-Physics 或锁定子科。

方法：

- COT；
- Full LEAD；
- Initial Transition；
- Fixed TALR；
- 锁定后的新方法。

先使用 compact matrix，不恢复所有历史 format/quota 变体。

---

## 12. 统一指标与统计

### 12.1 任务指标

- accuracy / score；
- MMVP pair accuracy；
- POPE precision、recall、F1、yes ratio；
- failed extraction、runtime error；
- output length、long≥256、maxed1024。

### 12.2 干预指标

- proposal、accepted、rejected；
- coverage；
- fixed、damaged、net；
- actionability；
- actual $\eta$ distribution；
- embedding L2、one-step JS；
- visual specificity；
- checkpoint length；
- reset count；
- latency 与额外 forward。

### 12.3 统计要求

- paired bootstrap 95% CI；
- McNemar exact test；
- AUROC 同时报告 AUPRC；
- 样本是统计单位，不能把同一样本多个 event 当独立样本；
- 多检查点结果报告 clustered bootstrap 或按样本聚合；
- 小规模64样本结果只用于筛选，不写成最终显著结论。

---

## 13. 实验注册与文件规范

每个实验必须先有一份 manifest：

```json
{
  "experiment_id": "C2_TTRP_Q25_VSTAR64",
  "research_question": "RQ3",
  "hypothesis": "H3",
  "model": "R1-Onevision-7B-RL",
  "dataset": "vstar64_locked",
  "baseline": "fixed_lambda095",
  "changed_variable": "js_radius_quantile",
  "primary_metric": "damaged_reduction",
  "success_gate": "damage_reduction>=25%; fix_retention>=60%",
  "stop_rule": "pooled_net<0",
  "evaluator": "corrected_last_answer",
  "seed": 42
}
```

输出必须包含：

```text
manifest.json
config.json
results.jsonl
eval_report.json
token_entropy.jsonl
run.log
summary.json/md
pairwise_fixed_damaged.json
```

任何配置变化必须产生新 run directory，不覆盖旧结果。

---

## 14. 队列与资源约束

### P0

1. 完成 $\lambda=0.80$ source-strength；
2. 生成统一 causal label table；
3. 做 visual specificity 分析；
4. 决定是否允许进入新 Probe。

### P1

1. TTR-P 两数据集小矩阵；
2. Corrected handoff 的逐样本 reset/no-reset 分析；
3. Probe 特征可行性离线分析。

### P2

1. 仅训练通过标签门槛的新 Probe；
2. 仅补通过小矩阵的 full runs；
3. 跨模型 handoff 验证。

### 资源规则

- 小规模筛选优先64样本；
- 同一 GPU 最多按真实显存容纳模型进程，不以进程数为目标；
- 不为了防止 GPU 空闲而自动扩大未通过门槛的实验；
- 高成本 full counterfactual rollout 必须有 Phase A/B 的明确证据支持；
- 运行中不得根据 partial accuracy 修改下一分支。

---

## 15. 变更控制

以下情况必须先更新本总纲：

- 新增一个方法模块；
- 修改主要研究问题；
- 更换 Probe 标签定义；
- 更换 TTR 距离；
- 将开发数据改为验证数据；
- 根据验证结果回调已锁参数；
- 将负结果路线重新升级为主线。

每次修订记录：

```text
版本号
变更日期
变更原因
新增证据
被删除的假设
受影响实验
```

---

## 16. 最终决策树

```text
真实图片是否具有 source-specific positive utility？
    ├─ 否：停止 causal multimodal Probe 主线
    │      └─ 独立推进 Handoff；TTR 只做风险控制审计
    └─ 是：训练 causally supervised Probe
            ↓
        外部 utility AUROC 和 policy net 是否过门槛？
            ├─ 否：Probe 作为负结果，不接入方法
            └─ 是：进入 Probe-only 在线验证

TTR-P 是否减少 damage 并保留 fixed？
    ├─ 否：停止在线 TTR，保留固定 λ 或 Initial Transition
    └─ 是：锁定半径与 action set

Checkpoint/reset 是否保留有效 influence？
    ├─ 否：不加入 reset
    └─ 是：锁定 checkpoint horizon

至少两个模块通过？
    ├─ 否：分别报告机制结果，不强行联合
    └─ 是：运行最小联合消融
```

---

## 17. 总结

本总纲的目的不是保证形成一个复杂的新方法，而是保证每一步都回答一个清楚、自然且可证伪的问题。

后续实验必须遵守以下顺序：

1. **先证明视觉来源具有正确性相关的因果价值；**
2. **再判断这种价值能否由线上特征预测；**
3. **独立验证 TTR 是否能控制 damage；**
4. **验证短离散 prefix 与 reset 是否能安全承载影响；**
5. **只有通过门槛的模块才允许联合。**

这能避免把“多模态特征、Probe、TTR、Handoff”堆成一个难以归因的系统，也能确保即使某条方法路线失败，机制发现本身仍然可形成可靠研究结论。


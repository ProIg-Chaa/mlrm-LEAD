# TALR 近期科研日志与后续研究上下文

> 文档状态：阶段性科研日志，作为后续研究的主上下文入口  
> 日期：2026-07-23  
> 项目：`mlrm-LEAD`  
> 当前代码基线：commit `198e5be`  
> 当前主方法：`W8K2-T1.25-lambda0.95-NoGuard`  
> 当前研究目标：不再赶投 AAAI，转为充分打磨后投稿 ICLR  
> 统一实验口径：origin prompt、greedy、seed 42、max new tokens 1024  

## 摘要

本日志汇总近期 TALR 的方法演化、证据来源、正式结果、负结果、当前运行任务和下一阶段研究计划。项目最初从 pure-soft 格式退化与 entropy routing 审计出发，随后发现 full LEAD 的可测收益在若干任务中主要集中于 early initializer 和 soft-to-discrete transition，并以 prefix-mediated Early Trajectory Commitment 解释其作用。当前 TALR 在继承 early transition 的基础上，将后续 latent intervention 限制为 handoff 后短窗口内最多两次、强度为 `0.95 soft + 0.05 hard` 的一步 refinement，之后永久回到 hard COT。

最新统一复评显示：在 R1-Onevision-7B-RL 四个核心数据集上，TALR 相对 Full LEAD 为三胜一平，简单平均提高 1.36 个百分点；在八个扩展数据集上，TALR 相对 Full LEAD 简单平均提高 1.10 个百分点，但相对 COT 低 1.19 个百分点。Vision-R1 和 OpenVLThinker 的现有 VStar/MMVP 结果中 refinement 几乎不触发，说明固定 entropy threshold 缺少跨模型适应性。

项目已经由“继续局部调参提高 TALR”转向更有研究纵深的问题：**如何直接估计一次 latent intervention 的反事实效用，并据此决定是否干预、干预多强，以及发生 drift 后何时真正清除 soft history。** 当前最高优先级的新实验不是立即训练 router，而是先构造 matched counterfactual Latent Intervention Atlas，计算 oracle upper bound，判断 utility-aware 路线是否具有足够收益空间。

**一句话上下文：** TALR 是当前可靠的无训练启发式基线；ICLR 阶段拟将其升级为“utility-aware selection + trajectory trust region + sparse causal reset”，但任何新模块都必须先通过可证伪的反事实实验门槛。

## 目录

PDF 转换脚本将自动生成目录与书签，本区块无需手工维护。

---

## 1. 当前结论

### 1.1 已经相对确定的事实

1. **Predictive uncertainty 不等于 correctness。** Pure-soft 可以在低熵、高置信状态下持续犯错，confidence-correctness 关系具有明显任务依赖性。
2. **Predictive uncertainty 也不等于 intervention utility。** Full LEAD 的后续 routing 在不同数据集上的 fixed/damaged 差异很大，高 entropy 不能稳定表示“此处 soft 会改善最终答案”。
3. **Early path 在若干任务中具有较高影响。** Initial Transition 在 VStar/MMVP 上接近 Full LEAD，在 VisuLogic 上曾高于 Full LEAD；但 RealWorldQA 构成明确反例。
4. **Early effect 更符合 prefix mediation，而非必须长期保留的 soft-KV basin。** Prefix-2 cache rebuild 在 VStar/MMVP 上基本保留准确率，严格 same-prefix probe 未发现明显的长期 route-specific logit 差异。
5. **更多 soft 不会单调提高正确性。** Pure-soft、较大 quota 和较宽 refinement 暴露都可能增加 damaged、重复、长输出与答案漂移。
6. **Format cooldown 能修 pure-soft 退化，但不是稳定 reasoning improvement。** 它在 VStar pure-soft 上非常有效，却不能跨任务稳定超过 COT，也不能清除已经写入 KV cache 的 soft history。
7. **当前 TALR 在 R1-RL 主模型上优于 Full LEAD。** 四核心平均高 1.36 pp；八扩展集平均高 1.10 pp。
8. **当前 TALR 尚不能证明跨模型完整泛化。** Vision-R1 和 OpenVLThinker 的 refinement 候选过少，现有结果很大程度退化为 Initial Transition。

### 1.2 当前不能宣称的内容

- 不能宣称 entropy 无用，只能说它缺少跨任务、跨阶段的稳定 utility calibration。
- 不能宣称所有多模态推理都在 step 0 锁定。
- 不能把原 LEAD 的 early initializer、newline mix 或 `</think>` bridge包装为 TALR 新算子。
- 不能宣称 `lambda=0.95` 是理论最优或跨模型普适常数。
- 不能宣称 Format Guard 是当前 NoGuard TALR 的有效组成部分。
- 不能用错题子集调参结果替代全量 held-out 结果。
- 不能把 trajectory divergence 当作 accuracy improvement。
- 不能把 selected branch replay 当作总体平均因果效应。

---

## 2. 研究路线演化

### 2.1 第一阶段：Format Stability

研究起点是 pure-soft 输出退化：

- 格式边界不稳定；
- 输出显著变长；
- 重复增加；
- 最终答案发生漂移或反转；
- failed extraction 与 maxed1024 增加。

主要方法：

- `format_cooldown2`；
- diffuse collapse；
- late64 repeat gate；
- answer-zone discrete；
- format + quota；
- lead soft veto。

关键结果：

- VStar pure-soft：58.64%，平均长度 237.3，long 33，maxed 18；
- pure-soft + format2：74.35%，平均长度 131.1，long 9，maxed 4；
- pure-soft guard：74.87%，平均长度 127.3，long 8，maxed 3。

阶段结论：

> Format2 是 pure-soft 的有效稳定器，但不是 LEAD 主收益机制，也不能跨数据集稳定提升 reasoning accuracy。

遗留问题：

> Cooldown 只阻止新的 soft 输入，不会移除此前 soft-derived KV，因此在已经漂移的长轨迹上可能干预过晚。

### 2.2 第二阶段：Early Transition 与 ETC

Trigger audit 显示，LEAD 在部分数据集上平均触发很少，很多样本只有开头 transition，没有后续 `to-soft`。由此开展：

- Initial Soft Only；
- Initial Transition Only；
- no-to-normal；
- no-linebreak；
- no-anchor；
- timing step0/4/16/32；
- same-token replay；
- prefix cache rebuild；
- hard boundary controls；
- anchor replacement；
- visual anchor。

核心观察：

- VStar Initial Transition 接近 Full LEAD；
- MMVP Initial Transition 与 Full LEAD接近；
- VisuLogic Initial Transition 曾高于 Full LEAD；
- RealWorldQA 不支持“越早越好”的普遍定律；
- no-to-normal 在 VStar 退回 Initial Soft Only附近；
- prefix2 cache rebuild 基本保留 VStar/MMVP 准确率；
- actual visual anchor 与 static anchor 接近，没有证明真实视觉内容贡献。

最终机制定位：

> Early latent intervention 的可测作用往往先改变最初的离散 reasoning prefix，后续生成再沿该 prefix 展开。该效应具有任务依赖性，不能表述为所有任务的 step-0 决定论。

### 2.3 第三阶段：从旧 TALR 到当前 W8K2

曾探索的版本包括：

- transition + quota05 + format guard；
-旧 TALR；
- True TALR；
- initial transition with refinement；
- W8K2、W16K4、W32K8；
- dynamic transition；
- guard on/off；
- refinement `lambda` sweep。

旧 quota 方法的问题：

- quota catch-up 会为达到比例而在后期补 soft；
- 可能在答案方向已形成后重新扰动；
- VisuLogic 等任务出现明显 damaged；
- Format 虽有触发，但不稳定转化为净正确。

当前 W8K2 的变化：

- 不再全程追赶 soft quota；
- 只在 handoff 后短窗口内考虑 refinement；
- 每个样本最多两次；
- 每次只持续一个 embedding step；
- 完成后立即返回 hard；
- 窗口关闭后不可逆 hard-only。

### 2.4 第四阶段：`lambda=0.95` 轻度收缩

Pure-soft refinement：

$$
z_t=s_t.
$$

当前 contracted refinement：

$$
z_t
=0.95s_t+0.05h_t
=h_t+0.95(s_t-h_t).
$$

它只在 accepted refinement event 生效，不改变：

- early initializer；
- transition timing；
- candidate window；
- soft cap；
-最终离散输出 token；
-后续 hard lock。

科学含义：

> 保留大部分 distributional embedding，同时沿模型自身 top-1 realization 轻微收缩。

边界：

- R1-RL 核心集上有正证据；
- 扩展集并非单调优于 `lambda=1.00`；
- Vision/OpenVL 尚无足够 active refinement 支撑跨模型结论。

### 2.5 第五阶段：由 AAAI 保底转向 ICLR 深化

时间目标放宽后，研究策略发生调整：

- 不再把“再调高几个百分点”作为唯一目标；
- 不再围绕 `W/K/T/lambda` 做无限局部网格；
- 保留 TALR 作为强启发式 baseline；
- 将研究问题升级为 causal intervention allocation；
- 允许重新启用曾被放弃的 Format 与 visual signal，但必须修正旧机制缺陷。

---

## 3. 当前 TALR 的准确方法定义

### 3.1 固定配置

```yaml
method: lead
early_policy: initial_transition_with_refinement
alpha: 0.4
max_switch_count: 5
window_size: 128
refinement_window: 8
refinement_soft_cap: 2
refinement_entropy_threshold: 1.25
refinement_soft_mix_lambda: 0.95
guard_candidate_only: true
answer_zone_lock: false
format_guard: false
late_veto: false
quota_catch_up: false
cot_prompt_mode: orign
do_sample: false
temperature: 0.6
top_p: 0.95
top_k: 20
seed: 42
max_new_tokens: 1024
```

### 3.2 四阶段

#### 阶段一：Early Soft Initialization

保留原 LEAD early path，使连续分布信息在生成开头进入模型。该组件为继承组件，不是 TALR 新发明。

#### 阶段二：Natural Soft-to-Discrete Handoff

保留原 LEAD 的 transition 条件与 bridge，完成从 early soft state 到 normal decoding 的交接。该组件被本项目审计和重新定位，但归属仍是原 LEAD。

#### 阶段三：Windowed Budgeted Contracted Refinement

handoff 后仅当：

$$
0<t-\tau\leq8,\qquad n_t<2,
$$

且候选 entropy 条件满足时，使用：

$$
z_t=0.95s_t+0.05h_t.
$$

随后立即返回 hard decoding。

#### 阶段四：Irreversible Discrete Continuation

窗口关闭或预算耗尽后，不再允许 latent re-entry，不执行 quota catch-up，始终使用 hard COT。

### 3.3 当前真正属于 TALR 的创新

- 将 entropy 从全程状态控制器降级为局部 candidate signal；
- 用 early window 限制 intervention timing；
- 用 soft cap 限制 intervention count；
- 用 contracted refinement 限制 intervention magnitude；
- 用 irreversible return 替代 recurrent mode switching。

---

## 4. 最新统一主结果

### 4.1 R1-Onevision-7B-RL 四核心

以下数字以 2026-07-22 统一 corrected/specialized evaluator 主表为准。

| 数据集 | COT | Full LEAD | TALR | TALR-COT | TALR-LEAD |
|---|---:|---:|---:|---:|---:|
| VStar | 68.06% | 72.25% | **74.35%** | +6.29 pp | +2.10 pp |
| MMVP sample | 68.00% | 70.33% | **71.33%** | +3.33 pp | +1.00 pp |
| MMVP pair | 39.33% | 42.00% | **44.00%** | +4.67 pp | +2.00 pp |
| RealWorldQA fixed200 | **66.00%** | 63.00% | 63.00% | -3.00 pp | 0.00 pp |
| VisuLogic300 | 21.33% | 27.00% | **30.33%** | +9.00 pp | +3.33 pp |
| 四集 sample 平均 | 55.85% | 58.40% | **59.75%** | +3.91 pp | +1.36 pp |

结论：

- 对 Full LEAD：3胜1平；
- 对 COT：3胜1负；
- RealWorldQA 是核心反例；
-不能宣称普遍超过 COT。

### 4.2 R1-RL 八个扩展数据集

| 数据集 | COT | Full LEAD | TALR | TALR-COT | TALR-LEAD |
|---|---:|---:|---:|---:|---:|
| VMCBench-dev | **74.00%** | 73.50% | 73.30% | -0.70 pp | -0.20 pp |
| MMK12-Math | 48.20% | 44.80% | **50.40%** | +2.20 pp | +5.60 pp |
| MMK12-Physics | **41.40%** | 34.60% | 38.60% | -2.80 pp | +4.00 pp |
| MMK12-Chemistry | **45.80%** | 41.20% | 42.20% | -3.60 pp | +1.00 pp |
| MMK12-Biology | **46.00%** | 41.00% | 42.80% | -3.20 pp | +1.80 pp |
| POPE-Adversarial | 82.60% | **84.20%** | 82.80% | +0.20 pp | -1.40 pp |
| POPE-Popular | **84.00%** | **84.00%** | 82.40% | -1.60 pp | -1.60 pp |
| POPE-Random | 84.20% | **84.60%** | 84.20% | 0.00 pp | -0.40 pp |
| 八集平均 | **63.28%** | 60.99% | 62.09% | -1.19 pp | +1.10 pp |

结论：

- 对 Full LEAD：4胜4负，平均 +1.10 pp；
- 对 COT：2胜1平5负，平均 -1.19 pp；
- 主要优势来自 MMK12；
- POPE 没有稳定收益；
-扩展集证明 TALR 不是统一超过 hard COT 的通用解。

### 4.3 Vision-R1-7B 已有核心结果

| 数据集 | COT | Full LEAD | TALR |
|---|---:|---:|---:|
| VStar | 81.15% | **82.20%** | 81.68% |
| MMVP sample | **73.67%** | 73.33% | **73.67%** |
| MMVP pair | **48.67%** | 48.00% | **48.67%** |

两集 sample 平均：

- COT 77.41%；
- Full LEAD 77.77%；
- TALR 77.68%。

关键限制：

> Refinement 实际触发接近 0，因此结果主要反映 Initial Transition，不足以验证 `lambda=0.95`。

### 4.4 OpenVLThinker-7B 已有核心结果

| 数据集 | COT | Full LEAD | TALR |
|---|---:|---:|---:|
| VStar | 80.63% | 80.63% | **81.15%** |
| MMVP sample | 50.00% | **51.33%** | **51.33%** |
| MMVP pair | 8.67% | 11.33% | **12.00%** |

限制：

- MMVP 存在 23 条 failed extraction；
- refinement 几乎不触发；
-结果仅作为初步外部验证。

### 4.5 结果口径警告

旧报告曾出现：

- VStar TALR 74.87%；
- RealWorldQA TALR 65.00%；
- VisuLogic TALR 29.00%。

这些是早期快照或旧复评口径。当前科研上下文统一采用 2026-07-22 主表：

- VStar 74.35%；
- RealWorldQA 63.00%；
- VisuLogic 30.33%。

以后写作、汇总和制表不得混用这两套数字。

---

## 5. 近期负结果与它们的研究价值

### 5.1 Format 不能仅为了叙事被加入

当前 W8K2 refinement 平均次数较少，Format2 实际 suppress 的 candidate 很少，很多 guard 消融接近空操作。增加 soft 暴露后，Format 虽可能更多触发，但并未稳定产生全量正收益。

研究价值：

> 旧 cooldown 的机制缺陷明确，为真正的 cache reset 提供了动机。

### 5.2 Dynamic transition 没有形成稳定普遍优势

不同任务偏好的 transition timing 不同：

- VStar、MMVP、VisuLogic 更支持 early intervention；
- RealWorldQA 曾在更晚 timing 上更好。

研究价值：

> Timing 是 task-dependent treatment effect，不适合用一条普遍 step0 定律概括。

### 5.3 Anchor identity 具有任务依赖性

- VStar 有时 direct hard 或 newline 更强；
- MMVP 有时 `</think>` bridge 更强；
- actual visual anchor 没有优于 static perturbation。

研究价值：

> 特殊 token 与直接视觉注入不是当前最有希望的主创新；应转向学习 action utility。

### 5.4 `lambda=0.95` 在扩展集不稳定

相对 `lambda=1.00`：

- VMCBench、MMK12-Math 略升；
- Biology 持平；
- Physics、Chemistry、POPE 下降。

研究价值：

> 固定混合比例缺乏跨任务几何一致性，为 trajectory trust region 提供直接动机。

---

## 6. 当前正在运行的实验

### 6.1 NewGpu3

服务器：

```text
Host: NewGpu3
GPU: 1 x A800 80GB
Code: /root/gushuo/proj/mlrm-LEAD
Commit: 198e5be
Output:
/root/autodl-tmp/gushuo/outputs/experiments/20260723_cross_model_expansion
```

tmux：

```text
cross_expand_vision
cross_expand_openvl
```

矩阵：

- Vision-R1-7B；
- OpenVLThinker-7B；
- POPE-Adversarial / Popular / Random；
- COT / Full LEAD / TALR。

截至 2026-07-23 晚间：

- Vision 已完成 POPE-Adversarial 三方法；
- Vision 已完成 POPE-Popular 三方法；
- Vision 正在 POPE-Random COT；
- OpenVL 已完成 POPE-Adversarial 三方法；
- OpenVL 正在 POPE-Popular COT；
- GPU约使用 66.8GB，利用率100%。

### 6.2 gpu13

访问路径：

```text
local -> super-mu01 -> gpu13
```

Slurm：

```text
job 26805
node gpu13
2 x A800 80GB
```

代码部署：

```text
/dev/shm/wangzixu_20260723/mlrm-lead-198e5be
```

模型：

```text
/dev/shm/wangzixu_models/Vision-R1-7B
/dev/shm/wangzixu_models/OpenVLThinker-7B
```

输出：

```text
/share/home/wangzixu/liudinghao/gushuo/output/experiments/
20260723_cross_model_expansion_gpu13
```

四个 tmux worker：

```text
xexp_v0
xexp_v1
xexp_o0
xexp_o1
```

分片：

| GPU | Worker | 模型 | 数据集 |
|---:|---|---|---|
| 0 | `xexp_v0` | Vision-R1 | MMK12 Math / Physics / Chemistry |
| 0 | `xexp_v1` | Vision-R1 | MMK12 Biology / VMCBench |
| 1 | `xexp_o0` | OpenVL | MMK12 Math / Physics / Chemistry |
| 1 | `xexp_o1` | OpenVL | MMK12 Biology / VMCBench |

每个数据集运行：

- COT；
- Full LEAD；
- TALR。

截至日志写入时：

- 四个 worker 均在首个 COT full run；
- 两卡利用率约99%至100%；
- 显存约79.2GB/80GB，每卡只剩约1.8GB；
-尚未记录 OOM，但属于高风险显存状态；
- 若出现 OOM，应减少为每卡单进程续跑，不要降低模型精度或使用 CPU offload污染口径。

### 6.3 当前扩展矩阵的目标

该矩阵回答：

- TALR 相对 Full LEAD 的平均优势能否跨模型延续；
- 固定 `T=1.25` 在外部模型上是否仍退化为空触发；
- MMK12 与 POPE 的任务差异是否具有模型一致性；
- Vision/OpenVL 的 failed extraction 和输出稳定性如何变化。

它不回答：

- utility probe 是否有效；
- trust region 是否优于 `lambda=0.95`；
- causal reset 是否有效。

---

## 7. ICLR 阶段的新主问题

### 7.1 问题升级

旧问题：

> 如何设置 entropy threshold、window、cap 和 lambda，使 TALR 更高？

新问题：

> 在某个生成位置，soft intervention 相对 hard continuation 的反事实净效用是多少；它是否可由干预前状态预测；若值得干预，应允许多大的状态偏移；若发生漂移，如何真正回到不含 soft history 的离散状态？

统一目标：

$$
\max_\pi
\mathbb E[
R(Y^\pi)
-\gamma C(\pi)
-\eta D(Y^\pi)
].
$$

### 7.2 三个拟研究模块

#### A. Utility-Aware Selection

通过 matched counterfactual replay 得到：

$$
\Delta_{i,t}
=R(Y_{i,t}^{L})-R(Y_{i,t}^{H})
-\gamma\Delta C
-\eta\Delta D.
$$

训练小型：

$$
\hat\Delta_{i,t}=q_\phi(f_{i,t})
$$

直接预测 intervention utility，而不是预测难度或当前正确性。

#### B. Trajectory Trust Region

用归一化 soft-hard distance 控制偏移：

$$
\bar d_t
=\frac{\|s_t-h_t\|_2}
{\|h_t\|_2+\varepsilon},
$$

$$
\alpha_t
=\min\left(
1,
\frac{\epsilon_m}{\bar d_t+\varepsilon}
\right),
$$

$$
z_t=h_t+\alpha_t(s_t-h_t).
$$

目的：

- 不再共享不可比的固定 `lambda`；
- soft-hard 很近时允许充分 intervention；
-差异很大时自动收缩。

#### C. Discrete Checkpoint and Causal Reset

风险出现时：

```text
mixed/soft KV history
    -> 保持可见离散 prefix
    -> 从 prompt + discrete prefix 重新 forward
    -> 重建 pure-discrete KV
    -> 永久 hard-only
```

它与 cooldown2 的根本区别是：

> Reset 会移除已写入 cache 的 soft treatment history；cooldown 只停止新增 soft。

---

## 8. Matched Counterfactual Atlas V0

### 8.1 推荐定义

在 hard COT trajectory 上选择事件 $t$：

对照：

$$
z_t^H=h_t.
$$

处理：

$$
z_t^L=0.95s_t+0.05h_t.
$$

从 $t+1$ 起两分支都 hard greedy continuation。

标签：

$$
\Delta_{i,t}^{acc}
=
\mathbf1[Y_{i,t}^{L}\text{ correct}]
-
\mathbf1[Y_i^H\text{ correct}].
$$

### 8.2 成本优化

每个样本只跑一次完整 hard baseline。多个 intervention event 共享：

$$
Y_{i,t}^{H}=Y_i^H.
$$

因此8个事件的成本是：

$$
1+8=9
$$

条 trajectory，而不是16条。

### 8.3 推荐规模

| 数据集 | stratified sample |
|---|---:|
| VStar | 64 |
| MMVP | 64 |
| RealWorldQA | 64 |
| VisuLogic | 64 |
| 总计 | 256 |

每样本位置：

- handoff 后 offset 1、2、4、8、16、32；
- entropy top-1 candidate；
-一个分层随机位置。

总事件：

$$
256\times8=2048.
$$

总生成上界：

$$
256+2048=2304.
$$

预计：

- 单卡8至16 GPU小时；
-两卡4至8小时；
-加上评测和审计按半天估算。

### 8.4 Atlas V0 决策门槛

| 结果 | 下一步 |
|---|---|
| Oracle 明显高于 TALR/entropy router | 训练 utility probe |
| Oracle 收益只在最早位置 | 收缩为 early utility selection |
| Oracle 模型内有效但跨模型异质 | 研究模型内归一化与轻量适配 |
| Oracle 自身收益很小 | 停止 probe 主线，转机制或 reset |

---

## 9. 后续实验优先级

### P0：完成当前跨模型扩展

-等待 NewGpu3 POPE；
-等待 gpu13 MMK12/VMC；
-统一 corrected/specialized evaluator；
-统计 refinement candidate/active 次数；
-失败与 runtime error 单列；
-先判断固定 threshold 的跨模型失效程度。

### P0：实现 Atlas V0

-复用现有 replay/assertion 设施；
-hard-context one-step intervention；
-`lambda=0.95` treatment；
-只保存事件级 trace，避免全量 top-k 文件；
-先计算 oracle gap。

### P1：Utility probe

仅在 oracle gap 足够时进行：

- logistic regression；
- two-layer MLP；
- GBDT；
- leave-one-dataset-out；
- leave-one-model-out；
-按样本划分，禁止同一样本事件泄漏。

### P2：Trust region

-固定 probe 比较 `lambda` 与 normalized radius；
-报告实际位移和 $\alpha_t$ 分布；
-跨模型做 non-inferiority；
-禁止按每个 test 数据集单独选 radius。

### P3：Causal reset

-先跑 pure-soft/TALR damaged、maxed、repeat 高风险子集；
- no guard vs cooldown2 vs full rebuild；
-评估 rescue rate、误伤率和 latency；
-通过后才考虑进入主方法。

### P4：正式大矩阵

方法冻结后比较：

- COT；
- Full LEAD；
- Initial Transition；
-当前 TALR；
-utility-only；
-utility + trust region；
-utility + trust region + reset。

---

## 10. 研究与实现纪律

### 10.1 结果口径

- VStar/VisuLogic 使用 corrected last-answer evaluator；
- MMVP 使用 specialized sample/pair evaluator；
- RealWorldQA 使用 fixed200 专用 evaluator；
- POPE 报 accuracy、precision、recall、F1、yes ratio；
- MMK12/VMC 使用 deterministic corrected MCQ；
- runtime error 不能混为普通错误；
- failed extraction 单列；
-旧错误 RealWorldQA 数据永不纳入主结论。

### 10.2 数据泄漏

-不能在 benchmark test 上构造 utility gold 后训练并在同一 test 报告；
-事件 train/test split 必须按 sample 分组；
-优先使用 train/dev、leave-dataset-out 或 leave-model-out；
-错题集只用于筛选机制，不用于最终锁参。

### 10.3 反事实证据

可以使用因果措辞的最低条件：

-同一 checkpoint；
-同一 prompt/image；
-目标事件前 prefix一致；
-同一 decoding；
-同一 continuation policy；
-只改变目标 routed embedding；
-replay assertion通过。

### 10.4 资源纪律

- GPU实验前优先把模型复制到 `/dev/shm`；
-四进程高显存运行要监控 OOM；
-不要在登录节点运行 `nvidia-smi` 来判断计算卡；
-集群禁用节点不得登录或使用；
-不要扫描、抢占或停止他人任务；
-任何结果目录必须保留 config、results、eval与日志。

---

## 11. 接手本项目时的快速检查

### 11.1 首先阅读

本地：

```text
报告/15_TALR当前实验结果/COT_LEAD_TALR主方法对照表_20260722.md
报告/16_TALR_Lambda095_Refinement/
报告/17_AAAI_Introduction/
报告/18_实验思绪阶段性整理/
```

远程：

```text
result/7-23/TALR近期科研日志与后续研究上下文_20260723.md
log.md
```

### 11.2 立即检查

1. NewGpu3 的 `cross_expand_vision/openvl`；
2. gpu13 的 `xexp_v0/v1/o0/o1`；
3. 是否出现 OOM/runtime error；
4. 每个完整 run 是否存在 `eval_report.json`；
5. 外部模型实际 refinement 次数；
6. 当前代码是否仍是 `198e5be`；
7. 汇总是否使用 7 月 22 日统一主表口径。

### 11.3 不要立即做

-不要继续大规模搜索固定 `T`；
-不要重跑已有且配置一致的 R1-RL 主表；
-不要把 Format2重新塞入 NoGuard TALR；
-不要基于扩展 test 单项结果修改 `lambda`；
-不要先训练复杂 router；
-不要在没有 oracle gap 前扩大 Atlas 到全部模型和数据集。

---

## 12. 最终研究定位

当前 TALR 最准确的定位是：

> 一个由 entropy-routing 机制审计导出的、时间受限、预算受限、强度受限的 training-free latent intervention baseline。它在 R1-RL 上平均优于 Full LEAD，但仍具有任务依赖性和跨模型触发校准问题。

下一阶段最有潜力的统一定位是：

> 将 multimodal latent reasoning 从 uncertainty-triggered mode switching 重构为 causal-utility-aware intervention allocation。

统一控制逻辑：

$$
\boxed{
\text{Utility decides whether}
\quad
\text{Trust region decides how far}
\quad
\text{Reset decides when to erase history}
}
$$

这条路线保留了此前所有有价值的发现：

- confidence-correctness mismatch；
- early trajectory commitment；
- prefix externalization；
-少量后续 refinement 的 residual utility；
-pure-soft 的格式与长度退化；
-固定 threshold 和 fixed lambda 的跨模型问题。

同时，它也吸收了此前被放弃路线的真正价值：

- Format 不再是局部 cooldown，而是 cache-level reset；
- Visual signal 不再直接注入，而是作为 utility feature；
- Dynamic routing 不再由 entropy直接决定，而是由反事实效用监督。

## 附录 A. 当前关键路径

```text
NewGpu3 repo:
/root/gushuo/proj/mlrm-LEAD

NewGpu3 expansion:
/root/autodl-tmp/gushuo/outputs/experiments/
20260723_cross_model_expansion

gpu13 snapshot:
/dev/shm/wangzixu_20260723/mlrm-lead-198e5be

gpu13 expansion:
/share/home/wangzixu/liudinghao/gushuo/output/experiments/
20260723_cross_model_expansion_gpu13
```

## 附录 B. 核心内部报告索引

| 报告 | 作用 |
|---|---|
| `Format与Transition早中期探索实验全景归档_20260717.md` | 早中期实验全景 |
| `TALR方法定稿_结构冻结版_20260717.md` | TALR结构与归属边界 |
| `TALR诊断优化实验前分析与决策基线_20260717.md` | 诊断框架 |
| `TALR三版本方法与实验结果对比_20260718.md` | 版本演化 |
| `TALR_Motivation_Insight与论文完整叙事_20260718.md` | 论文故事 |
| `TALR当前实验结果与四阶段机制简报_20260720.md` | 四阶段机制 |
| `TALR_Lambda095_Refinement解释与锁参依据_20260721.md` | 轻度收缩 |
| `COT_LEAD_TALR主方法对照表_20260722.md` | 最新统一结果 |
| `从TALR到因果效用感知隐空间干预_20260723.md` | ICLR研究升级方案 |

## 附录 C. 日志状态标签

**已完成：**

- R1-RL 四核心 COT/LEAD/TALR；
- R1-RL 八扩展集 COT/LEAD/TALR；
-核心 Format/Transition/timing/replay/cache rebuild 分析；
-当前 TALR `lambda=0.95` 锁定；
-ICLR研究问题升级。

**运行中：**

- Vision-R1 扩展数据集；
-OpenVLThinker 扩展数据集；
-NewGpu3 POPE；
-gpu13 MMK12/VMC 四进程矩阵。

**待启动：**

- Atlas V0；
-oracle gap；
-utility probe；
-trajectory trust region；
-causal reset。

**已降级或停止：**

-特殊 anchor 大规模搜索；
-直接 pooled visual anchor；
-全程 quota catch-up；
-Format2 作为当前主方法；
-固定 threshold 的无限微调；
-“所有任务 step0 锁定”的强叙事。

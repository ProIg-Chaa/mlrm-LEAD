# Early Trajectory Commitment 与 TALR：机制主线整理

## 目录

1. 研究背景
2. 当前方法的问题
3. 反常识现象与动机证据
4. Early Trajectory Commitment
5. Latent Trajectory Initialization
6. TALR 方法
7. 已有实验结果
8. 证据边界
9. AAAI Introduction 叙事
10. 当前贡献

## 1. 研究背景

多模态大模型通常在离散 token 空间中执行 Chain-of-Thought（COT）推理。Soft reasoning 则将概率分布对应的连续 embedding 作为下一步输入，希望在隐空间中保留多个候选方向，减少过早离散化造成的信息损失。LEAD 进一步使用 entropy 在 soft 与 normal 路由之间动态切换，其隐含假设是：不确定性高时 soft 推理更有价值，中后段的 entropy 变化可以指导何时进入或退出隐空间。

我们的研究起初沿着这一 entropy-routing 假设展开，但统一复现实验表明，更多 soft、更多动态切换或更强视觉锚定都不自动带来更高正确率。真正稳定的信号集中在生成最早期：第 0 步的一次 latent 初始化及随后回到 normal COT 的 transition，已经足以改变大量样本的最终答案。

## 2. 当前方法的问题

标准 LEAD 存在三个问题。

第一，动态触发极少。VStar 上平均约 1.7 次切换/样本，并且第 0 步固定以 soft 路由开始。因此，“动态 entropy routing”与“固定早期初始化”在原方法中被混合，无法判断收益来自哪一部分。

第二，entropy 不是可靠的 correctness proxy。错误轨迹可以表现为低熵、高置信和长推理；模型在错误方向锁定后，中后段只是自洽地扩展既有结论。此时等待 entropy spike 再干预已经太晚。

第三，soft 路由具有明显退化风险。Pure-soft 容易出现格式边界破坏、答案漂移、重复和超长输出；即使 format guard 高频触发，它主要修复生成稳定性，也不能保证提升视觉推理能力。

## 3. 反常识现象与动机证据

### 3.1 高置信不等于高正确

Pure-soft 的 confidence–correctness 分析显示，多个数据集上的严格 AUC 约为 0.369–0.485，未形成“置信度越高、正确率越高”的可靠单调关系。最高置信分组的正确率甚至可能比总体低 15.33–48.20 个百分点。该结果不意味着 confidence 总是与正确性反向，只说明它不能被直接当作路由 oracle。

### 3.2 More-soft 不是答案

Quota 系列表明，少量后续 soft 有时有收益，但结果明显依赖数据集：VStar 的较优配置约为 quota 0.05，RealWorldQA 更偏向 quota 0.03，quota 0.08 已出现不稳定。增加 latent step ratio 并不会单调提升 accuracy。

### 3.3 Visual anchor 不是主要贡献

组件控制中，移除 simple visual anchor 后，initial transition 的表现基本保持；而移除 soft-to-normal transition 会明显退化。这说明收益不能简单归因于“再次注入视觉信息”。

### 3.4 高频 format 触发主要修退化

在 VStar 上，pure-soft 从 58.64% 提升到 pure-soft-format2 的 74.35%，同时平均长度由 237.3 降至 131.1，`long>=256` 从 33 降至 9，`maxed1024` 从 18 降至 4。完整 guard 为 74.87%，平均长度 127.3。该结果有力证明 format stability 能修复 pure-soft 退化，但跨数据集比较中，format/guard 并不能稳定超过 COT，尤其在 VMCBench 与 POPE 上不理想。因此它是稳定组件，而不是 reasoning ability 的充分来源。

## 4. Early Trajectory Commitment

我们提出 **Early Trajectory Commitment（ETC，早期轨迹承诺）**：多模态生成在最早几个 token 内选择一个语义与视觉解释方向，后续 autoregressive decoding 强烈依赖这一前缀，并倾向于沿已选方向进行 elaboration。错误样本可以很早锁定，随后以低熵和高置信持续生成，因此中段 entropy trigger 未必能改变答案方向。

这一机制统一解释了三类现象：

1. Initial transition 有效，而延迟到中后段的 intervention 迅速衰减。
2. 错题会呈现“低熵、高置信、长输出”，因为长推理可能只是错误轨迹上的自洽扩展。
3. LEAD 的平均动态触发很少，主要收益可以由开头的固定 transition 复现。

已有 early-token 分叉分析中，VStar 的代表性分叉中位位置约为第 22 token，MMVP 约为第 16 token。共 2344 个 branch replay 分支未出现 prefix mismatch，说明 replay 工具能够稳定复现目标前缀。与此同时，部分分支的 immediate next token 并未变化，但随后轨迹和最终答案发生变化，提示 intervention 影响的是连续状态与后续路径，而不一定表现为下一 token 立刻翻转。

## 5. Latent Trajectory Initialization

由 ETC 得到的设计原则是 **Latent Trajectory Initialization（LTI）**：soft reasoning 的核心价值不一定是长期替代离散推理，而可能是生成开头的一次连续状态初始化。

`initial_transition_only` 包含三个步骤：

1. 第 0 步使用概率加权的 soft embedding，并保留既定的 linebreak/anchor 混合。
2. 从 soft 路由切回 normal 时执行 soft-to-normal transition mixed embedding，而不是直接丢弃 latent 状态。
3. transition 后锁定 normal，按普通 greedy COT 完成后续生成。

组件消融显示，关键项是第二步的 transition：`no_to_normal` 会向 `initial_soft_only` 退化，而 `no_anchor` 通常接近完整 initial transition。Timing 消融进一步用于检验延迟 intervention 是否失效。

## 6. TALR 方法

我们将最终工程候选称为 **TALR**。它不是无限增加 soft 推理，而是把 latent 使用限制在“初始化、少量修正、稳定退出”三个位置。

### 6.1 Early Initializer

生成开始时执行 LTI：第 0 步 soft 初始化，并保留 soft-to-normal transition。该模块负责在轨迹承诺前提供连续候选空间。

### 6.2 Budgeted Refiner

初始 transition 后，只允许约 5% 的后续 token 使用 soft 路由。预算约束避免 pure-soft 的长期漂移，同时保留少量隐空间修正机会。

### 6.3 Discrete Stability Guard

从第 2 步起，对格式边界执行 cooldown2；后期若同时出现 diffuse signal 与重复退化，则 veto soft 路由并回到 hard/normal embedding。Guard 保护格式、答案区和生成长度，但不被解释为单独提升 reasoning ability。

TALR 当前统一配置为：early transition + quota 0.05 + `format_cooldown2/min_step2` + late diffuse/repeat veto。

## 7. 已有实验结果

在历史 matched greedy、seed 42 结果中，TALR 相对 COT 的核心结果为：

| 数据集 | COT | TALR | 变化 |
|---|---:|---:|---:|
| VStar | 68.06% | 73.82% | +5.76 pp |
| MMVP sample | 68.00% | 70.33% | +2.33 pp |
| MMVP pair | 39.33% | 42.67% | +3.34 pp |
| VisuLogic300 | 21.33% | 23.00% | +1.67 pp |
| RealWorldQA fixed200 | 66.00% | 67.00% | +1.00 pp |

这些结果支持 TALR 作为当前最佳工程候选，但仍主要来自 R1-Onevision-7B-RL。新的紧凑主矩阵将使用 R1-Onevision-7B 与 Vision-R1-7B，在七个 benchmark 上统一比较 COT、LEAD、initial transition 与 TALR，并以 OpenVLThinker 的 VStar/MMVP 作为外部验证。

## 8. 证据边界

当前结论严格限定如下：

- 不宣称 confidence 总是与正确性反向，只宣称它不是稳定的 correctness indicator。
- 不宣称 `min_step2` 单独有效；它是 TALR 完整组合中的保护参数。
- 不宣称 format stability 本身提升 reasoning ability；它可靠改善的是格式、重复、长度和抽取稳定性。
- 旧 branch replay 能证明轨迹持续性；只有实际 intervention 的 matched counterfactual replay 才能支持因果措辞。
- 历史主结果集中于 R1-Onevision-7B-RL，需要新模型和新数据集矩阵验证泛化。
- Sampled 论文复现与 greedy 机制分析口径必须分开，不能直接构造 fixed/damaged 因果表。

## 9. AAAI Introduction 叙事

第一段提出问题：现有 latent reasoning 方法通常把 uncertainty 当作动态路由依据，但多模态生成中的错误可能高度自信，且解码轨迹具有强前缀依赖。

第二段给出反常识观察：更多 soft、视觉 anchor 和高频 format 切换均不能稳定超过 COT；LEAD 的动态触发极少，其大部分收益却可以由一次开头 transition 复现。

第三段提出 ETC：推理方向在最早期承诺，中后段 entropy spike 往往发生在答案方向已经形成之后。Confidence mismatch、timing curve、early divergence 与 branch replay 共同支持这一解释。

第四段提出 TALR：Early Initializer 在承诺前初始化 latent trajectory，Budgeted Refiner 提供有限修正，Discrete Stability Guard 防止格式和重复退化。

第五段概述实验：跨模型、跨 general/perception/hallucination/math/science benchmark 比较 COT、LEAD、initial transition 与 TALR，并分别报告 accuracy、fixed/damaged、长度、抽取失败、soft ratio 和触发次数。

## 10. 当前贡献

1. 识别并系统验证多模态 latent reasoning 中的 Early Trajectory Commitment 现象。
2. 揭示 confidence、soft 使用量与 correctness 之间并非简单单调关系，重新定位 entropy routing 的作用边界。
3. 将 LEAD 中的早期 transition 与中后段动态触发解耦，提出 Latent Trajectory Initialization 原则。
4. 提出 TALR，将早期初始化、预算化 latent 修正与离散稳定保护组合为统一方法。
5. 建立 corrected evaluator、逐样本 fixed/damaged、事件 trace 与 counterfactual replay 的机制评估体系。

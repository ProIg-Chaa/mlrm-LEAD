# Pure-Soft 推理中的高置信错误实验

## 目录

1. [为什么做这次实验](#1-为什么做这次实验)
2. [实验对象与口径](#2-实验对象与口径)
3. [指标解释](#3-指标解释)
4. [主要结果](#4-主要结果)
5. [排除抽取失败后的结果](#5-排除抽取失败后的结果)
6. [置信度随推理阶段的变化](#6-置信度随推理阶段的变化)
7. [高置信阈值组](#7-高置信阈值组)
8. [长度控制](#8-长度控制)
9. [代表性高置信错题](#9-代表性高置信错题)
10. [结论与边界](#10-结论与边界)

## 1. 为什么做这次实验

此前 VStar50 和 MMVP300 的 pure-soft 分析发现，错题可能具有更高 token confidence、更低 entropy、更长输出。旧实验主要比较 correct/wrong 均值和最高置信样本，尚不足以回答“confidence 能否作为最终正确性的可靠排序信号”。本次实验用当前统一模型、greedy、seed42、1024-token 全量结果重新分析，并增加 AUROC、置信度十分位、risk-coverage、semantic-only 和长度分层控制。

## 2. 实验对象与口径

本实验只分析真正的 `pure_soft`，不包含 COT、LEAD、format2 或 guard。复用已完成的 5 个 full run，不重新生成，因此没有额外 sampling 噪声。

总计 `2291` 个样本，覆盖 VStar、MMVP、VisuLogic300、VMCBench-dev、MMK12-Physics；其中严格口径下 failed extraction 共 `274` 个。所有结果使用 corrected/specialized evaluator。

置信度使用每一步所选 greedy token 在过滤前完整词表分布中的概率 `raw_selected_prob`。pure-soft greedy 下它等价于 raw top-1 probability。禁止使用 top-k/top-p 过滤后的 `selected_prob`，因为该值经常接近 1，会制造虚假高置信。

## 3. 指标解释

- **Strict accuracy/AUC**：抽取失败也算错误，回答“这种轨迹能否可靠地产生正确可用答案”。
- **Semantic-only AUC**：排除抽取失败，只比较成功抽取答案中的语义正确与错误。
- **Confidence AUROC**：用 sample mean raw confidence 排序最终正确性；0.5 表示随机，低于 0.5 表示错误样本反而更自信。
- **Top10 delta**：最高置信 10% 的准确率减总体准确率。若 confidence 可靠，该值应明显为正。
- **Length-controlled AUC**：在输出长度四分位内分别计算再加权，控制长输出退化的影响。
- **Early32/Tail20**：分别观察开头 32 token 与最后 20 token，区分早期不确定和后期错误锁定。

## 4. 主要结果

| 数据集 | strict acc | failed | mean-conf AUC (95% CI) | top10 strict delta | wrong-conf - correct-conf |
|---|---:|---:|---:|---:|---:|
| vstar | 70.68% | 13 | 0.485 [0.376, 0.588] | -35.68pp | +0.0177 |
| mmvp | 61.00% | 28 | 0.409 [0.338, 0.470] | -44.33pp | +0.0327 |
| visulogic300 | 22.00% | 23 | 0.410 [0.323, 0.488] | -15.33pp | +0.0147 |
| vmcbench_dev | 66.20% | 110 | 0.369 [0.330, 0.407] | -48.20pp | +0.0263 |
| mmk12_physics | 29.80% | 100 | 0.450 [0.398, 0.503] | -23.80pp | +0.0058 |

五个数据集的 strict confidence AUC 全部低于 0.5（5/5）。MMVP、VisuLogic、VMCBench 的 95% CI 上界也低于 0.5，表明在这些设置上，高 confidence 不仅不能预测正确，反而稳定地偏向错误。最高置信 10% 的准确率比总体低 15.33–48.20 个百分点。

所有数据集的 wrong mean confidence 都高于 correct，差值为 +0.0058 到 +0.0327。这个方向与旧 VStar/MMVP 观察一致。

## 5. 排除抽取失败后的结果

| 数据集 | semantic baseline acc | semantic AUC (95% CI) | top10 semantic delta |
|---|---:|---:|---:|
| vstar | 75.84% | 0.611 [0.492, 0.713] | -9.18pp |
| mmvp | 67.28% | 0.489 [0.422, 0.555] | +4.15pp |
| visulogic300 | 23.83% | 0.425 [0.346, 0.510] | -13.11pp |
| vmcbench_dev | 74.38% | 0.479 [0.429, 0.515] | -12.58pp |
| mmk12_physics | 37.25% | 0.539 [0.475, 0.603] | +7.75pp |

排除格式/抽取失败后，4/5 个数据集的 AUC 仍不超过 0.55。VisuLogic 的 AUC=0.425，仍明显表现为语义错题更自信；MMVP 和 VMCBench 约为 0.49/0.48，基本没有预测力。

VStar 是重要反例：semantic AUC=0.611，说明在能稳定抽取答案的样本中，confidence 有一定正相关。因此不能写成“pure-soft 在所有数据集上错题一定更自信”。正确表述是：confidence 不是跨数据集可靠的 correctness signal，并存在大量高置信语义错误。

## 6. 置信度随推理阶段的变化

| 数据集 | early32 strict/semantic AUC | mean strict/semantic AUC | tail20 strict/semantic AUC |
|---|---:|---:|---:|
| vstar | 0.480/0.501 | 0.485/0.611 | 0.421/0.525 |
| mmvp | 0.518/0.473 | 0.409/0.489 | 0.415/0.491 |
| visulogic300 | 0.440/0.463 | 0.410/0.425 | 0.437/0.461 |
| vmcbench_dev | 0.469/0.497 | 0.369/0.479 | 0.381/0.499 |
| mmk12_physics | 0.536/0.552 | 0.450/0.539 | 0.402/0.502 |

Early32 strict AUC 位于 0.440–0.536，几乎等于随机，说明模型在开头并没有一个可用于判断最终正确性的可靠 confidence signal。Tail20 strict AUC 降至 0.381–0.437：错误轨迹在后续展开中往往变得更确定，而不是持续保持高熵犹豫。

这与 early trajectory commitment 相容：模型可能较早进入错误路径，随后 token distribution 逐渐收缩；尾部高 confidence 反映的是轨迹已经锁定，不代表该轨迹与图像或 gold 一致。

## 7. 高置信阈值组

| 数据集 | mean conf≥0.90 n/strict/semantic acc | tail20 conf≥0.95 n/strict/semantic acc |
|---|---:|---:|
| vstar | 25 / 36.00% / 64.29% | 46 / 52.17% / 70.59% |
| mmvp | 27 / 11.11% / 50.00% | 29 / 13.79% / 57.14% |
| visulogic300 | 67 / 14.93% / 17.86% | 116 / 17.24% / 20.83% |
| vmcbench_dev | 123 / 22.76% / 54.90% | 230 / 40.87% / 69.63% |
| mmk12_physics | 84 / 11.90% / 40.00% | 173 / 18.50% / 35.16% |

典型例子是 MMVP：mean confidence≥0.90 的 27 条样本 strict accuracy 只有 11.11%；即使排除抽取失败，6 条可抽取样本也只有 50% 正确。VMCBench 同一阈值组 strict accuracy 为 22.76%，semantic accuracy 为 54.90%，均明显低于其总体/semantic baseline。

## 8. 长度控制

| 数据集 | correct/wrong mean length | raw AUC | length-controlled AUC |
|---|---:|---:|---:|
| vstar | 157.4/429.9 | 0.485 | 0.546 |
| mmvp | 126.0/311.5 | 0.409 | 0.454 |
| visulogic300 | 591.7/665.1 | 0.410 | 0.457 |
| vmcbench_dev | 227.9/539.9 | 0.369 | 0.503 |
| mmk12_physics | 516.1/623.1 | 0.450 | 0.496 |

五个数据集的错题都更长。控制长度后，AUC 收敛到 0.454–0.546：VMCBench 的强反向关系主要由长输出/抽取退化放大；MMVP 和 VisuLogic 仍保留一定反向趋势。

因此存在两种高置信错误：一类是语义上选错但输出格式正常；另一类是 soft trajectory 进入长输出/重复退化后，token distribution 极度收缩。两者都说明 token confidence 不能直接当成最终答案可靠性。

## 9. 代表性高置信错题

以下样本均已成功抽取答案，排除了纯格式失败：

| 数据集 | id | pred/gold | mean conf | early32 | tail20 | entropy | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| vstar | 27 | D/B | 0.9746 | 0.5190 | 1.0000 | 0.1367 | 1024 |
| vstar | 158 | A/B | 0.9722 | 0.9359 | 0.9996 | 0.1221 | 1024 |
| vstar | 44 | A/D | 0.9574 | 0.4641 | 1.0000 | 0.2055 | 1023 |
| mmvp | 263 | A/B | 0.9676 | 0.6512 | 0.9996 | 0.1271 | 1023 |
| mmvp | 104 | B/A | 0.9617 | 0.7224 | 0.9996 | 0.1667 | 1023 |
| mmvp | 295 | A/B | 0.9462 | 0.5737 | 1.0000 | 0.2482 | 1024 |
| visulogic300 | 16 | D/C | 0.9778 | 0.7716 | 1.0000 | 0.1160 | 1024 |
| visulogic300 | 104 | A/D | 0.9618 | 0.8419 | 1.0000 | 0.1505 | 1024 |
| visulogic300 | 177 | D/A | 0.9591 | 0.7618 | 1.0000 | 0.1657 | 1023 |
| vmcbench_dev | vmcbench_dev_7861 | A/B | 0.9749 | 0.6015 | 1.0000 | 0.1210 | 1023 |
| vmcbench_dev | vmcbench_dev_1228 | C/A | 0.9674 | 0.5961 | 0.9973 | 0.1461 | 1023 |
| vmcbench_dev | vmcbench_dev_97 | B/D | 0.9672 | 0.8367 | 1.0000 | 0.1741 | 1022 |
| mmk12_physics | mmk12_physics_9369f77e-e7fa-44bb-b8ee-be4e646339fe | D/B | 0.9515 | 0.8749 | 1.0000 | 0.2153 | 1024 |
| mmk12_physics | mmk12_physics_c521c316-b687-4006-9a5c-c1e0e9c5962b | B/C | 0.9477 | 0.8562 | 1.0000 | 0.2305 | 1016 |
| mmk12_physics | mmk12_physics_7184c4bf-9d52-463c-a8ae-502212580bdd | B/D | 0.9300 | 0.8649 | 0.9998 | 0.2628 | 1024 |

例如 VStar id=158 的预测为 A、gold 为 B：early32 confidence 已达 0.9359，整段 mean confidence=0.9722，tail20=0.9996，最终仍然错误。这是较干净的“早期就高置信地走错”案例。

## 10. 结论与边界

1. **高 token confidence 不是 pure-soft 最终正确性的可靠信号。** 五个数据集 strict AUC 全部低于 0.5，最高置信 10% 反而显著更差。
2. **模型确实会自信地犯错。** 排除抽取失败后仍保留 100 条最高置信语义错题；VisuLogic 上语义 AUC 显著低于 0.5。
3. **高置信错误的一部分来自轨迹锁定和生成退化。** 错题普遍更长，尾部 confidence 比早期更反向；这不是简单的早期低置信。
4. **结论不是‘confidence 永远反向’。** VStar semantic-only 显示一定正相关，MMK12-Physics 也有弱正趋势；真正可靠的主张是 confidence 缺乏跨任务校准性。
5. **这里的 confidence 是 token distribution concentration，不是模型显式报告的最终答案置信度。** 因此论文表述应使用 ‘token-level confidence does not imply final correctness’ 或 ‘pure-soft can confidently follow an incorrect trajectory’。

最终结论：

> Pure-soft reasoning often becomes highly confident after committing to a trajectory, but this confidence is poorly calibrated to final-answer correctness. High confidence can indicate trajectory lock-in or degeneration rather than reliable multimodal reasoning.

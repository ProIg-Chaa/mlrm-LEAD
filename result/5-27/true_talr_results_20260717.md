# True TALR 实验结果整理（2026-07-17）

## 1. 目的与版本边界

本次整理对应昨天启动、随后完成修复的 **True TALR** 实验。它不是历史的 `quota05_guard` 或 `quota05_format2`：True TALR 明确保留 early transition，再在后续少量位置引入受约束 soft refinement 与离散稳定保护。

固定生成口径为 greedy decoding、seed 42、`max_new_tokens=1024`、origin COT prompt。所有主指标使用 corrected last-answer evaluator；MMVP 使用专用 evaluator，并同时报告 sample / pair accuracy；RealWorldQA 使用专用 MCQ evaluator。

True TALR 的组成是：

1. **Early initializer**：沿用 LEAD 的 early soft-to-normal transition；
2. **Budgeted refiner**：后续 soft quota 为 5%；
3. **Discrete stability guard**：`format cooldown=2`、最早从 step 2 生效，并在后期 diffuse/repeat 信号出现时 veto soft。

特别说明：此前 Vision-R1 的 VStar Initial Transition/TALR 曾发生 trace/OOM 污染，出现 `6.81%` 的假低分；该目录已废弃，以下均为修复后的完整 191/300/200/300 条结果。

## 2. 主结果

| Model | Dataset | COT | Full LEAD | Initial Transition | True TALR | TALR-COT | TALR-LEAD |
|---|---|---:|---:|---:|---:|---:|---:|
| R1-Onevision-7B-RL | VStar | 68.06% | 72.77% | 72.25% | **72.25%** | +4.19 pp | -0.52 pp |
| R1-Onevision-7B-RL | MMVP | 68.00% / 39.33% pair | 70.33% / 42.00% | 70.33% / 42.00% | **70.67% / 42.67%** | +2.67 pp | +0.34 pp |
| R1-Onevision-7B-RL | RealWorldQA fixed200 | 66.00% | 64.50% | 63.50% | **65.50%** | -0.50 pp | +1.00 pp |
| R1-Onevision-7B-RL | VisuLogic300 | 21.33% | 25.67% | **28.33%** | 23.67% | +2.34 pp | -2.00 pp |
| Vision-R1-7B | VStar | 77.49% | 80.10% | **82.20%** | **82.20%** | +4.71 pp | +2.10 pp |
| Vision-R1-7B | MMVP | 73.67% / 48.67% pair | 73.33% / 48.00% | 73.67% / 48.67% | 73.33% / 48.00% | -0.34 pp | +0.00 pp |
| Vision-R1-7B | RealWorldQA fixed200 | 67.00% | **68.00%** | 66.50% | 67.50% | +0.50 pp | -0.50 pp |
| Vision-R1-7B | VisuLogic300 | 23.00% | **24.67%** | 22.33% | 20.67% | -2.33 pp | -4.00 pp |

## 3. 稳定性与路由统计

| Model | Dataset | True TALR failed extraction | Avg tokens | Long >=256 | Maxed 1024 | Soft ratio | Switch/sample | Format trigger/sample | Veto/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1-Onevision-7B-RL | VStar | 7 | 117.7 | 5 | 1 | 4.48% | 1.00 | 36.55 | 0.26 |
| R1-Onevision-7B-RL | MMVP | 0 | 110.5 | 3 | 0 | 3.73% | 1.00 | 35.55 | 0.19 |
| R1-Onevision-7B-RL | RealWorldQA fixed200 | 1 | 138.5 | 17 | 0 | 4.53% | 1.00 | 42.19 | 0.53 |
| R1-Onevision-7B-RL | VisuLogic300 | 8 | 517.0 | 294 | 12 | 4.11% | 1.00 | 226.08 | 3.94 |
| Vision-R1-7B | VStar | 0 | 120.4 | 7 | 3 | 4.76% | 1.00 | 48.03 | 0.06 |
| Vision-R1-7B | MMVP | 0 | 153.0 | 11 | 6 | 4.81% | 1.00 | 56.72 | 0.15 |
| Vision-R1-7B | RealWorldQA fixed200 | 1 | 183.1 | 15 | 11 | 4.86% | 1.00 | 67.00 | 0.18 |
| Vision-R1-7B | VisuLogic300 | 11 | 533.8 | 200 | 77 | 4.97% | 1.00 | 247.71 | 0.61 |

## 4. 解读

### R1-Onevision-7B-RL

True TALR 相对 COT 在 4 个数据集里 3 个不低于 COT，平均提升 **+2.17 pp**，达到这轮预注册的“主方法”门槛。最稳定的增益来自 VStar 与 MMVP；MMVP 的 sample 和 pair accuracy 都超过 COT，并略高于 Full LEAD。

不过，TALR 并不统一优于每个组成部分：在 VisuLogic300，单独 Initial Transition 的 28.33% 高于 TALR 的 23.67%。这表示 quota/guard 的价值是控制风险和输出退化，而不是保证每个任务都进一步提升推理准确率。

### Vision-R1-7B

修复后的 VStar 是本轮最强的单项证据：Initial Transition 与 True TALR 都达到 **82.20%**，比 COT 高 4.71 pp，也高于 Full LEAD 2.10 pp。RealWorldQA 也略高于 COT。

但 MMVP 无收益、VisuLogic300 明显下降，四集平均仅 **+0.64 pp**，并且只有 2/4 数据集不低于 COT。因此 Vision-R1 的结果支持“early transition 在部分任务和模型上可迁移”，但不支持把 True TALR 写成跨模型的稳定通用提升。

## 5. 当前可写与不可写的结论

可以写：

- 在 R1-Onevision-7B-RL 上，Early Transition 驱动的 TALR 在四个异质 benchmark 中取得正平均增益，并改善了 VStar/MMVP 的准确率；
- format/repeat guard 将后续 latent refinement 限制为约 4--5% 的 token 路由，同时在部分数据集降低长度或 maxed 输出；
- True TALR 是“机制导出的受约束工程设计”，而不是依赖高频 soft routing 的替代 COT。

暂时不能写：

- TALR 在所有模型、数据集上稳定优于 COT 或 LEAD；
- format guard 本身提升了 reasoning ability；
- VisuLogic 的收益来自 TALR，或 quota05 对所有长推理任务有益。

## 6. 与当前 2x2 实验的关系

True TALR 的 early initializer 仍包含目前正在审计的 step-0 `0.9 soft + 0.1 newline` 混合。正在排队的 2x2 不改变 TALR 主结果，而是回答这个初始化的增益是否来自 soft、newline，还是两者的交互。该结论决定论文应强调“latent initialization”还是更保守的“boundary-aware initialization”。

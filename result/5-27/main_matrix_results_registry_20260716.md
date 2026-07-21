# 主实验矩阵结果登记表（2026-07-16）

## 目的与使用规则

本表登记 TALR/ETC 论文目前的核心主矩阵，严格区分三类条目：

- **可引用：** 配置、样本数、评测器和 runtime 状态均已审计；
- **已修复：** 曾受运行环境影响，但已在独占 GPU 下完整重跑；
- **待定：** true TALR 正在运行，不能以旧 quota+guard 数值替代。

所有数值均使用 greedy、seed 42、origin COT prompt、`max_new_tokens=1024`。VStar/VisuLogic 使用 corrected last-answer evaluator；MMVP 使用 specialized sample/pair evaluator；RealWorldQA 使用 fixed200 专用 MCQ evaluator。

## 1. 方法定义审计

| 表中名称 | 运行含义 | 论文状态 |
|---|---|---|
| COT | 全程离散 greedy COT | 可引用基线 |
| Full LEAD | 原始 entropy-routed LEAD | 可引用基线 |
| Initial Transition | step-0 soft + `soft -> normal` bridge，之后 normal | 可引用机制基线 |
| Legacy LEAD+quota+guard | 标准 LEAD 的早期 transition 与后续 entropy routing + quota 0.05 + format/repeat guard | 有价值的历史对照；不是严格的 True TALR |
| True TALR | early transition + quota 0.05 refinement + cooldown2/min-step2 + late diffuse/repeat veto | 正在重跑的正式方法 |

此前名称为 `transition_preserving_quota05_guard_min2` 的 legacy run 不包含 `lead_initial_transition_with_refinement`，因此不满足论文 TALR 定义。本表不将其填入 TALR 列。

## 2. R1-Onevision-7B-RL：已审计 COT/LEAD/Initial Transition

| 数据集 | COT | Full LEAD | Initial Transition | 当前可读结论 |
|---|---:|---:|---:|---|
| VStar | 68.06% | 72.77% | 72.25% | IT 复现 LEAD 大部收益 |
| MMVP sample | 68.00% | 70.33% | 70.33% | IT 与 LEAD 相同 |
| MMVP pair | 39.33% | 42.00% | 42.00% | IT 与 LEAD 相同 |
| RealWorldQA fixed200 | 66.00% | 64.50% | 63.50% | COT 更强，早期干预存在损伤 |
| VisuLogic300 | 21.33% | 25.67% | 28.33% | IT 优于 LEAD 与 COT |

这组结果支持机制上的异质性：early transition 在 VStar/MMVP/VisuLogic 有价值，但不应被写成无条件提升所有 benchmark。

## 3. Vision-R1-7B：已审计 COT/LEAD/Initial Transition

| 数据集 | COT | Full LEAD | Initial Transition | 状态 / 当前可读结论 |
|---|---:|---:|---:|---|
| VStar | 77.49% | 80.10% | **80.10%** | IT 已独占 GPU 重跑，191/191、0 OOM，可引用 |
| MMVP sample | 73.67% | 73.33% | 73.67% | 三者接近 |
| MMVP pair | 48.67% | 48.00% | 48.67% | IT 与 COT 相同 |
| RealWorldQA fixed200 | 67.00% | 68.00% | 66.50% | full LEAD 略优 |
| VisuLogic300 | 23.00% | 24.67% | 22.33% | full LEAD 略优，IT 受损 |

### 已废弃的资源错误结果

旧 Vision-R1 VStar Initial Transition 与 legacy quota+guard run 都有 `176/191` 条 `OutOfMemoryError`。这造成了表面上的 6.81% accuracy、约 7.7 token 平均长度和大量抽取失败。该结果已移动到 `.oom_invalid.<timestamp>` 备份目录，禁止用于任何主表、均值或论文本体结论。

修复后的 Initial Transition 为 153/191 = 80.10%，且 0 runtime error，证明原异常是显存竞争而非方法失效。

## 4. True TALR 正式重跑矩阵

| 模型 | VStar | MMVP | RealWorldQA fixed200 | VisuLogic300 | 状态 |
|---|---|---|---|---|---|
| R1-Onevision-7B-RL | 运行中 | 待排队 | 待排队 | 待排队 | true TALR VStar 已启动 |
| Vision-R1-7B | 待排队 | 待排队 | 待排队 | 待排队 | 先完成 R1-RL 后自动执行 |

运行前 smoke 已完成：R1-RL VStar 2/2 样本正常，配置确认同时记录：

```text
lead_initial_transition_with_refinement = true
lead_soft_quota_ratio = 0.05
lead_format_cooldown = true
format_cooldown_steps = 2
format_cooldown_min_step = 2
lead_soft_veto_on_diffuse = true
```

每个正式 run 的验收条件：完整行数、`eval_report.json`、0 runtime error、配置审计通过；MMVP 与 RealWorldQA 额外运行 specialized evaluator。完成后自动生成 `main_summary_true_talr/talr_core_main_table.md/json` 与 fixed/damaged 结果。

## 5. 当前可用于论文的结论

1. **机制结论有跨模型线索。** 两个模型的 VStar 上，Initial Transition 都至少达到 full LEAD：R1-RL 为 72.25% vs. 72.77%，Vision-R1 为 80.10% vs. 80.10%。MMVP 上也分别达到 LEAD 或 COT 水平。
2. **性能结论尚不能写成 TALR 已跨模型获胜。** true TALR 的定义刚被实现并进入正式重跑；旧 quota+guard 结果不能替代。
3. **Full LEAD 并不普遍优于 COT。** R1-RL RealWorldQA 上 LEAD 低于 COT，Vision-R1 MMVP 上略低于 COT，说明任务和模型存在明显异质性。
4. **所有跨模型平均数必须等待 true TALR 完整矩阵。** 当前不计算混合 old/new method 的平均分，也不使用 OOM run 的 failed extraction/长度统计。

## 6. 与机制线的对应

主矩阵中的 Initial Transition 提供性能层面的简化对照；机制线解释其作用边界：

- VStar/MMVP prefix=1 cache rebuild 均损失一部分收益；prefix=2 基本恢复；
- hard-boundary-only 在 VStar 为 69.63%，低于完整 IT 的 72.25%；
- 47 条 same-prefix replay 中，强制相同 prefix 1/2 后仍有 42 条后续分叉；
- full LEAD 相对 IT 的 late routing 在 VStar/MMVP 几乎没有净收益，在 VisuLogic 为负净收益。

因此，当前论文的可靠主张是“early transition 值得作为独立机制变量审计”；true TALR 是否成为最终主方法，取决于本表第 4 节的重跑结果。

## 7. 产物位置

- 旧基线统一汇总：`output/experiments/20260716_talr_dual_line/main_summary/`
- True TALR 运行目录：`output/experiments/20260716_talr_dual_line/true_talr_core_runs/`
- True TALR 最终汇总：`output/experiments/20260716_talr_dual_line/main_summary_true_talr/`
- 机制外化结果：`output/experiments/20260716_talr_dual_line/transition_externalization/`

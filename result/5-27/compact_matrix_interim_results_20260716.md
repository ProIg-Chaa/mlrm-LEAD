# 紧凑主矩阵阶段性结果（2026-07-16）

## 口径与覆盖

本报告按模型分别汇总紧凑主矩阵的核心四数据集：VStar、MMVP、RealWorldQA fixed200、VisuLogic300。只纳入结果行数完整、存在 `eval_report.json` 且无 runtime error 的 run；MMVP 使用 specialized evaluator，RealWorldQA 使用专用 MCQ evaluator，其余使用 corrected last-answer evaluator。

- COT / Full LEAD / Initial Transition：`2 models × 4 datasets × 3 methods = 24/24` 已完成并通过审计。
- Vision-R1 VStar Initial Transition：旧 run 有 176/191 CUDA OOM，已独占 GPU 完整重跑；本表仅使用修复结果。
- True TALR：`2 models × 4 datasets = 8` 正在按新定义重跑。历史 quota+guard run 继承标准 LEAD 的 early transition，但仍允许 late entropy routing；因此不再填入严格 True TALR 行。
- 主生成口径：greedy、seed 42、max_new_tokens 1024、origin COT prompt。
- `Failed` 指 corrected/specialized evaluator 无法提取答案的数量；`Runtime errors` 不混入 Failed。

## R1-Onevision-7B-RL

| Dataset | Method | Accuracy | Pair acc | Delta vs COT | Failed | Avg tokens | Long>=256 | Maxed1024 | Runtime errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vstar | COT | 68.06% | - | +0.00 pp | 11 | 116.4 | 8 | 0 | 0 |
| vstar | Full LEAD | 72.77% | - | +4.71 pp | 10 | 122.5 | 6 | 1 | 0 |
| vstar | Initial Transition | 72.25% | - | +4.19 pp | 10 | 123.2 | 7 | 1 | 0 |
| vstar | True TALR | running | - | - | - | - | - | - | - |
| realworldqa_fixed200 | COT | 66.00% | - | +0.00 pp | 1 | 140.0 | 18 | 1 | 0 |
| realworldqa_fixed200 | Full LEAD | 64.50% | - | -1.50 pp | 2 | 139.4 | 15 | 1 | 0 |
| realworldqa_fixed200 | Initial Transition | 63.50% | - | -2.50 pp | 2 | 140.9 | 18 | 1 | 0 |
| realworldqa_fixed200 | True TALR | pending | - | - | - | - | - | - | - |
| mmvp | COT | 68.00% | 39.33% | +0.00 pp | 0 | 110.2 | 4 | 0 | 0 |
| mmvp | Full LEAD | 70.33% | 42.00% | +2.33 pp | 0 | 110.4 | 1 | 0 | 0 |
| mmvp | Initial Transition | 70.33% | 42.00% | +2.33 pp | 0 | 109.8 | 1 | 0 | 0 |
| mmvp | True TALR | pending | - | - | - | - | - | - | - |
| visulogic300 | COT | 21.33% | - | +0.00 pp | 5 | 527.0 | 294 | 12 | 0 |
| visulogic300 | Full LEAD | 25.67% | - | +4.34 pp | 12 | 492.2 | 293 | 4 | 0 |
| visulogic300 | Initial Transition | 28.33% | - | +7.00 pp | 5 | 514.4 | 295 | 10 | 0 |
| visulogic300 | True TALR | pending | - | - | - | - | - | - | - |

## Vision-R1-7B

| Dataset | Method | Accuracy | Pair acc | Delta vs COT | Failed | Avg tokens | Long>=256 | Maxed1024 | Runtime errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vstar | COT | 77.49% | - | +0.00 pp | 10 | 116.4 | 8 | 3 | 0 |
| vstar | Full LEAD | 80.10% | - | +2.61 pp | 7 | 121.4 | 7 | 4 | 0 |
| vstar | Initial Transition (repaired) | 80.10% | - | +2.61 pp | 0 | 120.3 | 7 | 3 | 0 |
| vstar | True TALR | pending | - | - | - | - | - | - | - |
| realworldqa_fixed200 | COT | 67.00% | - | +0.00 pp | 0 | 187.2 | 16 | 12 | 0 |
| realworldqa_fixed200 | Full LEAD | 68.00% | - | +1.00 pp | 0 | 180.7 | 14 | 9 | 0 |
| realworldqa_fixed200 | Initial Transition | 66.50% | - | -0.50 pp | 0 | 181.2 | 14 | 11 | 0 |
| realworldqa_fixed200 | True TALR | pending | - | - | - | - | - | - | - |
| mmvp | COT | 73.67% | 48.67% | +0.00 pp | 0 | 150.5 | 10 | 5 | 0 |
| mmvp | Full LEAD | 73.33% | 48.00% | -0.34 pp | 0 | 152.6 | 10 | 5 | 0 |
| mmvp | Initial Transition | 73.67% | 48.67% | +0.00 pp | 0 | 153.0 | 11 | 6 | 0 |
| mmvp | True TALR | pending | - | - | - | - | - | - | - |
| visulogic300 | COT | 23.00% | - | +0.00 pp | 8 | 520.5 | 192 | 74 | 0 |
| visulogic300 | Full LEAD | 24.67% | - | +1.67 pp | 7 | 483.4 | 187 | 51 | 0 |
| visulogic300 | Initial Transition | 22.33% | - | -0.67 pp | 8 | 516.1 | 191 | 72 | 0 |
| visulogic300 | True TALR | pending | - | - | - | - | - | - | - |

## 当前可读结论

1. **Early Transition 是应独立分析的有效变量。** 在 R1-RL 上，它在 VStar/MMVP/VisuLogic 明显达到或超过 Full LEAD；在 Vision-R1 VStar 上，修复后与 Full LEAD 同为 80.10%。
2. **收益具有模型和数据集异质性。** R1-RL 的 RealWorldQA 上 COT 仍更强；Vision-R1 的 MMVP 对三种 route 几乎不敏感；Vision-R1 的 VisuLogic 中 Initial Transition 低于 Full LEAD。
3. **旧 6.81% Vision-R1 VStar 结果完全作废。** 其 176/191 CUDA OOM 造成无输出；修复后 0 runtime error，Initial Transition 为 80.10%。
4. **TALR 尚无严格可用的主表结果。** 旧 `LEAD+quota+guard` 包含 early transition，但没有关闭 late entropy routing，不能替代严格 True TALR；当前正式 true TALR 从 R1-RL VStar 开始串行运行。

## 已知审计事项

- 所有后续 TALR 行必须包含 `lead_initial_transition_with_refinement=true`、quota 0.05、cooldown2/min-step2、late diffuse/repeat veto。
- True TALR 的结果仅在完整行数、0 runtime error、MMVP/RealWorldQA 专用评测报告存在时进入最终表。
- 当前报告只用于阶段性决策；跨模型平均值、TALR 胜率和论文最终主表须等 true TALR 8 个 run 完成后重新计算。

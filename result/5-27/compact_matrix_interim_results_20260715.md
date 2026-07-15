# 紧凑主矩阵阶段性结果（2026-07-15）

## 口径与覆盖

本报告只纳入结果行数完整且同时存在 `eval_report.json` 的 run；正在运行的 partial run 不进入表格。R1-RL 合并配置匹配的历史迁移结果和 NewGpu 补跑结果，Vision-R1 使用 NewGpu 新结果。

- R1-Onevision-7B-RL：26/28 个紧凑矩阵 run 已完成。
- Vision-R1-7B：8/28 个紧凑矩阵 run 已完成。
- 合计：34/56。
- 主生成口径：greedy、seed 42、max_new_tokens 1024、origin COT prompt。
- MMVP 使用 specialized evaluator，并同时报告 sample accuracy 与 pair accuracy；POPE precision/recall/F1 和其他 corrected evaluator 将在最终统一汇总中补充。

## R1-Onevision-7B-RL

| Dataset | Method | Accuracy | Pair acc | Delta vs COT | Failed | Avg tokens | Long>=256 | Maxed1024 | Runtime errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vstar | COT | 68.59% | - | +0.00 pp | 0 | 116.4 | 8 | 0 | 0 |
| vstar | LEAD | 72.77% | - | +4.19 pp | 0 | 122.5 | 6 | 1 | 0 |
| vstar | Initial Transition | 72.25% | - | +3.66 pp | 0 | 123.2 | 7 | 1 | 0 |
| vstar | TALR | 73.82% | - | +5.24 pp | 0 | 118.0 | 6 | 1 | 0 |
| realworldqa_fixed200 | COT | 66.00% | - | +0.00 pp | 0 | 140.0 | 18 | 1 | 0 |
| realworldqa_fixed200 | LEAD | 64.50% | - | -1.50 pp | 0 | 139.4 | 15 | 1 | 0 |
| realworldqa_fixed200 | Initial Transition | 63.50% | - | -2.50 pp | 0 | 140.9 | 18 | 1 | 0 |
| realworldqa_fixed200 | TALR | 66.00% | - | +0.00 pp | 0 | 136.2 | 16 | 0 | 0 |
| mmvp | COT | 68.00% | 39.33% | +0.00 pp | 0 | 110.2 | 4 | 0 | 0 |
| mmvp | LEAD | 70.33% | 42.00% | +2.33 pp | 0 | 110.4 | 1 | 0 | 0 |
| mmvp | Initial Transition | 70.33% | 42.00% | +2.33 pp | 0 | 109.8 | 1 | 0 | 0 |
| mmvp | TALR | 70.33% | 42.67% | +2.33 pp | 0 | 110.2 | 3 | 0 | 0 |
| visulogic300 | COT | 21.00% | - | +0.00 pp | 0 | 527.0 | 294 | 12 | 0 |
| visulogic300 | LEAD | 24.67% | - | +3.67 pp | 0 | 492.2 | 293 | 4 | 0 |
| visulogic300 | Initial Transition | 28.33% | - | +7.33 pp | 0 | 514.4 | 295 | 10 | 0 |
| visulogic300 | TALR | 22.33% | - | +1.33 pp | 0 | 489.8 | 290 | 5 | 0 |
| vmcbench_dev | COT | 74.40% | - | +0.00 pp | 0 | 239.7 | 381 | 16 | 0 |
| vmcbench_dev | LEAD | 72.60% | - | -1.80 pp | 0 | 229.2 | 358 | 5 | 0 |
| pope_adversarial | COT | 83.53% | - | +0.00 pp | 0 | 78.7 | 5 | 2 | 0 |
| pope_adversarial | LEAD | 83.43% | - | -0.10 pp | 0 | 79.0 | 10 | 3 | 0 |
| pope_adversarial | Initial Transition | 82.50% | - | -1.03 pp | 0 | 79.9 | 12 | 10 | 0 |
| pope_adversarial | TALR | 82.67% | - | -0.87 pp | 0 | 79.6 | 9 | 6 | 0 |
| mmk12_physics | COT | 41.20% | - | +0.00 pp | 0 | 530.5 | 478 | 25 | 0 |
| mmk12_physics | LEAD | 34.40% | - | -6.80 pp | 0 | 485.6 | 475 | 5 | 0 |
| mmk12_physics | Initial Transition | 38.60% | - | -2.60 pp | 0 | 522.4 | 480 | 29 | 0 |
| mmk12_physics | TALR | 34.80% | - | -6.40 pp | 0 | 483.6 | 478 | 8 | 0 |

## Vision-R1-7B

| Dataset | Method | Accuracy | Pair acc | Delta vs COT | Failed | Avg tokens | Long>=256 | Maxed1024 | Runtime errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pope_adversarial | COT | 86.33% | - | +0.00 pp | 0 | 117.7 | 11 | 7 | 0 |
| pope_adversarial | LEAD | 86.30% | - | -0.03 pp | 0 | 116.9 | 10 | 2 | 0 |
| pope_adversarial | Initial Transition | 86.33% | - | +0.00 pp | 0 | 117.6 | 11 | 7 | 0 |
| pope_adversarial | TALR | 86.20% | - | -0.13 pp | 0 | 116.7 | 9 | 3 | 0 |
| mmk12_physics | COT | 34.20% | - | +0.00 pp | 0 | 794.3 | 454 | 292 | 0 |
| mmk12_physics | LEAD | 41.80% | - | +7.60 pp | 0 | 683.2 | 451 | 117 | 0 |
| mmk12_physics | Initial Transition | 32.40% | - | -1.80 pp | 0 | 794.8 | 455 | 294 | 0 |
| mmk12_physics | TALR | 42.00% | - | +7.80 pp | 0 | 690.4 | 452 | 110 | 0 |

## 当前可读结论

1. R1-RL 的跨数据集结果仍然是明显异质的：没有一种 latent 方法在所有 benchmark 上稳定优于 COT。
2. Vision-R1 的 MMK12-Physics 上，LEAD/TALR 明显高于 COT，而 Initial Transition 下降；这说明 early transition 的收益并非无条件跨模型成立。
3. Vision-R1 的 POPE-Adversarial 上四种方法几乎持平，说明该 benchmark 对这些路由改动不敏感，或其主要瓶颈不在生成轨迹初始化。
4. Format/guard 更适合被解释为稳定组件；是否提高 reasoning accuracy 必须按模型和数据集分别验证。

## 已知审计事项

- R1-RL 的 POPE/TALR accuracy 与结果文件有效，但其 `token_entropy.jsonl` 曾被重复 worker 并发写入，不能用于触发次数或 soft-ratio 统计。
- 当前报告是阶段性 sample-accuracy 汇总，不替代最终 corrected/specialized evaluator 主表。
- 正在运行的 Vision-R1 MMVP 及后续数据集将在完整落盘后自动进入下一版报告。

# True TALR 与旧 TALR 结果对比

## 配置定义

| 版本 | Early policy | 后续 soft | Stability guard |
|---|---|---|---|
| 旧 TALR（Legacy TALR） | 原始 LEAD 的 legacy dynamic routing；未显式启用 `initial_transition_with_refinement` | quota = 0.05 | cooldown2/min-step2 + late diffuse/repeat veto |
| True TALR | 显式完成 early `soft -> normal` transition，再开启 refinement | quota = 0.05 | cooldown2/min-step2 + late diffuse/repeat veto |

两者均使用 R1-Onevision-7B-RL、greedy decoding、seed 42、origin prompt、`max_new_tokens=1024`。因此两者的核心区别是：旧 TALR 将原 LEAD 的 early path 与后续动态 routing 混在同一状态机中；True TALR 把 early transition 固定为初始化阶段，随后只允许预算化 refinement。

## 主结果：R1-Onevision-7B-RL

| Dataset | 旧 TALR accuracy | True TALR accuracy | True - 旧 | 旧 TALR correct | True TALR correct | 旧 -> True 平均输出 token | 旧 -> True long>=256 | 旧 -> True maxed1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VStar (191) | 73.82% | 72.25% | -1.57 pp | 141 | 138 | 118.0 -> 117.7 | 6 -> 5 | 1 -> 1 |
| MMVP (300) | 70.33% | 70.67% | +0.33 pp | 211 | 212 | 110.2 -> 110.5 | 3 -> 3 | 0 -> 0 |
| MMVP pair (150) | 42.67% | 42.67% | 0.00 pp | 64 | 64 | - | - | - |
| RealWorldQA fixed200 | 66.00% | 65.50% | -0.50 pp | 132 | 131 | 136.2 -> 138.5 | 16 -> 17 | 0 -> 0 |
| VisuLogic300 | 22.33% | 23.67% | +1.33 pp | 67 | 71 | 489.8 -> 517.0 | 290 -> 294 | 5 -> 12 |
| 四数据集简单平均 | 58.12% | 58.02% | -0.10 pp | - | - | - | - | - |

## True TALR 的实际控制强度

| Dataset | mean soft ratio | format trigger / sample | cooldown active steps / sample | late veto / sample |
|---|---:|---:|---:|---:|
| VStar | 4.48% | 36.55 | 36.94 | 0.26 |
| MMVP | 3.73% | 35.55 | 36.38 | 0.19 |
| RealWorldQA fixed200 | 4.53% | 42.19 | 42.99 | 0.53 |
| VisuLogic300 | 4.11% | 226.08 | 226.38 | 3.94 |

## 读表结论

- 旧 TALR 与 True TALR 的平均准确率几乎相同，差异仅 `-0.10 pp`；MMVP 上 True TALR 略高且 pair accuracy 持平。
- True TALR 在 VStar 与 RealWorldQA 略低，在 VisuLogic 略高但生成退化更严重。VisuLogic 的 trigger 极高，却没有减少长输出或 maxed，说明 guard 在该类长推理任务上并非充分修复。
- 因此，True TALR 的主要价值不是当前分数显著更高，而是提供了一个更清晰的、与 early-transition 机制一致的阶段化策略。

## 评测口径注意事项

这不是严格的最终 head-to-head 表：旧 TALR 的 VStar、RealWorldQA、VisuLogic 汇总仍使用历史 `run evaluator`，并报告 `failed_extraction=0`；True TALR 使用当前 corrected last-answer 或 specialized evaluator。MMVP 两侧均使用 specialized evaluator，因而是当前最可直接比较的一行。

在论文主表前，应对旧 TALR 已保存的 `results.jsonl` 用当前 corrected/specialized evaluator 离线复评。该操作不需要重新生成，完成后才能把 VStar、RealWorldQA、VisuLogic 的小幅差异归因于方法而非 extraction 口径。

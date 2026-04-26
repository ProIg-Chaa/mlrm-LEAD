# MLRM-LEAD 实验阶段报告

生成时间：2026-04-26

## 范围

本文档总结目前在本地 `mlrm-LEAD` 项目中已经完成并落盘的实验。

- 模型：`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`
- 数据集：`data/physunibench.jsonl`
- 主要 GPU 设置：`CUDA_VISIBLE_DEVICES=1`
- 当前 LEAD 主配置：`alpha=0.4`，`max_switch_count=5`
- 主要生成长度限制：`max_new_tokens=1024`

本文档只把已经完整完成、并且结果文件已经保存的实验作为依据。后续尝试过一次“关系词 token 熵”的解码统计，但该诊断统计没有完成，因此不作为正式实验结论写入。

## 实验清单

| 实验 | 目录 | 目的 | 状态 |
|---|---|---|---|
| LEAD 参数小网格 | `output/experiments/entropy_grid` | 搜索较优 LEAD 参数 | 已完成 |
| 30 样本三方法对比 | `output/experiments/cot_entropy_probe/compare_methods_20260426_182309` | 对比 `cot`、`cot_greedy`、`lead` | 已完成 |
| 100 样本 LEAD compact 熵统计 | `output/experiments/medium_best_lead_entropy_compact/compare_methods_20260426_183346` | 中等规模 LEAD 实验，并保存 compact 熵摘要 | 已完成 |
| 早前 100 样本 LEAD 实验 | `output/experiments/medium_best_lead_entropy/compare_methods_20260426_152200` | 作为中等规模结果的对照 | 已完成，主准确率与 compact 实验一致 |

## 1. LEAD 参数小网格实验

该实验测试了不同 `alpha` 和 `max_switch_count` 对 LEAD 效果的影响。

| alpha | max_switch_count | 准确率 | 备注 |
|---:|---:|---:|---|
| 0.4 | 3 | 16.67% | switch 次数较少，效果较差 |
| 0.4 | 5 | 30.00% | 当前小网格中最优 |
| 0.6 | 3 | 16.67% | switch 次数较少，效果较差 |
| 0.6 | 5 | 20.00% | 弱于 `alpha=0.4, switch=5` |

有价值的信息：

- 当前小网格中，`alpha=0.4`、`max_switch_count=5` 是最好的设置。
- 从 3 次 switch 增加到 5 次 switch 的收益，比把 `alpha` 从 0.4 提到 0.6 更明显。
- 后续中等规模实验沿用了 `alpha=0.4`、`max_switch_count=5`。

## 2. 30 样本三方法对比实验

实验目录：

`output/experiments/cot_entropy_probe/compare_methods_20260426_182309`

配置：

- 方法：`cot`、`cot_greedy`、`lead`
- 样本数：30
- `max_new_tokens=1024`
- `temperature=0.6`，`top_p=0.95`，`top_k=20`
- LEAD 参数：`alpha=0.4`，`max_switch_count=5`
- 保存 token 熵：是

### 主结果

| 方法 | 准确率 | 平均延迟 | 总延迟 | 平均输出 tokens | 总输出 tokens | 最大 CUDA reserved |
|---|---:|---:|---:|---:|---:|---:|
| `cot` | 23.33% (7/30) | 28.56s | 856.80s | 773.83 | 23215 | 18234 MB |
| `cot_greedy` | 16.67% (5/30) | 34.41s | 1032.16s | 835.60 | 25068 | 18234 MB |
| `lead` | 30.00% (9/30) | 36.55s | 1096.46s | 788.13 | 23644 | 18216 MB |

### 错误类型

| 方法 | 正确 | 答案错误 | 无法抽取答案 |
|---|---:|---:|---:|
| `cot` | 7 | 19 | 4 |
| `cot_greedy` | 5 | 15 | 10 |
| `lead` | 9 | 12 | 9 |

有价值的信息：

- 在这个 30 样本集合上，LEAD 的准确率最高，为 30.00%。
- LEAD 比普通 COT 多答对 2 个样本，比 greedy COT 多答对 4 个样本。
- LEAD 的平均延迟最高，说明准确率提升有时间成本。
- LEAD 相比 COT 减少了 `wrong_answer`，但增加了 `no_answer_extracted`。
- greedy COT 的准确率最低，并且答案抽取失败最多。这说明低随机性、低熵并不等价于更高准确率。

## 3. 100 样本 LEAD compact 熵统计实验

实验目录：

`output/experiments/medium_best_lead_entropy_compact/compare_methods_20260426_183346`

配置：

- 方法：`lead`
- 样本数：100
- `max_new_tokens=1024`
- `temperature=0.6`，`top_p=0.95`，`top_k=20`
- LEAD 参数：`alpha=0.4`，`max_switch_count=5`
- token 熵保存格式：compact 摘要格式

### 主结果

| 方法 | 准确率 | 平均延迟 | 总延迟 | 平均输出 tokens | 总输出 tokens | 最大 CUDA reserved |
|---|---:|---:|---:|---:|---:|---:|
| `lead` | 20.00% (20/100) | 33.77s | 3377.45s | 776.62 | 77662 | 30560 MB |

### 错误类型

| 正确 | 答案错误 | 无法抽取答案 |
|---:|---:|---:|
| 20 | 57 | 23 |

### 按物理子领域统计

| 子领域 | 准确率 |
|---|---:|
| Electromagnetism and electrodynamics | 23.3% (10/43) |
| Mechanics | 15.9% (7/44) |
| Molecular atomic and subatomic physics | 0.0% (0/1) |
| Optics | 25.0% (1/4) |
| Relativity Physics | 0.0% (0/2) |
| Solid physics and measurement of physical quantities | 0.0% (0/3) |
| Thermodynamics | 66.7% (2/3) |

### 按难度统计

| 难度 | 准确率 |
|---:|---:|
| 1 | 13.6% (3/22) |
| 2 | 15.4% (2/13) |
| 3 | 21.7% (5/23) |
| 4 | 13.6% (3/22) |
| 5 | 35.0% (7/20) |

有价值的信息：

- 100 样本上的 LEAD 准确率是 20.00%，明显低于 30 样本实验中的 30.00%。
- 这说明 30 样本上的提升还不稳定，不能直接认为 LEAD 在当前配置下有稳健收益。
- `no_answer_extracted` 达到 23%，说明答案格式和答案抽取是重要失败来源。
- 难度统计不是单调的，difficulty 5 反而准确率最高。这说明该子集里的难度标签不能直接预测模型表现。

## 4. token 熵统计

后续实验中，token 熵记录从完整 token 数组改成了 compact 的逐样本摘要。当前 compact 摘要已经能够正确识别 `<think>...</think>` 推理区间，修复了之前因为 tokenizer 把 `<think>` 拆分导致 reasoning token 统计不到的问题。

### 聚合熵结果

| 实验/方法 | 样本数 | reasoning ratio | reasoning raw entropy | reasoning p90 | high entropy > 1 | high entropy > 2 | soft ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30 样本 `cot` | 30 | 98.11% | 0.287 | 1.035 | 11.20% | 1.82% | 0.00% |
| 30 样本 `cot_greedy` | 30 | 98.51% | 0.266 | 0.955 | 9.79% | 1.48% | 0.00% |
| 30 样本 `lead` | 30 | 98.92% | 0.305 | 1.093 | 12.22% | 2.16% | 0.65% |
| 100 样本 `lead` | 100 | 97.60% | 0.319 | 1.135 | 13.00% | 2.40% | 0.76% |

有价值的信息：

- reasoning token 的统计现在是可用的。
- 大多数生成 token 都位于 `<think>` 推理区间内，因此整体输出熵和 reasoning 区间熵非常接近。
- 在 30 样本对比中，LEAD 的 reasoning 熵高于 COT。
- LEAD 的 soft token 激活比例很低：30 样本中约 0.65%，100 样本中约 0.76%。
- 由于 soft 激活非常稀疏，用整体平均熵解释 LEAD 行为是不够的。
- greedy COT 的熵更低，但准确率也更低，所以“低熵”本身不能作为正确性的充分指标。

## 当前解释

目前的实验显示出一个较弱但值得继续追踪的 LEAD 信号：

- 在 30 样本小规模对比中，LEAD 准确率最高。
- 在 100 样本实验中，LEAD 准确率回落到 20.00%，和早前中等规模结果一致。
- LEAD 似乎更多出现在高不确定性的生成区域，但当前 compact 熵统计还不能证明它专门修正了高熵推理转折点。
- 当前主要问题不只是推理正确性，答案格式和答案抽取失败也足以显著影响最终准确率。

## 尚未回答的问题

当前数据还不能严谨回答以下问题：

1. `therefore`、`however`、`because`、`so`、`then` 等表示推理关系的 token 是否比普通 reasoning token 有更高熵？
2. LEAD 的 soft token 是否集中出现在这些关系词或转折词附近？
3. soft token 事件是否改善了后续几个 token 的生成，还是只是出现在本来就不确定的区域？
4. 正确样本和错误样本在关系词附近的局部熵是否存在系统差异？
5. 如果加入更强的最终答案格式约束，是否能显著降低 `no_answer_extracted`？

## 建议的下一步实验

1. 先跑一个小规模 COT-only probe，保存 token text、token id、entropy，以及该 token 是否属于关系词。
2. 对 LEAD 跑同样 probe，并保存每个 soft token 前后 5 个 token 的局部窗口。
3. 按 token 类型聚合熵：
   - 推理关系词
   - 数字 token
   - 选项字母 token
   - 标点 token
   - 普通内容 token
4. 按正确性比较局部熵：
   - correct
   - wrong answer
   - no answer extracted
5. 在继续跑 100 样本实验前，先加入更严格的最终答案格式提示，并用 30 样本重新对比。

## 阶段结论

目前最有价值的结论是：

- 环境、数据集、模型和评测流程已经完整跑通。
- 当前本地搜索到的较优 LEAD 设置是 `alpha=0.4`、`max_switch_count=5`。
- LEAD 在 30 样本子集上有准确率提升，但在 100 样本上还没有表现出稳定收益。
- reasoning token 熵统计链路已经修好。
- LEAD 的 soft token 激活非常稀疏，低于 1%。
- 下一步最应该做的是局部 token 类型统计和 soft token 邻域统计，而不是继续只看整体平均熵。

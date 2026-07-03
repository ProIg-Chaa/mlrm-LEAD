# Early Trajectory Commitment Rerun Summary

主指标优先使用各数据集 official evaluator；MMVP 使用 specialized evaluator 并报告 pair accuracy；RealWorldQA 只使用 fixed200 MCQ evaluator。Pairwise fixed/damaged 用 enriched evaluator rows 或本地抽取，仅作为配对分析，不作为主 accuracy。

## 这份结果在回答什么问题

这组实验是对前面 `Early Trajectory Commitment` 机制假设的调参补充。前面的主结论是：多模态推理轨迹在生成极早期就被决定，LEAD 的主要收益来自开头的 `soft -> normal` transition，而不是中后段 entropy-gated 的动态 soft 触发。这次没有重新扩大所有方法矩阵，而是沿着两个最可能提升结果的方向做小范围调参：

1. `transition delay`：把开头 transition 从第 0 步稍微往后挪，检查第 1/2 个 token 后再干预是否更稳。
2. `quota ratio`：保留少量后续 soft token，比较 2%/3%/5%/8% quota 以及是否叠加 `format_cooldown2`。

读这份表时要注意：它不是完整机制复核表，而是“调参表”。因此表里没有再次列出 COT、full LEAD、initial_transition_only 等基线。比较这些数字时，需要对照上一轮 guard/机制重跑的主表：VStar 上 `initial_transition_only=72.25%`、full LEAD `72.77%`；MMVP 上 `initial_transition_only/full LEAD=70.33% / 42.00% pair acc`；VisuLogic 上 `initial_transition_only=28.33%`；RealWorldQA fixed200 上 COT/force-normal `66.00%`、`quota05_guard=67.00%`。

## 运行与评估口径

本轮输出目录：

```text
output/experiments/20260610_175907/tune_transition_delay_quota_format
```

模型固定为：

```text
/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL
```

解码口径保持和机制重跑一致：

```text
--no-do_sample --temperature 0.6 --top_p 0.95 --top_k 20
--seed 42 --max_new_tokens 1024
--save_token_entropy --trace_topk 20
```

数据集范围：

- `MMVP full`：`data/mmvp.jsonl`，主指标看 sample accuracy 和 pair accuracy。
- `VisuLogic300`：`data/visulogic.jsonl --limit 300`，保留 by_subtopic。
- `RealWorldQA fixed200`：`data/realworldqa_fixed_mcq_random200_seed42.jsonl`，只用修正后的 fixed200 MCQ。
- `VStar full`：`data/vstar.jsonl`，保留 by_subtopic。

列含义：

- `acc`：官方或专用 evaluator 的主准确率。
- `pair acc`：只对 MMVP 有意义，表示 paired samples 同时正确的比例。
- `len mean`：平均生成长度，用来观察方法是否导致长输出或退化。
- `long>=256`：生成长度不少于 256 token 的样本数。
- `maxed1024`：达到 `max_new_tokens=1024` 的样本数，越高越可能有重复/跑飞。
- `failed`：答案抽取失败数。
- `soft ratio`：实际 soft token 占比。它不是设定值，而是最终生成轨迹中的实际比例。
- `errors`：运行层面的异常数量，应该为 0。

## 实验方案说明

### 1. `transition_step1` / `transition_step2`

这两个 run 是 `initial_transition_only` 的 timing 消融。原始 `initial_transition_only` 等价于 `transition_step0`：第 0 个生成位置先进入 soft，然后马上强制回到 normal，后续保持普通 CoT 推理。它测试的是“开头一次 soft->normal transition 是否足以改变轨迹”。

本轮只调晚一点：

- `transition_step1`：先正常生成 1 个 token，然后再执行一次 soft->normal transition。
- `transition_step2`：先正常生成 2 个 token，然后再执行一次 soft->normal transition。

除了 delay 不同，其余参数保持一致。这个实验的核心问题是：如果轨迹真是在极早期锁定，那么把 transition 从第 0 步挪到第 1/2 步，收益应该下降，尤其在 VStar/MMVP/VisuLogic 这类更依赖早期视觉-推理路径选择的数据集上。

结果读法：

- MMVP：`step1/step2` 都是 69.00%，低于 `step0 initial_transition_only/full LEAD` 的 70.33%。
- VisuLogic：`step1=26.67%`，`step2=23.33%`，低于 `step0 initial_transition_only=28.33%`。
- RealWorldQA：`step1=65.50%`，比 `step0=63.50%` 好，但仍低于本轮 `quota003=67.50%`。

解释：整体上 timing 曲线仍支持“越早越有效”。RealWorldQA 是一个例外方向，它似乎更受少量后续 soft/quota 影响，而不是纯 early transition。

### 2. `quota002` / `quota003` / `quota005` / `quota008`

quota 系列是在 LEAD 框架下限制 soft token 的总预算。数字表示 soft quota ratio：

- `quota002`：最多约 2% 的生成位置允许进入 soft。
- `quota003`：最多约 3%。
- `quota005`：最多约 5%。
- `quota008`：最多约 8%。

这里的 quota 不是“前 N 个 token 固定 soft”，而是给 entropy-gated soft 触发设置预算上限：模型仍然根据触发条件决定何时 soft，但总 soft 占比不能超过对应 quota。它测试的问题是：在 early transition 之外，是否保留一点后续 soft 能修补部分样本，同时避免 pure-soft 的长输出/重复退化。

结果读法：

- VStar：`quota005_format2=73.82%` 最好；纯 `quota005=70.68%` 反而低，说明 VStar 上 quota 需要格式稳定机制配合才可能超过 full LEAD。
- MMVP：`quota005=70.67% / 43.33% pair acc` 最好，略高于 full LEAD/initial transition 的 `70.33% / 42.00%`。
- RealWorldQA：`quota003=67.50%` 最好，高于上一轮 `quota05_guard=67.00%`，也高于 COT/force-normal 的 66.00%。
- `quota008` 往往不稳：MMVP 掉到 67.33% / 37.33% pair acc，RealWorldQA 也没有继续提升。

解释：quota 的可用区间很窄。2% 偏保守，5% 对 VStar/MMVP 较好，3% 对 RealWorldQA 较好，8% 开始引入过多后续扰动，容易伤害已锁定的正确轨迹。

### 3. `*_format2`

`format2` 表示在 quota 方法上叠加 `format_cooldown2`。它的作用不是新的 reasoning 机制，而是格式稳定器：当模型进入答案边界、格式边界或容易跑飞的位置时，短暂降低 soft 干预，减少重复、长输出、答案漂移和抽取失败。

它主要回答一个控制问题：quota 的收益到底来自“更好的推理路径”，还是来自“减少格式退化”？

结果读法：

- VStar：`quota005_format2=73.82%` 明显好于 `quota005=70.68%`，说明 VStar 的 quota05 如果不加格式稳定，会损伤一些样本；format2 能把它救回来。
- MMVP：format2 没有带来提升，`quota005=70.67%/43.33%` 反而略好于 `quota005_format2=70.33%/42.67%`。
- RealWorldQA：format2 不稳定，`quota003=67.50%` 高于 `quota003_format2=67.00%`，`quota008_format2=64.00%` 明显较差。

解释：format2 是“防退化”的工程补丁，不是主机制。它在 pure-soft 和部分 VStar quota 设置中很有价值，但跨数据集并不稳定。

## 主要结论

第一，`transition delay` 没有带来更好的通用结果。MMVP 和 VisuLogic 都显示 `step1/step2` 弱于原始 `step0`，这继续支持 early path-dependence：一旦最初几个 token 的推理方向被选定，后面再做 transition 已经偏晚。

第二，quota 的最优比例是数据集相关的。VStar/MMVP 更接近 `quota005`，RealWorldQA 更接近 `quota003`。这说明“少量后续 soft”可以作为辅助路线，但它不是 LEAD 主收益来源，也不能简单越多越好。

第三，`format2` 的意义要谨慎表述。它确实能修复 pure-soft 或部分 quota 设置的格式退化，但不能被解释为 early trajectory commitment 的核心机制。它更像是稳定输出边界的 guardrail。

第四，当前最值得保留的候选配置是：

- VStar：`quota005_format2`，73.82%，当前最高，但需要说明它是 quota + format 稳定的组合收益。
- MMVP：`quota005`，70.67%，pair acc 43.33%，比 full LEAD/initial transition 略高。
- RealWorldQA fixed200：`quota003`，67.50%，本轮最好。
- VisuLogic：仍然优先 `initial_transition_only`，28.33%；quota 系列不建议作为 VisuLogic 主路线。

## 对主论文叙事的影响

这轮调参没有推翻原机制主线，反而让边界更清楚：

- 如果目标是解释 LEAD 的主要收益，应继续强调第 0 步附近的 `soft -> normal` transition。
- 如果目标是追求各数据集的最高数值，可以把 quota 当作 dataset-specific tuning：VStar/MMVP 用 `quota005` 附近，RealWorldQA 用 `quota003` 附近。
- 不应把 entropy-gated 中后段动态触发写成主要贡献，因为 timing delay 和 quota sweep 都说明后续 soft 的收益有限且不稳定。
- `format_cooldown2` 可以作为稳定性控制变量，而不是核心机制变量。

## phase1_transition_delay_refine / mmvp

| run | acc | pair acc | fixed/damaged vs COT | fixed/damaged vs LEAD | len mean | long>=256 | maxed1024 | failed | soft ratio | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transition_step1 | 69.00% | 41.33% | NA | NA | 108.7 | 3 | 0 | 0 | 1.09% | 0 |
| transition_step2 | 69.00% | 40.00% | NA | NA | 114.8 | 4 | 0 | 0 | 1.04% | 0 |

### by_subtopic
- transition_step1: unknown:207/300
- transition_step2: unknown:207/300

## phase1_transition_delay_refine / realworldqa_fixed200

| run | acc | pair acc | fixed/damaged vs COT | fixed/damaged vs LEAD | len mean | long>=256 | maxed1024 | failed | soft ratio | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transition_step1 | 65.50% | NA | NA | NA | 136.2 | 15 | 1 | 3 | 0.97% | 0 |
| transition_step2 | 64.00% | NA | NA | NA | 132.8 | 12 | 0 | 1 | 0.95% | 0 |

## phase1_transition_delay_refine / visulogic

| run | acc | pair acc | fixed/damaged vs COT | fixed/damaged vs LEAD | len mean | long>=256 | maxed1024 | failed | soft ratio | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transition_step1 | 26.67% | NA | NA | NA | 517.2 | 290 | 13 | 0 | 0.22% | 0 |
| transition_step2 | 23.33% | NA | NA | NA | 498.5 | 291 | 7 | 0 | 0.22% | 0 |

### by_subtopic
- transition_step1: Attribute Reasoning:6/29, Other:10/31, Positional Reasoning:12/37, Quantitative Reasoning:24/100, Spatial Reasoning:20/78, Stylistic Reasoning:8/25
- transition_step2: Attribute Reasoning:8/29, Other:11/31, Positional Reasoning:6/37, Quantitative Reasoning:24/100, Spatial Reasoning:18/78, Stylistic Reasoning:3/25

## phase2_quota_format_sweep / mmvp

| run | acc | pair acc | fixed/damaged vs COT | fixed/damaged vs LEAD | len mean | long>=256 | maxed1024 | failed | soft ratio | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| quota002 | 69.33% | 40.67% | NA | NA | 109.1 | 2 | 0 | 0 | 2.06% | 0 |
| quota002_format2 | 69.33% | 40.67% | NA | NA | 109.8 | 3 | 0 | 0 | 1.86% | 0 |
| quota003 | 68.67% | 41.33% | NA | NA | 108.8 | 5 | 0 | 1 | 2.76% | 0 |
| quota003_format2 | 69.00% | 41.33% | NA | NA | 109.7 | 3 | 0 | 0 | 2.61% | 0 |
| quota005 | 70.67% | 43.33% | NA | NA | 110.3 | 4 | 0 | 0 | 4.05% | 0 |
| quota005_format2 | 70.33% | 42.67% | NA | NA | 110.2 | 3 | 0 | 0 | 3.78% | 0 |
| quota008 | 67.33% | 37.33% | NA | NA | 112.5 | 1 | 0 | 0 | 6.13% | 0 |
| quota008_format2 | 67.33% | 38.00% | NA | NA | 111.4 | 2 | 0 | 0 | 5.78% | 0 |

### by_subtopic
- quota002: unknown:208/300
- quota002_format2: unknown:208/300
- quota003: unknown:206/300
- quota003_format2: unknown:207/300
- quota005: unknown:212/300
- quota005_format2: unknown:211/300
- quota008: unknown:202/300
- quota008_format2: unknown:202/300

## phase2_quota_format_sweep / realworldqa_fixed200

| run | acc | pair acc | fixed/damaged vs COT | fixed/damaged vs LEAD | len mean | long>=256 | maxed1024 | failed | soft ratio | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| quota002 | 66.00% | NA | NA | NA | 141.3 | 15 | 1 | 2 | 2.61% | 0 |
| quota002_format2 | 65.50% | NA | NA | NA | 140.9 | 16 | 1 | 1 | 2.43% | 0 |
| quota003 | 67.50% | NA | NA | NA | 137.2 | 14 | 1 | 2 | 3.32% | 0 |
| quota003_format2 | 67.00% | NA | NA | NA | 136.0 | 15 | 1 | 1 | 3.10% | 0 |
| quota005 | 65.00% | NA | NA | NA | 132.5 | 13 | 0 | 0 | 4.84% | 0 |
| quota005_format2 | 66.50% | NA | NA | NA | 135.1 | 15 | 0 | 1 | 4.68% | 0 |
| quota008 | 65.50% | NA | NA | NA | 143.9 | 17 | 1 | 2 | 7.22% | 0 |
| quota008_format2 | 64.00% | NA | NA | NA | 135.6 | 17 | 0 | 0 | 7.14% | 0 |

## phase2_quota_format_sweep / vstar

| run | acc | pair acc | fixed/damaged vs COT | fixed/damaged vs LEAD | len mean | long>=256 | maxed1024 | failed | soft ratio | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| quota002 | 72.25% | NA | NA | NA | 121.2 | 7 | 1 | 0 | 2.47% | 0 |
| quota002_format2 | 71.20% | NA | NA | NA | 121.2 | 8 | 1 | 0 | 2.20% | 0 |
| quota003 | 71.73% | NA | NA | NA | 117.2 | 6 | 1 | 0 | 3.10% | 0 |
| quota003_format2 | 70.16% | NA | NA | NA | 117.1 | 8 | 1 | 0 | 2.97% | 0 |
| quota005 | 70.68% | NA | NA | NA | 120.1 | 5 | 1 | 0 | 4.91% | 0 |
| quota005_format2 | 73.82% | NA | NA | NA | 117.9 | 6 | 1 | 0 | 4.58% | 0 |
| quota008 | 72.25% | NA | NA | NA | 118.2 | 6 | 0 | 0 | 7.27% | 0 |
| quota008_format2 | 72.77% | NA | NA | NA | 118.3 | 5 | 1 | 0 | 6.96% | 0 |

### by_subtopic
- quota002: direct_attributes:82/115, relative_position:56/76
- quota002_format2: direct_attributes:82/115, relative_position:54/76
- quota003: direct_attributes:78/115, relative_position:59/76
- quota003_format2: direct_attributes:78/115, relative_position:56/76
- quota005: direct_attributes:81/115, relative_position:54/76
- quota005_format2: direct_attributes:82/115, relative_position:59/76
- quota008: direct_attributes:84/115, relative_position:54/76
- quota008_format2: direct_attributes:83/115, relative_position:56/76

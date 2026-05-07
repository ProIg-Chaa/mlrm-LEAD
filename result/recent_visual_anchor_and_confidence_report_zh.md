# 近期实验整理报告：视觉 Anchor 改动与置信度分析

更新日期：2026-05-05

本文整理了 2026-04-28 至 2026-05-01 期间的两条主线实验：

1. `lead_attenachor` / 动态视觉 anchor 的工程与消融实验
2. `pure_soft`、`cot`、`lead` 上围绕“错题是否更自信、更低熵”的后续分析

---

## 摘要

本轮实验可以压缩成一个核心判断：

> 动态视觉 anchor 目前没有带来稳定收益；它的问题主要不是“没看图”，而是注入时机和注入方式破坏了原本生成轨迹。置信度分析则表明，`pure_soft` 和部分 `cot` 设置中存在明显的“高置信错误”现象，但 `LEAD` 在平均层面削弱了该现象，只保留了危险的高置信错误尾部。因此，下一步更合理的方向不是继续盲调 anchor guard，而是设计 `confidence/type-aware routing`。

---

## 目录

1. [实验背景与目标](#1-实验背景与目标)
2. [环境与代码改动](#2-环境与代码改动)
3. [视觉 Anchor 主线](#3-视觉-anchor-主线)
4. [VStar 视觉 Anchor 全量结果](#4-vstar-视觉-anchor-全量结果)
5. [VStar 视觉 Anchor 诊断子集](#5-vstar-视觉-anchor-诊断子集)
6. [视觉 Anchor 主线最终结论](#6-视觉-anchor-主线最终结论)
7. [置信度主线：Pure-Soft](#7-置信度主线pure-soft)
8. [置信度主线：COT](#8-置信度主线cot)
9. [置信度主线：LEAD（MMVP + VStar）](#9-置信度主线leadmmvp--vstar)
10. [置信度主线：LEAD（PhysUniBench + VisuLogic）](#10-置信度主线leadphysunibench--visulogic)
11. [跨方法、跨数据集汇总判断](#11-跨方法跨数据集汇总判断)
12. [当前建议](#12-当前建议)
13. [可直接引用的关键结论](#13-可直接引用的关键结论)

---

## 1. 实验背景与目标

近期实验主要围绕两个问题展开。

### 1.1 视觉 anchor 线

目标是验证新方法 `lead_attenachor` 是否能在 `VStar` 上超过原始 `lead`，并分析：

- 动态视觉 anchor 是否在高熵推理点有效触发
- 注入后是否能降低局部熵
- 是否能减少错误推理或抽取失败
- 不同 `top_m`、`lambda` 与 guard 策略是否形成稳定收益

### 1.2 置信度线

目标是检查以下现象是否稳定：

> 错题是否往往更自信、更低熵、更长、更慢

为此先后在以下方法上比较了 `VStar`、`MMVP`、`PhysUniBench`、`VisuLogic` 等数据集的对错题差异：

- `pure_soft`
- `cot`
- `lead`

---

## 2. 环境与代码改动

### 2.1 环境与模型

- 项目根目录：`/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD`
- 环境：`mlrm-lead`
- 模型：`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`

### 2.2 关键代码改动

#### A. token-level entropy / confidence trace

`main.py` 支持：

- `--save_token_entropy`
- `--save_full_token_entropy`

重要输出：

- `token_entropy.jsonl`
- `token_entropy_full.jsonl`

Per-token trace 中会记录：

- `raw_entropy`
- `filtered_entropy`
- `selected_prob`
- `raw_selected_prob`
- `confidence`
- `token_text`
- `is_reasoning_token`
- `is_relation_token`
- `relation_category`

#### B. pure_soft 方法

新增 `generate_pure_soft()`，位于：

- [generation_utils.py](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/lead/generation_utils.py)

后续 pure-soft 的 confidence 分析主要使用 `raw_selected_prob`。

#### C. correct/wrong 曲线脚本修正

修正了：

- [plot_pure_soft_correct_wrong_curves.py](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/plot_pure_soft_correct_wrong_curves.py)

修正点：

- 当 `raw_selected_prob` 不存在时，回退到 `selected_prob`

原因：

- `cot` / `lead` trace 里通常只有 `selected_prob`
- 修正前，这两类方法的 confidence 统计会错误地接近 `0`

#### D. MMVP 专用评估修正

修正了：

- [evaluate_specialized_results.py](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/evaluate_specialized_results.py)

MMVP 的新口径：

- 只在答案尾部区域抽取
- 接受显式 `\boxed{a}` / `\boxed{b}` 与清晰 `Answer: (a)/(b)`
- 保留选项文本映射回 `(a)/(b)` 的能力
- 移除了不安全的 `tail_ab` fallback
- 直接输出 `pair_accuracy`

另外新增：

- [evaluate_mmvp_official.py](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/script/evaluate_mmvp_official.py)

用途：

- 生成 MMVP 官方仓库所需 `answer.jsonl`
- 复现官方 `gpt_grader.py` 的 pair 口径

当前限制：

- 当时环境中没有 `OPENAI_API_KEY`，因此官方 GPT judge 没有真正跑完

---

## 3. 视觉 Anchor 主线

### 3.1 方法定义

`lead_attenachor` 相比原始 `lead` 的核心变化是：

1. 只在 `normal -> soft` 的第一个 token 上尝试注入视觉 anchor
2. 不使用固定视觉 anchor
3. 动态 anchor 由当前 token 对视觉 token 的 attention 动态构造

当前实现只使用最后四层 decoder attention。

### 3.2 主要调参维度

- `visual_anchor_top_m`
- `visual_anchor_lambda_scale`
- `visual_anchor_entropy_upper`
- `visual_anchor_single_use`
- `visual_anchor_skip_nonword`

当前经验里最关键的仍然是：

1. `top_m`
2. `lambda_scale`
3. `entropy_upper`

---

## 4. VStar 视觉 Anchor 全量结果

基线：

- `lead`：`139 / 191 = 72.77%`
- 目录：`output/experiments/vstar_lead_paper_params_20260426_213624`

首轮与后续版本：

| 方法/配置 | 目录 | 正确 / 总数 | 准确率 |
|---|---|---:|---:|
| `lead` | `output/experiments/vstar_lead_paper_params_20260426_213624` | 139 / 191 | 72.77% |
| 初版 `lead_attenachor` | `output/experiments/20260428/vstar_lead_attenachor_entropy_140553` | 137 / 191 | 71.73% |
| `top4_guarded` | `output/experiments/20260428/vstar_lead_attenachor_top4_guarded_151421` | 136 / 191 | 71.20% |
| `top4_midguard` | `output/experiments/20260428/vstar_lead_attenachor_top4_midguard_173005` | 133 / 191 | 69.63% |

相关已有报告：

- [vstar_lead_attenachor_experiment_report_zh.md](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/vstar_lead_attenachor_experiment_report_zh.md)
- [vstar_lead_attenachor_experiment_report_zh_detailed.md](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/vstar_lead_attenachor_experiment_report_zh_detailed.md)

### 4.1 初版 `lead_attenachor`

目录：

- `output/experiments/20260428/vstar_lead_attenachor_entropy_140553`

关键观察：

- 相比旧 `lead` 少 `2` 题
- anchor 触发样本：`42`
- anchor 事件：`48`
- relation token 上的 anchor：`5`
- anchor 前 5 token 平均熵：`0.619`
- anchor 后 5 token 平均熵：`2.977`

解释：

- anchor 不是把高熵峰压下去
- 更像是在高熵段入口触发，之后熵继续上升
- 失败模式主要是：
  - 触发后失稳
  - 输出退化
  - 抽取失败
  - 少量样本语义漂移

### 4.2 `top4_guarded`

目录：

- `output/experiments/20260428/vstar_lead_attenachor_top4_guarded_151421`

参数特征：

- `top_m = 4`
- 更小 `lambda`
- `single_use`
- `skip_nonword`
- `entropy_upper = 2.0`

结果：

- 准确率：`71.20%`
- anchor 触发样本：`23`

解释：

- 这版主要是止损
- 确实减少失稳
- 但也削弱了少量本来可能有效的触发

### 4.3 `top4_midguard`

目录：

- `output/experiments/20260428/vstar_lead_attenachor_top4_midguard_173005`

参数特征：

- `top_m = 4`
- `lambda_scale = 0.5`
- `entropy_upper = 2.5`
- 不再叠加 `single_use / skip_nonword`

结果：

- 准确率：`69.63%`
- anchor 触发样本：`31`

解释：

- 触发更多，但无效触发也更多
- 说明放松 guard 后重新引入失稳

---

## 5. VStar 视觉 Anchor 诊断子集

诊断子集由多轮变化样本合并得到，共 `15` 条：

- `19, 49, 74, 86, 113, 123, 128, 137, 140, 152, 154, 162, 172, 175, 178`

已有结果：

| 方法/配置 | 正确 / 15 |
|---|---:|
| 旧 `lead` | 9 / 15 |
| 初版 `lead_attenachor` | 7 / 15 |
| `top4_guarded` | 6 / 15 |
| `top4_midguard` | 3 / 15 |
| `vstar_anchor_diag_alllayers_222345` | 6 / 15 |
| `vstar_anchor_diag_dualdelta2_230826` | 7 / 15 |
| `vstar_anchor_diag_dualdelta2_w256_singleanchor_233204` | 7 / 15 |
| `vstar_anchor_diag_dualdelta2_w256_singleanchor_repeatgate_234218` | 7 / 15 |
| `vstar_anchor_diag_dualdelta2_w256_postreset_000936` | 7 / 15 |

结论：

- `top_m = 32` 过大，这一点已经很明确
- 降到 `4/6/8` 后，虽然失稳减轻，但没有任何组合超过旧 `lead`
- 后期 `dual_delta2 + larger window + singleanchor/repeatgate/postreset` 只能把行为调得更合理，没有逆转最终排名

---

## 6. 视觉 Anchor 主线最终结论

到目前为止，可以比较确定地说：

1. `lead_attenachor` 工程上已经跑通
2. 动态视觉 anchor 确实会触发
3. relation token 整体仍高熵
4. `top_m = 32` 明显过大
5. 当前主要问题不是完全没看图，而是：
   - 在失稳边缘触发
   - 注入后局部熵继续升高
   - 输出退化
   - 抽取失败或少量语义偏移
6. 当前没有任何一版 `lead_attenachor` 在 `VStar` 全量上超过旧 `lead`

因此，这条线的收获主要是问题定位，而不是得到更优版本。

---

## 7. 置信度主线：Pure-Soft

### 7.1 VStar Pure-Soft 50

目录：

- `output/experiments/20260429/vstar_pure_soft_50_203818_setsid`

报告与图：

- [vstar_pure_soft_confidence_vs_correctness_report_zh.md](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/vstar_pure_soft_confidence_vs_correctness_report_zh.md)
- [vstar_pure_soft_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/vstar_pure_soft_correct_wrong_curves.png)
- [vstar_pure_soft_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/vstar_pure_soft_correct_wrong_summary.json)

结果：

- `29 / 50 = 58.0%`

现象：

- wrong 的 `mean_raw_conf` 更高
- wrong 的 `last10/last20_raw_conf` 更高
- wrong 的 `mean_raw_entropy` 更低
- wrong 更长、更慢

解释：

- 错题常常不是“犹豫地错”
- 它们往往是“低熵、高置信、长轨迹”的错误推理

### 7.2 MMVP Pure-Soft 300

目录：

- `output/experiments/20260429/pure_soft_phys300_mmvp_parallel_212553/mmvp_full_gpu1`

图与摘要：

- [mmvp_pure_soft_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/mmvp_pure_soft_correct_wrong_curves.png)
- [mmvp_pure_soft_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/mmvp_pure_soft_correct_wrong_summary.json)

修正后结果：

- specialized sample accuracy：`191 / 300 = 63.67%`
- `failed_extraction = 14`

现象：

- correct `mean_raw_conf = 0.7559`
- wrong `mean_raw_conf = 0.7838`
- correct `last20_raw_conf = 0.8560`
- wrong `last20_raw_conf = 0.8737`
- correct `mean_raw_entropy = 0.9155`
- wrong `mean_raw_entropy = 0.8312`
- top 5 by `mean_raw_conf`：`100%` wrong
- `last20_raw_conf >= 0.95` 组准确率仅 `27.6%`

解释：

- `MMVP` 上也稳定复现了同一现象

### 7.3 PhysUniBench Pure-Soft 300

目录：

- `output/experiments/20260429/pure_soft_phys300_mmvp_parallel_212553/physunibench_uniform300_gpu0`

修正后结果：

- specialized accuracy：`14 / 300 = 4.67%`
- `failed_extraction = 196`

解释：

- 这里问题不主要是 evaluator bug
- `pure_soft` 在 `PhysUniBench` 上经常无法收敛到稳定 MCQ 输出

---

## 8. 置信度主线：COT

目录：

- `output/experiments/20260429/cot_phys300_mmvp_parallel_232652/mmvp_full_gpu1`
- `output/experiments/20260429/cot_phys300_mmvp_parallel_232652/physunibench_uniform300_gpu0`

### 8.1 MMVP COT

修正后结果：

- sample accuracy：`202 / 300 = 67.33%`
- pair accuracy：`59 / 150 = 39.33%`

现象：

- correct `mean_raw_conf = 0.8809`
- wrong `mean_raw_conf = 0.8787`
- correct `mean_raw_entropy = 0.7455`
- wrong `mean_raw_entropy = 0.7648`

解释：

- `MMVP COT` 不明显复现 pure-soft 的模式
- 错题略长、略慢，但不是整体更自信、更低熵

### 8.2 PhysUniBench COT

修正后结果：

- specialized accuracy：`31 / 300 = 10.33%`
- `failed_extraction = 164`

现象：

- correct `mean_raw_conf = 0.9169`
- wrong `mean_raw_conf = 0.9276`
- correct `last20_raw_conf = 0.9459`
- wrong `last20_raw_conf = 0.9650`
- correct `mean_raw_entropy = 0.4932`
- wrong `mean_raw_entropy = 0.4303`
- top 20 by `mean_raw_conf`：`100%` wrong
- `last20_raw_conf >= 0.95` 组准确率：`7.7%`

解释：

- `PhysUniBench COT` 明显复现了高置信、低熵、长错误轨迹现象

---

## 9. 置信度主线：LEAD（MMVP + VStar）

目录：

- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0`
- `output/experiments/20260501/lead_mmvp_vstar_parallel_004123/vstar_full_gpu1`

图与摘要：

- [mmvp_lead_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0/result/mmvp_lead_correct_wrong_curves.png)
- [mmvp_lead_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_mmvp_vstar_parallel_004123/mmvp_full_gpu0/result/mmvp_lead_correct_wrong_summary.json)
- [vstar_lead_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_mmvp_vstar_parallel_004123/vstar_full_gpu1/result/vstar_lead_correct_wrong_curves.png)
- [vstar_lead_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_mmvp_vstar_parallel_004123/vstar_full_gpu1/result/vstar_lead_correct_wrong_summary.json)

### 9.1 MMVP LEAD

修正后结果：

- sample accuracy：`211 / 300 = 70.33%`
- pair accuracy：`63 / 150 = 42.0%`

现象：

- correct `mean_raw_conf = 0.8797`
- wrong `mean_raw_conf = 0.8851`
- correct `last20_raw_conf = 0.9437`
- wrong `last20_raw_conf = 0.9429`
- correct `mean_raw_entropy = 0.7545`
- wrong `mean_raw_entropy = 0.7382`

解释：

- 平均上只有很弱的趋势：错题略高 confidence、略低 entropy
- 高置信尾部并不强烈错题占优

### 9.2 VStar LEAD

结果：

- `139 / 191 = 72.77%`

现象：

- correct `mean_raw_conf = 0.8797`
- wrong `mean_raw_conf = 0.8739`
- correct `last20_raw_conf = 0.9567`
- wrong `last20_raw_conf = 0.9484`
- correct `mean_raw_entropy = 0.7291`
- wrong `mean_raw_entropy = 0.7642`

解释：

- `VStar LEAD` 不支持“错题更自信、更低熵”
- 错题更长、更慢，但也更高熵
- 更像“错误但更犹豫”，不是“错误且锁死”

### 9.3 小结

`MMVP + VStar` 不支持对 `LEAD` 做出如下强结论：

> LEAD 仍然稳定表现为错题更自信、更低熵

更准确的说法是：

- `MMVP LEAD` 只有弱趋势
- `VStar LEAD` 平均上不成立

---

## 10. 置信度主线：LEAD（PhysUniBench + VisuLogic）

目录：

- `output/experiments/20260501/lead_phys300_visulogic_parallel_155552/physunibench_uniform300_gpu0`
- `output/experiments/20260501/lead_phys300_visulogic_parallel_155552/visulogic_full_gpu1`

### 10.1 PhysUniBench LEAD

修正后结果：

- specialized accuracy：`21 / 300 = 7.0%`
- `failed_extraction = 176`

图与摘要：

- [physunibench_lead_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_phys300_visulogic_parallel_155552/physunibench_uniform300_gpu0/result/physunibench_lead_correct_wrong_curves.png)
- [physunibench_lead_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_phys300_visulogic_parallel_155552/physunibench_uniform300_gpu0/result/physunibench_lead_correct_wrong_summary.json)

现象：

- correct `mean_raw_conf = 0.9234`
- wrong `mean_raw_conf = 0.9224`
- correct `last20_raw_conf = 0.9175`
- wrong `last20_raw_conf = 0.9184`
- correct `mean_raw_entropy = 0.4504`
- wrong `mean_raw_entropy = 0.4760`

平均上结论：

- 不支持“错题整体更自信、更低熵”

但高置信尾部现象仍然强：

- top 5 by `mean_raw_conf`：`100%` wrong
- top 5 by `last10_raw_conf`：`100%` wrong
- top 5 by `last20_raw_conf`：`100%` wrong
- `mean_raw_conf >= 0.9`：`243` 个样本，准确率 `6.58%`
- `last10_raw_conf >= 0.95`：`144` 个样本，准确率 `4.17%`
- `last20_raw_conf >= 0.95`：`143` 个样本，准确率 `5.59%`

解释：

- `LEAD` 可能削弱了整体均值上的高置信错题模式
- 但没有消除危险的高置信错误尾部

### 10.2 VisuLogic LEAD

结果：

- `229 / 1000 = 22.9%`

图与摘要：

- [visulogic_lead_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_phys300_visulogic_parallel_155552/visulogic_full_gpu1/result/visulogic_lead_correct_wrong_curves.png)
- [visulogic_lead_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260501/lead_phys300_visulogic_parallel_155552/visulogic_full_gpu1/result/visulogic_lead_correct_wrong_summary.json)

现象：

- correct `mean_raw_conf = 0.9018`
- wrong `mean_raw_conf = 0.9027`
- correct `last20_raw_conf = 0.9217`
- wrong `last20_raw_conf = 0.9175`
- correct `mean_raw_entropy = 0.6088`
- wrong `mean_raw_entropy = 0.6076`

解释：

- 平均差异几乎可以忽略
- 最多只能说 wrong 的 `mean_conf` 略高、`mean_entropy` 略低
- 量级非常小，不足以支持强结论

高置信尾部仍偏错：

- top 5 by `mean_raw_conf`：`100%` wrong
- top 5 by `last10_raw_conf`：`80%` wrong
- top 5 by `last20_raw_conf`：`60%` wrong

---

## 11. 跨方法、跨数据集汇总判断

### 11.1 最稳定复现高置信错题模式的设置

当前证据最强的是：

- `VStar pure_soft`
- `MMVP pure_soft`
- `PhysUniBench cot`

这些设置上，错题往往：

- 更高 confidence
- 更低 entropy
- 更长
- 更慢

### 11.2 `LEAD` 的不同之处

`LEAD` 的现象更复杂：

- 在 `MMVP` 上只有弱趋势
- 在 `VStar` 上平均上不成立
- 在 `PhysUniBench` 上平均上也不成立，但高置信尾部非常危险
- 在 `VisuLogic` 上平均差异几乎没有，但高置信尾部仍偏错

因此，对 `LEAD` 更准确的说法不是：

> LEAD 仍然强烈表现为错题更自信、更低熵

而是：

> LEAD weakens the average high-confidence-wrong pattern, but does not eliminate the dangerous high-confidence error tail.

### 11.3 任务结构的影响

目前可以把数据集大致分成两类。

#### A. 短程视觉判别型

- `MMVP`
- `VStar`

特点：

- 问题目标明确
- 答案空间小
- 更像局部视觉判别 / 关系判断

这类数据集上，不太容易稳定形成“错误长推理锁定”。

#### B. 长程视觉-符号推理型

- `PhysUniBench`
- `VisuLogic`（介于中间，但明显更复杂）

特点：

- 推理链更长
- 中间状态更多
- 更容易出现错误轨迹延长、格式失败或高置信尾部风险

---

## 12. 当前建议

### 12.1 关于视觉 anchor

当前不建议继续大规模堆复杂 guard。

如果继续做 `lead_attenachor`，更值得保留的方向是：

- 继续围绕更小的 `top_m`
- 更明确地区分“高熵入口触发”和“失稳后补救”
- 避免只在单点注入上继续叠加复杂条件

### 12.2 关于置信度研究

后续建议把结论分成两层。

1. **平均行为**
   - correct/wrong 的平均 confidence、entropy、长度、延迟
2. **危险尾部**
   - 高置信 top-k 是否被错题占据
   - 高置信阈值组准确率是否异常低

对 `LEAD` 来说，第二层往往比第一层更有解释力。

### 12.3 关于评估口径

以后引用结果时建议遵守：

- `MMVP`：不要再用仓库默认 evaluator；优先 specialized + pair accuracy
- `PhysUniBench`：不要直接信默认 `eval_report.json`；先看 specialized re-eval
- 所有 confidence-vs-correctness 对比尽量保留：
  - `results.jsonl`
  - `token_entropy_full.jsonl`
  - 修正后的评估报告

### 12.4 关于后续路线

当前结果更支持下面这个方向：

- `high entropy` 不等于一定该走 `soft`
- `high entropy` 不等于一定该注入 `anchor`
- `high confidence` 不等于一定正确

因此，更合理的下一步是设计一个 `confidence/type-aware multimodal routing`，让模型根据当前状态选择：

- `discrete`
- `soft`
- `weak visual anchor`
- `cooldown`

而不是统一执行 soft，或统一注入 anchor。

---

## 13. 可直接引用的关键结论

### 13.1 视觉 anchor

> 在 VStar 上，当前 `lead_attenachor` 工程已跑通，但没有任何一版超过原始 `lead`。问题已较明确地定位在：`top_m` 过大、触发点接近失稳边缘、注入后局部熵继续上升，以及由此带来的输出退化和抽取失败。

### 13.2 置信度

> `pure_soft` 与部分 `cot` 设置下，错题常表现为更高 confidence、更低 entropy、更长、更慢；但 `LEAD` 在多个数据集上削弱了这种平均模式。尽管如此，`LEAD` 仍未消除高置信错误尾部，尤其在 `PhysUniBench` 上，高置信样本组准确率依然非常低。

### 13.3 后续方向

> 当前结果说明，单纯改 visual anchor 并不足够。更合理的下一步是设计一个 `confidence/type-aware routing` 策略，根据当前 token 分布的置信度、高熵来源类型和视觉注意力可信度，动态选择 `discrete`、`soft`、`weak visual anchor` 或 `cooldown`。

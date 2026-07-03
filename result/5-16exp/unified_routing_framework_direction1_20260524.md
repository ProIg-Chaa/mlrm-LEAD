# 统一路由框架准备：从散点规则到 state -> route -> action

时间：2026-05-24

本文用于准备下一阶段的方向一：把近期实验中分散出现的 `format cooldown`、`low-confidence collapse`、`answer-zone discrete`、`image_pad visual bias` 等规则，整理成一个统一的路由框架。

核心目标不是继续调参，而是让方法变成可解释、可复用、可分析的形式：

```text
token / sample state -> uncertainty type -> intervention action
```

## 1. 为什么需要统一路由

目前已有结果说明，单一规则不能解释所有现象。

1. 高熵不等于需要视觉。
   - 高熵 token 可能是视觉词、关系词、格式词、答案词，也可能是低置信扩散。
   - 之前 entropy-gated image_pad bias 跨数据集不稳定，说明“高熵触发视觉注入”过粗。

2. 视觉注入有阶段依赖。
   - VStar 上 early image_pad bias 破坏很强。
   - 在 18 个 early damage 样本上，即使 `lambda=0.01`，early 仍只保留 `10/18 = 55.56%`。
   - mid 明显安全：`lambda=0.02/0.03/0.05` 都是 `17/18 = 94.44%`。
   - late 最安全：`18/18 = 100%`，但全量收益弱。

3. format / low-conf 路由已经证明有效。
   - VStar full 上 `cooldown2 + late64_repeat_gate` 达到约 `74.87%`，接近或超过 LEAD 的一部分设置。
   - 这说明很多错误不是单纯视觉缺失，而是 pure-soft 推理轨迹中的格式、扩散、重复退化问题。

因此，下一阶段应该避免继续把所有信号混成一个触发条件，而是建立清楚的路由表。

## 2. 当前代码位置

主要实现位置：

- 参数定义：`main.py`
- 参数传递：`lead/inference.py`
- 路由核心：`lead/generation_utils.py` 的 `generate_pure_soft(...)`
- 高熵 token 类型分析：`script/exp5_16/analyze_spike_types.py`
- bad event 分析：`script/exp5_16/analyze_bad_events.py`

当前 `generate_pure_soft(...)` 每步大致流程：

```python
logits_original = outputs.logits[:, -1, :]
probs_original = softmax(logits_original)
raw_entropy = entropy(probs_original)

logits_filtered = apply_sampling_filter(...)
probs = softmax(logits_filtered)
next_tokens = argmax/prob_sample(probs)

soft_emb = probs_original @ E
normal_emb = E[next_tokens]
```

之后根据不同 mask 决定下一步输入 embedding：

```python
biased_soft_emb = soft_emb or image_pad_biased_soft_emb
format_emb = lambda * normal_emb + (1 - lambda) * biased_soft_emb

last_emb = biased_soft_emb
last_emb = where(format_mask, format_emb, last_emb)
last_emb = where(collapse_mask | answer_zone_mask, normal_emb, last_emb)
```

这意味着当前优先级实际是：

```text
collapse / answer_zone hard discrete
> format cooldown
> image_pad-biased soft
> pure soft
```

这个优先级合理，但目前没有被显式抽象成路由框架。

## 3. 已有信号清单

### 3.1 不确定性基础信号

| 信号 | 当前字段 / 代码 | 含义 |
|---|---|---|
| `raw_entropy` | token trace | 原始 logits 分布熵 |
| `filtered_entropy` | token trace | top-k/top-p 后分布熵 |
| `raw_top1_prob` | token trace | 原始 top1 概率 |
| `raw_margin` | token trace | 原始 top1 - top2 概率差 |
| `collapse_entropy_delta` | token trace | 当前熵相对 rolling history 的突升 |
| `output_tokens` | results | 推理长度，间接反映退化 |

### 3.2 token 类型信号

已有分析脚本 `analyze_spike_types.py` 把高熵 spike 分成：

| 类型 | 含义 |
|---|---|
| `visual_spike` | top-k 候选中视觉词质量较高，如颜色、位置、对象名 |
| `relation_spike` | 关系/推理连接词，如 because, therefore, but, if |
| `format_spike` | 标点、换行、括号、think/answer 等结构 token |
| `answer_spike` | A/B/C/D 或 answer 附近 |
| `diffuse_low_conf_spike` | top1 低或 margin 小，分布扩散 |
| `other` | 暂未分类 |

### 3.3 阶段信号

当前 image_pad bias 已支持：

| 阶段 | 代码参数 | 当前实验含义 |
|---|---|---|
| early | `0 <= step <= 128` | VStar 上危险，MMVP 上有收益 |
| mid | `129 <= step <= 512` | 当前最像通用安全收益区 |
| late | `step >= 513` | 安全但收益弱 |

### 3.4 退化信号

| 信号 | 当前参数 | 含义 |
|---|---|---|
| low-conf diffuse | `collapse_low_conf_tau`, `collapse_low_margin_tau` | top1 低或 margin 小 |
| entropy spike | `collapse_entropy_window`, `collapse_entropy_alpha` | 当前熵突升 |
| repeat degeneration | `collapse_require_repeat_degen`, `collapse_repeat_ngram`, `collapse_recent_repeat_tau` | 近期重复/退化 |
| format overuse | `format_cooldown_active_count`, `format_cooldown_max_active` | format cooldown 触发过多 |

## 4. 动作清单

当前实际有四类动作。

| 动作 | 实现 | 作用 | 风险 |
|---|---|---|---|
| `pure_soft` | `last_emb = soft_emb` | 保留概率分布隐式推理 | 容易长输出、格式退化、扩散 |
| `hard_discrete` | `last_emb = normal_emb` | 强制回到 argmax token embedding | 可能破坏本来有益的 soft 不确定性 |
| `format_cooldown` | 格式 token 附近若干步 hard/mixed discrete | 稳定结构、减少长输出和答案缺失 | 过宽会 damage 一些本来正确样本 |
| `image_pad_bias` | `(1-lambda)*soft_emb + lambda*E[<image_pad>]` | 轻量补视觉 anchor | early 阶段会破坏 VStar 推理 |

后续可考虑但尚未稳定验证的动作：

| 动作 | 设想 |
|---|---|
| `weak_visual_bias_mid_only` | 只在 mid 阶段加入小 lambda image_pad bias |
| `visual_bias_if_visual_state` | 只对视觉型不确定 token 加视觉 bias |
| `relation_soft_keep` | 对关系型不确定 token 保留 pure-soft，不 hard collapse |
| `format_hard_only` | 对格式型 token 使用 hard discrete/cooldown |
| `answer_zone_hard` | 进入答案区后 hard discrete，避免答案格式漂移 |

## 5. Router v0：第一版统一规则

第一版 router 不追求最优分数，而是追求规则清楚、可解释，并尽量复用已验证有效的动作。

### 5.1 优先级原则

高优先级动作覆盖低优先级动作：

```text
answer_zone_hard / collapse_hard
> format_cooldown_hard
> mid_visual_bias
> pure_soft
```

原因：

- `answer_zone` 和 `collapse` 都是“防退化/防答案丢失”的保护动作，应该覆盖视觉 bias。
- `format_cooldown` 已经证明是 VStar 上的核心稳定器。
- `image_pad_bias` 是轻量辅助，不能覆盖 hard discrete 保护。
- 默认仍为 pure-soft。

### 5.2 路由表

| route state | 判断条件 | action | 当前证据 |
|---|---|---|---|
| `answer_zone` | recent text 出现 `</think` 或 `answer` | `hard_discrete` | 单独 answer-zone 能修一批答案格式，但太晚，适合保护答案区 |
| `diffuse_repeat_degen` | entropy spike + low top1/margin + repeat gate + step>=64 | `hard_discrete` | `late64 + repeat_gate` damage 低，适合保守修复长输出/重复 |
| `format_uncertain` | token 是 format/high-risk format，cooldown2 | `hard_discrete` | VStar bestcombo 的主要收益来源之一 |
| `mid_visual_soft` | 129<=step<=512，且未被上面 hard route 覆盖 | `weak_image_pad_bias` | mid-only 全量 VStar/VisuLogic 较安全，damage 集上 `17/18` |
| `early_visual` | step<=128 | `no_visual_bias` | early damage 集上 λ=0.01 仍只 `10/18` |
| `late_visual` | step>=513 | `usually_noop` 或 very weak | damage 集安全但全量收益弱 |
| default | 无特殊信号 | `pure_soft` | 保留 soft 推理能力 |

### 5.3 当前不建议放进 v0 的规则

| 规则 | 暂不放入原因 |
|---|---|
| high entropy -> image_pad_bias | 已验证跨数据集不稳 |
| early image_pad_bias | VStar damage 很强 |
| full image_pad_bias | 主要伤害来自 early，VStar 明显掉点 |
| highrisk format only | VStar 上收益不足，普通格式 token 也重要 |
| mixed format embedding 替代 hard discrete | VStar 上不如 hard cooldown |

## 6. 需要补的 trace 字段

为了让方向一真正可分析，建议先做一次“无行为变化”的 trace 增强。也就是不改当前行为，只把路由状态显式记录下来。

建议新增字段：

| 字段 | 含义 |
|---|---|
| `generation_phase` | `early/mid/late` |
| `route_signal` | 当前 token 的主信号，如 `format_uncertain`, `diffuse_repeat_degen`, `mid_visual_soft` |
| `route_action` | 实际动作，如 `pure_soft`, `hard_discrete`, `format_cooldown`, `image_pad_bias` |
| `route_priority` | 被哪个高优先级规则覆盖 |
| `route_suppressed_by` | 例如 image_pad 被 format/collapse 覆盖 |
| `is_highrisk_format_token` | 区分普通 format 与高危结构 format |
| `repeat_degen_detected` | repeat gate 是否命中 |
| `diffuse_mask` | top1/margin 是否低置信扩散 |
| `entropy_spike_mask` | rolling entropy spike 是否命中 |
| `visual_bias_candidate` | 是否本来满足视觉 bias 条件 |

这些字段可以避免后续每次都从多个 bool 反推路由原因。

## 7. 推荐实施顺序

### Step 1：无行为变化的 route annotation

修改 `generate_pure_soft(...)`，只新增 route state 变量和 trace 字段，不改变 `last_emb` 的计算结果。

目标：

- 确认现在 bestcombo、image_pad phase gate 的行为能被统一解释；
- 后续所有实验都能直接统计 route 分布、覆盖关系、damage 来源。

验收：

- 在一个小 subset 上跑 old/new，结果和输出完全一致或只因 trace 字段变化而一致。
- full VStar 上 accuracy 与原 bestcombo 对齐。

### Step 2：写 route summary 分析脚本

输入一个 run_dir，输出：

- 每类 `route_signal` 的触发次数；
- 每类 `route_action` 的触发次数；
- correct/wrong 上的 route 分布；
- fixed/damaged 样本上的 route 分布；
- 每个样本 route count 与输出长度、是否答错的关系。

这一步可以基于现有 `analyze_bad_events.py` 改，不需要重新发明评估逻辑。

### Step 3：Router v0 实验

在 annotation 成功后，启动一个明确的 Router v0：

```text
format cooldown2
+ late64 repeat-gated collapse
+ mid-only image_pad_bias lambda=0.02 or 0.03
+ answer-zone hard discrete optional
```

关键不是先扫参数，而是看：

- route 之间是否互相覆盖过多；
- mid visual bias 是否主要命中原本 pure-soft 区域；
- damage 是否来自某个 route 优先级错误；
- 对 MMVP / VisuLogic / VStar 的 route 分布是否不同。

### Step 4：从规则路由走向 learned / adaptive routing

如果 v0 的 route annotation 能稳定解释结果，后续可以考虑更高级版本：

- 用 token trace 训练一个轻量 classifier，预测某步应使用 `pure_soft/hard/visual_bias`；
- 或者只学习阈值和阶段边界；
- 或者按数据集/问题类型选择不同 route profile。

但这一步必须建立在 route annotation 足够清楚之后。

## 8. 当前阶段的工作定义

方向一现在应先完成两件事：

1. **代码层**：把 `generate_pure_soft` 内部隐式的 mask/action 变成显式 route annotation，不改变行为。
2. **分析层**：写统一 route summary，让每次实验都能回答：

```text
这个样本为什么用了这个动作？
哪个 route 修复了它？
哪个 route 破坏了它？
不同数据集主要依赖哪些 route？
```

完成这两步后，再讨论 Router v0 是否要加入 mid visual bias，以及是否要把不同数据集分成不同 route profile。

## 9. 已完成的准备工作

本次已经完成第一批准备：

1. 新增了 route annotation 字段。

代码位置：

- `lead/generation_utils.py`

新增 trace 字段包括：

- `generation_phase`
- `route_signal`
- `route_action`
- `route_priority`
- `route_suppressed_by`
- `is_highrisk_format_token`
- `visual_bias_candidate`
- `visual_bias_effective`
- `entropy_spike_mask`
- `diffuse_mask`
- `repeat_degen_detected`

这些字段只用于记录，不改变当前 `last_emb` 的行为。

2. 新增了统一 route summary 脚本。

代码位置：

- `script/exp5_16/analyze_route_summary.py`

使用方式：

```bash
python script/exp5_16/analyze_route_summary.py \
  --run_dir /path/to/run_dir \
  --baseline_run_dir /path/to/baseline_run_dir \
  --output /path/to/route_summary.md \
  --output_json /path/to/route_summary.json
```

`--baseline_run_dir` 可选。提供后会把样本分为：

- `fixed`
- `damaged`
- `correct`
- `wrong`

并统计不同组的 route action / route signal / phase / suppressed route 分布。

3. 完成 smoke test。

测试目录：

- `/tmp/mlrm_route_smoke`

验证结果：

- `py_compile` 通过；
- `generate_pure_soft` 能正常跑完；
- `token_entropy_full.jsonl` 中新增字段正常出现；
- route summary 脚本能正常生成 `/tmp/mlrm_route_smoke/route_summary.md`。

一个 smoke 中的典型记录：

```text
route_signal=early_visual_bias
route_action=image_pad_bias
visual_bias_effective=True
```

当 format cooldown 覆盖 visual bias 时，会记录：

```text
route_signal=format_uncertain
route_action=format_cooldown
route_suppressed_by=['image_pad_bias_by_format']
visual_bias_effective=False
```

这说明后续已经可以直接分析：

```text
哪个 route 实际生效？
哪个 route 只是候选但被更高优先级覆盖？
fixed / damaged 样本分别主要由哪些 route 造成？
```

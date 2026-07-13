# Transition-Preserving Format Guard 待做实验

## 状态

- 优先级：高
- 类型：机制组合与控制变量修正
- 记录日期：2026-07-11
- 当前状态：待运行

## 为什么需要补做

历史实验已经运行过 `lead_format2`、`lead_guard`、`quota05_format2` 和
`quota05_guard`。这些方法都以标准 LEAD 为基础，因此包含第 0 步 soft 和
随后的 soft-to-normal transition。

但是，历史配置的 `format_cooldown_min_step` 全部为 0。format cooldown 会把
换行、标点、括号、`think`、`answer`、`option` 等 token 识别为格式边界，并
优先使用离散 token embedding。LEAD 的第 0 步通常正好生成 `<think>`、换行或
相关格式 token，因此 format cooldown 可能在 step 0 直接覆盖 soft route，破坏
我们希望保留的 early transition。

现有结果因此不能被简单解释为“transition 与 format guard 组合后的效果”，因为
两者可能在生成开头发生冲突。

历史结果审计还显示：

- `lead_initial_transition_only=true` 且 format/guard 开启的正式 run 数为 0。
- `method=lead` 且 `lead_format_cooldown=true` 的历史 run 中，
  `format_cooldown_min_step>0` 的数量为 0。

## 核心假设

> 先完整保留 early latent-to-discrete transition，再延迟启用 format cooldown 和
> late diffuse/repeat veto，可能同时保留早期轨迹重定向收益与后期输出稳定性。

已有 `lead_format2/guard` 的不稳定结果，可能部分来自 format guard 过早介入，
而不是 format 稳定机制本身无效。

## 方法定义

### 旧组合

```text
step 0 起：LEAD soft/normal route 与 format cooldown 同时生效
format route 优先级高于 soft route
```

### 新组合

```text
step 0-1：完整保留初始 soft 与 soft-to-normal transition
step 2/4 起：允许 format_cooldown2
step 64 起：允许 diffuse + repeat veto
```

建议名称：

- `transition_preserving_lead_format2`
- `transition_preserving_lead_guard`
- `transition_preserving_quota05_format2`
- `transition_preserving_quota05_guard`

## 最小实验矩阵

| run | 关键配置 | 目的 |
|---|---|---|
| `initial_transition_only` | 仅 early transition | transition 基准 |
| `lead_format2_min0` | format 从 step 0 生效 | 复现旧冲突配置 |
| `lead_format2_min2` | format 从 step 2 生效 | 保护 transition 后立即启用 |
| `lead_format2_min4` | format 从 step 4 生效 | 更宽 early protection zone |
| `lead_guard_min2` | min2 format + late veto | 无 quota 的完整保护 |
| `quota05_guard_min0` | 旧 quota guard | 旧组合基准 |
| `quota05_guard_min2` | quota + min2 format + late veto | 主候选 |
| `quota05_guard_min4` | quota + min4 format + late veto | timing 控制 |

## 数据集与模型顺序

第一阶段先跑 VStar full：

1. `R1-Onevision-7B`（非 RL）
2. `Vision-R1-7B`
3. `R1-Onevision-7B-RL`（历史强收益 checkpoint）

VStar 出现有效趋势后，再外推：

1. MMVP full，报告 sample accuracy 和 pair accuracy
2. VisuLogic300
3. RealWorldQA fixed200

## 固定生成口径

```text
--no-do_sample
--temperature 0.6
--top_p 0.95
--top_k 20
--seed 42
--max_new_tokens 1024
--cot_prompt_mode orign
--alpha 0.4
--max_switch_count 5
--window_size 128
--save_token_entropy
--save_full_token_entropy
--trace_topk 0
```

## 必须报告的指标

- accuracy，以及 MMVP pair accuracy
- fixed / damaged against COT
- fixed / damaged against standard LEAD
- 平均输出长度、`long>=256`、`maxed1024`
- failed extraction
- mean soft ratio 和 mean switch count
- step 0 soft 实际生效率
- step 0 被 format cooldown 覆盖的样本数
- format cooldown active count / sample
- diffuse/repeat veto count / sample

## 成功判据

满足以下任一项即说明延迟组合有价值：

1. `min2/min4` 明显优于 `min0`，且 step 0 soft 生效率恢复。
2. 相对标准 LEAD，fixed 大于 damaged，同时长度和抽取失败不恶化。
3. 相对 quota05，保留准确率并减少长输出、重复或 maxed 样本。

若 `min0` 与 `min2/min4` 基本相同，则说明 step 0 format 覆盖不是历史下降的主要
原因；若 `min2/min4` 仍普遍低于标准 LEAD，则 format guard 更适合 pure-soft，
不适合保留 transition 的 LEAD 路线。

## 解释边界

`initial_transition_only` 在 transition 完成后已经回到 normal COT，因此后续 format
cooldown 大部分时间不会改变路由。它与 format 的组合主要用于验证 step 0 冲突，
不是最终实用方法。真正有应用意义的组合是：

> early transition + sparse later soft + delayed format guard + late degeneration veto。

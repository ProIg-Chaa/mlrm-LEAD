# Early Token Divergence Summary

这份摘要对应详细审计文件：

```text
result/5-27/early_token_divergence_analysis_20260615.md
result/5-27/early_token_divergence_analysis_20260615.json
```

## 核心结论

Early token divergence 分析支持当前主线：`initial_transition` 改变最终答案时，差异往往不是出现在答案末尾，而是在 reasoning opening 的早期 wording / visual grounding 阶段就已经分叉。

最直接的证据来自 VStar 和 MMVP：

- VStar `COT wrong -> initial_transition fixed`：18 个 fixed 样本，median first divergence = 22 token；其中 8/18 在前 16 token 内分叉，10/18 在前 32 token 内分叉。
- VStar `initial_transition correct -> no_to_normal damaged`：13 个 damaged 样本，median first divergence = 23 token；其中 4/13 在前 8 token 内分叉，8/13 在前 32 token 内分叉。
- MMVP `COT wrong -> initial_transition fixed`：12 个 fixed 样本，median first divergence = 16 token；其中 6/12 在前 16 token 内分叉，11/12 在前 32 token 内分叉。
- MMVP `initial_transition correct -> no_to_normal damaged`：11 个 damaged 样本，median first divergence = 19 token；其中 4/11 在前 16 token 内分叉，8/11 在前 32 token 内分叉。

这说明 `soft -> normal transition` 的影响经常发生在第一句话或第一段视觉描述里，而不是后段 entropy spike 后临时修正答案。

## 代表现象

VStar 的若干 fixed 样本非常清楚：

- COT 通常以 `Okay, so I need to determine...` 开头，先复述任务，再逐步解释。
- `initial_transition_only` 常以 `The image shows...` 开头，更早进入图像实体和空间关系描述。
- 在关系题中，这种起手差异会直接改变空间锚定，例如左/右、物体相对位置、是否可见。

典型例子：

- `id=161`：问题是 small red car 在 baby carriage 左边还是右边。COT 的早期描述走向 right，最终答 B；initial transition 在前 16 token 已经写出 baby carriage，并在前 32 token 进入 `left of the baby carriage`，最终答 A。
- `id=168`：问题是 dog 在 blue bicycle 左边还是右边。COT 走向 left，initial transition 在前 32 token 已经建立 `blue bicycle` 和后续 dog 关系，最终改为 right。
- `id=135`：`initial_transition_only` 正确，但关闭 `to_normal transition` 后损坏；两者在第 3 个 token 就从 `The/Okay` 起手分叉，后续视觉路径不同，最终答案相反。

## 对机制假设的含义

这组分析强化了三个判断：

1. **轨迹锁定发生得很早。** 很多 fixed/damaged 样本在前 3-32 token 内已经分叉，说明关键不是后段答案格式，而是开头 reasoning trajectory。
2. **`to_normal transition` 是关键控制项。** 当保留 step0 soft 但去掉 `soft -> normal` 回切时，VStar/MMVP 都出现一批 damaged 样本，且早期文本也分叉。
3. **late entropy trigger 解释力不足。** 如果答案方向在第一段视觉描述里已经定了，中后段 entropy spike 再触发 soft 就已经偏晚，只能影响表述或格式，难以系统性改正轨迹。

## 限制

VisuLogic 的 `phase3_cross_dataset_minimal` 没有 full token trace，因此 `COT wrong -> initial_transition fixed` 只能使用完整输出文本开头做 qualitative fallback，不能像 VStar/MMVP 那样统计 first divergence token。VisuLogic timing curve 有 full trace，`step0 correct -> step16 damaged` 的 median first divergence = 38.5 token，说明在更长推理任务中分叉仍发生在较早 reasoning 段，但不一定压在前 16 token。

## 下一步建议

下一步不急着再调 quota。更有价值的是把这些代表样本做成论文里的机制图：

- 左侧放 COT / no_to_normal 的 early opening。
- 右侧放 initial_transition 的 early opening。
- 标出第一个不同 token、早期视觉实体/空间关系、最终答案变化。
- 配合 timing curve，说明为什么 `step0` 有效而 `step16/32` 变弱。

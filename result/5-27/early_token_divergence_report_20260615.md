# Early Token Divergence 分析报告

## 1. 为什么要做这次分析

前面的重跑实验已经得到一个比较稳定的现象：LEAD 的大部分收益可以由 `initial_transition_only` 复现，也就是只在生成开头做一次 `soft -> normal` transition，后面回到普通 CoT 推理。进一步的 timing 消融也显示，把 transition 从第 0 步推迟到第 1/2/4/16/32 步后，效果整体衰减。

这些结果支持一个新的机制假设：

> 多模态推理轨迹在生成极早期就被锁定；LEAD 的主要收益不是来自中后段 entropy-gated 动态触发，而是来自开头的 `soft -> normal` transition 改变了模型进入推理的路径。

但仅看 accuracy 还不够。我们还需要回答一个更直接的问题：

> 当 `initial_transition` 修正一个样本，或者去掉 `to_normal transition` 损坏一个样本时，输出轨迹到底是在什么时候开始不同的？是在答案末尾才变，还是在最早的 reasoning opening 就已经分叉？

所以这次分析的目的不是再调参，也不是重新跑模型，而是利用已有 full token trace 和结果文件，检查 fixed/damaged 样本的早期生成轨迹。

如果 fixed/damaged 样本在前几个 token 或第一句话就已经分叉，那么它就能支持 `early trajectory commitment`：模型不是后面临时改答案，而是在很早的位置选择了不同的视觉锚定和推理路径。

## 2. 分析了哪些样本

这次分析主要围绕两类样本。

第一类是 fixed 样本：

> baseline 错，但 `initial_transition_only` 对。

这类样本用来回答：`initial_transition` 修正答案时，是不是从生成开头就改变了 reasoning trajectory？

具体分析了：

- VStar：`cot_orign_greedy -> initial_transition_only`
- MMVP：`cot_orign_greedy -> initial_transition_only`
- VisuLogic：`cot_orign_greedy -> initial_transition_only`

第二类是 damaged 样本：

> `initial_transition_only` 对，但去掉某个关键机制后变错。

这类样本用来回答：关键控制项被移除时，是否也会在早期导致路径走偏？

具体分析了：

- VStar：`initial_transition_only -> initial_transition_no_to_normal`
- MMVP：`initial_transition_only -> initial_transition_no_to_normal`
- VStar timing：`transition_step0 -> transition_step16`
- VisuLogic timing：`transition_step0 -> transition_step16`

其中 `initial_transition_no_to_normal` 是关键控制：它保留第 0 步 soft，但关闭后续 `soft -> normal` 回切。这个对比用来区分“第 0 步 soft 本身”与“soft 后回到 normal 的 transition”。

## 3. 分析了哪些数据

这次分析没有重新推理模型，而是读取已有实验输出。

主要使用的数据包括：

- `results.jsonl`：每个样本的完整输出、答案、输出长度。
- `token_entropy_full.jsonl`：逐 token 的文本、entropy、mode 信息。
- MMVP / RealWorldQA 的 specialized evaluator rows：用于判断样本是否 correct。
- 本地答案抽取：用于 VStar / VisuLogic 的 correct/fixed/damaged 判断。

对每个 selected sample，提取了以下信息：

- `first token divergence`：两条输出 token 序列第一次不同的位置。
- 前 `1/2/4/8/16/32` 个 generated tokens。
- 前 32 token 的 entropy summary，包括 mean entropy、max entropy、soft token 数量、前几个 token 的 mode。
- `output_tokens`：输出长度。
- `answer marker position`：答案标记第一次或最后一次出现的位置。
- `reasoning opening`：开头 reasoning 的短摘录。
- `final answer region`：最终答案附近的短摘录。
- gold / ref prediction / current prediction。

输出文件：

```text
result/5-27/early_token_divergence_analysis_20260615.md
result/5-27/early_token_divergence_analysis_20260615.json
result/5-27/early_token_divergence_summary_20260615.md
```

分析脚本：

```text
script/exp5_27/analyze_early_token_divergence_v2.py
```

## 4. 发现的具体信息

### 4.1 VStar: COT wrong -> initial_transition fixed

这组共有 18 个 fixed 样本。

统计结果：

- median first divergence = 22 token
- 8/18 在前 16 token 内分叉
- 10/18 在前 32 token 内分叉

代表现象是，COT 常常以任务复述式开头：

```text
Okay, so I need to determine whether...
```

而 `initial_transition_only` 更常以图像描述式开头：

```text
The image shows...
```

这不是单纯语气变化。很多 VStar 样本是左右关系、相对位置、物体可见性问题。开头是否先进入正确视觉实体和空间关系，会直接影响最终答案。

典型样本：

- `id=161`：问题是 small red car 在 baby carriage 左边还是右边。COT 最终答 right，错；`initial_transition_only` 在前 16 token 已经提到 baby carriage，前 32 token 进入 `left of the baby carriage`，最终答 left，正确。
- `id=168`：问题是 dog 在 blue bicycle 左边还是右边。COT 最终走向 left，错；`initial_transition_only` 在前 32 token 已经建立 blue bicycle 与 dog 的空间关系，最终答 right，正确。
- `id=164`：问题是 yellow car 在 pool 左边还是右边。COT 走向 right，错；`initial_transition_only` 早期直接描述 pool 和 yellow car 的相对位置，最终答 left，正确。

这说明 `initial_transition` 修正 VStar 样本时，很多时候是在第一段视觉 grounding 里就改变了路径。

### 4.2 VStar: initial_transition correct -> no_to_normal damaged

这组共有 13 个 damaged 样本。

统计结果：

- median first divergence = 23 token
- 4/13 在前 8 token 内分叉
- 8/13 在前 32 token 内分叉

这组尤其重要，因为它直接验证 `to_normal transition` 的作用。`initial_transition_no_to_normal` 保留了第 0 步 soft，但关闭了 soft 后回到 normal 的 transition。如果只是第 0 步 soft 本身有效，那么关闭 `to_normal` 不应该造成明显损坏；但实际出现了 13 个 damaged 样本。

代表样本：

- `id=135`：`initial_transition_only` 正确，`no_to_normal` 错。两者第 3 个 token 就从不同起手分叉，后续视觉路径不同，最终答案相反。
- `id=140`、`id=168` 也呈现类似模式：早期 wording / visual grounding 改变，最终左右关系判断反转。

这说明关键不是“soft 一下”这么简单，而是 `soft -> normal` 的回切结构改变了模型进入普通推理时的初始状态。

### 4.3 MMVP: COT wrong -> initial_transition fixed

这组共有 12 个 fixed 样本。

统计结果：

- median first divergence = 16 token
- 6/12 在前 16 token 内分叉
- 11/12 在前 32 token 内分叉

MMVP 的证据比 VStar 更集中：大多数 fixed 样本在前 32 token 内已经分叉。也就是说，`initial_transition` 对 MMVP 的修正几乎都可以追溯到很早的生成阶段。

这支持一个重要判断：MMVP 上 full LEAD 与 `initial_transition_only` 表现接近，不是巧合。它们可能都主要依赖开头 transition 改变路径，而不是后续动态触发。

### 4.4 MMVP: initial_transition correct -> no_to_normal damaged

这组共有 11 个 damaged 样本。

统计结果：

- median first divergence = 19 token
- 4/11 在前 16 token 内分叉
- 8/11 在前 32 token 内分叉

这和 VStar 的 no_to_normal damage 结果一致：去掉 `to_normal transition` 后，不少样本会在非常早的位置走向不同的答案。

这进一步说明 `to_normal transition` 是跨数据集有效的关键控制项，而不是 VStar 特例。

### 4.5 VisuLogic 结果

VisuLogic 的 `phase3_cross_dataset_minimal` 没有 full token trace，因此 `COT wrong -> initial_transition fixed` 不能像 VStar/MMVP 那样统计 first divergence，只能看完整输出文本的开头。这是这次分析的一个限制。

不过 VisuLogic timing curve 有 full token trace。对 `transition_step0 correct -> transition_step16 damaged` 的样本：

- selected samples = 44
- median first divergence = 38.5 token

这个分叉位置比 VStar/MMVP 更晚，但仍然处在较早 reasoning 段，而不是答案末尾。VisuLogic 本身输出更长、推理更啰嗦，所以分叉点自然会后移。

这和 timing curve 的结果一致：`step0` 比 `step16` 更有效，因为等到第 16 步再做 transition 时，部分推理开头已经展开，轨迹已经开始锁定。

## 5. 最终结论

这次 early token divergence 分析支持以下结论。

第一，`initial_transition` 的作用发生在生成早期，而不是答案末尾。VStar 和 MMVP 的 fixed/damaged 样本大量在前 16-32 token 内分叉，代表样本也显示早期视觉实体和空间关系已经不同。

第二，`soft -> normal transition` 是关键机制。`initial_transition_no_to_normal` 的 damaged 样本说明，只保留第 0 步 soft 不够；soft 后回到 normal 的 transition 才是改变后续普通 CoT 轨迹的重要环节。

第三，late entropy-gated dynamic trigger 的解释力有限。如果一个样本在第一句话或第一段视觉 grounding 中已经选定方向，那么中后段 entropy spike 再触发 soft 往往已经太晚，更多只能影响表述、格式或局部细节。

第四，这组结果把前期 format 稳定路线和当前 early trajectory 路线区分开了。format 稳定解释的是 soft intervention 为什么会导致长输出、重复和答案漂移，以及如何修复这些退化；early transition 解释的是为什么 LEAD 能改变推理正确率。

因此，可以把当前机制叙事写成：

> Soft latent intervention 有两个效应。中后期 soft 容易带来 generation instability，需要 format/cooldown/guard 抑制；而真正带来推理收益的，是生成开头一次 `soft -> normal` transition，它改变了模型进入视觉推理的早期路径。多模态 CoT 后续的长推理，很多时候只是沿着这个早期路径展开。

# VStar 路由实验：late64+repeat_gate 与 format_cooldown8

日期：2026-05-19  
数据集：`data/vstar.jsonl`，全量 191 题  
模型：`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`  
解码：greedy，`cot_prompt_mode=orign`，`max_new_tokens=1024`

本文记录两次主要路由实验：

1. `pure-soft + late64 + repeat_gate`
2. `pure-soft + format_cooldown8`

这两次实验都以 pure-soft baseline 作为主要对照。

## 1. 背景

前序实验发现，pure-soft 在 VStar 上容易出现长输出、重复、答案格式漂移和答案抽取失败。尤其是在高熵、低置信、top1-top2 margin 很低的扩散型 token 上，继续使用 soft embedding 会把模型带入退化轨道。

因此我们尝试两类路由：

- `late64 + repeat_gate`：针对低置信扩散退化，只在中后期且已有重复退化苗头时，把下一步输入从 soft embedding 改为 discrete embedding。
- `format_cooldown8`：针对格式 token，命中格式 token 后接下来 8 步使用 discrete embedding，验证格式/结构 token 附近的 soft 是否会破坏生成稳定性。

## 2. 对照基线

pure-soft baseline 实验目录：

`output/experiments/20260517_210348/pure_soft_collapse_vstar_full_parallel/pure_soft_baseline_gpu0`

结果：

| 方法 | 正确率 | 平均长度 | p90 长度 | 长输出 >=256 | 打满 1024 |
|---|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 237.29 | 1023 | 33 | 18 |

baseline 的主要问题：

- 很多错误不是简单视觉判断错误，而是输出退化。
- `p90=1023`，说明尾部样本大量接近最大长度。
- 打满 1024 的样本有 18 个。
- 答案格式异常较多：
  - `empty_paren_answer`: 19
  - `missing_answer_marker`: 40

## 3. 实验一：pure-soft + late64 + repeat_gate

### 3.1 实验设置

实验目录：

`output/experiments/20260518_200744/pure_soft_collapse_late64_repeat_gate_vstar_full`

运行目录：

`output/experiments/20260518_200744/pure_soft_collapse_late64_repeat_gate_vstar_full/late64_repeat_gate_gpu0`

脚本：

`script/exp5_16/run_pure_soft_collapse_late64_repeat_gate_vstar_full.sh`

路由逻辑：

```text
当前 token 是低置信扩散 entropy spike
AND step >= 64
AND 最近输出已有重复退化迹象
=> 下一步输入使用 discrete token embedding，而不是 soft embedding
```

参数：

```text
collapse_entropy_window = 16
collapse_entropy_alpha = 2.0
collapse_min_history = 4
collapse_min_entropy = 1.0
collapse_low_conf_tau = 0.20
collapse_low_margin_tau = 0.05
collapse_min_step = 64
collapse_require_repeat_degen = true
collapse_repeat_ngram = 3
collapse_recent_repeat_window = 32
collapse_recent_repeat_tau = 0.35
```

### 3.2 结果

对比命令：

```bash
bash output/experiments/20260518_200744/pure_soft_collapse_late64_repeat_gate_vstar_full/compare_after_done.sh
```

| 方法 | 正确率 | 平均长度 | p90 长度 | 长输出 >=256 | 打满 1024 |
|---|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 237.29 | 1023 | 33 | 18 |
| late64 + repeat_gate | 119/191 = 62.30% | 193.44 | 322 | 25 | 13 |

变化：

```text
changed = 34
fixed = 8
damaged = 1
net = +7
```

触发统计：

```text
collapse 总次数 = 71
有触发样本 = 36/191
mean = 0.37
median = 0
p90 = 1
max = 12
```

fixed 样本：

```text
[30, 92, 115, 123, 128, 129, 148, 184]
```

damaged 样本：

```text
[73]
```

### 3.3 样本级观察

fixed 的 8 个样本几乎都有同一个特点：baseline 长输出退化，导致答案抽取失败；路由后长度恢复到正常范围，答案可抽取且正确。

| id | 问题简述 | baseline 长度 | route 长度 | 触发步 |
|---:|---|---:|---:|---|
| 30 | pet collar color | 1011 | 189 | 105 |
| 92 | messenger bag color | 1022 | 187 | 86 |
| 115 | telephone vs hand lamp | 1024 | 180 | 75, 108, 129 |
| 123 | orange luggage vs purple umbrella | 1023 | 310 | 129, 130 |
| 128 | umbrella vs traffic light | 1024 | 186 | 115, 130 |
| 129 | cyclist vs handbag | 1024 | 168 | 89 |
| 148 | shovel vs house | 1021 | 196 | 73, 107 |
| 184 | scooter vs cyclist | 1024 | 223 | 146 |

damaged 的唯一样本：

| id | 问题 | 标准答案 | baseline | route | 触发步 |
|---:|---|---|---|---|---|
| 73 | cleaning cloth color | B | 正确，长度 291 | 抽取失败，长度 214 | 122 |

这个 damaged 样本在普通 collapse、late64、repeat_gate 和组合路由中都会被破坏，说明它是当前低置信扩散路由家族的共同坏例。触发点附近是视觉描述词，并不是明显格式区错误。

### 3.4 与单独 gate 对比

| 方法 | 正确率 | changed | fixed | damaged | net |
|---|---:|---:|---:|---:|---:|
| 普通 collapse | 114/191 = 59.69% | 141 | 23 | 21 | +2 |
| late64 | 119/191 = 62.30% | 55 | 12 | 5 | +7 |
| repeat_gate | 119/191 = 62.30% | 43 | 10 | 3 | +7 |
| late64 + repeat_gate | 119/191 = 62.30% | 34 | 8 | 1 | +7 |

组合路由的正确率没有超过单独 `late64` 或 `repeat_gate`，但它显著减少了损伤：

```text
damaged: 5 / 3 -> 1
changed: 55 / 43 -> 34
```

这说明组合 gate 更保守、更精确。

### 3.5 实验一结论

`late64 + repeat_gate` 是目前最干净的低置信扩散路由：

```text
低置信扩散信号是有效的；
但只有结合生成阶段和退化迹象，才能避免大量误伤。
```

它的作用主要是防止 pure-soft 进入长输出/重复/不可抽取的坏轨道，而不是直接提升视觉识别能力。

## 4. 实验二：pure-soft + format_cooldown8

### 4.1 实验设置

实验目录：

`output/experiments/20260519_181159/pure_soft_format_cooldown_vstar_full`

运行目录：

`output/experiments/20260519_181159/pure_soft_format_cooldown_vstar_full/format_cooldown8_gpu0`

脚本：

`script/exp5_16/run_pure_soft_format_cooldown_vstar_full.sh`

路由逻辑：

```text
如果当前 token 是格式 token：
    接下来 8 步使用 discrete token embedding
否则：
    默认使用 pure-soft embedding
```

当前格式 token 规则来自离线 spike 分析脚本中的 `FORMAT_WORDS`，包括：

```text
换行、空白、标点、冒号、括号、尖括号、think、answer、option、星号等
```

主要参数：

```text
pure_soft_format_cooldown = true
format_cooldown_steps = 8
```

注意：本实验不叠加低置信扩散 collapse，目的是单独验证格式信号。

### 4.2 结果

对比命令：

```bash
bash output/experiments/20260519_181159/pure_soft_format_cooldown_vstar_full/compare_after_done.sh
```

| 方法 | 正确率 | 平均长度 | p90 长度 | 长输出 >=256 | 打满 1024 |
|---|---:|---:|---:|---:|---:|
| pure-soft baseline | 112/191 = 58.64% | 237.29 | 1023 | 33 | 18 |
| format_cooldown8 | 136/191 = 71.20% | 121.28 | 225 | 12 | 0 |

变化：

```text
changed = 191
fixed = 37
damaged = 13
net = +24
```

fixed 样本：

```text
[5, 11, 14, 17, 19, 23, 30, 32, 38, 41, 76, 85, 92, 95, 98, 99,
 107, 123, 128, 130, 131, 144, 146, 148, 156, 158, 162, 164, 172,
 173, 176, 178, 181, 184, 186, 188, 190]
```

damaged 样本：

```text
[25, 34, 49, 51, 62, 75, 116, 120, 135, 142, 154, 175, 189]
```

### 4.3 格式异常变化

| 指标 | baseline | format_cooldown8 |
|---|---:|---:|
| `empty_paren_answer` | 19 | 0 |
| `multiple_answer_lines` | 0 | 0 |
| `missing_answer_marker` | 40 | 17 |
| 打满 1024 | 18 | 0 |

这些指标说明，格式 cooldown 显著减少了答案格式漂移和长输出退化。

### 4.4 触发强度

format cooldown 的触发非常频繁：

```text
format_cooldown 总次数 = 17692
有触发样本 = 191/191
mean = 92.63
median = 75
p90 = 176
max = 276
```

因此，这个实验不能简单解释为“只在格式不确定时轻微修正”。它更像是：

```text
在大量格式/结构 token 附近强行局部离散化，使 pure-soft 变成一种半离散推理。
```

这也解释了为什么：

- 正确率提升很大。
- 所有样本输出都发生变化。
- damaged 也不少，有 13 个。

### 4.5 与其他方法对比

| 方法 | 正确率 |
|---|---:|
| pure-soft baseline | 112/191 = 58.64% |
| pure-soft + late64/repeat_gate | 119/191 = 62.30% |
| CoT | 131/191 = 68.59% |
| pure-soft + format_cooldown8 | 136/191 = 71.20% |
| LEAD | 139/191 = 72.77% |

`format_cooldown8` 已经非常接近 LEAD，并超过了 CoT。这说明 pure-soft 的大量错误确实来自生成稳定性问题，而不是纯视觉能力问题。

### 4.6 实验二结论

格式路由很有潜力，但当前版本太强、太宽。

当前结果支持：

```text
格式/结构 token 附近的 soft embedding 对 pure-soft 稳定性影响很大；
强格式 cooldown 能显著提升正确率，并消除打满 1024 的长输出退化。
```

但也要注意：

```text
format_cooldown8 不是精确路由，而是高频半离散化。
```

下一步需要做更细粒度消融：

- `format_cooldown2`
- `format_cooldown4`
- `format_cooldown8`
- `answer_zone_discrete`
- `format_uncertain_only`

目标是保留 format route 的收益，同时减少 `changed=191` 和 `damaged=13`。

## 5. 两类路由的对比理解

| 路由 | 主要作用 | 优点 | 问题 |
|---|---|---|---|
| late64 + repeat_gate | 防止中后期低置信扩散导致长输出/重复退化 | 精确，damaged 低 | 提升幅度有限，只修退化型错误 |
| format_cooldown8 | 在格式/结构 token 附近局部离散化 | 提升大，消除 1024 长输出 | 触发过宽，changed 全部样本，damaged 较多 |

可以这样理解：

```text
late64 + repeat_gate 是“止损路由”；
format_cooldown8 是“稳定化路由”。
```

前者适合和其他方法组合，因为它保守、损伤低。  
后者潜力更大，但需要先收窄触发范围。

## 6. 对后续组合实验的启示

不建议立刻把当前 `format_cooldown8` 直接和 `late64 + repeat_gate` 混合，因为 `format_cooldown8` 已经覆盖所有样本、所有阶段，组合后很难判断收益来自哪里。

更推荐先做：

```text
format_cooldown2
format_cooldown4
answer_zone_discrete
format_uncertain_only
```

然后再尝试：

```text
late64 + repeat_gate + 精简版 format route
```

更理想的最终路由形态可能是：

```text
低置信扩散 + 中后期 + 重复退化 -> discrete
答案区 / Answer 区 / </think> 后 -> discrete
格式 token 高熵且低 margin -> short cooldown
其他 token -> 保持 soft 或 LEAD 原逻辑
```

## 7. 相关代码位置

### late64 + repeat_gate

核心代码：

`lead/generation_utils.py`

相关参数：

```text
--pure_soft_collapse_on_diffuse
--collapse_min_step 64
--collapse_require_repeat_degen
--collapse_repeat_ngram 3
--collapse_recent_repeat_window 32
--collapse_recent_repeat_tau 0.35
```

脚本：

`script/exp5_16/run_pure_soft_collapse_late64_repeat_gate_vstar_full.sh`

### format_cooldown8

核心代码：

`lead/generation_utils.py`

新增在线格式判断：

```text
_is_format_token_text(...)
```

相关参数：

```text
--pure_soft_format_cooldown
--format_cooldown_steps 8
```

脚本：

`script/exp5_16/run_pure_soft_format_cooldown_vstar_full.sh`


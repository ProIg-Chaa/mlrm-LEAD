# MMHal 小样本实验强峰值 Token 分析

## 1. 实验范围

- 实验目录：
  - `output/experiments/20260427/mmhal_balanced_entropy_curves_154341`
- 方法：
  - `cot`
  - `lead`
- 数据：
  - MMHal-Bench 小样本子集
  - 每类题目抽取 2 个样本，共 16 个样本
- 熵指标：
  - 使用 `raw_entropy`
- 强峰值定义：

\[
H_t > \mu_{\text{sample}} + 1.5 \sigma_{\text{sample}}
\]

其中 `μ_sample` 和 `σ_sample` 都是**在单个样本、单次独立推理过程内**，仅对 `is_reasoning_token=True` 的 token 计算。

## 2. 总体结论

这次小样本实验里，强峰值 token 的主体并不是推理关系词，而是以下几类：

1. 视觉内容词 / 实体词
   - 例如：`black`, `red`, `silver`, `sign`, `poster`, `backpack`, `garage`, `yacht`, `rope`
2. 空间或描述性词
   - 例如：`positioned`, `arranged`, `evenly`, `clear`, `wearing`
3. 生成不稳定或格式异常 token
   - 例如：`,`, `-br`, `"`, `Ground`, `Including`
   - 这类现象在 `lead` 中更明显，主要出现在少数长推理样本

关系词在强峰值中确实略有富集，但占比仍然不高，说明：

- “关系词平均熵偏高”是成立的；
- 但“极端强峰值主要由关系词构成”在这次小样本实验中并不成立。

## 3. 总体统计

### 3.1 COT

- reasoning tokens：`2725`
- 强峰值 tokens：`255`
- 强峰比例：`9.3578%`
- 全部 reasoning token 中关系词占比：`2.1284%`
- 强峰 token 中关系词占比：`3.1373%`
- 强峰 token 中 soft token 占比：`0.0%`

### 3.2 LEAD

- reasoning tokens：`4012`
- 强峰值 tokens：`350`
- 强峰比例：`8.7238%`
- 全部 reasoning token 中关系词占比：`0.9472%`
- 强峰 token 中关系词占比：`1.4286%`
- 强峰 token 中 soft token 占比：`12.0%`

## 4. 强峰值 token 高频统计

### 4.1 COT 高频强峰 token

按归一化 token 计数，前 20 项如下：

- `a` 16
- `the` 8
- `and` 7
- `red` 4
- `building` 4
- `sign` 4
- `no` 4
- `black` 4
- `on` 4
- `is` 4
- `in` 4
- `street` 3
- `ground` 3
- `silver` 3
- `parked` 3
- `key` 3
- `clear` 3
- `with` 3
- `wearing` 3
- `rear` 2

其中更有语义信息量的强峰词主要有：

- `star`
- `staircase`
- `clear`
- `black`
- `described`
- `backpack`
- `poster`
- `positioned`
- `yacht`
- `garage`
- `road`

### 4.2 LEAD 高频强峰 token

按归一化 token 计数，前 20 项如下：

- `the` 20
- `a` 14
- `is` 12
- `and` 11
- `.` 9
- `,` 9
- `on` 8
- `in` 7
- `cyclist` 6
- `this` 6
- `street` 5
- `red` 5
- `building` 5
- `presence` 5
- `moving` 5
- `left` 4
- `parked` 4
- `wet` 4
- `.\n\n` 4
- `ground` 3

更有语义信息量的强峰词主要有：

- `model`
- `rope`
- `evenly`
- `stand`
- `large`
- `website`
- `glowing`
- `arranged`
- `black`
- `wheels`
- `facing`
- `riding`

## 5. 关系词在强峰中的表现

### 5.1 COT

强峰中的关系词：

- `also` 2
- `therefore` 2
- `first` 1
- `but` 1
- `so` 1
- `while` 1

对应类别统计：

- `sequence` 3
- `conclusion` 2
- `contrast` 2
- `causal_condition` 1

### 5.2 LEAD

强峰中的关系词：

- `therefore` 2
- `first` 1
- `also` 1
- `since` 1

对应类别统计：

- `sequence` 2
- `conclusion` 2
- `causal_condition` 1

### 5.3 解释

关系词在强峰里不是主力，但相对其在全体 reasoning token 中的占比，确实有轻微富集：

- COT：从 `2.1284%` 上升到 `3.1373%`
- LEAD：从 `0.9472%` 上升到 `1.4286%`

这说明关系词更容易进入高熵区域，但极端峰值更多还是落在视觉内容词和局部描述词上。

## 6. 最高熵强峰示例

### 6.1 COT：按绝对熵值排序

前 12 个强峰 token：

1. `i`，entropy=`5.28125`，sample=`12`，step=`136`
2. `star`，entropy=`4.8125`，sample=`8`，step=`84`
3. `staircase`，entropy=`4.8125`，sample=`10`，step=`124`
4. `clear`，entropy=`4.78125`，sample=`9`，step=`79`
5. `black`，entropy=`4.65625`，sample=`1`，step=`65`
6. `"`，entropy=`4.5`，sample=`10`，step=`80`
7. `described`，entropy=`4.4375`，sample=`1`，step=`74`
8. `backpack`，entropy=`4.40625`，sample=`11`，step=`122`
9. `sign`，entropy=`4.40625`，sample=`12`，step=`210`
10. `serene`，entropy=`4.3125`，sample=`1`，step=`131`
11. `poster`，entropy=`4.28125`，sample=`0`，step=`85`
12. `silver`，entropy=`4.25`，sample=`1`，step=`59`

### 6.2 LEAD：按绝对熵值排序

前 12 个强峰 token：

1. `is`，entropy=`6.28125`，sample=`5`，step=`134`，`soft`
2. `,`，entropy=`6.09375`，sample=`13`，step=`156`，`soft`
3. `Ground`，entropy=`5.71875`，sample=`13`，step=`154`，`soft`
4. `,`，entropy=`5.40625`，sample=`13`，step=`153`，`soft`
5. `model`，entropy=`5.09375`，sample=`5`，step=`64`
6. `clear`，entropy=`4.78125`，sample=`9`，step=`79`
7. `rope`，entropy=`4.78125`，sample=`15`，step=`100`
8. `evenly`，entropy=`4.65625`，sample=`10`，step=`52`
9. `stand`，entropy=`4.625`，sample=`1`，step=`60`
10. `Including`，entropy=`4.625`，sample=`13`，step=`157`，`soft`
11. `large`，entropy=`4.59375`，sample=`1`，step=`63`
12. `website`，entropy=`4.53125`，sample=`5`，step=`150`

## 7. 按超阈值幅度（margin）排序

定义：

\[
\text{margin}_t = H_t - (\mu_{\text{sample}} + 1.5\sigma_{\text{sample}})
\]

这个指标比单纯看熵值更合理，因为它考虑了每个样本自己的熵基线。

### 7.1 COT top margin

前 12 项：

1. `i`，margin=`2.8240`
2. `star`，margin=`2.3336`
3. `clear`，margin=`2.0688`
4. `sign`，margin=`1.9490`
5. `black`，margin=`1.8662`
6. `positioned`，margin=`1.8172`
7. `yacht`，margin=`1.7547`
8. `garage`，margin=`1.6813`
9. `backpack`，margin=`1.6666`
10. `described`，margin=`1.6474`
11. `qr`，margin=`1.6365`
12. `poster`，margin=`1.5508`

### 7.2 LEAD top margin

前 12 项：

1. `,`，margin=`4.7813`，`soft`
2. `Ground`，margin=`4.4063`，`soft`
3. `,`，margin=`4.0938`，`soft`
4. `is`，margin=`3.5578`，`soft`
5. `Including`，margin=`3.3126`，`soft`
6. `br`，margin=`3.0626`，`soft`
7. `black`，margin=`2.8050`
8. `,`，margin=`2.7501`，`soft`
9. `which`，margin=`2.7188`，`soft`
10. `glass`，margin=`2.7112`
11. `which`，margin=`2.6876`，`soft`
12. `joint`，margin=`2.6251`，`soft`

## 8. 样本层面观察

### 8.1 COT 中强峰较多的样本

前 8 个样本：

- sample 15, id 12：361 reasoning tokens，35 peaks
- sample 12, id 7：356 reasoning tokens，31 peaks
- sample 8, id 5：329 reasoning tokens，29 peaks
- sample 13, id 15：230 reasoning tokens，23 peaks
- sample 1, id 9：200 reasoning tokens，20 peaks
- sample 11, id 14：219 reasoning tokens，19 peaks
- sample 0, id 1：162 reasoning tokens，18 peaks
- sample 4, id 2：121 reasoning tokens，16 peaks

### 8.2 LEAD 中强峰较多的样本

前 8 个样本：

- sample 11, id 14：1024 reasoning tokens，105 peaks
- sample 13, id 15：1024 reasoning tokens，75 peaks
- sample 10, id 6：397 reasoning tokens，35 peaks
- sample 8, id 5：269 reasoning tokens，20 peaks
- sample 5, id 10：195 reasoning tokens，16 peaks
- sample 0, id 1：132 reasoning tokens，15 peaks
- sample 1, id 9：147 reasoning tokens，13 peaks
- sample 9, id 13：131 reasoning tokens，13 peaks

这里可以看到，LEAD 中的两个 1024-token 样本贡献了大量强峰值，而且其中包含大量格式异常或重复生成痕迹。这说明 LEAD 的强峰统计容易被少数长尾退化样本放大。

## 9. 阶段性结论

这次小样本实验可以得到几个明确结论：

1. 强峰值 token 的主体不是关系词，而是视觉内容词、实体词、局部描述词。
2. 关系词在强峰里有轻微富集，但远不是主要来源。
3. COT 的强峰更像“视觉证据落地时的高不确定词”。
4. LEAD 的强峰除了视觉内容词，还混入了较多 soft 模式下的格式异常 token。
5. 如果研究目标是“关系词是否是高熵关键点”，仅看全局强峰还不够，应该进一步看：
   - 关系词本体的高熵排序
   - 关系词前后窗口的局部熵峰值
   - 关系词附近 soft token 的触发情况

## 10. 建议的下一步

下一步更值得做的不是继续扩大“全局强峰”列表，而是针对关系词做更聚焦的分析：

1. 统计所有关系词本体的熵分布
2. 统计关系词前后 `±3` 或 `±5` token 的熵窗口
3. 比较 `cot` 与 `lead` 在关系词邻域内的平均熵曲线
4. 排除 1024-token 的明显退化样本后再重复统计，避免长尾异常样本污染结论

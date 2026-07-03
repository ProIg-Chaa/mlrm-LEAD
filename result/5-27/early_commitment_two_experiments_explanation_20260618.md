# Early Trajectory Commitment：两次机制实验的统一说明

## 核心问题

这两次实验都围绕同一个机制假设：

> 多模态推理的答案方向在生成极早期就被锁定。LEAD 的主要收益来自开头的 `soft -> normal` transition 对早期推理轨迹的重定向，而不是中后段 entropy-gated 动态触发。

更直观地说，模型在回答视觉问题时，往往不是“先完整推理，最后再决定答案”，而是很早就形成一个视觉叙述或关系判断。后面的长 reasoning 很多时候是在展开、包装、合理化这个早期判断。

为了验证这个主线，我们做了两步：

1. **Early Token Divergence Analysis**
   
   先看 COT 和 initial transition 的轨迹是不是很早就分叉。

2. **Early Prefix Replay**
   
   再把早期 prefix 强制塞回模型，验证这些早期 token 是否真的能因果影响最终答案。

第一步是诊断/相关性分析，第二步是更强的因果干预实验。两者合在一起，构成“早期轨迹承诺”的主要证据链。

## 实验一：Early Token Divergence Analysis

### 为什么做

在前面的实验里，我们发现 `initial_transition_only` 在 VStar/MMVP 上能逼近 full LEAD 的大部分收益，而 LEAD 的中后段动态触发贡献并不稳定。

这引出了一个问题：

> initial transition 到底是在什么时候改变了模型？是最后答案变了，还是一开始的视觉叙述/推理方向就变了？

如果轨迹只是在最后答案处不同，那说明 early transition 可能只是影响了答案格式或最终选择。  
但如果两条轨迹在前几十个 token 内就不同，而且这种不同已经涉及关键视觉事实、空间关系或答案方向，那就说明推理路径很早就被改写了。

所以这次分析的目标是：

> 找出 COT 与 initial-transition 轨迹第一次分叉的位置，并观察分叉处是否已经包含关键视觉判断。

### 分析了哪些样本

主要分析两类 pairwise comparison。

第一类：

```text
COT wrong -> initial_transition fixed
```

含义是：普通 COT 答错，但 `initial_transition_only` 答对。

这类样本用于分析：

> early transition 修复答案时，修复是不是发生在很早的 token 阶段？

第二类：

```text
initial_transition correct -> no_to_normal damaged
```

含义是：`initial_transition_only` 答对，但去掉 `to_normal transition` 后答错。

这类样本用于分析：

> 关键贡献是不是来自 `soft -> normal` transition，而不只是第 0 步 soft 或其它表面因素？

### 具体方法

对每个样本，我们拿到两条完整生成轨迹：

- 一条是 baseline 或 ablation 的轨迹。
- 一条是 initial-transition 相关轨迹。

然后做 token-level 对齐和比较。

具体抽取：

1. **前 1/2/4/8/16 个 generated tokens**
   
   观察最早期文本是否已经出现不同的视觉描述、空间关系判断、对象存在性判断或答案倾向。

2. **first divergence position**
   
   计算两条轨迹第一次 token 不同的位置。
   
   例如前 15 个 token 完全一样，第 16 个 token 开始不同，则 first divergence position 是 16。

3. **token entropy summary**
   
   汇总早期 token entropy，用来观察错误轨迹是不是“低熵、高置信”地走向错误，而不是一直处于不确定状态。

4. **first answer marker position**
   
   记录第一次出现 `Answer:`、`\boxed{}` 或明显选项标记的位置。
   
   这个指标用于判断答案方向是不是在正式答案标记出现之前就已经由前文 reasoning 决定。

5. **output length**
   
   记录输出长度，避免把现象误判成单纯的长输出、短输出或格式稳定性问题。

### 指标解释

#### first divergence position

这是最核心的指标之一。

它衡量两条推理轨迹在 token 层面多早分叉。

- 数值越小，说明两条轨迹越早不同。
- 如果大量样本在前 16 或 32 token 内分叉，说明答案方向可能在极早期就被重定向。

它不是 accuracy 指标，而是机制指标。

#### divergence before 16/32 tokens

这是 first divergence position 的聚合统计。

例如：

```text
10/18 before 32
```

表示 18 个样本中有 10 个样本，两条轨迹在前 32 个 generated tokens 内已经不同。

这个指标直观回答：

> 分叉是不是集中发生在开头？

#### token entropy summary

entropy 用来衡量模型在每一步 token 选择上的不确定性。

这里关注的不是“entropy 越低越好”，而是：

> 错误答案是不是也可能是低熵、高置信生成出来的？

如果错题低熵、高置信，而且很长，说明模型不是在后面犹豫，而是早期选错后一路自信展开。

#### first answer marker position

这个指标帮助区分两种情况：

1. 模型一开始就直接给答案。
2. 模型先形成视觉叙述/推理方向，后面才写答案。

如果答案标记很晚才出现，但前面视觉描述已经分叉，说明真正的答案方向可能早于正式答案文本形成。

### 具体数据

VStar：

| comparison | 样本数 | median first divergence | before 16 | before 32 |
|---|---:|---:|---:|---:|
| COT wrong -> initial_transition fixed | 18 | 22 | 8/18 | 10/18 |
| initial_transition correct -> no_to_normal damaged | 13 | 23 | NA | 8/13 |

MMVP：

| comparison | 样本数 | median first divergence | before 16 | before 32 |
|---|---:|---:|---:|---:|
| COT wrong -> initial_transition fixed | 12 | 16 | NA | 11/12 |
| initial_transition correct -> no_to_normal damaged | 11 | 19 | NA | 8/11 |

这些数字说明：很多关键样本不是到最后答案处才不同，而是在前 16 到 32 个 token 内就已经进入不同轨迹。

### 观察到的典型现象

以“图中是否能看到字母 J”这类问题为例：

COT 错误轨迹早期会描述：

```text
The image shows ... keys "U", "I", "O" ...
The key "J" is not visible ...
```

initial-transition 修复轨迹早期会描述：

```text
The image shows ... keys include ... J ...
The key with the letter "J" is clearly visible ...
```

这类差异不是最后答案格式不同，而是第一段视觉事实描述已经不同。  
一旦模型早期说“J 不可见”，后面就自然走向 “No”；一旦早期说“J 可见”，后面就自然走向 “Yes”。

### 实验一结论

Early Token Divergence Analysis 支持三个判断：

1. initial transition 的影响发生得非常早，常见于前 16 到 32 个 token。
2. 差异往往不是格式差异，而是关键视觉叙述、空间关系或对象存在性判断的差异。
3. 错误轨迹经常表现为早期低熵、高置信地选错方向，后面只是继续展开。

但是，这个实验仍然主要是相关性证据。  
它说明“早期分叉与最终答案变化同时出现”，但还不能单独证明“早期分叉导致最终答案变化”。

因此需要第二个实验。

## 实验二：Early Prefix Replay

### 为什么做

实验一已经看到两条轨迹很早就分叉，但还有一个更强的问题：

> 如果我们人为强制模型走某条轨迹的早期 prefix，后续答案会不会也跟着走？

如果答案真的高度 path-dependent，那么：

- 强制 COT 错误 prefix 后，即使后面切回普通 greedy，模型也应该继续错。
- 强制 initial-transition 修复 prefix 后，即使后面切回普通 greedy，模型也应该更容易继续对。

这就是 Early Prefix Replay 的核心。

它不再只是观察已有轨迹，而是直接干预早期 token。

### 分析了哪些样本

样本选择仍然聚焦在最有信息量的一类：

```text
COT wrong -> initial_transition fixed
```

也就是普通 COT 答错，但 `initial_transition_only` 答对。

使用样本：

- VStar：18 个
- MMVP：12 个

这些样本适合做 replay，因为同一道题已经存在两条明确不同的轨迹：

- COT 错误轨迹
- initial-transition 修复轨迹

### 具体方法

对每个样本，先从两条已有轨迹中截取早期 generated tokens。

两种 prefix 来源：

1. `cot_prefix`
   
   来自普通 COT 错误轨迹。

2. `initial_transition_prefix`
   
   来自 initial-transition 修复轨迹。

四种 prefix 长度：

```text
8 / 16 / 32 / 64 tokens
```

然后进行 replay：

```text
图像 + 问题
  -> 强制模型先生成指定来源的前 N 个 token
  -> 之后完全切回普通 greedy COT
  -> 评估最终答案
```

关键点是：后续生成不再使用 LEAD，也不再使用 soft intervention。  
唯一改变是最开头强制写入哪条轨迹的 prefix。

所以这个实验控制得比较干净：

- 图像相同
- 问题相同
- 模型相同
- continuation decoding 相同
- 只改变前 N 个 generated tokens 的来源

### 指标解释

#### accuracy

最终答案是否正确。

这里的 accuracy 不是在全量数据集上算，而是在“COT 错、initial_transition 对”的子集上算。  
因此它衡量的是：

> replay 后能不能保留或破坏 initial transition 原本修复的样本。

#### pair accuracy

MMVP 有 pair 结构，所以除了 sample-level accuracy，还报告 pair accuracy。

pair accuracy 要求一对相关样本都答对，才算这个 pair 正确。  
它比普通 sample accuracy 更严格，能反映模型是否稳定掌握该组视觉概念。

#### prefix length

prefix length 表示强制 replay 的早期 token 数量。

- 8 token：非常短，通常还在模板化开头或刚进入视觉描述。
- 16 token：可能刚开始描述主体。
- 32 token：通常已经覆盖第一段视觉判断的一部分。
- 64 token：通常已经包含较完整的早期视觉叙述或关系判断。

prefix length 的变化可以回答：

> 正确轨迹需要多长的早期文本才能被稳定 replay？

### 具体数据

| dataset | run | n | acc | pair acc | correct/total | pair correct/total |
|---|---:|---:|---:|---:|---:|---:|
| VStar | cot_prefix_len08 | 18 | 5.56% | NA | 1/18 | NA |
| VStar | cot_prefix_len16 | 18 | 5.56% | NA | 1/18 | NA |
| VStar | cot_prefix_len32 | 18 | 5.56% | NA | 1/18 | NA |
| VStar | cot_prefix_len64 | 18 | 5.56% | NA | 1/18 | NA |
| VStar | initial_transition_prefix_len08 | 18 | 33.33% | NA | 6/18 | NA |
| VStar | initial_transition_prefix_len16 | 18 | 38.89% | NA | 7/18 | NA |
| VStar | initial_transition_prefix_len32 | 18 | 44.44% | NA | 8/18 | NA |
| VStar | initial_transition_prefix_len64 | 18 | 77.78% | NA | 14/18 | NA |
| MMVP | cot_prefix_len08 | 12 | 0.00% | 0.00% | 0/12 | 0/6 |
| MMVP | cot_prefix_len16 | 12 | 0.00% | 0.00% | 0/12 | 0/6 |
| MMVP | cot_prefix_len32 | 12 | 0.00% | 0.00% | 0/12 | 0/6 |
| MMVP | cot_prefix_len64 | 12 | 0.00% | 0.00% | 0/12 | 0/6 |
| MMVP | initial_transition_prefix_len08 | 12 | 8.33% | 0.00% | 1/12 | 0/6 |
| MMVP | initial_transition_prefix_len16 | 12 | 33.33% | 0.00% | 4/12 | 0/6 |
| MMVP | initial_transition_prefix_len32 | 12 | 83.33% | 66.67% | 10/12 | 4/6 |
| MMVP | initial_transition_prefix_len64 | 12 | 91.67% | 83.33% | 11/12 | 5/6 |

### 主要发现

#### 发现一：COT prefix 几乎锁死错误轨迹

在 VStar 上，强制 COT 错误 prefix 后，无论 prefix 是 8、16、32 还是 64 token，都只有 1/18 正确。

在 MMVP 上更极端，四个 prefix 长度全部是 0/12，pair accuracy 也是 0/6。

这说明错误轨迹不是后段才偶然答错。  
只要开头沿着 COT 的错误视觉叙述走，后续普通 greedy 基本没有自我纠正回来。

#### 发现二：initial-transition prefix 能把正确轨迹带回来

VStar 上，initial-transition prefix 的正确率随长度增加：

```text
8 token:  6/18
16 token: 7/18
32 token: 8/18
64 token: 14/18
```

MMVP 上趋势更明显：

```text
8 token:  1/12
16 token: 4/12
32 token: 10/12
64 token: 11/12
```

这说明 initial transition 的修复不是只在第 0 步瞬间发生，而是在早期生成中逐步形成一段可延续的正确轨迹。  
当 prefix 覆盖到 32 或 64 token 时，后续即使完全切回普通 greedy，也能保留大量修复效果。

#### 发现三：32 token 是一个很关键的观察窗口

MMVP 从 16 token 的 4/12 跳到 32 token 的 10/12。  
这说明很多样本的关键判断并不在最开始几个模板 token，而是在第一段视觉描述逐渐展开时形成。

也就是说：

> 干预要早，但 replay 要覆盖到足够的早期语义内容，通常是第一段视觉叙述或关系判断开始成形的位置。

### 实验二结论

Early Prefix Replay 提供了更强的因果证据：

1. 错误 COT prefix 可以把模型重新带回错误答案。
2. initial-transition prefix 可以把模型带回正确答案。
3. prefix 越覆盖早期视觉叙述，轨迹越稳定。

这说明 early tokens 不只是“结果的表征”，而是会实际影响后续推理方向。

## 两次实验如何连起来

两次实验分别回答了不同层次的问题。

### 第一层：轨迹是不是很早分叉？

由 Early Token Divergence Analysis 回答。

结论是：是的。大量样本在前 16 到 32 token 内就已经分叉，而且分叉内容常常是关键视觉事实或空间关系判断。

### 第二层：早期分叉是否有因果作用？

由 Early Prefix Replay 回答。

结论是：是的。强制错误 prefix 会让模型继续错，强制修复 prefix 会让模型重新走向正确。

### 合并后的机制图景

可以概括为：

```text
第 0 步 soft-to-normal transition
  -> 改变早期 token 分布
  -> 改变第一段视觉叙述/空间关系判断
  -> 推理轨迹被早期锁定
  -> 后续生成沿着已选轨迹展开
  -> 最终答案改变
```

这也解释了为什么中后段 dynamic trigger 的收益弱：

```text
等到中段 entropy spike 出现时，
答案方向往往已经由前面的视觉叙述决定了。
```

所以 LEAD 的有效部分更像是 early trajectory steering，而不是 late correction。

## 对之前 format 稳定路线的关系

这两次实验并不是否定 format 稳定路线，而是重新定位它。

之前 pure-soft 退化确实包含：

- 格式边界不稳
- 长输出
- 重复
- 答案漂移
- extraction failure

`format_cooldown2`、`late64_repeat_gate` 等方法可以修复这些退化。  
但这些更像是让 soft 生成“不坏掉”的稳定性工程。

而这两次实验说明：

> initial transition 的主要收益不只是格式稳定，而是早期视觉推理轨迹被重定向。

换句话说：

- format 稳定路线回答的是：“如何避免 soft 破坏输出？”
- early commitment 路线回答的是：“为什么一个开头 transition 就能改变最终答案？”

两者不是同一个层次。当前更核心的机制主线应该放在后者。

## 总结结论

两次实验合起来支持一个比较清晰的结论：

1. 多模态推理轨迹在生成极早期就出现关键分叉。
2. 分叉常发生在前 16 到 32 token，内容往往是视觉事实、对象存在性或空间关系判断。
3. 错误轨迹通常不是后面才错，而是早期就自信地选错方向，后续只是 elaboration。
4. 强制 replay 早期 prefix 可以把模型带回对应轨迹，说明早期 token 对最终答案有因果影响。
5. initial transition 的主要作用是 early trajectory steering，而不是中后段动态修正。

因此，目前最准确的表述是：

> LEAD 的收益主要来自开头 `soft -> normal` transition 对早期视觉推理轨迹的重定向；一旦早期轨迹形成，后续推理高度 path-dependent，中后段 entropy-gated intervention 往往已经太晚。

## 下一步实验方向

接下来最自然的实验不是继续盲目调参，而是更细地定位“轨迹承诺边界”：

1. 在 16 到 64 token 之间加密 prefix length，例如 24/32/40/48/56。
2. 标注 first visual claim、first relation claim、first answer-direction claim 的位置。
3. 对 COT 正确但 transition 损坏的样本做反向 replay，验证错误 transition prefix 是否也会锁死坏轨迹。
4. 对更多数据集做小规模确认，重点看 32 token 和 64 token replay 是否复现。

这样可以把主线从“early 有效”进一步推进到：

> 到底是哪一种早期语义声明让推理轨迹完成承诺？

# Early Actual-Visual Anchor：RealWorldQA hard54 高优先级验证

## 1. 为什么做这次实验

Early Trajectory Commitment（ETC）认为，多模态推理轨迹在生成极早期被决定。现有 Initial Transition 虽然能通过 step-0 soft initializer 改变部分答案，但 RealWorldQA fixed200 中仍有 54 题被 COT、Initial Transition 与 TALR 同时答错。这组样本更接近三种既有路线的共同感知盲区。

本实验检验一个更具体的假设：如果 step 0 不只接受普通 soft initializer，而是接收来自当前图片真实 visual-token hidden states 的问题相关信号，能否把这些共同错题引向更好的早期轨迹？

## 2. 方法与控制变量

### 2.1 Actual-Visual Anchor

首次 prompt forward 后，以最后一个文本位置的 hidden state 作为查询 (q_0)，对当前图片的视觉 token hidden states (h_i^V) 计算余弦相似度：

\[
s_i=\cos(q_0,h_i^V),\qquad
a_i=\operatorname{softmax}(s_i/\tau).
\]

固定选择 top-8 视觉 token，温度为 0.10，并构造加权视觉 anchor：

\[
v_0=\sum_i a_i h_i^V.
\]

将它的范数对齐到已有 step-0 soft embedding (z_0)：

\[
\hat v_0=\frac{\lVert z_0\rVert}{\lVert v_0\rVert+\epsilon}v_0,
\qquad
\tilde z_0=0.9z_0+0.1\hat v_0.
\]

这里的 (z_0) 已包含原有的 `0.9 soft + 0.1 newline` 初始化。注入只发生一次；后续继续执行已有 `soft -> normal` transition，并锁定 normal COT，不在中后期再次注入视觉信号。

### 2.2 Static Anchor 控制

Static 控制用静态 `<|image_pad|>` embedding 替代真实视觉 hidden states，其余权重、范数对齐、样本顺序、prompt、seed 和解码参数完全相同。它用于判断结果来自“真实视觉内容”，还是来自一般的 step-0 embedding 扰动。

需要注意：该模型的原始 `<|image_pad|>` embedding 范数极小，因此范数对齐比例约为 (8.84\times10^7)。它仍被缩放到目标范数，但这个对照代表的是一个被显著放大的静态方向，解释时不能忽略这一数值特性。

### 2.3 统一运行口径

- 模型：R1-Onevision-7B-RL
- 子集：RealWorldQA hard-wrong54
- hard-wrong 定义：COT、Initial Transition、TALR 在专用 evaluator 下全部错误
- 解码：greedy，seed 42，max new tokens 1024
- LEAD 参数：alpha 0.4、max switch count 5、window size 128
- 评测：RealWorldQA answer-region / option-text 专用 evaluator
- trace：保存完整 token entropy 与 top-20，新增 step-0 anchor 字段

## 3. 实现与回归检查

- helper 单测 3/3 通过，覆盖 top-m、无视觉 token 回退、范数对齐、确定性与非法参数。
- `lambda=0` 的 2 条 GPU smoke 与原 Initial Transition 的完整输出逐字符一致，说明新增 hidden-state 采集和 trace 本身不改变生成。
- actual 与 static 均在 54/54 样本的 step 0 成功应用 anchor，runtime error 都为 0。
- Actual query similarity：均值 0.6867，中位数 0.6934，范围 0.5547–0.7578。
- Actual norm ratio：均值 0.00419，范围 0.00331–0.00482。

## 4. hard54 结果

| 方法 | 正确 | Accuracy | 相对 Initial fixed | Failed extraction | 平均输出 token |
|---|---:|---:|---:|---:|---:|
| Existing Initial Transition | 0/54 | 0.00% | 0 | 2 | 143.87 |
| Early Static Anchor | 4/54 | 7.41% | 4 | 0 | 133.69 |
| Early Actual-Visual Anchor | 4/54 | 7.41% | 4 | 1 | 150.43 |

Actual 修复 ID：`146, 187, 234, 324`。

Static 修复 ID：`234, 324, 522, 716`。

两种 anchor 共同修复 `234, 324`；Actual 独有修复 `146, 187`，Static 独有修复 `522, 716`。因此 Actual 相对 Static 是 fixed 2、damaged 2、净变化 0。Actual 的 failed extraction 出现在 ID `716`。

预注册门槛要求：Actual 至少修复 5 题、严格高于 Static，并且 runtime error 与 failed extraction 都为 0。本次三个条件均未同时满足，所以按计划不运行 correct-control20，也不扩展到 fixed200。

## 5. 结论

这次结果证明了 step-0 小扰动确实可以让少量共同错题翻转，但没有证明真实 visual-token hidden states 比静态 embedding 扰动更有效。Actual 与 Static 都修复 4/54，且只有一半修复样本重叠，说明 early initializer 对方向敏感、能改变轨迹；然而当前 cosine top-8 pooling 并没有把这种敏感性稳定转化为视觉内容收益。

因此，当前证据支持“早期轨迹具有高度 path dependence”，但不支持“这版 actual-visual anchor 能稳定修复共同感知错误”。它应被记录为一次有信息量的负结果，而不能进入主方法或 fixed200 主表。

## 6. 产物

实验目录中保存：完整 hard54 预测、fixed IDs、Actual/Static 差异、逐 token trace、汇总 JSON/Markdown，以及 4 个 Actual 修复样本的 case card。控制集 JSONL 已构造，但因预注册门槛未通过而没有运行。

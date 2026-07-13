# LEAD 论文口径严格复现：高优先级待做实验

## 目的

当前 OpenVLThinker-7B 在 VStar 上的本地 greedy COT 为 154/191（80.63%），论文表 2 为 68.1%，相差 12.53 个百分点。这个差异首先意味着基线协议未对齐，不能直接归因于 LEAD 是否有效。

本实验优先回答两个问题：

1. 在论文和公开代码能够确认的同一生成协议下，能否复现论文中的 COT 和 LEAD 趋势？
2. OpenVLThinker 的高 COT 主要来自 greedy、短输出上限、checkpoint/processor，还是评测抽取？

## 已确认口径

| 项目 | 严格设置 | 依据 |
|---|---|---|
| 数据集 | VStar full 191 | 本地数据规模与论文一位小数分数粒度一致 |
| seed | 42 | 公开仓库 README/run scripts |
| temperature/top-p/top-k | 0.6 / 0.95 / 20 | 公开仓库 README/run scripts |
| token 上限 | 25600 | 公开仓库完整评测脚本 |
| COT | conventional discrete decoding，开启 sampling | 公开仓库 `run_cot.sh` 和默认 `do_sample=True` |
| LEAD sampling | 与 COT 完全一致 | 避免方法间改变采样口径 |
| switch 上限 | 5 | 论文 4.1 节与公开代码 |
| persistence window | 128 | 论文消融称 128 达到峰值 |
| visual injection | 论文 lambda=0.4，对应源码 alpha=0.6 | 源码为 `alpha*soft + (1-alpha)*visual` |
| prompt | `orign`，不额外追加 step-by-step | 对齐公开仓库当前 prompt 构造 |

## 尚不能从论文确认的部分

- 论文只明确说图中示例使用 greedy，没有直说表 2 全部使用 greedy；公开完整评测脚本默认 sampling。因此 sampling 是当前最有依据的主口径，greedy 只作为控制。
- 论文没有列出各模型精确 Hugging Face revision、processor revision 和 chat template hash。
- 论文公式使用三个视觉 special-token embedding 的均值；公开代码实际以 image-pad embedding 作为 anchor。主实验先复现公开实现，后续另做公式实现审计，二者不可混称。
- 公开代码默认 window=256，但论文消融报告 128 最优。因此同时保留 `w128` 论文文本口径与 `w256` 发布代码默认控制。

## 已完成的数据与 checkpoint 审计

- 本地 OpenVLThinker 权重目录携带的模型卡明确指向官方 `ydeng9/OpenVLThinker-7B`，不是同名的纯文本 OpenThinker。
- 集成仓库和公开原始仓库的 VStar JSONL 原始哈希不同，但逐行去掉 `image` 路径字段后的哈希完全一致。差异仅来自服务器图片绝对路径，191 条题目、选项、答案与顺序一致。
- 旧 OpenVLThinker COT 结果 191 行完整、failed extraction 为 0，因此 80.63% 不是由漏抽取后错误改变分母造成的。评测器仍会在新实验中用 last-answer 口径复核。

## 运行矩阵

所有模型先跑 20 条 smoke，再跑 VStar 191 全量。模型顺序优先 OpenVLThinker，其次 Vision-R1、VL-Cogito。

| 方法 | 作用 |
|---|---|
| `cot_paper_sampled` | 论文/发布脚本所支持的 sampled COT 主基线 |
| `lead_paper_sampled_a06_w128` | alpha=0.6、window=128、switch=5 的主 LEAD 复现 |
| `cot_greedy_25600` | OpenVLThinker 控制：只关闭 sampling |
| `lead_released_default_w256` | OpenVLThinker 控制：使用发布代码默认 window=256 |

旧的 `greedy + max_new_tokens=1024` 结果继续保留为历史对照，但不进入论文严格主表。

## 验收

- COT 与 LEAD 必须使用同一 sampling、seed、token budget、prompt 和 evaluator。
- 每个全量结果必须为 191 行，runtime error 为 0。
- 同时报原 evaluator 与 answer-region/last-answer corrected evaluator。
- 报 accuracy、failed extraction、平均输出长度、maxed 25600、LEAD 平均切换次数与 soft ratio。
- 如果 sampled COT 明显回落到论文附近，12.53 点主要是 decoding protocol；若仍为约 80%，继续核查 checkpoint revision、processor/chat template 和 VStar 数据版本。

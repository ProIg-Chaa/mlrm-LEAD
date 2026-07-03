# Early Prefix Replay 因果实验总结（2026-06-15）

## 为什么做这次实验

前面的 early token divergence 分析已经说明：COT 错、initial transition 修对的样本，通常在生成前 16 到 32 个 token 内就出现轨迹分叉。这支持“多模态推理轨迹在极早期被锁定”的解释，但它仍然偏相关性：我们看到 early wording 不同，并不能直接证明这些 early tokens 本身携带了因果作用。

因此这次做 prefix replay：从已有轨迹中截取前缀 token，强制模型先生成这些 token，然后关闭 LEAD/soft 干预，用普通 greedy COT 继续生成。这样可以回答一个更强的问题：

如果只把早期前缀换成 COT 的错误轨迹，后续是否会继续错？如果只把早期前缀换成 initial-transition 的修复轨迹，后续是否会重新走向正确答案？

## 实验设置

实验目录：

`output/experiments/20260615_early_prefix_replay/early_prefix_replay`

运行节点：

`gpu11`

模型：

`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`

样本选择：

- VStar：只取 `cot_orign_greedy` 错、`initial_transition_only` 对的 18 个样本。
- MMVP：只取 `cot_orign_greedy` 错、`initial_transition_only` 对的 12 个样本。

对每个样本做两类 replay：

- `cot_prefix_lenN`：强制前 N 个 generated tokens 等于原始 COT 错误轨迹的前缀，然后普通 greedy 继续。
- `initial_transition_prefix_lenN`：强制前 N 个 generated tokens 等于 initial-transition 修复轨迹的前缀，然后普通 greedy 继续。

prefix 长度：

`8 / 16 / 32 / 64`

校验：

- 共 16 个 run。
- VStar 每个 run 18 行，MMVP 每个 run 12 行，总计 240 条生成。
- smoke test 后正式运行完成，正式 run 未见异常中断。

## 结果表

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

## 具体发现

第一，COT prefix 几乎锁死错误轨迹。

在这些样本上，原本 `initial_transition_only` 是正确的，但只要强制前缀来自 COT 错误轨迹，后续即使重新回到普通 greedy 继续生成，结果依然几乎全错。VStar 在 8/16/32/64 token 都只有 1/18 正确；MMVP 在所有长度上都是 0/12，pair accuracy 也是 0/6。

这说明错误不是只发生在最后答案抽取阶段，也不是简单的后段表述问题。错误方向在前缀里已经被带入，后续生成会沿着这个方向继续 elaboration。

第二，initial-transition prefix 的修复能力随前缀长度明显增强。

VStar 上，强制 initial-transition 前缀后：

- 8 token：6/18
- 16 token：7/18
- 32 token：8/18
- 64 token：14/18

MMVP 上趋势更强：

- 8 token：1/12
- 16 token：4/12
- 32 token：10/12，pair acc 4/6
- 64 token：11/12，pair acc 5/6

这说明 initial transition 的收益并不只是“第 0 步扰动一下”这么简单，而是它在早期生成中逐渐塑造了一个可延续的正确轨迹。到 32 或 64 token 时，这个轨迹已经足够稳定，哪怕之后完全切回普通 greedy，也能大幅保留修复效果。

第三，8 token 太短，32 token 开始很有信息量。

MMVP 的结果尤其清楚：8 token 的 initial-transition prefix 只能救 1/12，但 32 token 直接到 10/12。也就是说，真正承载视觉判断和答案方向的信息，可能不是第一个词或前几个模板 token，而是第一段视觉描述/关系判断开始成形的阶段。

这和之前 timing curve 的结论是互补的：干预必须早，因为轨迹早期形成；但要 replay 这个轨迹，太短的 prefix 不一定足够，通常需要覆盖到早期视觉叙述或关系判断的形成位置。

第四，结果支持 path-dependent，而不支持 late correction。

如果推理主要是在中后段重新计算答案，那么强制一个错误 COT 前缀后，模型应该还有机会靠后续普通推理纠正；但实验中几乎没有纠正。相反，如果给出 initial-transition 的早期正确轨迹，后续普通 greedy 可以延续它。这更符合“早期承诺，后续展开”的机制图景。

## 结论

这次 prefix replay 把之前的相关性证据推进到了更强的因果证据：

1. COT 错误轨迹的早期 prefix 足以把模型重新拉回错误答案。
2. initial-transition 修复轨迹的早期 prefix 足以在后续普通 greedy 中保留大量收益。
3. 关键窗口不只是第 0 个 token，而是第 0 步 transition 之后形成的早期视觉叙述/空间关系判断，约在前 32 到 64 token 内变得稳定。

因此，目前主线可以更明确地表述为：

LEAD 的主要收益来自开头 soft-to-normal transition 对早期推理轨迹的重定向；一旦早期轨迹选定，后续生成高度 path-dependent，中后段 entropy-gated 动态触发往往已经太晚。

## 下一步建议

下一步最值得做的是更细粒度的 prefix boundary 扫描：在 16 到 64 token 之间加密，例如 24/32/40/48/56，并记录 first visual claim、first answer-direction claim 的位置。目标不是继续刷 accuracy，而是定位“正确轨迹真正成形”的边界。

同时可以做反向 replay：对 `initial_transition_only` 原本正确的完整样本，强制 COT 错误 prefix；以及对 COT 原本正确但 transition 损坏的样本，强制 transition prefix，检查是否同样呈现轨迹锁定。

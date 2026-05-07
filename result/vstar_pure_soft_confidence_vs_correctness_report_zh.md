# VStar Pure-Soft 置信度与正误关系分析

本报告分析实验目录 [vstar_pure_soft_50_203818_setsid](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260429/vstar_pure_soft_50_203818_setsid) 中 `pure_soft` 方法在 VStar 前 50 个样本上的表现，重点检查一个现象：模型做错时是否往往更有把握。

对比曲线见 [vstar_pure_soft_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/vstar_pure_soft_correct_wrong_curves.png)，数值摘要见 [vstar_pure_soft_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/vstar_pure_soft_correct_wrong_summary.json)。

## 1. 实验结论

这 50 个样本里，按项目正式评测逻辑统计，正确 29 个，错误 21 个。结论不是“错题的最终答案 token 一定更自信”，而是：

- 错题在整段生成过程中整体更自信。
- 错题在生成后段更自信，尤其是在即将收束到答案时。
- 错题的平均熵更低，说明它们不是随机乱答，而是更早收敛到一条错误路径。
- 错题通常更长、更慢，表现为长推理后稳定地走向错误结论。

## 2. 关键数值

- 平均原始置信度 `mean_raw_conf`：
  - 正确样本 `0.7581`
  - 错误样本 `0.8056`
- 最后 10 个 token 平均原始置信度 `last10_raw_conf`：
  - 正确样本 `0.9092`
  - 错误样本 `0.9454`
- 最后 20 个 token 平均原始置信度 `last20_raw_conf`：
  - 正确样本 `0.8773`
  - 错误样本 `0.9058`
- 平均原始熵 `mean_raw_entropy`：
  - 正确样本 `0.8487`
  - 错误样本 `0.7238`
- 平均输出长度：
  - 正确样本 `128.7` token
  - 错误样本 `366.7` token
- 平均时延：
  - 正确样本 `5.64s`
  - 错误样本 `13.31s`

这些数值说明，错题不是“高熵犹豫后答错”，而更像是“低熵、高置信地走错路”。

## 3. 曲线解释

图中上半部分是 `raw_selected_prob` 的归一化进度平均曲线，下半部分是 `raw_entropy` 的归一化进度平均曲线。

- 在置信度曲线里，错误样本整体高于正确样本，差距在中后段更明显。
- 在熵曲线里，错误样本整体低于正确样本，说明错误样本更早进入稳定收敛状态。
- 两条曲线合起来看，可以把错题理解成“过早自信收缩”，而不是“最后一步才突然选错”。

换句话说，`pure_soft` 在错题上常见的失败模式是：模型先形成一条错误叙述，然后在后续生成中持续强化这条叙述，最终以较高把握输出错误结论。

## 4. 更强的量化证据

按高置信样本排序后，错误样本占比很高：

- `mean_raw_conf` 最高的前 5 个样本里，错题占 `80%`
- `last10_raw_conf` 最高的前 5 个样本里，错题占 `80%`
- `last20_raw_conf` 最高的前 5 个样本里，错题占 `80%`

这说明“高置信错题”不是个别现象，而是这次 `pure_soft` 小规模实验里的一个稳定模式。

## 5. 解释与启发

从这次结果看，`pure_soft` 的风险不只是答错，而是会在错题上表现出一种“错误但笃定”的状态。这个现象对后续方法设计有两个直接启发：

- 不能只把高置信当作可靠信号，尤其是生成后段的高置信。
- 更值得关注的是“高置信 + 低熵 + 长推理”这个组合，它更像错误轨道锁定的信号。

如果下一步继续做机制分析，优先级最高的是：

- 对 `correct` 和 `wrong` 分别画未归一化的前 200 token 置信度/熵曲线，找出两者从第几个 token 开始显著分叉。
- 单独筛出 `last20_raw_conf > 0.95` 的错题，定位它们是在哪一段开始进入错误高置信收敛。

## 6. MMVP 补充结果

除了 VStar，这里还对 `pure_soft` 在 MMVP 全量 300 条样本上的输出做了同类分析。实验目录是 [pure_soft_phys300_mmvp_parallel_212553/mmvp_full_gpu1](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260429/pure_soft_phys300_mmvp_parallel_212553/mmvp_full_gpu1)。

注意，MMVP 的标准答案是 `"(a) Open"` / `"(b) Closed"` 这一类格式，仓库默认 `A/B/C/D` 评测器不适用，因此这里使用专门重评后的结果。重评报告见 [specialized_eval_report.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260429/pure_soft_phys300_mmvp_parallel_212553/mmvp_full_gpu1/specialized_eval_report.json)。

MMVP 的 correct/wrong 对比图见 [mmvp_pure_soft_correct_wrong_curves.png](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/mmvp_pure_soft_correct_wrong_curves.png)，摘要见 [mmvp_pure_soft_correct_wrong_summary.json](/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/result/mmvp_pure_soft_correct_wrong_summary.json)。

重评后的 MMVP 准确率是 `191/300 = 63.67%`。按 correct/wrong 分组后，仍然能看到和 VStar 相同的模式：

- 平均原始置信度 `mean_raw_conf`：
  - 正确样本 `0.7559`
  - 错误样本 `0.7838`
- 最后 10 个 token 平均原始置信度 `last10_raw_conf`：
  - 正确样本 `0.8552`
  - 错误样本 `0.8796`
- 最后 20 个 token 平均原始置信度 `last20_raw_conf`：
  - 正确样本 `0.8560`
  - 错误样本 `0.8737`
- 推理段平均原始置信度 `reason_mean_raw_conf`：
  - 正确样本 `0.7100`
  - 错误样本 `0.7477`
- 回答段平均原始置信度 `answer_mean_raw_conf`：
  - 正确样本 `0.9119`
  - 错误样本 `0.9212`
- 平均原始熵 `mean_raw_entropy`：
  - 正确样本 `0.9155`
  - 错误样本 `0.8312`
- 平均输出长度：
  - 正确样本 `149.0` token
  - 错误样本 `284.7` token
- 平均时延：
  - 正确样本 `4.94s`
  - 错误样本 `9.29s`

这说明在 MMVP 上，错题同样表现为：

- 更高的整体置信度
- 更高的后段置信度
- 更低的熵
- 更长、更慢的推理过程

也就是说，MMVP 上的失败模式依然不是“犹豫后答错”，而更像“较低熵、较高置信地走错路”。

更强的量化证据是：

- 按 `mean_raw_conf` 排序，最高前 5 个样本里错题占 `100%`
- 按 `last10_raw_conf` 排序，最高前 5 个样本里错题占 `80%`
- 按 `last20_raw_conf` 排序，最高前 5 个样本里错题占 `80%`

另外，`last20_raw_conf >= 0.95` 的 29 个样本里，准确率只有 `27.6%`。这说明在 MMVP 上，后段极高置信度本身可以视作一个危险信号，而不是正确性的保证。

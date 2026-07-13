# Pure-Soft 高置信错误图表

本目录包含基于 5 个数据集、2291 条 pure-soft 样本生成的图表。PNG 为 300 DPI，适合 Markdown、汇报和幻灯片；PDF 为矢量版本，适合论文排版。

## 推荐使用

1. `figure5_pure_soft_confidently_wrong_summary`：四联核心结论图，适合作为汇报或论文主图。
2. `figure3_confidence_decile_reliability`：展示准确率没有随置信度单调上升，适合支撑“confidence 未校准”。
3. `figure2_temporal_confidence_auroc`：展示 early/full/tail 三阶段 AUROC，适合讨论错误轨迹的后期锁定。
4. `figure4_confidence_failure_distributions`：区分语义错误与答案抽取失败，并展示长轨迹趋向高置信。
5. `figure1_pure_soft_confidence_overview`：包含 AUROC、top-decile delta、正确/错误置信度和输出长度的完整统计总览。

## 图注建议

**主图图注：** Pure-soft token confidence 对最终答案正确性缺乏稳定预测力。五个数据集的 strict AUROC 均低于 0.5，最高置信 10% 样本的准确率低于数据集总体水平；错误轨迹的平均 token confidence 反而更高。Early-32 confidence 接近随机预测，而 tail-20 confidence 与正确性呈更强反向关系，说明高置信度可能反映错误轨迹锁定或长输出退化，而非可靠的多模态推理。

图中 `Strict` 将答案抽取失败计为错误；`Semantic-only` 排除抽取失败，仅比较可成功抽取答案的正确与错误样本。置信度均使用过滤前完整词表分布中的 `raw_selected_prob`。

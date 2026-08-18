# Probe V2 冻结外部扩展实验

## 目标

验证当前 Probe V2 的外部正收益是否能在更大的、严格未参与训练与阈值选择的样本集合上保持。

## 冻结项

- 模型：R1-Onevision-7B-RL
- Probe：2026-07-27 合并数据训练得到的 linear/MLP V2 artifact
- 动作：contracted soft lambda 0.90、0.95，以及 pure soft 1.00
- 检查点：1、2、4、8、16、32
- 风险系数：rho=1.5
- Probe 权重、归一化统计、actionability threshold、utility threshold 均冻结

## 外部数据

八个数据集各抽取 64 条，共 512 条：

- VStar
- MMVP
- RealWorldQA fixed200
- VisuLogic300
- MMK12-Math
- MMK12-Physics
- VMCBench-dev
- POPE-Adversarial

抽样前排除 2026-07-24、2026-07-27 extended、2026-07-27 follow-up
三轮 Atlas 中出现过的全部原始样本 ID。按数据集的 subtopic/subject 与答案分层，
使用固定种子 20260728。

## 对照

- Hard COT
- 冻结 linear Probe V2
- 冻结 MLP Probe V2
- 相同覆盖率的 entropy-only 路由
- 相同覆盖率的随机路由，重复 1000 次
- 可用动作中的 oracle upper bound，仅作诊断上界

## 验收标准

- 512 条选择样本与历史 Atlas 无 ID 重叠。
- 四个 shard 各 128 条，runtime error 为 0。
- 每条样本在合法长度内生成 1/2/4/8/16/32 六个检查点的三种 treatment。
- 外部评测直接加载冻结 artifact，不重新拟合、不重新选择阈值。
- 主要报告 fixed、damaged、net、accuracy、coverage 和 by-dataset 结果。

## 继续门槛

- 总体 fixed > damaged，且 fixed/damaged 至少为 2。
- 八个数据集中多数净收益非负。
- 冻结 MLP 明显优于 matched-coverage random，并优于 entropy-only。
- 若满足以上条件，再进入第二模型的小规模迁移 Atlas；否则优先分析损坏类型。

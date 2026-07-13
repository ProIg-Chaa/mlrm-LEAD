# GPU 资源分配策略（2026-07-11）

## 当前资源快照

| 节点/GPU | 实际占用 | 我方任务 | 决策 |
|---|---:|---|---|
| gpu09/GPU0 | 61168/81920 MiB，100% util | paper-aligned OpenVLThinker；旧 format VMCBench 诊断 | 保留 paper，暂停旧 format |
| gpu09/GPU1 | 37668/81920 MiB，99% util | correct-model COT baseline；旧 format MMK12-Physics 诊断 | 保留 baseline，暂停旧 format |
| gpu15/GPU0 | 55912/81920 MiB，100% util | Vision-R1 我们的方法 | 独占给当前方法队列，不再叠加模型 |

gpu09 上另有 `end2end.py` 长期占用两卡约 6.6/3.4 GiB 并持续使用计算资源。它不属于本轮 MLRM 实验，不在本策略中主动终止。

Slurm 当前没有真正空闲的 GPU GRES。gpu02、gpu03、gpu06、gpu10、gpu23、gpu24 的 GPU 均已被调度器全部分配。gpu04 虽仍有一个 Slurm GRES，但物理显存曾被未纳入该 GRES 的进程占至约 73 GiB，列入临时排除节点。

## 优先级

1. **P0：论文口径严格复现**。先完成 OpenVLThinker VStar 的 sampled COT/LEAD，再完成模型控制与 Vision-R1、VL-Cogito。
2. **P1：我们的方法**。Vision-R1 当前矩阵完成后，继续其他正确模型上的 format/transition 候选。
3. **P2：正确模型 COT/LEAD baseline**。保留当前已经运行的队列，避免丢失长数据集进度。
4. **P3：旧 format 诊断和广覆盖补跑**。可暂停，待 P0/P1 结束后恢复。

## 最佳分配

### gpu09/GPU0

- 只运行 paper-aligned 队列。
- 暂停 `pure_soft_diffuse_collapse` VMCBench 诊断，减少同卡计算竞争。
- 当前 shared smoke 速度约为 20 条 COT / 7 分 40 秒，可接受；如果获得干净新卡，优先迁移 P0。

### gpu09/GPU1

- 只保留 correct-model baseline 队列。
- 暂停 `highrisk_only_cooldown2` MMK12-Physics 诊断。
- baseline 完成当前 POPE run 后继续原队列，不再加入新方法。

### gpu15/GPU0

- 完成 Vision-R1 我们的方法队列。
- 当前可用显存仅约 25 GiB，不叠加第二个 7B 进程。
- 队列完成后优先转给 P0 剩余模型；若 P0 已完成，则转给 P1。

## 新 GPU 获取策略

- 取消旧的双卡 pending 请求。当前可用节点无法满足双卡，且用户 QOS 同时最多容纳约 4 张 GPU，双卡请求会长期处于 `QOSMaxGRESPerUser`。
- 后续只申请 `gpu:1`，优先任意节点，排除 down/drained 节点与已知异常的 gpu04。
- 新 allocation 启动模型前必须检查真实空闲显存：低于 30 GiB 立即退出并换节点。
- 新卡第一用途是拆分 P0：OpenVLThinker 留在原卡，Vision-R1/VL-Cogito 分配到新卡；必须依靠独立输出目录或完成标记避免重复运行。

## 恢复条件

- P0 全部结束后恢复 gpu09/GPU0 的 format 诊断。
- gpu09/GPU1 baseline 队列结束后恢复该卡的 format 诊断。
- 恢复前再次检查原 PID 是否仍属于记录的方法，避免 PID 重用。

## 风险控制

- 不在 mu01 执行 `nvidia-smi`；只通过已分配作业的 `srun --jobid ... --overlap` 进入计算节点检查。
- 不把发生 CPU offload 的运行纳入结果。日志出现 `offloaded to the cpu` 时立即停止并换节点。
- 同一数据集/方法目录若没有 `eval_report.json`，迁移前先确认没有另一进程仍在写，避免重复运行和结果竞争。

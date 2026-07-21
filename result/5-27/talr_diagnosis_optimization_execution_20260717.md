# TALR 诊断、优化与消融执行说明

## 目标

本轮不再扩大方法故事，而是固定 Early Initializer，只优化 TALR 的后续两个模块：

1. Budgeted Refiner：把全程 quota 追赶改为 early-window、strict-cap refinement。
2. Discrete Stability Guard：只在 refinement 候选实际出现时触发和计数。

预注册成功标准为：两模型四数据集平均高于 full LEAD，至少 6/8 个设置不低于
full LEAD 0.5pp，不出现超过 2pp 的单项下降，failed extraction、long 和 maxed
不恶化。

## 已实现接口

- `--lead_refinement_window {8,16,32}`：第一次 soft-to-normal transition 后允许
  refinement 的窗口长度。
- `--lead_refinement_soft_cap {1,2}`：每样本后续 soft refinement 的严格上限。
- `--lead_guard_candidate_only`：format/diffuse guard 只处理真正的 refinement 候选。
- `--lead_disable_answer_zone_lock`：仅用于消融默认的 `</think>` 后 normal lock。

新策略不再使用 quota catch-up。原 LEAD entropy 条件只产生候选，不能突破窗口、
次数上限、answer-zone lock 或 guard。

## 执行阶段

1. 自动发现迁移后的 COT、full LEAD、Initial Transition、legacy/True TALR 结果。
2. 生成 paired fixed/damaged 诊断、event utility 和分层样本清单。
3. 在 R1-RL 的 VStar、RealWorldQA 固定 64 样本上筛选 `W x K` 六组配置。
4. 前三名进入完整开发集，选出一个 refiner。
5. 比较无 answer lock、answer lock、format guard、full guard，锁定唯一配置。
6. 仅运行一次 R1-RL MMVP/VisuLogic 和 Vision-R1 四数据集验证。
7. 验证完成后再运行递增组件消融与 selected event replay。

验证集结果不得用于重新搜索参数。

## 主要产物

- `talr_component_diagnosis.md/json`
- `talr_fixed_damaged_samples.jsonl`
- `refinement_event_utility.md/json`
- `guard_event_utility.md/json`
- `locked_talr_config.json`
- `talr_optimization_summary.md/json`
- `talr_event_replay_summary.md/json`
- 正式两模型四数据集组件消融结果

## 解释边界

- Event utility 是相关性证据，不能单独证明某次 intervention 导致修复。
- 只有 prefix 一致的单事件 replay 才用于事件级因果表述。
- Guard 只解释输出稳定性；没有 accuracy 证据时不宣称其提升 reasoning ability。
- 若 TALR 只稳定优于 full LEAD 而未超过 COT，定位为由机制审计导出的稳定化
  latent-routing 策略。

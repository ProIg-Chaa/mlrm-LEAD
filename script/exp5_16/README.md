# exp5_16 脚本目录说明

这个目录保存 2026-05-16 之后围绕 VStar、pure-soft、LEAD、路由信号的主要实验脚本。所有脚本默认从项目根目录执行：

```bash
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
bash script/exp5_16/<script>.sh
```

## 启动和检查

脚本启动后会输出 `BASE_DIR`、`RUN_DIR`、`PID` 和 compare 命令。每个 run 目录里都有：

- `run_command.sh`：真实启动命令
- `pid.txt`：后台 pid
- `nohup.log`：运行日志
- `results.jsonl`：最终输出
- `token_entropy_full.jsonl`：token 级 entropy / route trace

常用检查：

```bash
ps -p $(cat <run_dir>/pid.txt) -o pid,stat,etime,cmd
tail -50 <run_dir>/nohup.log
wc -l <run_dir>/results.jsonl
bash <base_dir>/compare_after_done.sh
```

## 脚本索引

- `run_exp1_vstar_spike_type_parallel.sh`
  - VStar full 上并行跑 COT / LEAD / pure-soft，并做 spike 类型分析。

- `run_pure_soft_collapse_wrong_union_parallel.sh`
  - 在 COT / LEAD / pure-soft 错题并集上跑 diffuse collapse。

- `run_pure_soft_collapse_vstar_full_parallel.sh`
  - VStar full 上跑 pure-soft baseline 与普通 diffuse collapse。

- `run_pure_soft_collapse_precision_vstar_full.sh`
  - diffuse collapse 的 precision 消融：strict threshold、patience、late64、repeat gate。

- `run_pure_soft_collapse_late64_repeat_gate_vstar_full.sh`
  - 单独跑 late64 + repeat gate。

- `run_pure_soft_format_cooldown_vstar_full.sh`
  - 单独跑 format cooldown8。

- `run_pure_soft_format_cooldown_ablation_vstar_full.sh`
  - 并行跑 format cooldown2 / cooldown4，并与 cooldown8 和 baseline 对比。

- `run_pure_soft_cooldown2_late64_repeat_gate_vstar_full.sh`
  - 当前组合实验：format cooldown2 + late64 repeat-gated diffuse collapse。

- `run_pure_soft_answer_zone_discrete_vstar_full.sh`
  - 进入 answer zone 后强制离散 embedding。

- `run_lead_soft_veto_late64_repeat_gate_vstar_full.sh`
  - 在 LEAD 上尝试 low-confidence diffuse soft veto。

## 改路由时优先复制哪个脚本

- 新 pure-soft 路由：优先复制 `run_pure_soft_cooldown2_late64_repeat_gate_vstar_full.sh`
- 只做 cooldown 消融：复制 `run_pure_soft_format_cooldown_ablation_vstar_full.sh`
- 只做低置信扩散消融：复制 `run_pure_soft_collapse_precision_vstar_full.sh`
- 在 LEAD 上做改动：复制 `run_lead_soft_veto_late64_repeat_gate_vstar_full.sh`

详细代码地图见项目根目录：

```bash
EXPERIMENT_RUNBOOK_zh.md
```


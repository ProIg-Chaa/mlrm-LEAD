# MLRM-LEAD 实验接手手册

这个文件用于解决一个实际问题：不依赖聊天上下文，也能知道实验从哪里启动、路由在哪里实现、如何切换信号和动作、结果怎么复查。

## 1. 基本路径

- 项目根目录：`/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD`
- Python 环境：`/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python`
- 当前主要模型：`/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL`
- 当前主要数据：`data/vstar.jsonl`
- 近期脚本目录：`script/exp5_16`
- 实验输出目录：`output/experiments/<时间戳>/<实验名>/<run_name>`
- 近期结论报告目录：`result/5-16exp`

所有新脚本建议都从项目根目录启动：

```bash
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
bash script/exp5_16/<脚本名>.sh
```

## 2. 一次实验如何启动

以当前最重要的组合路由为例：

```bash
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
bash script/exp5_16/run_pure_soft_cooldown2_late64_repeat_gate_vstar_full.sh
```

脚本会做三件事：

1. 创建输出目录，例如：
   `output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full`
2. 在 run 目录写入真正执行的 `run_command.sh`
3. 用 `setsid ... &` 后台启动，并写入 `pid.txt`

检查是否在跑：

```bash
RUN=output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0
ps -p $(cat "$RUN/pid.txt") -o pid,stat,etime,cmd
tail -50 "$RUN/nohup.log"
nvidia-smi
```

跑完后比较：

```bash
bash output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/compare_after_done.sh
```

## 3. 代码入口地图

### 参数定义

文件：`main.py`

近期新增路由参数集中在 `main.py` 约 467 行以后：

- `--pure_soft_collapse_on_diffuse`
- `--collapse_entropy_window`
- `--collapse_entropy_alpha`
- `--collapse_min_history`
- `--collapse_min_entropy`
- `--collapse_low_conf_tau`
- `--collapse_low_margin_tau`
- `--collapse_min_step`
- `--collapse_patience`
- `--collapse_patience_window`
- `--collapse_require_repeat_degen`
- `--collapse_repeat_ngram`
- `--collapse_recent_repeat_window`
- `--collapse_recent_repeat_tau`
- `--pure_soft_format_cooldown`
- `--format_cooldown_steps`
- `--pure_soft_answer_zone_discrete`
- `--lead_soft_veto_on_diffuse`

### 参数传递

文件：`lead/inference.py`

- `method=lead` 的参数在 `lead/inference.py` 约 159-183 行传入 `generate_lead(...)`
- `method=pure_soft` 的参数在 `lead/inference.py` 约 213-235 行传入 `generate_pure_soft(...)`
- `method=lead_attenachor/lead_attenanchor` 的视觉 anchor 参数在约 184-212 行传入 `generate_lead_attenachor(...)`

### 真正实现

文件：`lead/generation_utils.py`

- `generate_pure_soft(...)`：约 386 行开始
- pure-soft 路由判断：约 506-585 行
- pure-soft trace 记录：约 590-616 行
- cooldown / answer-zone 状态更新：约 629-642 行
- `generate_lead(...)`：约 966 行开始
- LEAD soft veto 逻辑：约 1169-1225 行
- `generate_lead_attenachor(...)`：约 1671 行开始，视觉 anchor 相关代码在这个函数里

## 4. 当前路由的共同机制

当前这些实验本质上都在控制“下一步输入 embedding 用什么”。

模型每步先正常产生 logits 和 `next_token`，然后构造两种下一步输入：

- `normal_emb = E[next_token]`
  - 离散 token embedding
  - 等价于普通 COT/greedy 续写里的下一步输入
- `soft_emb = probs_original @ E`
  - 用完整词表概率分布加权平均 embedding
  - 这是 pure-soft 的核心动作

在 `generate_pure_soft(...)` 中，最终选择在这里完成：

```python
route_mask = collapse_mask | format_mask | answer_zone_mask
last_emb = torch.where(route_mask[:, None], normal_emb, soft_emb)
```

也就是说：

- `route_mask=False`：继续 pure-soft
- `route_mask=True`：本步生成的 token 作为下一步输入时，改用离散 embedding

## 5. 当前几个路由分别是什么

### 5.1 low-confidence diffuse collapse

开关：

```bash
--pure_soft_collapse_on_diffuse
```

信号：

- 当前 token entropy 是局部 spike
- raw top1 概率低，或者 top1-top2 margin 小
- 可选：必须已经出现重复退化
- 可选：只在 `collapse_min_step` 之后触发

动作：

- 命中时当前 token 的下一步输入从 `soft_emb` 改成 `normal_emb`

典型 late64 + repeat gate 参数：

```bash
--pure_soft_collapse_on_diffuse \
--collapse_entropy_window 16 \
--collapse_entropy_alpha 2.0 \
--collapse_min_history 4 \
--collapse_min_entropy 1.0 \
--collapse_low_conf_tau 0.20 \
--collapse_low_margin_tau 0.05 \
--collapse_min_step 64 \
--collapse_require_repeat_degen \
--collapse_repeat_ngram 3 \
--collapse_recent_repeat_window 32 \
--collapse_recent_repeat_tau 0.35
```

主要脚本：

- `script/exp5_16/run_pure_soft_collapse_late64_repeat_gate_vstar_full.sh`
- `script/exp5_16/run_pure_soft_collapse_precision_vstar_full.sh`

### 5.2 format cooldown

开关：

```bash
--pure_soft_format_cooldown
--format_cooldown_steps 2
```

信号：

- 当前生成 token 被 `_is_format_token_text(...)` 判定为格式 token
- 包括换行、括号、标点、answer 相关模板、think 标签等

动作：

- 命中格式 token 后，后续若干步强制用离散 embedding
- `format_cooldown_steps=2` 表示当前格式 token 及短暂后续区域离散化

主要脚本：

- `script/exp5_16/run_pure_soft_format_cooldown_vstar_full.sh`
- `script/exp5_16/run_pure_soft_format_cooldown_ablation_vstar_full.sh`

目前最好单路由结果：

- `format_cooldown2`：`142/191 = 74.35%`

### 5.3 cooldown2 + late64_repeat_gate

开关组合：

```bash
--pure_soft_format_cooldown \
--format_cooldown_steps 2 \
--pure_soft_collapse_on_diffuse \
--collapse_min_step 64 \
--collapse_require_repeat_degen \
...
```

含义：

- 格式 token 附近用短 cooldown 稳住输出格式
- 后期如果出现低置信扩散 + 重复退化，再做一次离散坍缩

主要脚本：

- `script/exp5_16/run_pure_soft_cooldown2_late64_repeat_gate_vstar_full.sh`

### 5.4 answer_zone_discrete

开关：

```bash
--pure_soft_answer_zone_discrete
```

信号：

- 最近生成文本中出现 `</think` 或 `answer`
- 一旦触发，该样本进入 answer zone

动作：

- 进入 answer zone 后，后续步骤都用离散 embedding

主要脚本：

- `script/exp5_16/run_pure_soft_answer_zone_discrete_vstar_full.sh`

注意：

- 这是一个初版信号，可能在思考区里出现 “answer” 时提前触发。
- 如果后续要更精确，可以改成只在检测到 `</think>` 之后触发，或者检测最后答案模板。

### 5.5 LEAD soft veto

开关：

```bash
--lead_soft_veto_on_diffuse
```

位置：

- `lead/generation_utils.py` 的 `generate_lead(...)` 约 1169-1225 行

动作：

- 当 LEAD 当前本来要使用 soft embedding，且命中低置信扩散/重复退化信号时，临时改用离散 embedding

当前状态：

- 之前一次 full VStar 结果没有改变输出，说明当时的 current-step veto 对实际 LEAD 触发帮助不大。
- 如果继续做，更推荐改成“soft entry veto”：在 `to_soft` 刚要切换到 soft 前拦截，而不是已经算出当前 mode 后再 veto。

## 6. 如何切换路由信号或动作

最小修改路径：

1. 在 `main.py` 增加命令行参数。
2. 在 `lead/inference.py` 把参数塞进 `model_inputs`。
3. 在 `lead/generation_utils.py` 的对应 generation 函数里读取参数。
4. 在每步生成后构造一个新的 mask，例如 `my_route_mask`。
5. 把新 mask 合入动作选择：

```python
route_mask = collapse_mask | format_mask | answer_zone_mask | my_route_mask
last_emb = torch.where(route_mask[:, None], normal_emb, soft_emb)
```

如果新动作不是“离散坍缩”，而是别的 embedding，例如视觉 anchor，需要把 `last_emb` 改成多分支：

```python
last_emb = soft_emb
last_emb = torch.where(discrete_mask[:, None], normal_emb, last_emb)
last_emb = torch.where(anchor_mask[:, None], anchor_emb, last_emb)
```

建议每个新路由都在 trace 里记录：

```python
"my_route_active": bool(my_route_mask[bi].item()),
"my_route_score": float(my_score[bi].item()),
```

这样跑完后可以统计“到底触发了多少次、集中在哪些样本、是否修复/损坏”。

## 7. 完整例子：新增一个更严格的 answer-zone 路由

这个例子演示从“想法”到“代码开关”再到“脚本启动”的完整过程。

### 7.1 实验想法

现有 `answer_zone_discrete` 的信号是：

- 最近 16 个生成 token 中出现 `</think`
- 或者出现 `answer`

问题是：模型可能在思考区里写出 “the answer should be ...”，这会提前触发 answer-zone，导致过早离散化。

所以可以做一个更严格的新实验：

- 新信号：只在检测到 `</think>` 后触发
- 新动作：触发后，后续 token 全部用离散 embedding
- 实验名：`answer_zone_think_end_only`

这个实验和原 `answer_zone_discrete` 的区别只有信号不同，动作相同。

### 7.2 第一步：在 `main.py` 增加参数

位置：`main.py` 约 499 行附近，已有：

```python
parser.add_argument("--pure_soft_answer_zone_discrete", action="store_true",
                    help="pure_soft 中进入答案区后强制使用离散 token embedding")
```

在它后面加：

```python
parser.add_argument("--pure_soft_answer_zone_think_end_only", action="store_true",
                    help="pure_soft 中只在检测到 </think> 后进入答案区并强制离散 token embedding")
```

如果 `main.py` 后面有 config 记录字典，也要把它加进去：

```python
"pure_soft_answer_zone_think_end_only": args.pure_soft_answer_zone_think_end_only,
```

### 7.3 第二步：在 `lead/inference.py` 传参数

位置：`lead/inference.py` 约 228-230 行，已有：

```python
model_inputs["format_cooldown"] = args.pure_soft_format_cooldown
model_inputs["format_cooldown_steps"] = args.format_cooldown_steps
model_inputs["answer_zone_discrete"] = args.pure_soft_answer_zone_discrete
```

在后面加：

```python
model_inputs["answer_zone_think_end_only"] = args.pure_soft_answer_zone_think_end_only
```

### 7.4 第三步：在 `generate_pure_soft` 读取参数

位置：`lead/generation_utils.py` 的 `generate_pure_soft(...)` 开头，已有：

```python
format_cooldown = kwargs.pop("format_cooldown", False)
format_cooldown_steps = kwargs.pop("format_cooldown_steps", 0)
answer_zone_discrete = kwargs.pop("answer_zone_discrete", False)
```

改成：

```python
format_cooldown = kwargs.pop("format_cooldown", False)
format_cooldown_steps = kwargs.pop("format_cooldown_steps", 0)
answer_zone_discrete = kwargs.pop("answer_zone_discrete", False)
answer_zone_think_end_only = kwargs.pop("answer_zone_think_end_only", False)
```

### 7.5 第四步：修改信号判断

位置：`lead/generation_utils.py` 的 `answer_zone_discrete` 判断块，当前类似：

```python
if answer_zone_discrete:
    for bi, orig in enumerate(unfinished_idx):
        recent_ids = all_generated[orig][prompt_lens[orig]:] + [int(next_tokens[bi].item())]
        recent_text = tokenizer.decode(
            recent_ids[-16:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).lower()
        triggered = ("</think" in recent_text) or ("answer" in recent_text)
        answer_zone_trigger_mask[bi] = triggered
        answer_zone_mask[bi] = answer_zone_active[orig] or triggered
```

改成：

```python
if answer_zone_discrete or answer_zone_think_end_only:
    for bi, orig in enumerate(unfinished_idx):
        recent_ids = all_generated[orig][prompt_lens[orig]:] + [int(next_tokens[bi].item())]
        recent_text = tokenizer.decode(
            recent_ids[-16:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).lower()
        if answer_zone_think_end_only:
            triggered = "</think" in recent_text
        else:
            triggered = ("</think" in recent_text) or ("answer" in recent_text)
        answer_zone_trigger_mask[bi] = triggered
        answer_zone_mask[bi] = answer_zone_active[orig] or triggered
```

动作不需要改，因为动作统一在这里：

```python
route_mask = collapse_mask | format_mask | answer_zone_mask
last_emb = torch.where(route_mask[:, None], normal_emb, soft_emb)
```

### 7.6 第五步：trace 里记录新信号

在 token trace 的 record 里，已有：

```python
"answer_zone_discrete_active": bool(answer_zone_mask[bi].item()),
"answer_zone_trigger": bool(answer_zone_trigger_mask[bi].item()),
```

建议补一项，便于后面区分不同 answer-zone 实验：

```python
"answer_zone_think_end_only": bool(answer_zone_think_end_only),
```

如果想让 `mode` 更直观，也可以把：

```python
"mode": (
    "collapsed"
    if bool(collapse_mask[bi].item())
    else ("format_cooldown" if bool(format_mask[bi].item()) else "pure_soft")
),
```

改成：

```python
"mode": (
    "collapsed"
    if bool(collapse_mask[bi].item())
    else (
        "format_cooldown"
        if bool(format_mask[bi].item())
        else ("answer_zone_discrete" if bool(answer_zone_mask[bi].item()) else "pure_soft")
    )
),
```

这一步不影响模型输出，只影响日志可读性。

### 7.7 第六步：复制脚本

复制现有 answer-zone 脚本：

```bash
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
cp script/exp5_16/run_pure_soft_answer_zone_discrete_vstar_full.sh \
   script/exp5_16/run_pure_soft_answer_zone_think_end_only_vstar_full.sh
chmod +x script/exp5_16/run_pure_soft_answer_zone_think_end_only_vstar_full.sh
```

然后编辑新脚本：

```bash
vim script/exp5_16/run_pure_soft_answer_zone_think_end_only_vstar_full.sh
```

需要改三类地方。

第一，改实验目录名：

```bash
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_answer_zone_think_end_only_vstar_full"
RUN_DIR="${BASE_DIR}/answer_zone_think_end_only_gpu1"
```

第二，把启动参数从：

```bash
--pure_soft_answer_zone_discrete
```

改成：

```bash
--pure_soft_answer_zone_think_end_only
```

第三，把 compare 里的 run 名从 `answer_zone_discrete` 改成 `answer_zone_think_end_only`。例如：

```python
runs = {
    "baseline": Path("${BASELINE_RUN}"),
    "answer_zone_think_end_only": Path("${RUN_DIR}"),
}
```

trace 统计可以继续统计：

```python
count = sum(1 for t in (r.get("tokens") or []) if t.get("answer_zone_discrete_active"))
```

因为底层字段仍表示“answer-zone 动作是否激活”。如果已经新增了专门字段，也可以改成统计新字段。

### 7.8 第七步：先做语法检查

改完代码后先跑：

```bash
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python -m py_compile \
  main.py lead/inference.py lead/generation_utils.py
```

通过后再启动实验。

### 7.9 第八步：启动实验

```bash
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
bash script/exp5_16/run_pure_soft_answer_zone_think_end_only_vstar_full.sh
```

脚本会输出类似：

```text
BASE_DIR=/share/home/.../output/experiments/20260520_xxxxxx/pure_soft_answer_zone_think_end_only_vstar_full
RUN_DIR=/share/home/.../answer_zone_think_end_only_gpu1
PID=xxxxxx
Compare after done:
  bash /share/home/.../compare_after_done.sh
```

### 7.10 第九步：检查是否正常

把脚本输出的 `RUN_DIR` 复制出来：

```bash
RUN=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260520_xxxxxx/pure_soft_answer_zone_think_end_only_vstar_full/answer_zone_think_end_only_gpu1
ps -p $(cat "$RUN/pid.txt") -o pid,stat,etime,cmd
tail -50 "$RUN/nohup.log"
nvidia-smi
```

判断正常的标准：

- `ps` 能看到进程还在。
- `nohup.log` 里能看到模型加载完成，并出现 `[1/191]`、`[2/191]` 这样的样本进度。
- `nvidia-smi` 里对应 GPU 有显存占用和利用率。

### 7.11 第十步：跑完后比较

```bash
bash /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD/output/experiments/20260520_xxxxxx/pure_soft_answer_zone_think_end_only_vstar_full/compare_after_done.sh
```

重点看这些行：

- `accuracy`
- `length mean / p90 / max`
- `long>=256`
- `maxed1024`
- `delta_vs_baseline: changed / fixed / damaged / net`
- answer-zone 触发次数

判断这个实验是否有价值：

- 如果准确率上升，同时 `damaged` 很少，说明信号更精确。
- 如果准确率没变但 `long>=256`、`maxed1024` 明显下降，说明它可能是一个格式稳定路由。
- 如果 `changed` 很多、`damaged` 也很多，说明动作过强或者信号触发太早。
- 如果几乎不触发，说明信号太窄，需要扩大到 `</think>` 后的答案模板，或者加入 `Answer:` 但排除思考区普通 answer。

### 7.12 这个例子对应的“切换路由”逻辑

在这个例子里：

- 路由动作没有变：仍然是 `normal_emb` 替代 `soft_emb`
- 变的是路由信号：
  - 旧信号：`</think` 或 `answer`
  - 新信号：只看 `</think`

如果要换动作，比如不是离散化，而是加入视觉 anchor，那么改动点不是 `triggered = ...`，而是：

```python
last_emb = torch.where(route_mask[:, None], normal_emb, soft_emb)
```

需要改成多分支 embedding 选择。

## 8. 常用脚本说明

近期 VStar 路由实验：

- `run_exp1_vstar_spike_type_parallel.sh`
  - 跑 COT / LEAD / pure-soft 三个基线，并做 spike 类型分析
- `run_pure_soft_collapse_vstar_full_parallel.sh`
  - pure-soft baseline 与普通 diffuse collapse
- `run_pure_soft_collapse_precision_vstar_full.sh`
  - strict / patience / late64 / repeat_gate 消融
- `run_pure_soft_collapse_late64_repeat_gate_vstar_full.sh`
  - 只跑 late64 + repeat_gate
- `run_pure_soft_format_cooldown_vstar_full.sh`
  - format cooldown8
- `run_pure_soft_format_cooldown_ablation_vstar_full.sh`
  - format cooldown2 / cooldown4 / cooldown8 对比
- `run_pure_soft_cooldown2_late64_repeat_gate_vstar_full.sh`
  - 当前组合路由
- `run_pure_soft_answer_zone_discrete_vstar_full.sh`
  - answer zone 后离散化
- `run_lead_soft_veto_late64_repeat_gate_vstar_full.sh`
  - 在 LEAD 上尝试 soft veto

## 9. 结果文件怎么看

每个 run 目录通常有：

- `run_command.sh`
  - 最真实的启动命令，优先看这个
- `nohup.log`
  - 实时日志
- `pid.txt`
  - 后台进程 pid
- `results.jsonl`
  - 每个样本的最终输出
- `token_entropy_full.jsonl`
  - 每个 token 的 entropy、置信度、路由触发信息

每个实验根目录通常有：

- `compare_after_done.sh`
  - 跑完后一键比较结果

常用检查命令：

```bash
tail -50 <run_dir>/nohup.log
wc -l <run_dir>/results.jsonl
bash <base_dir>/compare_after_done.sh
```

如果 `wc -l results.jsonl` 等于 `191`，VStar full 基本跑完。

## 10. 当前最重要的已有结果

- pure-soft baseline：`112/191 = 58.64%`
- LEAD baseline：`139/191 = 72.77%`
- COT baseline：`131/191 = 68.59%`
- late64 + repeat_gate：`119/191 = 62.30%`
- format_cooldown8：`136/191 = 71.20%`
- format_cooldown4：`138/191 = 72.25%`
- format_cooldown2：`142/191 = 74.35%`

目前最值得继续研究的方向：

1. `format_cooldown2` 为什么能超过 LEAD。
2. `format_cooldown2` 是否过度离散化，是否能进一步缩小触发范围。
3. `cooldown2 + late64_repeat_gate` 是否能在不损坏正确样本的情况下继续补救长输出/退化样本。
4. `answer_zone_discrete` 是否能用更精确 answer-zone 信号替代宽泛 format cooldown。

## 11. 新开一个实验的推荐模板

不要直接手写长命令，建议复制最近的脚本：

```bash
cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
cp script/exp5_16/run_pure_soft_cooldown2_late64_repeat_gate_vstar_full.sh \
   script/exp5_16/run_my_new_route_vstar_full.sh
```

然后只改这些地方：

- `BASE_DIR` 的实验名
- `RUN_DIR` 的 run 名
- `CUDA_VISIBLE_DEVICES`
- `main.py` 后面的路由参数
- `compare_after_done.sh` 里的 runs 字典
- trace 统计字段

启动后立刻确认：

```bash
bash script/exp5_16/run_my_new_route_vstar_full.sh
tail -50 output/experiments/<时间戳>/<实验名>/<run_name>/nohup.log
nvidia-smi
```

## 12. 当前正在跑的实验

截至 2026-05-20 11:42 左右：

- GPU0：`cooldown2 + late64_repeat_gate`
  - `output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0`
  - PID：`2825516`
- GPU1：`answer_zone_discrete`
  - `output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full/answer_zone_discrete_gpu1`
  - PID：`2828661`

检查：

```bash
ps -p $(cat output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0/pid.txt) -o pid,stat,etime,cmd
tail -50 output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0/nohup.log

ps -p $(cat output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full/answer_zone_discrete_gpu1/pid.txt) -o pid,stat,etime,cmd
tail -50 output/experiments/20260520_114012/pure_soft_answer_zone_discrete_vstar_full/answer_zone_discrete_gpu1/nohup.log
```

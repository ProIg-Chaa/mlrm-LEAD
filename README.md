<div align="center">

# LEAD: Latent Entropy-Aware Decoding

### Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-orange.svg)](https://zhongxing-xu.github.io/LEAD)


</div>

## 🔥 最新动态

- **[2026/01]** 🎉 LEAD 被 **CVPR 2026** 接收！
- **[2026/03]** 代码和数据集已开源。

---

## 📌 概述

**LEAD**（Latent Entropy-Aware Decoding）是一种 **无需训练（training-free）** 的解码策略，旨在缓解多模态大推理模型（MLRMs）中的幻觉问题。LEAD 通过实时监测 token 概率分布的熵值，自适应地在 **潜在解码（latent decoding）** 和 **离散解码（discrete decoding）** 之间动态切换，并在高不确定性节点注入视觉锚点以强化视觉 grounding。

<div align="center">
<img src="figure/method.png" width="80%"/>
</div>

## 💡 核心亮点

- **熵感知切换**：高熵阶段采用概率加权的连续 embedding 进行 latent reasoning；低熵阶段恢复离散 token decoding 保证收敛性。
- **视觉锚点注入**：在不确定推理的关键时刻注入视觉 anchor token，减少模型脱离图像内容的"脑补式"推理。
- **即插即用**：无需额外训练或外部工具，可直接应用于现有 MLRMs。

---

## 📈 部分实验结果

<div align="center">
<img src="figure/Thinking efficiency2.png" width="38%"/>  <img src="figure/Passk2.png" width="50%"/>
</div>

<p align="center"><em>（左）LEAD 以更短的推理长度获得更高准确率；（右）Pass@k 采样效率对比。</em></p>

<div align="center">
<img src="figure/Fluency2.png" width="50%"/> <img src="figure/example/visualization.png" width="42%"/>
</div>

<p align="center"><em>（左）文本质量评测；（右）Token 级可视化分析。</em></p>

更多结果与分析请参见论文。

---

## 🛠️ 环境配置

### 1. 克隆仓库

```bash
git clone https://github.com/Zhongxing-XU/LEAD.git
cd LEAD
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型权重

下载 [R1-Onevision-7B](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B)，或直接使用 HuggingFace 模型名自动下载：

```bash
# 方式 A：自动下载
--model_name Fancy-MLLM/R1-Onevision-7B

# 方式 B：本地路径
--model_name /path/to/R1-Onevision-7B
```

---

## 🚀 快速开始

### Demo 示例

```bash
python main.py \
    --model_name Fancy-MLLM/R1-Onevision-7B \
    --dataset data/demo.jsonl \
    --method lead \
    --max_new_tokens 2048
```

### 完整评测

```bash
bash script/run.sh
```

### 自定义配置

```bash
python main.py \
    --model_name Fancy-MLLM/R1-Onevision-7B \
    --dataset data/physunibench.jsonl \
    --output_dir output \
    --method lead \
    --alpha 0.6 \
    --max_switch_count 5 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --max_new_tokens 25600 \
    --seed 42
```

---

## ⚙️ 参数说明

### 模型与数据

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | `Fancy-MLLM/R1-Onevision-7B` | HuggingFace 模型名或本地权重路径 |
| `--dataset` | `data/physunibench.jsonl` | 数据集 JSONL 文件路径 |
| `--output_dir` | `output/` | 结果保存目录 |
| `--limit` | `None` | 仅运行前 N 条样本（调试用） |

### 解码方法

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `lead` | 解码方法：`lead` / `cot` / `cot_greedy` |
| `--alpha` | `0.6` | Soft-mode 混合系数 α₀，越大越偏向概率加权 embedding |
| `--max_switch_count` | `5` | 最大 soft→normal 模式切换次数，超出后触发收敛注入 |

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--temperature` | `0.6` | 采样温度 |
| `--top_p` | `0.95` | Nucleus sampling 阈值 |
| `--top_k` | `20` | Top-k 过滤 |
| `--max_new_tokens` | `25600` | 最大生成 token 数 |
| `--seed` | `42` | 随机种子 |

---

## 📁 可用脚本

| 脚本 | 说明 |
|------|------|
| `script/run.sh` | LEAD 方法完整评测 |
| `script/run_cot.sh` | CoT baseline 评测 |
| `script/run_debug.sh` | 调试模式：5 条样本、短生成 |
| `script/run_eval.sh` | 仅评估已有结果 |

---

## 📦 数据集格式

将 JSONL 文件放入 `data/` 目录，格式如下：

```json
{"id": 1, "image": "path/to/image.jpg", "question": "What is shown?", "options": "A. ...\nB. ...\nC. ...\nD. ...", "answer": "A"}
```

内置 benchmark 数据集：`physunibench`、`math_vision`、`math_vista`、`mmvp`、`realworldqa`、`visulogic`、`vstar`、`demo`

---

## 🗂️ 项目结构

```
LEAD/
├── main.py                    # 主入口
├── lead/
│   ├── generation_utils.py    # LEAD 和 CoT 核心生成算法
│   ├── inference.py           # 输入构造和单样本推理
│   ├── data.py                # 数据加载与预处理
│   ├── evaluator.py           # 答案评估与准确率统计
│   ├── prompts.py             # Prompt 模板管理
│   ├── logger.py              # 日志系统
│   └── utils.py               # 通用工具函数
├── data/                      # 数据集 JSONL 文件
├── figure/                    # 论文图表
├── script/                    # 运行脚本
├── tests/                     # 单元测试
├── requirements.txt
├── setup.py
├── LICENSE
└── CONTRIBUTING.md
```

---

## 📝 引用

如果本项目对你的研究有所帮助，请引用我们的论文：

```bibtex
@inproceedings{xu2026lead,
      title     = {Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding},
      author    = {Zhongxing Xu and Zhonghua Wang and Zhe Qian and Dachuan Shi and Feilong Tang and Ming Hu and Shiyan Su and Xiaocheng Zou and Wei Feng and Dwarikanath Mahapatra and Yifan Peng and Mingquan Lin and Zongyuan Ge},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2026}
}
```

---

## 📄 License

本项目基于 MIT License 开源 — 详见 [LICENSE](LICENSE) 文件。

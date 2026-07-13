# 实验环境与跨平台迁移记录（2026-07-13）

## 1. 项目与环境位置

- 项目：`/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD`
- Git commit：`c251543b3a28efad5825463b95ab2ac255724cda`
- Python 环境：`/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead`
- Python 可执行文件：`/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python`
- 模型目录：`/share/home/wangzixu/liudinghao/gushuo/models`
- 环境体积：约 5.8GB，含 1171 个符号链接
- 项目体积：约 74GB，其中 `output/` 约 73GB，`data/` 约 19MB，`result/` 约 4.2MB

## 2. 系统兼容条件

- OS：Linux 3.10.0-1160.el7.x86_64
- 架构：x86_64
- glibc：2.17
- Python：3.10.20，conda-forge build，GCC 14.3.0
- PyTorch：2.6.0+cu124
- PyTorch CUDA runtime：12.4
- cuDNN：9.1.0（`90100`）
- 旧平台 GPU driver：550.54.14

新平台至少应满足：Linux x86_64、glibc 不低于 2.17、NVIDIA 驱动支持 CUDA 12.4。迁移后必须重新执行 CUDA smoke，不能只以环境成功解压作为验收。

## 3. 核心 Python 依赖

| Package | Version |
|---|---:|
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 5.6.2 |
| accelerate | 1.13.0 |
| tokenizers | 0.22.2 |
| qwen-vl-utils | 0.0.14 |
| numpy | 1.23.0 |
| pillow | 12.1.1 |

完整清单位于项目根目录：

- `environment-pip-freeze-20260713.txt`：81 行，适合核对 Python 包。
- `environment-conda-explicit-20260713.txt`：107 行，包含 Conda 包的精确下载 URL/build。
- `environment-conda-history-20260713.txt`：Conda 安装历史。

## 4. 模型资产

| 模型 | 体积 |
|---|---:|
| R1-Onevision-7B | 16GB |
| R1-Onevision-7B-RL | 16GB |
| Vision-R1-7B | 16GB |
| VL-Cogito-7B | 16GB |
| OpenVLThinker-7B | 16GB |

完整五模型约 80GB。本机 C 盘当前仅约 10.9GB 可用，不能经本机同时中转全部模型或 73GB 实验输出。环境压缩包可能可以单独中转，但生成后必须先检查实际大小。

## 5. 是否可直接 SCP 环境

技术上可以把环境下载到本机再上传，但不推荐直接执行：

```bash
scp -r old:/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead .
```

原因：

1. 环境包含 1171 个符号链接，`scp -r` 对链接的处理不适合作为可复现归档。
2. Conda 环境中的脚本、shebang 和元数据可能包含旧绝对前缀。
3. 大量小文件会使 SCP 较慢，断点续传能力差。
4. Windows 本机不能运行这个 Linux 环境，只能把归档当作中转文件。

## 6. 推荐迁移方案

### 方案 A：锁文件重建，最稳

在新平台创建环境：

```bash
micromamba create -p /new/path/mlrm-lead \
  --file environment-conda-explicit-20260713.txt
```

若新平台无法访问旧 channel URL，则使用 Python 3.10 环境并安装 `environment-pip-freeze-20260713.txt`，然后逐项执行 smoke test。精确 Conda lock 优先于 pip-only 重建。

### 方案 B：Conda-pack 经本机中转，适合网络隔离平台

旧平台安装或临时调用 `conda-pack` 后：

```bash
conda-pack -p /share/home/wangzixu/.local/share/mamba/envs/mlrm-lead \
  -o /share/home/wangzixu/mlrm-lead-20260713.tar.gz
```

下载到本机：

```powershell
scp super-mu01:/share/home/wangzixu/mlrm-lead-20260713.tar.gz D:\migration\
```

上传并在新平台解压：

```bash
mkdir -p /new/path/envs/mlrm-lead
tar -xzf mlrm-lead-20260713.tar.gz -C /new/path/envs/mlrm-lead
/new/path/envs/mlrm-lead/bin/conda-unpack
```

`conda-unpack` 用于修复旧绝对前缀，是该方案比原始 `scp -r` 更可靠的关键。

### 方案 C：保留软链接的 tar，作为备份方案

若不能使用 conda-pack：

```bash
tar --zstd -cpf mlrm-lead-raw-20260713.tar.zst \
  -C /share/home/wangzixu/.local/share/mamba/envs mlrm-lead
```

该方式保留符号链接，但不会自动修复 Conda 前缀；只能作为备份，迁移后需要检查 shebang、RPATH 和旧路径引用。

## 7. 数据与结果迁移策略

- 项目代码、`data/`、`result/`、环境 lock 文件：优先通过 Git/SCP 迁移，体积小。
- `output/`：约 73GB，不经当前本机中转。建议旧平台打包后直接服务器到服务器传输，或只迁移 compact 主矩阵及最终报告。
- 模型：每个约 16GB，当前本机空间不足。优先在新平台从 Hugging Face 重下；若网络不可用，使用服务器间 `rsync --partial --append-verify`。
- 图片数据不一定包含在项目 `data/` 中。迁移前必须根据 JSONL 的绝对图片路径单独审计真实 image root。

## 8. 新平台验收

```bash
/new/path/envs/mlrm-lead/bin/python - <<'PY'
import torch, transformers
print(torch.__version__, torch.version.cuda)
print(transformers.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY

python -m py_compile main.py lead/inference.py lead/generation_utils.py
```

随后使用 2 条样本分别运行 COT、LEAD、initial transition 和 TALR，要求：

- runtime error 为 0；
- 生成 token 与相同 seed/greedy 配置一致；
- `results.jsonl`、`config.json`、`eval_report.json`、`token_entropy.jsonl` 均生成；
- switch、soft ratio、format/veto trigger 字段存在；
- 图片路径存在率为 100%。

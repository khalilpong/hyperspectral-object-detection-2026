# 在云服务器上训练（Kaggle GPU 不够用时的备选方案）

> 本文是历史备选方案，保留作技术参考。项目已完赛归档，当前结果与恢复入口见 [README](../README.md)，没有待执行的租用或训练任务。

## 什么时候需要租云服务器

当时本机 8GB 显存限制了更大模型及 batch 的尝试。Kaggle 免费 GPU 是第一选择，
遇到以下情况再考虑租服务器：

| Kaggle 的限制 | 说明 |
|---|---|
| GPU 是 T4 | 16GB 显存够用，但 m 模型固定 30 轮实测约 2.0~2.5 小时，且免费周额度已成为主要限制 |
| 每周约 30 小时额度 | 留出对照 + 全量训练两次就可能用掉一半 |
| 单次最长 12 小时 | 更大的模型或更多轮数跑不完 |
| L4 等更好的卡 | 对本账号不开放 |

## 租什么配置

| 项目 | 最低 | 推荐 | 为什么 |
|---|---|---|---|
| GPU 显存 | 16 GB | **24 GB**（如 RTX 4090 / A10 / L4） | m 模型要 batch≥4 才不被拖累；24GB 可上 batch 8 或试 l 模型 |
| 内存 | 32 GB | **64 GB** | 16 通道数据做 mosaic 增强，**每个数据加载进程约占 3.5~5.5GB**，内存决定能开几个进程（见下文） |
| CPU | 4 核 | 8 核以上 | 数据加载在 CPU 上做，本机实测 CPU 是瓶颈 |
| 磁盘 | 50 GB | 80 GB | 原始 zip 6.2 + 解压 14.6 + 16 波段数据约 9 + 模型和日志 |
| 系统 | Ubuntu 22.04，NVIDIA 驱动支持 CUDA 12.x | | |

**内存与数据加载进程数（workers）的关系**——这是本项目最容易踩的坑：

mosaic 增强会先建一张 `2048 × 2048 × 16` 的拼图画布（每张约 64MB），每个进程各自分配。
另外 Ultralytics 在**验证阶段会额外起 2 倍 workers**。本机 31.7GB 内存实测：

| workers | 结果 |
|---|---|
| 0 | 稳定，但慢（CPU 单线程读数据） |
| 1 | 稳定，快 1.6 倍 |
| 2 | 内存逐轮下降，险些崩溃 |
| 4 | 直接 `MemoryError` 崩溃 |

经验值：**可用内存 GB ÷ 6 ≈ 安全的 workers 数**（留出验证阶段的余量）。
64GB 内存可以用 workers=4；32GB 用 workers=1~2。

## 操作步骤

以下命令在云服务器的终端里执行。所有东西都放在 `~/hsi` 下。

### AutoDL：当前推荐的可审计 EIoU 路径

当前第一优先级是已经预注册的单变量 EIoU fixed 实验。不要在服务器上临时拼
`HSI_*` 参数；仓库提供的 wrapper 会清除外部遗留的 `HSI_*` 环境变量，再写入完整固定合同。

#### 租用前的配置与费用边界

- 首选 1 张 RTX 4090 24GB，至少 8 vCPU、**64GB RAM**、80GB 数据盘；
- Python 必须为 3.11 或 3.12，基础镜像必须已带 CUDA 可用的 PyTorch；
- 2026-09-22 官网公开标价快照为 RTX 4090 `1.88 元/小时`，会员价为 95 折；
  这不是具体地区库存承诺，最终以登录后所选主机的即时价格为准；
- fixed 预留 4 小时，若通过门禁后再做 full，合计预留 8 小时。创建付费实例前要按
  控制台即时单价重新计算，并取得操作时确认；训练完立即关机。

本地生成只含 EIoU 所需代码和 `yolo26m.pt` 的自审计上传包：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_autodl_eiou_bundle.py `
  --output .\artifacts\autodl\yolo26m_eiou_fixed_20260922.zip
```

生成器会先核对平铺代码与源文件哈希，再写 `bundle_manifest.json`。当前包不含原始比赛数据、
Kaggle 凭据或历史输出。将该包和原始比赛 zip 放入 AutoDL 私有数据盘；不要放进“公开数据”。

本机已核验的原始比赛包是
`data/raw/hyperspectral-object-detection-challenge-2026.zip`，大小 6,694,200,518 bytes，
SHA-256 为 `C99BE7F2F930813E0846F64931851466F8DEB3720E35AB0F5800EB0F02E0B8AE`。
wrapper 会按与 `hsi_detection.layout.discover_layout()` 相同的规则识别
`Annotations/VIS`、训练图 `data_train/.../VIS` 和测试图 `data_test/.../VIS`；不要手工改目录名或重打包。

在实例内准备目录并解开代码包：

```bash
mkdir -p /root/autodl-tmp/hsi/{code,raw,work,outputs}
unzip /root/autodl-tmp/hsi/yolo26m_eiou_fixed_20260922.zip -d /root/autodl-tmp/hsi/code
python3 -m venv --system-site-packages /root/autodl-tmp/hsi/venv
source /root/autodl-tmp/hsi/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r /root/autodl-tmp/hsi/code/requirements-autodl-eiou.txt
```

先做**不解压、不训练**的付费环境检查：

```bash
python /root/autodl-tmp/hsi/code/run_autodl_eiou.py \
  --code-root /root/autodl-tmp/hsi/code \
  --input-root /root/autodl-tmp/hsi/raw \
  --work-root /root/autodl-tmp/hsi/work \
  --output-root /root/autodl-tmp/hsi/outputs \
  --preflight-only
```

它要求恰好 1 张 24GB 级 NVIDIA GPU、CUDA PyTorch、至少 48GiB RAM、至少 45GiB
工作盘余量，并逐文件验证 bundle；原始 zip 会检查路径穿越、符号链接和比赛目录锚点。
只有看到 `PREFLIGHT_OK` 才启动 fixed：

```bash
tmux new -s hsi-eiou
source /root/autodl-tmp/hsi/venv/bin/activate
python /root/autodl-tmp/hsi/code/run_autodl_eiou.py \
  --code-root /root/autodl-tmp/hsi/code \
  --input-root /root/autodl-tmp/hsi/raw \
  --work-root /root/autodl-tmp/hsi/work \
  --output-root /root/autodl-tmp/hsi/outputs
```

wrapper 会安全解压、运行 exact `8:2,6:2,4:2,4:0` fixed 合同、调用
`check_fixed_split_gate.py`，并输出：

- `outputs/autodl_ablation_yolo26m_eiou_e30/fixed_gate.json`；
- `autodl_audit/` 下的 `nvidia-smi`、`pip freeze`、bundle/profile 合同和 wrapper 日志；
- 每个结果文件的 SHA-256 清单；
- `outputs/autodl_ablation_yolo26m_eiou_e30.tar.gz` 及独立 archive SHA-256。

wrapper **不会**启动 full，也不会上传或提交 Kaggle。fixed 只有在完整 30 epochs、数据/模型/
优化器合同和哈希全部通过，且 best mAP50-95 `>= 0.70443` 时，才取得继续同配方 full 的资格。

### 1. 配置 Kaggle 命令行

安装：

```bash
pip install kaggle
```

然后**自己**去 Kaggle 网站 → Settings → API → Create New Token，把令牌存到
`~/.kaggle/access_token`（`chmod 600` 保护起来）。**令牌等同于账号密码，不要发给别人、不要提交到 git。**

验证：

```bash
kaggle competitions list -s hyperspectral
```

### 2. 创建 Python 环境并安装带 CUDA 的 PyTorch

```bash
python3 -m venv ~/hsi/venv && source ~/hsi/venv/bin/activate
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

确认 GPU 可用（输出 `True` 才能继续）：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ultralytics 不用手动装，训练脚本启动时会自动安装固定版本 8.4.147。
> 但**必须先装好带 CUDA 的 torch**，否则 ultralytics 可能拉下 CPU 版 torch。

### 3. 下载代码包和比赛数据

代码包（私有数据集，含训练流水线、划分清单、预训练权重）：

```bash
kaggle datasets download zephyrpong/hsi-detection-code -p ~/hsi/code --unzip
```

比赛原始数据（二选一，都是同一份数据）：

```bash
kaggle competitions download hyperspectral-object-detection-challenge-2026 -p ~/hsi/raw
```

```bash
kaggle datasets download zephyrpong/hsi-competition-raw -p ~/hsi/raw --unzip
```

**不需要手动解压比赛数据**：脚本会自动找到 zip 并解压到工作目录。
但第二条命令**必须带 `--unzip`**：`kaggle datasets download` 会把数据集再包一层 zip，
不解开外层的话就变成"zip 里套 zip"，脚本找不到里面的 `class.txt`。
（第一条 `competitions download` 下载的 zip 里 `class.txt` 直接在根目录，不需要 `--unzip`。）

### 4. 启动训练

用 `tmux` 运行，**SSH 断开也不会中断训练**：

```bash
tmux new -s hsi
```

在 tmux 里设置参数并启动（下例是**留出集对照实验**）：

```bash
source ~/hsi/venv/bin/activate
export HSI_MODE=ablation HSI_MODEL=yolo26m.pt HSI_EPOCHS=30 HSI_RUN_NAME=cloud_ablation_m1024_e30
export HSI_ATTEMPTS=8:2,6:2,4:2,4:0 HSI_MULTISCALE=0
export HSI_CODE_ROOT=~/hsi/code HSI_INPUT_ROOT=~/hsi/raw HSI_WORK_ROOT=~/hsi/work HSI_OUT_DIR=~/hsi/outputs/cloud_ablation_m1024_e30
python ~/hsi/code/run_hsi_yolo26.py
```

按 `Ctrl+B` 再按 `D` 离开 tmux（训练继续跑）。回来看：`tmux attach -t hsi`。

**三种模式怎么选：**

| HSI_MODE | 训练数据 | 用途 | HSI_MULTISCALE |
|---|---|---|---|
| `smoke` | 2400 张，1 轮 | 验证环境能跑通、测显存和耗时 | 0 |
| `ablation` | 2400 训练 + 600 验证 | **判断新方案是否更好**（有验证分数，对照基准 0.69817） | 0 |
| `full` | 全部 3000 张，关闭验证 | **出正式提交**（只在 ablation 证明有效后再跑） | **1** |

**`HSI_ATTEMPTS` 按服务器内存调**：格式是 `batch:workers[:设备],...`，显存不够或内存报错时自动退到下一组。多卡服务器可写 `16:4:0+1`（两卡 DDP，总 batch 16）。
64GB 内存可以写 `8:4,6:4,4:4,4:0`。

### 5. 查看进度

```bash
cat ~/hsi/outputs/cloud_ablation_m1024_e30/status.json
```

`status.json` 记录每一步的耗时和结果，**出问题先看它**。训练过程看这里：

```bash
tail -f ~/hsi/outputs/cloud_ablation_m1024_e30/train_b*_w*.log
```

显存 / 内存曲线在 `resource_log.csv`（每 30 秒一行）。

### 6. 判断结果

**ablation 模式**：看 `<RUN_NAME>/results.csv` 里 `metrics/mAP50-95(B)` 列的最大值。

- **≥ 0.70443** → 通过当前预注册门禁，才可评估是否接着跑同配方 `full`
- 低于这个值 → 增益在训练噪声范围内（轮间标准差约 0.0016），不值得跑全量

**不要跳过 ablation 直接跑 full**。上一次伪标签实验为赶时间跳过验证，提交后 Kaggle 分数反而掉了 0.0094。

### 7. 提交

`full` 模式跑完后，先把提交文件和权重下载回本地，复核 1,000 张图、框数量、哈希及
单 checkpoint 合规性。上传文件可以先暂存，但 **Competition Submit 必须在操作时取得单独确认**；
不要把 fixed smoke、门禁通过或 CSV checker 通过写成 Kaggle 分数。

明确确认后才执行类似命令：

```bash
kaggle competitions submit hyperspectral-object-detection-challenge-2026 -f ~/hsi/outputs/<RUN_NAME>/submission_<RUN_NAME>.csv -m "说明这次改了什么"
```

同时把权重 `<RUN_NAME>/last.pt` 下载回本地备份。

### 8. 关机

**训练结束后立刻关机或释放实例，云服务器按小时计费。**
关机前确认 `outputs` 目录里的提交文件和权重已经下载回本地。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `status.json` 里 `cuda_available: false` | torch 装成了 CPU 版，回到第 2 步重装带 CUDA 的版本 |
| 所有 ATTEMPTS 都 `cuda_oom: true` | 显存太小，换更大显存的卡，或在 ATTEMPTS 里加更小的 batch（如 `2:1`） |
| `shm_error: true` 或 `MemoryError` | 内存不够，减小 workers；Docker 环境可能是共享内存太小，启动容器时加 `--shm-size=16g` |
| `数据数量不符合预期 2400/600/1000` | 数据下载不完整，删掉 `~/hsi/work` 和 `~/hsi/raw` 重新下载 |
| 重跑很慢 | 正常只有第一次慢：16 波段数据已存在时会自动跳过生成，放在 `HSI_WORK_ROOT` 里别删 |

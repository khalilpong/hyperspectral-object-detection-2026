# 用 Kaggle 免费 GPU 远程训练

> 本文保留历史训练环境与操作记录，旧成绩和计划不代表最终方案。项目已完赛归档，当前结果与恢复入口见 [README](../README.md)。云训练的历史参考见 [CLOUD_SERVER_TRAINING.md](CLOUD_SERVER_TRAINING.md)。

## 为什么要上 Kaggle

本机 RTX 4060 Laptop 只有 **8GB 显存**，yolo26m 被迫 batch=2，全量成绩 0.60553 **反而输给 s 模型的 0.61766**。
Kaggle Notebook 免费提供更大显存的 GPU，可以把 m 模型的 batch 提到 4~8。

## Kaggle 实际给的环境（2026-09-17 实测）

| 项目 | 实测值 |
|---|---|
| GPU | **2 × Tesla T4，每张 15GB** |
| CPU | 4 核 |
| 内存 | 31.3 GB |
| 临时盘 `/tmp` | 约 1 TB（放 16 波段数据绰绰有余） |
| 输出盘 `/kaggle/working` | 19.5 GB（运行结束后可下载） |
| Python / torch | 3.12.13 / 2.10.0+cu128 |
| 联网 | 可以（`pip install` 正常） |
| 额度 | 每周约 30 小时 GPU，单次最长 12 小时 |

> L4（24GB）对本账号**不开放**；脚本目前只用第 1 张 T4（未做多卡并行）。

## 整体结构

```
本机 kaggle_remote/                          Kaggle
────────────────────────────                 ─────────────────────────────────────────
code_dataset/  ──── 上传 ────▶  私有数据集 zephyrpong/hsi-detection-code
  代码（平铺）+ 划分清单 + 预训练权重             │
  + run_hsi_yolo26.py                             │
raw_dataset/   ──── 上传 ────▶  私有数据集 zephyrpong/hsi-competition-raw
  原始比赛 zip（硬链接，不占本地额外空间）          │
                                                  ▼
run_hsi_yolo26.py ──┐                    Notebook（后台运行，可关电脑）
                    ├ make_kernel.py ──▶   1. 装依赖
kernel_<mode>/ ◀────┘   生成   ── 推送 ─▶   2. 找到/解压比赛数据，生成 16 波段数据
                                            3. 训练（显存不够自动降 batch）
outputs/<mode>/ ◀──────── 下载 ──────────   4. 推理 + 校验，产物留在 /kaggle/working
```

**为什么比赛数据要自己上传一份**：见"踩过的坑"第 5 条。两个数据集都是**私有**的，只有本人可见。

**为什么只维护一个主脚本**：Kaggle 的 script 类型 Notebook 只上传一个文件、不能传参数，配置只能写死。
用 `make_kernel.py` 把配置写进主脚本的 `CONFIG` 块再生成 Notebook 目录，避免维护多份脚本。
同一个主脚本在云服务器上用环境变量配置，两边不会分叉。

## 常用命令

⚠️ 所有 Kaggle 命令都要**先 cd 进对应目录再用 `-p .`**（见"踩过的坑"第 1 条）。

**生成 Notebook**（在 `kaggle_remote` 目录下）：

```bash
../.venv/Scripts/python.exe make_kernel.py --mode smoke --model yolo26m.pt --epochs 1
```

```bash
../.venv/Scripts/python.exe make_kernel.py --mode ablation --model yolo26m.pt --epochs 30
```

```bash
../.venv/Scripts/python.exe make_kernel.py --mode full --model yolo26m.pt --epochs 30 --multiscale
```

**推送并启动**：

```bash
cd kernel_smoke && kaggle kernels push -p .
```

**查看状态**（运行中拿不到中间产物，只能看状态）：

```bash
kaggle kernels status zephyrpong/hsi-yolo26m-smoke
```

**结束后下载产物**：

```bash
cd outputs/smoke && kaggle kernels output zephyrpong/hsi-yolo26m-smoke -p .
```

**改了主脚本或本地代码后，更新代码数据集**（先把改动复制进 `code_dataset/`）：

```bash
cd code_dataset && kaggle datasets version -p . -m "说明改了什么"
```

## 产物说明

| 文件 | 内容 |
|---|---|
| `status.json` | 每一步耗时与结果、完整配置、`input_tree`（输入目录结构）、失败原因。**出问题先看它** |
| `resource_log.csv` | 每 30 秒一次的显存、GPU 利用率、内存 |
| `prepare.log` / `train_b*_w*.log` / `predict.log` / `check.log` | 各步骤完整日志（首行是执行的命令） |
| `<RUN_NAME>/results.csv` | 每轮训练指标 |
| `<RUN_NAME>/last.pt`、`best.pt` | 模型权重（smoke 模式不保存） |
| `submission_<RUN_NAME>.csv` | 已通过校验的提交文件 |
| `<notebook-slug>.log` | Kaggle 自带的完整控制台日志 |

## 脚本的自动降级

- **CUDA 显存不足** → 按 `ATTEMPTS` 依次降低 batch：8 → 6 → 4
- **DataLoader 共享内存报错** → 退回 `workers=0`
- **其他错误不重试**，直接失败并把日志尾部写进 `status.json`，避免掩盖真正的 bug
- **多卡 DDP**：`ATTEMPTS` 每组可写第三段设备，`+` 连接多卡，例如 `8:2:0+1` = 两张 T4 分布式训练、总 batch 8（每卡 4）。
  多卡这一组**出任何错都会降级**到后面的单卡组合（DDP 的报错五花八门，不好逐一识别）。
  日志文件名带设备，如 `train_b8_w2_d0+1.log`；`status.json` 里 `trained.device` 是最终实际用的设备。
- **代码包里没有的权重**（如 `yolo26l.pt`）会交给 Ultralytics 联网自动下载。

## 训练数据选择（`--data`）

| 值 | 数据 | 生成脚本 | 图片格式 |
|---|---|---|---|
| `hsi16`（默认） | 16 波段，共用 P0.5~P99.5 缩放 | `prepare_multispectral.py` | `.npy` |
| `hsi16_phase` | 保留 4×4 snapshot mosaic 物理 row/column phase 的 16 波段；phase-aware bilinear v1，保持宽高比、长边默认 1024，共用 native-cube P0.5~P99.5 缩放 | `prepare_phase_aware_multispectral.py` | `.npy` |
| `pseudo_rgb` | 波段 5/8/13 三通道伪RGB | `prepare_pseudo_rgb.py` | `.png` |
| `pseudo_rgb:3,6,8` | 自定义三个波段的伪RGB | `prepare_pseudo_rgb.py --bands 3 6 8` | `.png` |

`pseudo_rgb` 的 Notebook 名、运行名和生成目录都会带 `prgb`（如 `hsi-yolo26m-prgb-full`、`kernel_full_prgb/`），不会覆盖 16 波段的。
伪RGB 的训练/验证划分是在 Kaggle 上按种子**现场重新生成**的，脚本会和 `split_manifest.csv` 逐条比对，不一致直接报错。
（09-18 已在本机验证：重新生成的划分、像素、标注与原数据**完全一致**。）

`hsi16_phase` 只改变输入表示：固定 split、16 通道顺序、标签和 seed 2026 不变；Kernel 实验强制 random extra-channel init，并关闭 SpectralStem、object crops、tile inference。它是 fixed-split 消融，不代表 Public 分数或比赛提交。数据会一次性生成到临时盘；本次 4000 张 NPY 约 33.65 GB，脚本要求至少 45 GiB 可用空间，结束时不会把该缓存带入 Kaggle 输出。

分类损失权重消融通过 `--cls-pw` 单变量控制，并由 runner 写入 `status.json`、训练命令和 `args.yaml`。`hsi-yolo26m-clspw025-ablation` 使用普通 HSI16、固定 2400/600 split、seed 2026、batch 8、workers 2、device 0、30 epochs 和 `cls_pw=0.25`；它只用于 fixed-split 门禁，不自动产生 Public 分数，也不包含 Kaggle Submit。

随机仿射尺度消融通过 `--scale` 单变量控制，并与训练期 `--multi-scale` 明确区分：前者改变每个样本的 random-affine 缩放范围，后者改变训练 batch 的输入尺寸。默认 `scale=0.5`；`hsi-yolo26m-scale030-ablation` 只把它改为 `0.3`，普通 HSI16、固定 2400/600 split、seed 2026、batch 8、workers 2、device 0、30 epochs 和 `cls_pw=0.0` 均保持不变。该运行最佳/最终 `0.69992 < 0.70443`，已否决，不进入 full-data 或比赛 Submit。

随机仿射旋转消融通过 `--degrees` 单变量控制。默认 `degrees=0`；`hsi-yolo26m-deg5-ablation` 只把它改为 `5`，普通 HSI16、固定 2400/600 split、seed 2026、`scale=0.5`、`dfl=1.5`、`cls_pw=0.0` 与其余训练合同保持不变。门禁器必须同时给出 `--architecture yolo --expected-dfl 1.5 --expected-degrees 5 --gate 0.70443`，以排除远端配置漂移。

RT-DETR denoising-query 消融通过 `--rtdetr-num-denoising` 控制，默认 `100`。非默认值只允许 RT-DETR 路径，YOLO 路径会拒绝；训练入口在构建并加载目标模型后修改 decoder 的 `num_denoising`，同时写入模型 YAML metadata。门禁器用 `--expected-rtdetr-num-denoising` 核验精确值。该能力目前只是后备实验接口，不代表 `nd=200` 已经训练或有效。

## 踩过的坑

1. **Kaggle 命令行在 Windows 上不能用带 `/` 的相对路径**。
   `kaggle datasets create -p kaggle_remote/code_dataset` 报
   `No such file or directory: ...uploads\kaggle_remote/code_dataset_xxx.json`，它把路径拼进了临时文件名。
   **解决：先 cd 进目录再 `-p .`**。

2. **上传数据集时子文件夹会被跳过**。所以代码改成**平铺上传**
   （`hsi_detection.spectral.py`、`scripts.train_baseline.py`），脚本启动时按文件名还原目录结构。

3. **校验脚本的默认参数指向本地才有的目录**。`check_submission.py` 的 `--images` 默认
   `data/processed/pseudo_rgb/images/test`，远程不存在，会把每一行判为"未知图片 ID"。
   **必须显式传 `--images`**。推送前已把所有脚本里写死的相对路径排查过一遍。

4. **L4（24GB）对本账号不开放**。元数据写 `"machine_shape": "NvidiaL4"` 推送报
   `400 Bad Request`，没有任何具体原因。去掉后 Kaggle 自动分配了 2×T4。

5. **⭐ 比赛数据无法挂载进 Notebook，而且是静默失败**。元数据里写了
   `"competition_sources": ["hyperspectral-object-detection-challenge-2026"]`，推送成功、没有报错，
   但用 `kaggle kernels pull <id> -m` 拉回 Kaggle 实际保存的配置，发现 **`competition_sources` 变成了空列表**。
   **解决：把原始比赛 zip 上传成私有数据集 `hsi-competition-raw`，改用 `dataset_sources` 挂载**。
   脚本会自动识别"已解压的目录"或"原始 zip"两种形态。

6. **⭐ 先诊断再下结论**。第一次冒烟测试报"找不到 class.txt"时，我猜测是"比赛数据以符号链接挂载、
   Python 3.12 的 rglob 不跟进符号链接"，据此改了代码——**这个猜测是错的**。
   真正起作用的是同时加上的诊断：失败时把 `/kaggle/input` 的目录树写进 `status.json`，
   第二次运行立刻显示只有 `datasets/zephyrpong/hsi-detection-code`，根本没有比赛数据。
   **教训：远程调试一次要十几分钟，每次失败都要让它把"现场"带回来，而不是靠猜。**

7. **改了主脚本必须同步到代码数据集，否则云服务器拿到的是旧版本**。
   Kaggle Notebook 用的是 `make_kernel.py` 生成时**嵌入**的脚本，不读数据集里的那份；
   但云服务器是 `kaggle datasets download` 下载数据集里的 `run_hsi_yolo26.py` 来跑的。
   交接前检查发现数据集里的版本缺少"自动识别 zip"的逻辑，云服务器路线会找不到数据，已同步修复。
   **每次改完主脚本都执行**：

   ```bash
   (cd kaggle_remote && cp run_hsi_yolo26.py code_dataset/ && cd code_dataset && kaggle datasets version -p . -m "说明改了什么")
   ```

9. **Kaggle 的登录凭证约 12 小时过期**（`~/.kaggle/credentials.json`，OAuth 方式）。过期后所有 `kaggle` 命令都报
   `Authentication required` 或 `Permission 'kernels.get' was denied`（后者看起来像"找不到 Notebook"，其实是登录问题）。
   **由用户自己在终端跑 `kaggle auth login` 重新登录**（会开浏览器）。长时间无人值守的任务要考虑这一点。

8. **`prepare_pseudo_rgb.py` 只生成 `dataset.yaml`，不生成全量训练用的 `dataset_all.yaml`**（16 波段脚本两个都生成）。
   本机的伪RGB `dataset_all.yaml` 是早期手工补的，所以本机没暴露；Kaggle 上 full 模式 3 分钟就报 `dataset_all.yaml does not exist`。
   已在 `run_hsi_yolo26.py` 里补写。**教训：新数据类型第一次上 Kaggle，ablation 和 full 最好别同时推**，或者先确认所需文件都会生成。

## 运行记录

| 时间 | Notebook | 结果 | 发现 |
|---|---|---|---|
| 09-17 10:37 | `hsi-yolo26-smoke` v1 | ❌ 12 秒失败 | 找不到 class.txt；确认 GPU 为 2×T4、依赖可安装 |
| 09-17 10:41 | `hsi-yolo26m-smoke` v1 | ❌ 12 秒失败 | 目录树诊断：比赛数据未挂载；`kernels pull` 确认 `competition_sources` 被丢弃 |
| 09-17 10:44~10:51 | 原始数据上传为私有数据集 `hsi-competition-raw` | ✅ 6.23GB，7 分钟（15.6MB/s） | 本地用硬链接暂存，未额外占用 D 盘 |
| 09-17 10:55~11:11 | `hsi-yolo26m-smoke` v2（挂载两个私有数据集） | ✅ **全流程成功，12.9 分钟** | 详见下方"冒烟测试结果" |
| 09-17 14:56~17:00 | `hsi-yolo26m-ablation`（30 轮，batch 8） | ✅ 2.0 小时；**mAP50-95 0.70143**（第 30 轮） | +0.00326 vs s 模型，刚过门槛 0.70117；每轮 3.9 分钟，无 OOM |
| 09-17 17:04~19:40 | `hsi-yolo26m-full`（3000 张 30 轮 + 7 尺度推理） | ✅ 2.6 小时（训练 8660 秒 + 推理 343 秒）；**Kaggle 0.62953 新最佳** | 比 s 模型 +0.00302，留出集增益几乎全部转化 |
| 09-17 19:48~21:31 | `hsi-yolo26l-ablation`（30 轮，**两张 T4 DDP**，总 batch 8） | ✅ 1.6 小时（每轮 3.1 分钟）；mAP50-95 **0.69628 未达标** | **双卡 DDP 一次跑通**；l 单模型比 m 低 0.005，可能是每卡 BN 只有 4 张图 |
| 09-18 00:48 | `hsi-yolo26m-prgb-ablation` + `hsi-yolo26m-prgb-full` v1（伪RGB 数据，**两个同时跑**） | ablation ✅ 1.9 小时，mAP50-95 **0.69726**；full v1 ❌ 3 分钟失败 | Kaggle 允许两个 GPU Notebook 同时跑；full 失败原因见坑 8 |
| 09-18 00:55~03:10 | `hsi-yolo26m-prgb-full` v2（伪RGB 全量，batch 8，30 轮） | ✅ 2.2 小时 | 成为融合成员 C |
| 09-18 03:30~08:30 | `hsi-yolo26m-prgb-full` v3（伪RGB 全量，**batch 3，60 轮**） | ✅ 5.0 小时（每轮 4.8 分钟） | 故意用不同超参做"差异化成员"；权重在 `outputs/prgb-full-b3e60/` |
| 09-18 08:59~11:37 | `hsi-yolo26m-prgb368-full`（**波段 3/6/8** 伪RGB 全量，batch 8，30 轮） | ✅ 2.55 小时 | 新数据类型；`--data pseudo_rgb:3,6,8` |
| 09-21 | `hsi-yolo26m-ablation` v2（普通 HSI16，fixed 2400/600，45 轮） | ✅ 3.09 小时；最佳/最终 `0.69899` | 比同规格 e30 低 `0.00244`，门禁失败；36,080 detections 本地 checker 通过；无比赛 Submit |
| 09-21 | `hsi-yolo26m-phase-ablation` v1（phase-aware HSI16，fixed 2400/600，30 轮） | ✅ 2.50 小时；最佳/最终 `0.69935` | phase-aware bilinear v1，长边 1024；数据准备 21.8 分钟；batch 8/workers 2/device 0；38,304 detections 本地 checker 通过；低于 `0.70443` 门禁；无比赛 Submit |
| 09-21 | `hsi-yolo26m-clspw025-ablation` v1（普通 HSI16，fixed 2400/600，`cls_pw=0.25`，30 轮） | ✅ 2.29 小时；最佳 epoch 27 `0.69863`，最终 epoch 30 `0.69862` | batch 8/workers 2/device 0；无 CUDA OOM/SHM；36,784 detections 本地 checker 通过；低于 `0.70443` 门禁；无比赛 Submit/Public |
| 09-21 | `hsi-yolo26m-scale030-ablation` v1（普通 HSI16，fixed 2400/600，random-affine `scale=0.3`，30 轮） | ✅ 2.06 小时；最佳/最终 epoch 30 `0.69992` | 相对普通同规格 e30 低 `0.00151`，低于 `0.70443` 门禁；batch 8/workers 2/device 0；31,848 detections 本地 checker 通过；无 full-data、无比赛 Submit/Public |

## 冒烟测试结果（2026-09-17，yolo26m，1 轮）

**结论：Kaggle 免费 GPU 够用，训练 yolo26m 不需要租云服务器。**

### 各步骤耗时

| 步骤 | 耗时 |
|---|---|
| 安装依赖 | 13 秒 |
| 生成 16 波段数据（4000 张，4 核） | **5.6 分钟** |
| 训练 1 轮（2400 训练 + 600 验证） | **4.5 分钟**（results.csv 记为 267 秒） |
| 单尺度推理 1000 张 | 67 秒 |

### 资源占用（batch=8，workers=2）

| 指标 | 实测 | 判断 |
|---|---|---|
| **batch 8 一次成功** | 没有触发降级 | m 模型可以用 batch 8（本机只能用 2） |
| 显存峰值 | **13,793 / 15,360 MiB（90%）** | 装得下但已接近上限，**batch 8 就是 T4 的极限** |
| 训练阶段 GPU 利用率 | **中位数 99%** | GPU 满载，瓶颈是 T4 算力，**加 workers 不会更快** |
| 内存 | 峰值占用 6.9 GB，最低可用 22.0 GB / 31.3 GB | 非常宽裕 |

> ⚠️ 全程平均 GPU 利用率只有 30%，**不要据此以为是 CPU 瓶颈**：平均值被前 6 分钟数据生成（GPU 空闲）
> 和每轮末的验证阶段拉低了。只看训练阶段，GPU 一直在 99~100%。

### 第 1 轮指标（仅供参考，不要据此判断好坏）

留出集 mAP50-95 = **0.5003**，mAP50 = 0.773。

低于本机 s 模型第 1 轮的 0.5636 **是正常的**：batch 8 每轮只更新 300 次参数（batch 4 是 600 次），
加上还在学习率预热期。**必须跑满 30 轮再比。**

### 数据挂载的实际路径

| 数据集 | 挂载路径 |
|---|---|
| `hsi-detection-code` | `/kaggle/input/datasets/zephyrpong/hsi-detection-code` |
| `hsi-competition-raw` | `/kaggle/input/hsi-competition-raw`（**Kaggle 自动解压了 zip**） |

两个路径风格不一样，所以脚本用递归搜索而不是写死路径。

### 后续步骤的时间与额度估算

| 步骤 | 预计耗时 | 占每周约 30 小时额度 |
|---|---|---|
| ablation（30 轮，含逐轮验证） | 约 **2.5 小时** | 8% |
| full（3000 张 30 轮 + 7 尺度推理） | 约 **2.5~3 小时** | 10% |

都远低于单次 12 小时上限。

### 已知限制

**显存不足时的自动降级会从头重新训练，不是续训。** 如果跑到第 20 轮才爆显存，前 20 轮就白跑了。
第 1 轮显存峰值 90%，最后 10 轮关闭 mosaic 后显存占用会下降，所以风险不大；
但如果真发生，看 `status.json` 里 `train_attempt_b8_w2` 的 `cuda_oom: true` 就能确认。

## 计划

| 步骤 | 目的 | 状态 |
|---|---|---|
| 1. smoke（1 轮） | 验证整条链路，测显存/内存/每轮耗时，定正式参数 | ✅ **完成**：batch 8 可用，每轮 4.5 分钟 |
| 2. ablation（留出集，30 轮） | m 模型大 batch 能否 ≥ **0.70117**（基准 0.69817 + 0.003） | ✅ **0.70143 达标**（2.0 小时） |
| 3. full（全量，30 轮 + 7 尺度推理） | 仅当第 2 步达标 → 出提交 | ✅ **Kaggle 0.62953，新最佳** |
| 4. yolo26l ablation（双卡 DDP，batch 8） | 更大模型能否 ≥ **0.70443**（m 0.70143 + 0.003） | ❌ 0.69628，单模型不如 m；当第 3 个融合成员只 +0.00076，不训全量 |
| 6. 伪RGB 的 yolo26m（batch 8，波段 5/8/13、3/6/8、0/7/15 各一个） | 当融合成员 | ✅ 全部训好；三个成员都进了八模型融合，Kaggle **0.64831** |
| 7. 新的单变量 HSI16 fixed-split 消融 | 在不违反单模型边界下寻找可归因提升 | SpectralStem、phase-aware、e45、`cls_pw=0.25`、random-affine `scale=0.3` 均已否决；项目已结束训练并归档，最终方案见 README |
| 5. 本地模型融合（m + s 各 7 尺度共 14 路投票） | 留出集上能否比 m 单模型 +0.003 | ✅ 留出集 0.70830（+0.00426）→ **Kaggle 0.63917，新最佳** |

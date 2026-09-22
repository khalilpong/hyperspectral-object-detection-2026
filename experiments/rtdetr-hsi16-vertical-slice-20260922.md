# RT-DETR-L HSI16 本地 vertical slice（2026-09-22）

## 定位

当前 YOLO26m 的 mAP50 很高而严格 IoU 定位仍是主要瓶颈。RT-DETR-L 的
IoU-aware query selection 与 transformer decoder 是一个架构级、仍保持单
detection model / 单 checkpoint 的候选。它尚未获得 fixed-split 分数，本记录
只证明本地软件链路可运行，不代表 Kaggle T4、完整训练或比赛成绩。

## 已实现

- `src/hsi_detection/rtdetr.py`
  - `HSIRTDETRTrainer` 按 dataset `channels: 16` 构建 RT-DETR；
  - 显式将官方 RGB HGStem `model.0.stem1.conv` 复制到目标前三通道；
  - 额外 13 通道默认保留随机初始化；
  - 把迁移契约写入 checkpoint YAML 元数据。
- `scripts/train_rtdetr_hsi.py`
  - 单 checkpoint 训练入口；
  - 因 RT-DETR 的 `F.grid_sample` 与 bipartite matching 约束，明确使用
    `deterministic=False`、`amp=False`。
- `scripts/smoke_rtdetr_hsi.py`
  - 16 通道 forward/loss/backward；
  - 检查 RGB 权重精确迁移和 HSI stem 有限非零梯度。
- `scripts/predict_submission.py --architecture rtdetr`
  - 保存后的 16 通道 RT-DETR checkpoint 可沿用现有 NPY/CSV 推理路径。
- `kaggle_remote/run_hsi_yolo26.py` / `make_kernel.py`
  - 新增显式 `ARCHITECTURE=yolo|rtdetr` 契约；
  - RT-DETR 分支只调用 `scripts/train_rtdetr_hsi.py`，并把同一架构传给推理；
  - 禁止把 YOLO 专属的 DFL、`cls_pw`、SpectralStem、scale/crop/tile 变量混入；
  - RT-DETR smoke 默认按 `batch/workers=2:2,1:2,1:0` 降级。

## 事实证据

官方初始化：

- 文件：`rtdetr-l.pt`
- 下载来源：Ultralytics Assets `v8.4.0` release
- SHA-256：`6DE60B10D4BC566F00CDA0F5B4D64AFE4B66D48DC9695D2171EFFB7859D8E73F`
- Ultralytics：`8.4.147`

结构 smoke（`rtdetr-l.yaml`，CPU，`1x16x64x64`）：

- RGB stem transfer：逐值完全一致；
- stem shape：`32 x 16 x 3 x 3`；
- loss/backward：成功；
- stem gradient：全部有限且绝对值和大于 0。

真实预训练 trainer smoke（现有 `artifacts/smoke_hsi16`，1 train / 1 val，
CPU，1 epoch，imgsz 64）：

- 目标模型：32,846,810 parameters，约 110.7 GFLOPs；
- 兼容权重转移：`940/941`；
- COCO 与比赛类名匹配的 decoder 分类行：`4/18`；
- RGB HGStem 显式迁移成功，额外通道 random；
- 训练、验证、`best.pt`/`last.pt` 保存均成功；
- fresh reload：`channels=16`、stem shape 与迁移元数据均正确；
- smoke `best.pt` SHA-256：`1A5C45C34BC7389C9133A2FAD6F54DB1C91EA46FF28D27E9F0818803ED8E8AB5`；
- fresh-process NPY inference 返回标准 `result.boxes`；
- 现有提交脚本输出 300 框、0 dropped，checker 通过 1/1 图。

完整测试：`112 passed`。私有模板 `kaggle_remote/kernel_smoke_rtdetr`
脚本语法编译通过；配置为 1 epoch、1024、HSI16、`rtdetr-l.pt`、单
checkpoint。RT-DETR 所需三个新/更新源码与 runner 已逐文件 SHA-256 同步到
私有 `hsi-detection-code` staging，新 dataset version 已 `ready`，远端文件清单
核验通过。官方权重未混入 CC0 staging，由 Kernel 联网从 Ultralytics Assets
获取。

## 远程 smoke 结果与下一门禁

独立私有 Kernel `zephyrpong/hsi-rtdetr-l-smoke` version 1 已于 2026-09-22
04:29（北京时间）完成并通过全部硬门禁：

- Kaggle 环境为 2×T4；训练只使用 device 0，batch 2 / workers 2 一次成功；
- GPU 峰值约 7427 MiB，无 CUDA OOM 或 shared-memory 错误；
- 官方预训练迁移 `940/941` 项，4/18 个 decoder 分类行按类名迁移，RGB
  HGStem 显式迁移到 16 通道，额外 13 通道 random；
- 1 epoch 用时 1131 秒，保存并验证 `best.pt`/`last.pt`；独立推理进程从
  `last.pt` 重载，1000 张 16 通道 NPY 推理成功；
- CSV 为 300000 框、1000/1000 图、0 invalid，checker 通过；
- `status.json` / `results.csv` / smoke CSV SHA-256 分别为
  `C707F38A...9A76` / `C20905AF...8A8F` / `3424E5F0...462B`。

尚未证明：30 epoch fixed-val mAP、full-data 或 Public Score。独立 fixed
2400/600 私有 Kernel `zephyrpong/hsi-rtdetr-l-ablation` version 1 已于 04:43
推送，04:44 状态为 `RUNNING`。它必须达到 `0.70443` 才能训练全量，不得与
YOLO26m、DFL2.0/2.5 或任何其他 checkpoint 融合。架构感知
`scripts/check_fixed_split_gate.py` 已兼容 RT-DETR，并回放两条旧 YOLO DFL
结果保持一致。

## 条件式 full-data 包（尚未推送）

等待 fixed 作业期间，已于 2026-09-22 09:47（北京时间）用当前主 runner
生成本地私有包 `kaggle_remote/kernel_full_rtdetr`，远端目标 slug 为
`zephyrpong/hsi-rtdetr-l-full`。它只为缩短 fixed 门禁通过后的周转时间，当前
**没有上传、没有运行，也不是提交候选**。

- 单一 `rtdetr-l.pt` checkpoint，HSI16 全量 3000 张训练，30 epochs，1024；
- attempts `2:2,1:2,1:0`，seed 2026，额外通道 random；
- `num_denoising=100`，无多尺度、crop、tile 或第二 checkpoint；
- 生成 runner 与主 runner 的 diff 仅为预期 CONFIG（以及 `0.70` 的等值文本
  格式化为 `0.7`）；私有 metadata 保持 `is_private: true`；
- 当前全仓测试 `135 passed`。

只有 fixed best mAP50-95 `>=0.70443`、门禁合同与产物哈希全部通过，且 Kaggle
GPU 配额实时恢复可用时，才允许推送该包一次。fixed 失败时不得推送；推送前还
需根据 fixed checkpoint 的同模型单尺度/多尺度验证结果决定是否重生成推理配置。

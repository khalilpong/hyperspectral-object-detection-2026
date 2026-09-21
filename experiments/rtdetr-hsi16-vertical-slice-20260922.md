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

完整测试：`101 passed`。

## 尚未证明与下一门禁

尚未证明：Kaggle T4 显存/吞吐、1024 分辨率 batch、30 epoch fixed split、
正式 fixed-val mAP、full-data 或 Public Score。下一步只能是独立 RT-DETR-L
HSI16 Kaggle smoke（优先 batch 2/1），成功后再跑固定 2400/600；固定划分仍
必须达到 `0.70443` 才能训练全量。它不得与 YOLO26m、DFL2.0 或 DFL2.5 的
预测融合。

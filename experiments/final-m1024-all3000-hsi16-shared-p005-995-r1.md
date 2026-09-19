# YOLO26m HSI16 全量训练与 Kaggle 提交记录

## 实验目的与假设

测试在保留全部 16 个高光谱通道（共享 0.5%–99.5% 百分位缩放）的前提下，将模型容量由 YOLO26s（9.9M 参数）扩大至 YOLO26m（21.8M 参数，76.6 GFLOPs）对细粒度高光谱目标检测性能的影响。

## 配置与训练环境

- 训练代码基准提交：`5ccb19f`
- 模型架构：YOLO26m，初始权重 `yolo26m.pt`（首层扩展为 16 通道，额外 13 通道置零初始化）
- 输入数据：`data/processed/hsi16_shared_p005_995/dataset_all.yaml`（全量 3000 张标注图）
- 训练超参：`epochs=30`、`imgsz=1024`、`batch=2`、`workers=0`、`seed=2026`、`amp=True`、`hsv_h=0, hsv_s=0, hsv_v=0`
- 运行目录：`runs/final_m1024_all3000_hsi16_shared_p005_995_r1`
- 最终轮次（Epoch 30）损失：`box_loss=0.94318`, `cls_loss=0.30573`, `l1_loss=0.00313`

## 推理与提交校验

- 推理设置：单尺度标准推理、`imgsz=1024`、`batch=1`、`half=True`、`conf=0.0001`、`iou=0.70`、`max_det=300`
- 有效预测框数：56,319 个，覆盖 1000/1000 张测试图
- 过滤非有限/零面积框：2,130 个
- 校验脚本：`scripts/check_submission.py`，退出码 0

## 文件指纹

- 提交文件：`submissions/submission_final_m1024_all3000_hsi16_shared_p005_995_r1_conf0001.csv` (SHA-256: `455930C6F3CCDAE95FE5D46EE85D8D2DAD7F33FB5D4E496AF4C22B768F1A9534`)
- 权重文件：`runs/final_m1024_all3000_hsi16_shared_p005_995_r1/weights/last.pt` (SHA-256: `DD3BA2DCA9796F7424CCA753DEA79F47B6484C42B67B19CB6598799E5EF327C0`)
- 结果记录：`runs/final_m1024_all3000_hsi16_shared_p005_995_r1/results.csv` (SHA-256: `AE2CB7F90C1EBBCCF95E39B11238D6AAEAC794734C79AE7BE0DCB5EB37BB4A51`)
- 参数文件：`runs/final_m1024_all3000_hsi16_shared_p005_995_r1/args.yaml` (SHA-256: `13A17EF9040CF59BAC3EE7B38E6260A6D8E6A71F785DCAF5D32FE30A3801ED93`)

## Kaggle 评测结果与深度归因分析

- Kaggle 提交 Ref：`56241140`
- 状态：`SubmissionStatus.COMPLETE`
- **Kaggle Public Score**：**0.60553**（低于 YOLO26s 基线 **0.61766**，差距 -0.01213）

### 归因分析：为什么大模型在全量 3000 样本下出现退化？

1. **批大小（Batch Size）对 BatchNorm 统计的约束**：
   - 8 GB 显存限制下，YOLO26m 必须降至 `batch=2`；而 YOLO26s 可用 `batch=4`。
   - 在 1024 分辨率与 16 通道高维输入下，`batch=2` 的批归一化（BatchNorm）小样本统计波动剧烈，导致特征图归一化噪声显著增大，损害了泛化能力。
2. **容量过剩引发低置信度虚警激增**：
   - 类别级对比表明：对于高置信度目标（`conf > 0.5`），YOLO26m 与 YOLO26s 预测出的真值数量几乎完全一致（例如 `car_toy` 均为 29 个，`egg` 均为 84 vs 102）。
   - 但在极低置信度（`conf < 0.01`）区间，YOLO26m 输出了大量虚假候选框（例如 `car_toy` 虚警多达 4,000+ 个，`banana` 虚警多达 2,700+ 个，`orange` 虚警多达 2,600+ 个）。
   - 虚警在极低置信度下被置信度排序污染，导致每个图像上限 300 个候选框过早被占满（达到 300 框限制的图像数由 YOLO26s 的 13 张上升至 29 张），显著拉低了 mAP 曲线下的 Precision。
3. **结论与后续策略指向**：
   - 盲目扩大模型参数量（m / l）并不是单卡 8GB 资源下的最优解，反而受制于 batch size=2 的噪声和容量过拟合。
   - **最优路线回归**：紧密围绕 **YOLO26s 架构（batch=4 稳定批统计）**，向 **P1: 训练轮数拓展（60 轮充分收敛）** 以及 **P3: 光谱归一化优化** 寻求切实提分。

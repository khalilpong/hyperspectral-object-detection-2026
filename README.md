# Hyperspectral Object Detection Challenge 2026

这是一个以学习完整 AI 训练流程为目标的可复现实验工程。第一阶段先建立可靠的伪 RGB 单模型基线；第二阶段再验证 16 波段是否真正带来增益。

## 已确认的数据事实

- 训练图像：3000 张；测试图像：1000 张。
- 原图：16 位单通道 PNG，4x4 马赛克打包 16 个光谱波段。
- 解包后图像深度：16；标注为 Pascal VOC XML，坐标对应解包后的空间尺寸。
- 类别数：18。
- 主指标：mAP@[0.5:0.95]。
- 比赛限制多模型集成；同一模型的 TTA/多尺度推理允许。

## 学习路线

1. `audit_dataset.py`：理解数据、类别、框和异常样本。
2. `prepare_pseudo_rgb.py`：把 4x4 马赛克解成 16 波段，再选 3 个波段生成伪 RGB，并转换为 YOLO 标注。
3. `train_baseline.py`：用单个预训练 YOLO 模型建立可重复的基线。
4. `predict_submission.py`：在测试集推理并生成 Kaggle CSV。
5. 基线稳定后，再实现 16 通道/光谱融合模型，并严格用同一验证集比较。

## 环境

项目使用独立的 Python 3.11 虚拟环境。Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 数据流程

下载后的压缩包放在 `data/raw/`，解压到 `data/raw/extracted/`，然后执行：

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py
.\.venv\Scripts\python.exe scripts\prepare_pseudo_rgb.py --bands 5 8 13 --seed 2026
```

训练一个轻量基线：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py --model yolo26n.pt --epochs 30
```

每次训练会使用独立的实验目录。若训练被中断，可从对应的 `last.pt` 继续：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py --resume runs\baseline\weights\last.pt
```

恢复时不写其他参数会完整继承 checkpoint 的训练尺寸、batch、设备、验证和绘图设置。只有确实需要修改显存相关设置时，才显式附加 `--imgsz`、`--batch`、`--device` 或 `--workers`；脚本会拒绝 Ultralytics 无法安全覆盖的 epoch、模型、seed 和运行名，避免默认值静默污染续训。

生成提交文件：

```powershell
.\.venv\Scripts\python.exe scripts\predict_submission.py --weights runs\baseline\weights\best.pt
.\.venv\Scripts\python.exe scripts\check_submission.py submission_baseline.csv
```

如果 640 像素推理显存不足，启用 FP16 并把 batch 降为 1：

```powershell
.\.venv\Scripts\python.exe scripts\predict_submission.py --weights runs\baseline\weights\best.pt --imgsz 640 --batch 1 --half --output submission_baseline.csv
```

所有随机划分都会写入 `data/processed/pseudo_rgb/split_manifest.csv`，确保后续模型使用同一训练/验证划分。

训练过程中的概念和阶段门槛见 [`docs/learning_path.md`](docs/learning_path.md)，实验结果统一登记在 `experiments/experiments.csv`。
预训练模型的来源、版本与许可说明见 [`docs/model_provenance.md`](docs/model_provenance.md)。

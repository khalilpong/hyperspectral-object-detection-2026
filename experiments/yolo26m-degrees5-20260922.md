# YOLO26m `degrees=5` fixed-split gate（2026-09-22）

## 目的

当前普通 HSI16 YOLO26m fixed 2400/600 基线最佳为 `0.70143`，而 mAP50 已很高，主要缺口在严格 IoU 定位。现有训练使用 Ultralytics 默认 `degrees=0`；本实验只增加小角度随机旋转，检验几何扰动能否提高未见图像上的边界稳定性。

## 预注册合同

- 单个 YOLO26m checkpoint；不与任何 checkpoint 融合；
- 数据：普通 `hsi16_shared_p005_995`，fixed 2400 train / 600 val；
- `epochs=30`、`imgsz=1024`、首选 `batch=8/workers=2/device=0`，保留既有自动降级；
- `seed=2026`、random extra-channel init；
- 唯一训练变量：random-affine `degrees: 0.0 → 5.0`；
- `scale=0.5`、`dfl=1.5`、`cls_pw=0.0`、普通 HSI16 输入均保持不变；
- full-data 门禁：标准 full-val 最佳 `mAP50-95 >= 0.70443`；低于门禁立即否决，不训练 full-data、不提交测试 CSV。

## 实现与本地验证

- `scripts/train_baseline.py` 暴露并校验 `--degrees`，resume 禁止覆盖该训练变量；
- `kaggle_remote/run_hsi_yolo26.py` 将 `HSI_DEGREES` 写入 status 配置和训练命令；RT-DETR 路径拒绝该 YOLO 专属变量；
- `kaggle_remote/make_kernel.py` 生成隔离的 `_deg5` 运行名、目录和私有 Kernel slug；
- `scripts/check_fixed_split_gate.py` 使用 `--expected-degrees 5` 审计精确变量，并兼容旧 status 中缺失的新默认字段；
- 全仓测试：`122 passed`；训练/RT-DETR/远程生成器的针对性子集：`57 passed`；
- 真实 `rtdetr-l.pt` 构造验证了后备 `num_denoising=200` 控制落在 16 通道目标模型 decoder，不是只改命令字符串；
- 私有代码数据集新版本上传后状态为 `ready`；生成的 Kernel metadata 为 `is_private: true`。

## 远程状态

- Kernel：`zephyrpong/hsi-yolo26m-deg5-ablation`
- version：`1`
- 推送：2026-09-22 05:29（北京时间）
- 05:29 实时状态：`RUNNING`
- Competition Submit：无

完成后下载到 `kaggle_remote/outputs/degrees5_ablation`，使用：

```powershell
.\.venv\Scripts\python.exe scripts\check_fixed_split_gate.py `
  --status kaggle_remote\outputs\degrees5_ablation\status.json `
  --results kaggle_remote\outputs\degrees5_ablation\kaggle_ablation_yolo26m_deg5_e30\results.csv `
  --architecture yolo --expected-dfl 1.5 --expected-degrees 5 `
  --gate 0.70443 `
  --output artifacts\gates\yolo26m_deg5_fixed_20260922.json
```

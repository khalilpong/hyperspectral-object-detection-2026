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

## 远程结果

- Kernel：`zephyrpong/hsi-yolo26m-deg5-ablation` version 1；2026-09-22 05:29（北京时间）推送，07:45 前后完成；
- 数据、模型和唯一变量合同均通过门禁器：普通 HSI16、fixed 2400/600、YOLO26m、30 epochs、1024、seed 2026、random extra-channel init、`scale=0.5`、`dfl=1.5`、`degrees=5`、CIoU；
- 首个 `batch=8/workers=2/device=0` attempt 成功，训练 return code 0；无 CUDA OOM 或 shared-memory 错误；训练约 7,223 秒，全流程约 7,618 秒；
- 最佳 epoch 29：`mAP50-95=0.68169`、`mAP50=0.95954`；最终 epoch 30：`mAP50-95=0.68092`；
- 相对普通同规格 e30 基线 `0.70143` 下降 `0.01974`，相对 full-data 门槛 `0.70443` 低 `0.02274`；
- fresh checkpoint reload 与 1000 张 test NPY 推理通过，结构 checker 验证 36,566 条检测、1000/1000 图像；
- 结论：**门禁失败并否决**。不跑对应 full-data，不上传生成的测试 CSV，无 Competition Submit。

门禁复核命令：

```powershell
.\.venv\Scripts\python.exe scripts\check_fixed_split_gate.py `
  --status kaggle_remote\outputs\degrees5_ablation\status.json `
  --results kaggle_remote\outputs\degrees5_ablation\kaggle_ablation_yolo26m_deg5_e30\results.csv `
  --architecture yolo --expected-dfl 1.5 --expected-degrees 5 `
  --gate 0.70443 `
  --output artifacts\gates\degrees5_fixed_gate_20260922.json
```

审计哈希：

- `status.json`：`6B495198231FF37D6E095F9BC4179B84FE65E1305D2F9EEF263FAE06D3BF2408`
- `results.csv`：`0AFDEF18C1B38C3D1B5455457BFBA636CB0785863922026DC77CA048A1510D2B`
- `best.pt`：`377994227058D02A404B16B4807E1150D97A70F54FBB22AFD3519458035145A1`
- `last.pt`：`64A19672E0DBB6BD1E6F21B392A41B2AB6508D0F831C618053B685CDDE7D2CDF`
- 测试 CSV：`295F4BCDAEE69334F6FB427C3B619561A34C0917635D1B1BBA5A1EA425115D53`

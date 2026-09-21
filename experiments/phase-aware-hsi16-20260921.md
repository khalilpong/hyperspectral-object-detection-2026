# Phase-aware HSI16 fixed-split 消融（2026-09-21）

## 结论

phase-aware bilinear v1 的工程链路与私有 Kaggle 训练均完整通过，但性能门禁失败：最佳/最终 epoch 30 的 `mAP50-95=0.69935`，比普通同规格 e30 的 `0.70143` 低 `0.00208`，比全量门槛 `0.70443` 低 `0.00508`。因此该方向停止：不跑 full-data、不上传测试 CSV、不得描述为 Public 成绩。

## 要验证的问题

普通 HSI16 先把每个 4×4 snapshot mosaic cell 解成一个 16 通道像素，后续 resize 会把 16 个物理采样点当作共址。这个实验只改变输入表示：保留每个波段在 4×4 cell 内的真实 row/column phase，再把各波段独立插值到模型实际使用的几何画布；其余训练变量保持不变。

这是一种 geometry-aware interpolation，不是超分辨率，不能恢复未采样位置的真实高频光谱信息。

## 单变量合同

| 项目 | 普通基线 | phase-aware 实验 |
|---|---|---|
| 数据划分 | 固定 2400 train / 600 val | 相同 manifest |
| 测试集 | 1000 张 | 相同 |
| 模型/初始化 | YOLO26m / `yolo26m.pt` / random extra-channel init | 相同 |
| 训练 | 30 epochs，imgsz 1024，batch 8，workers 2，device 0，seed 2026 | 相同 |
| 通道 | 16；`5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15` | 相同 |
| 编码 | 每图、跨通道共享 native-cube P0.5–P99.5，uint8 | 相同 |
| SpectralStem / object crops / tile inference | 关闭 | 关闭 |
| 唯一变量 | 4×4 解包后的 compact cube | 保留物理 phase 的 bilinear v1，保持宽高比、长边 1024 |

源码对应本地提交 `c0a3127`。核心实现：

- `src/hsi_detection/spectral.py`：目标形状与 per-band phase-aware bilinear reconstruction；
- `scripts/prepare_phase_aware_multispectral.py`：独立数据生成器、合同防混写、原子写盘与报告；
- `kaggle_remote/run_hsi_yolo26.py` / `make_kernel.py`：`hsi16_phase` 远程入口与单变量约束；
- `tests/test_spectral.py` / `tests/test_prepare_phase_aware.py` / `tests/test_kaggle_remote.py`：物理采样点、phase offset、常量场、续跑合同和 Kernel 渲染测试。

本地全套测试：`75 passed in 10.99s`。

## 几何定义

物理波段 `b` 的 phase 为 `r=b//4`、`c=b%4`，其 native 样本为：

```text
B_b[i,j] = raw[4*i+r, 4*j+c]
```

目标像素中心先映回 raw mosaic：

```text
y_raw = (v+0.5) * R/H - 0.5
x_raw = (u+0.5) * C/W - 0.5
```

再映入该波段自己的 lattice：

```text
y_b = (y_raw-r)/4
x_b = (x_raw-c)/4
```

每个波段独立做 bilinear interpolation，边界使用 replicate；输出保持原始宽高比，长边固定为 1024，不在数据制作阶段 padding，最终 letterbox 仍由 Ultralytics 负责。

## 私有 Kaggle 运行

- Kernel：`zephyrpong/hsi-yolo26m-phase-ablation` version 1，私有；
- run：`kaggle_ablation_yolo26m_phase_e30`；
- 结果：`success`，return code 0，无 CUDA OOM、shared-memory error 或 NaN；
- 环境：2×Tesla T4（实际 device 0）、Python 3.12.13、Torch 2.10.0+cu128、Ultralytics 8.4.147；
- 数据准备：4000 张全部重新生成，2400/600/1000；耗时 1308.5 秒；NPY 共 33,652,936,704 bytes；
- 训练：30/30 epochs，2.087 小时；
- 总流程：8992.8 秒（约 2.50 小时）；
- 标注 sanitation 与普通路线一致：clipped 3、dropped 41。

## 指标与门禁

| 指标 | 普通 e30 | phase-aware e30 | 差值 |
|---|---:|---:|---:|
| mAP50 | 0.95327 | 0.95838 | +0.00511 |
| mAP50-95 | 0.70143 | 0.69935 | -0.00208 |
| 相对全量门槛 0.70443 | -0.00300 | -0.00508 | -0.00208 |

phase run 的训练 CSV 没有 mAP75 字段，不能补写该指标。epoch 30 同时是该 CSV 的最大值和最终值。最后 10 轮从 `0.68575` 总体上升到 `0.69935`，但仍未超过普通 e30 或门槛。

逐类 mAP50-95 相对普通 e30 的主要变化：`e-bike +0.042`、`people +0.011`、`banana_plastic +0.010`；`stone_block -0.023`、`orange -0.019`、`charger_head -0.017`。总体 mAP50 上升而 mAP50-95 下降，说明粗召回改善没有转化为更严格的定位质量。

## 产物完整性

推理使用该 run 的 `last.pt`、单尺度 1024、batch 1。独立 checker 结果为 `valid submission: 38304 detections across 1000 images`。这只证明本地/Notebook 产物格式有效，不是 Kaggle Submit 或 Public 分数。

| 产物 | bytes | SHA-256 |
|---|---:|---|
| `status.json` | 2,577 | `4367D11B36BFA0701F3C85A888747380D5B7E766D59B4DF3472236092B8768E9` |
| `results.csv` | 3,894 | `FEDE9BC06541A9EC7B35A1D8CA04B52389AA9C28DEE59F7AAD31FF8990184749` |
| `best.pt` | 44,130,841 | `8112CAF4D3FF0716D11CEB395713429774A400394D4CC5F8B250A271C9CEFA91` |
| `last.pt` | 44,130,841 | `AF22016B99E9B3735F79C82B41B001F378D5E88CD1BA22F7118CA7F62828ACB4` |
| 测试 CSV | 2,298,609 | `87B77896FC0D26DE45747A308769754D8F794873D63171D761CCB7D123E4A573` |

`best.pt` 与 `last.pt` 字节数相同但 SHA-256 不同，符合 best/last checkpoint 语义。日志中的 Pillow `mode` 参数和 Ultralytics `half` 参数 deprecation warning 均未影响本次成功；未来升级依赖时需分别移除显式 `mode="RGB"`，并复核 `quantize` 接口。

## 决策

1. `0.69935 < 0.70443`，fixed-split 门禁失败；
2. 不启动 full-data phase-aware 训练；
3. 不上传本次测试 CSV，不消耗比赛提交额度；
4. 不把 checker 通过写成 Kaggle 提交或 Public 成绩；
5. 截止前不重复该 phase-aware bilinear v1 路线。

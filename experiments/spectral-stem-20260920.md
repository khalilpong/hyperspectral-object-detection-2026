# SpectralStem 单模型路线（2026-09-20）

## 决策

当前合规 Public 最佳 `0.63072` 与目标 `0.66` 仍差 `0.02928`；同 checkpoint 后处理从 `fusion_iou=0.70` 调到 `0.74` 只增加 `0.00006`。下一条 fixed-split 主线改为一个 checkpoint 内的可学习光谱投影：

```text
16-channel HSI NPY
  -> Conv2d(16, 3, kernel_size=1, bias=False)
  -> original pretrained YOLO26m first block and unchanged detector
```

这仍是一个 detection model、一个 checkpoint，不使用多模型融合。

## 接口与初始化不变量

- 当前 NPY 的物理波段顺序是 `5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15`。
- 因此物理波段 `5/8/13` 在数组中的通道索引是 `(0,1,2)`；projection 的 step-0 权重只设置 `W[0,0]=W[1,1]=W[2,2]=1`，其余为 0。
- 原 YOLO 第一层保持 3 通道输入和原预训练权重；projection 与它一起放在 `model.model[0]`，不改变后续层编号。
- checkpoint 只序列化 `torch.nn.Sequential` 和 `torch.nn.Conv2d` 等标准模块。项目代码负责新训练与 resume 时的构造/校验，但普通预测和受限安全加载不依赖反序列化自定义类。
- model YAML 保持外部输入契约 `channels: 16`，并保存 band order、物理 identity bands 和数组索引 metadata。

实现：

- `src/hsi_detection/spectral_stem.py`
- `scripts/train_baseline.py --spectral-stem`
- `kaggle_remote/run_hsi_yolo26.py` / `make_kernel.py`

## 本地门禁结果

以下验证均已通过：

1. projection 初值逐元素等于前三数组通道的 identity，额外 13 通道初值为 0 且能收到非零梯度。
2. 包裹前后原 detector 第一块的权重逐元素不变；真实 YOLO26 模型能接收 16 通道并完成 forward。
3. checkpoint 在全新 Python 进程中加载；子进程主动禁止任何 `hsi_detection` import 仍能 forward。
4. `ULTRALYTICS_SAFE_LOAD=1` 能恢复 checkpoint；stem 类型仍是标准 `Sequential + Conv2d`。
5. 真实 16 通道 NPY 的 load/predict 成功，checkpoint YAML 保持 `channels=16`。
6. 64 train / 4 val、YOLO26n、CPU、`imgsz=128`、batch 4、1 epoch 的端到端训练退出码为 0。保存后的 projection 相对 identity：
   - `identity_delta_abs_sum = 0.02143645`
   - 额外 13 通道 `extra_abs_sum = 0.01773190`
   - `max_delta = 0.00048828125`

第 6 项证明 projection 不只“有梯度”，还真实进入 trainer optimizer、被保存到 checkpoint。该小样本 mAP 为 0，只是链路 smoke，不是性能证据。临时 run、临时样本列表和子集 `train.cache/val.cache` 已精确清理。

## Kaggle fixed-split 实验结果

- 私有代码数据集 `zephyrpong/hsi-detection-code` 版本 8 已于 2026-09-20 创建并达到 `ready`；新增/更新：
  - `hsi_detection.spectral_stem.py`
  - `scripts.train_baseline.py`
  - `run_hsi_yolo26.py`
- 私有 Kernel staging：`kaggle_remote/kernel_ablation_stem`
- 私有 Notebook：`zephyrpong/hsi-yolo26m-stem-ablation`，版本 1；训练已正常完成，`status.json` 为 `result: success`。
- 配置：YOLO26m、HSI16 P0.5-P99.5、固定 2400/600、30 epochs、`imgsz=1024`、seed 2026、首选 batch 8 / workers 2 / device 0。
- 唯一实验变量：`SpectralStem 16->3`；不启用 object crop、tile inference、训练期 multiscale 或多模型融合。
- 首次 batch 8 / workers 2 / device 0 即成功，训练 `7119 s`，总 Kernel `7547.2 s`；无 CUDA OOM 或 shared-memory 错误。

上传前核验：本地载荷 18 个文件、64,840,426 bytes，移除了 13 个可重建 `.pyc`；三份关键源码与 staging SHA-256 一致，凭证扫描无命中。远端版本 8 的 `hsi_detection.spectral_stem.py`、`scripts.train_baseline.py`、`run_hsi_yolo26.py` 大小分别为 12,485 / 9,157 / 26,377 bytes，与本地一致。

标准 full-val 的最佳与最终均为 epoch 30：

| 指标 | SpectralStem | 旧同规格 YOLO26m | 差值 |
|---|---:|---:|---:|
| mAP50 | 0.95719 | 0.95327 | +0.00392 |
| mAP50-95 | 0.69930 | 0.70143 | -0.00213 |
| 相对全量门槛 0.70443 | -0.00513 | -0.00300 | -0.00213 |

类别变化以 mAP50-95 计：`car +0.014`、`people +0.012`、`banana_plastic +0.007`，但 `stone_block -0.048`、`orange -0.015`、`car_toy -0.012`。mAP50 上升而 mAP50-95 下降，说明粗粒度检出略有改善，但高 IoU 定位没有改善；这是指标解释，不是因果证明。

产物核验：

- `status.json` 记录 train/val/test=`2400/600/1000`、train return code 0、`SPECTRAL_STEM=true`。
- 测试 CSV 经本地 checker 验证为 44,367 detections / 1000 images；未上传 Kaggle，不能视作 Public 分数。
- `best.pt` / `last.pt` 都为 44,119,956 bytes，SHA-256 分别为 `E629E810103CB703BB97EFA110FD6382425A9CAF8337E2A9CB38348281C269B0` / `7AC65DCAC5D1439E3C1667E0E10A8F00737670FABB79AF7243E3745A8D86970D`。
- 两个 checkpoint 均恢复出 `channels=16`，band order、identity physical bands 与输入索引 metadata 一致；`best.pt` 还通过了禁止 `hsi_detection` import 的全新进程加载与 forward。
- `status.json` SHA-256：`998AD19D934CA7DAB126770FEDF1DC18BDF4F6DF48DAB88A6E5E32775C0030F0`；`results.csv` SHA-256：`29F822EA1C4E4AFE38721D6A31DA1D98AAC7EAE365D3E0C3DC25FA87BC07C880`。

## 决策门禁

- 门槛是标准 full-val `mAP50-95 >= 0.70443`；实测 `0.69930`，低 `0.00513`，门禁失败。
- 决策：记录负结果并停止 SpectralStem；不启动全量 3000 张训练，不生成或上传正式提交候选。
- 远端自动生成的测试 CSV 仅作为链路结构校验，不进入比赛提交清单。

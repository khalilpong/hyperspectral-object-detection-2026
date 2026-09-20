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

## Kaggle fixed-split 实验状态

- 私有代码数据集 `zephyrpong/hsi-detection-code` 版本 8 已于 2026-09-20 创建并达到 `ready`；新增/更新：
  - `hsi_detection.spectral_stem.py`
  - `scripts.train_baseline.py`
  - `run_hsi_yolo26.py`
- 私有 Kernel staging：`kaggle_remote/kernel_ablation_stem`
- 私有 Notebook：`zephyrpong/hsi-yolo26m-stem-ablation`，版本 1，2026-09-20 20:35（北京时间）状态 `RUNNING`。
- 配置：YOLO26m、HSI16 P0.5-P99.5、固定 2400/600、30 epochs、`imgsz=1024`、seed 2026、首选 batch 8 / workers 2 / device 0。
- 唯一实验变量：`SpectralStem 16->3`；不启用 object crop、tile inference、训练期 multiscale 或多模型融合。
- 参考 zero-init fixed-split 的实测总时长 `7247 s`，预计约 2 小时 Kaggle T4；projection 的额外计算量很小。

上传前核验：本地载荷 18 个文件、64,840,426 bytes，移除了 13 个可重建 `.pyc`；三份关键源码与 staging SHA-256 一致，凭证扫描无命中。远端版本 8 的 `hsi_detection.spectral_stem.py`、`scripts.train_baseline.py`、`run_hsi_yolo26.py` 大小分别为 12,485 / 9,157 / 26,377 bytes，与本地一致。当前已消耗 Kaggle GPU 运行 fixed split；**没有启动全量训练，也没有新的竞赛 Submit。**

## 决策门禁

- 先运行 fixed 2400/600；标准 full-val `mAP50-95 >= 0.70443` 才允许全量 3000 张训练。
- 未达 `0.70443`：记录负结果并停止 SpectralStem，不生成正式提交候选。
- 达标：下载并核对 `status.json`、`results.csv`、`best.pt/last.pt`、哈希和 checkpoint stem metadata；再单独请求全量训练授权。
- 全量完成后仍需独立检查 1000/1000 测试图、0 非法框；最终 Kaggle Submit 必须再次取得动作时确认。

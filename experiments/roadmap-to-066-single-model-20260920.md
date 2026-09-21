# 0.66 冲刺路线：单模型、可复现、门禁式推进（2026-09-20）

## 结论先行

当前奖项合规的 Public 最佳是 `0.63072`，目标 `0.66` 仍差 `0.02928`。这是一次显著跃升，不能承诺一定达到。`fusion_iou 0.70 -> 0.74` 的最后一次受控后处理检查只带来 Public `+0.00006`；随后显式 horizontal-flip-only、YOLO26m 延长到 45 轮、phase-aware HSI16 输入重建、`cls_pw=0.25` 和 random-affine `scale=0.3` fixed-split 消融也都未过门禁。低成本推理、单纯延长训练、phase-aware 输入、分类频次加权和这条温和 scale 路线均已否决；下一条主线若继续，必须是新的单模型训练变量，并继续用固定 2400/600 门禁控制 GPU 成本。

执行顺序：

1. zero-init 固定划分已完成并否决：最佳 `0.69690 < 0.70443`；
2. object-crop 固定划分已完成并否决：`0.69772 < 0.70443`；当前 checkpoint 的重叠切片推理也已否决；
3. 同 checkpoint `fusion_iou=0.74` 已提交，Public `0.63072`，只比 `f0.70` 高 `0.00006`；停止继续细扫后处理；
4. `SpectralStem 16→3→YOLO26m` 的本地链路通过，但 fixed split 最佳/最终仅 `0.69930 < 0.70443`，已否决且不跑全量；
5. 显式 horizontal-flip-only 已完成：cache control 通过，但最佳七尺度+flip 仅比 supported-full 高 `+0.00016978 < +0.001`，已否决；
6. 同规格 YOLO26m 45 轮 fixed split 已完成：最佳/最终 `0.69899 < 0.70443`，不启动全量训练；
7. phase-aware bilinear v1 fixed split 已完成：最佳/最终 `0.69935 < 0.70443`，较普通同规格 e30 低 `0.00208`，不启动全量训练；
8. `cls_pw=0.25` fixed split 已完成：最佳 epoch 27 为 `0.69863 < 0.70443`，较普通同规格 e30 低 `0.00280`，不启动全量训练；
9. random-affine `scale=0.3` fixed split 已完成：最佳/最终 `0.69992 < 0.70443`，较普通同规格 e30 低 `0.00151`，不启动全量训练；
10. 当前没有正在运行或已授权的新训练，任何新 GPU 运行和 Kaggle Submit 仍需单独确认。

## 合规边界（事实）

- 竞赛规则要求最终只有一个 detection model，禁止不同模型的 voting、weighted fusion 和 post-NMS fusion；代码审核还要求提交训练/推理源码、权重与可复现说明。[Competition Rules](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/rules)
- 主办方明确允许同一训练模型的 flip、scale、crop TTA，以及把这些输入变换的预测合并；公开 ImageNet/COCO 预训练 backbone 也允许，但必须记录模型、来源与许可证。[Organizer clarification](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863)
- DINO teacher 的竞赛讨论目前没有主办方答复，外部卫星预训练权重又涉及未声明外部数据风险，因此不作为奖项合规主线。[DINO teacher clarification request](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/737202)

因此，本路线只加载一个 checkpoint；full-image multi-scale 和 tile 都是该 checkpoint 的输入变换。历史多模型成绩只保留为审计记录，不能再作为最终候选。

## 当前证据

### 合规基线

- 模型：一个 YOLO26m、16 通道 HSI、P0.5–P99.5 shared per-image uint8 编码；
- 推理：同 checkpoint 七尺度 `832/896/960/1024/1088/1152/1216`；
- support gain：`0.125`；
- Public：`0.63072`，ref `56392305`；上一版 `f0.70` 为 `0.63066`，ref `56379896`；
- 记录：[single-model-compliant-recovery-20260920.md](single-model-compliant-recovery-20260920.md)。

### fixed-split 误差画像

七尺度 fixed-split 结果约为：

| 指标 | 数值 |
|---|---:|
| mAP50-95 | 0.704035 |
| mAP50 | 0.955741 |
| mAP75 | 0.852002 |
| 预测框数 | 71,179 |

最弱类别 AP50-95：`car 0.402576`、`people 0.448190`、`stone_block 0.474740`、`e-bike 0.519522`。同类最佳匹配在 IoU 0.75 的通过率分别约为 `56.4% / 64.9% / 73.6% / 78.9%`。验证 GT 中 `70%` 的框面积低于 `1000 px²`。这组证据说明主要瓶颈是小目标的精定位，而不是 mAP50 召回或简单置信度阈值。

### 已否定方向

本项目已有实测负结果：`imgsz=1280`、训练期 multi-scale、P2 head、close-mosaic=20、box loss=10、P1–P99、伪标签、YOLO26l、SpectralStem、phase-aware HSI16、`cls_pw=0.25`、random-affine `scale=0.3`、horizontal-flip-only，以及同规格 YOLO26m 单纯延长到 45 轮。它们不应在截止前重复消耗 GPU。详见 [HANDOFF.md](../HANDOFF.md) 与 [experiments.csv](experiments.csv)。

## 已实现的对象裁剪

实现：

- [prepare_object_crops.py](../scripts/prepare_object_crops.py)
- [tiling.py](../src/hsi_detection/tiling.py)
- [predict_submission.py](../scripts/predict_submission.py)
- [eval_tiled_inference.py](../scripts/eval_tiled_inference.py)

固定参数：

- crop：原图坐标 `128×256`；
- 每个有有效目标的 train 原图最多一个 crop；
- anchor 只依据训练标签中的类别频率与框面积选择，不读取 val AP；
- 确定性 jitter：最多为 tile 边长的 `15%`，seed `2026`；
- 标签保留：原框可见面积至少 `90%`；
- 直接切片已编码的 16 通道 NPY，不对 crop 重新估计 percentile；
- fixed split 的 val 仍是原始 600 张完整图，不生成或训练 val crop。

本机真实数据结果：2400 张 train 原图生成 `2376` 个 crop；24 张源图在清洗后没有可用框而跳过。Ultralytics 扫描 `4776/4776` 个 full+crop 样本，报告 `0 background / 0 corrupt`，并实际加载出 `(16, 1024, 1024)` crop tensor。

训练链路也做了真实反向传播 smoke，而不只停留在数据扫描：YOLO26m 构造出的首层为 `16→64`，预训练权重转移 `768/768`。先用 full+crop YAML 的 `fraction=0.01` 跑通 48 个样本、1 epoch；由于列表顺序使这 48 个样本都来自 full 目录，又补做了 crop-only smoke。后者明确扫描 `labels/train_object_crops` 的 24 个 crop，完成 12 个 batch、1 epoch，退出码为 0，`box/cls/l1` loss 均为有限值。两次 smoke 使用 `imgsz=256`、只用于证明读取和反向传播契约，不是性能实验；其验证 mAP 没有决策意义。

用户明确授权后，最小增量数据集已作为私有 `zephyrpong/hsi-object-crop-code` 创建，状态 `ready`、版本 1；远端四个数据文件（3 个源码文件和 SHA-256 manifest）的名称与大小均已逐项核对。它共含 `52,032` bytes 源码，不含原始比赛数据、crop 数据、模型权重、split manifest、submission 或凭证。私有 fixed-split Kernel `zephyrpong/hsi-yolo26m-crop-ablation` 随后推送版本 1。Kernel 复用原私有代码/权重与原始数据集，再叠加该增量代码；没有竞赛提交步骤。

### Object-crop 结果：否决

- 训练成功，return code 0，无 CUDA OOM 或 SHM 错误；2400 张 train 完整图生成 2376 张 train-only crop，600 张 val 保持原始完整图。
- 标准 full-val 最佳与最终均为 epoch 30 的 `0.69772`，比旧同规格 `0.70143` 低 `0.00371`，比全量门槛 `0.70443` 低 `0.00671`。
- 结论：不训练全量 object-crop 模型，不对该模型做 tile 扩展，不提交其测试 CSV。
- `status.json` SHA-256：`DBAD966C5272EBF8FDF8A986B59749D43AF64ADF7487D0451D45D7D86861051E`；`results.csv` SHA-256：`49791CFA5EE7DFB8BAC11456B11BA256844DEE40F5D1F841F6DBC8BF6373FB69`。

### Zero-init 结果：否决

- 训练成功，最佳 epoch 28 为 `0.69690`，最终 epoch 30 为 `0.69648`。
- 最佳值比旧同规格 `0.70143` 低 `0.00453`，比全量门槛 `0.70443` 低 `0.00753`。
- 结论：不训练全量 zero-init 模型，不提交其测试 CSV。

### SpectralStem 结果：否决

- 私有 fixed 2400/600 训练正常完成，return code 0，无 CUDA OOM 或 shared-memory 错误；最佳与最终均为 epoch 30 的 `mAP50-95=0.69930`。
- 相对旧同规格 YOLO26m，mAP50 从 `0.95327` 提升到 `0.95719`（`+0.00392`），但 mAP50-95 从 `0.70143` 降到 `0.69930`（`-0.00213`）；相对全量门槛仍差 `0.00513`。
- 类别层面 `car +0.014`、`people +0.012`，但 `stone_block -0.048`、`orange -0.015`、`car_toy -0.012`。这支持“粗召回略好、严格 IoU 定位未改善”的解释，不支持全量训练。
- 生成的 1000 图测试 CSV 通过本地结构校验（44,367 detections），但这不是 Public 分数，也没有上传比赛。
- `best.pt` SHA-256：`E629E810103CB703BB97EFA110FD6382425A9CAF8337E2A9CB38348281C269B0`；`last.pt` SHA-256：`7AC65DCAC5D1439E3C1667E0E10A8F00737670FABB79AF7243E3745A8D86970D`。两者均恢复出 `channels=16` 与一致的 stem metadata。

### Phase-aware HSI16 结果：否决

- 私有 Kernel `zephyrpong/hsi-yolo26m-phase-ablation` version 1 正常完成；fixed 2400/600，唯一训练变量是输入表示：保留 4×4 snapshot mosaic 的物理 row/column phase，按 phase-aware bilinear v1 重建到保持宽高比且长边 1024 的 16 通道 NPY。
- 16 通道顺序、同一 manifest、标签、seed 2026 与 native-cube shared P0.5–P99.5 uint8 编码保持不变；random 首层初始化，关闭 SpectralStem、object crops 与 tile inference。
- 训练 return code 0，batch 8/workers 2/device 0，无 CUDA OOM 或 shared-memory 错误。最佳与最终均为 epoch 30：`mAP50-95=0.69935`、`mAP50=0.95838`。
- 相对普通同规格 e30，mAP50 提升 `+0.00511`，但 mAP50-95 下降 `-0.00208`；相对全量门槛仍差 `0.00508`。这说明粗召回改善没有转化为严格 IoU 定位提升。
- 1000 图测试 CSV 仅通过本地结构校验（38,304 detections），没有上传比赛，也没有 Public 分数。完整合同、哈希和逐类变化见 [phase-aware-hsi16-20260921.md](phase-aware-hsi16-20260921.md)。

### `cls_pw=0.25` 结果：否决

- 私有 Kernel `zephyrpong/hsi-yolo26m-clspw025-ablation` version 1 正常完成；fixed 2400/600，普通 HSI16、YOLO26m、30 epochs、`imgsz=1024`、batch 8、workers 2、seed 2026 均与同规格 e30 基线一致，唯一训练变量为 `cls_pw: 0.0→0.25`。
- 训练 return code 0，无 CUDA OOM 或 shared-memory 错误。最佳为 epoch 27：`mAP50-95=0.69863`、`mAP50=0.95662`；最终 epoch 30 为 `0.69862` / `0.95578`。训练 CSV 与日志没有 mAP75 字段，不能补写该指标。
- 相对普通同规格 e30，最佳 mAP50 提升 `+0.00335`，但 mAP50-95 下降 `-0.00280`；相对全量门槛仍差 `0.00580`。epoch 27 后三轮平台/轻微回落，而普通基线持续上升到 epoch 30，分类频次加权没有改善严格 IoU 定位。
- 逐类严格 IoU 变化混合：`car +0.009`、`egg_plastic +0.007`、`rubik +0.007`，但 `orange -0.024`、`people -0.017`、`car_toy -0.012`。这不支持全量训练。
- 1000 图测试 CSV 仅通过本地结构校验（36,784 detections），没有上传比赛，也没有 Public 分数。`status.json` / `results.csv` / `best.pt` / `last.pt` / 测试 CSV 的 SHA-256 分别为 `89FC7B02...DE296D`、`ABB813EE...FC0F8`、`836D8161...BA6F6`、`3666916F...A617D`、`32C40F59...84B9E9`。

### Random-affine `scale=0.3` 结果：否决

- 私有 Kernel `zephyrpong/hsi-yolo26m-scale030-ablation` version 1 正常完成；fixed 2400/600，普通 HSI16、YOLO26m、30 epochs、`imgsz=1024`、batch 8、workers 2、seed 2026、`cls_pw=0.0` 均保持同规格 e30 基线。已记录的训练与数据参数中，唯一差异是 Ultralytics random-affine `scale: 0.5→0.3`；它不同于保持关闭的 training-time `multi_scale`。
- 训练 return code 0，无 CUDA OOM 或 shared-memory 错误。最佳与最终均为 epoch 30：`mAP50-95=0.69992`、`mAP50=0.95646`；训练 CSV 与日志没有 mAP75 字段，不能补写该指标。
- 相对普通同规格 e30，mAP50 提升 `+0.00319`，但 mAP50-95 下降 `-0.00151`；相对全量门槛仍差 `0.00451`。逐类严格 IoU 变化混合：`e-bike +0.018`、`egg_plastic +0.010`、`rubik +0.009`，但 `people -0.028`、`charger_head -0.017`、`orange -0.017`，温和缩放没有改善总体严格定位。
- 1000 图测试 CSV 仅通过本地结构校验（31,848 detections），没有上传比赛，也没有 Public 分数。`status.json` / `results.csv` / `best.pt` / `last.pt` / 测试 CSV 的 SHA-256 分别为 `9992143E...8B6B22F`、`43EDCA5D...20B81D`、`5808F0F4...D1AACD`、`D518B401...FEF3B1`、`AA0DFE08...28EA83`。
- 结论：固定划分门禁失败，不启动 full-data `scale=0.3` 训练，不生成或提交该路线的比赛候选。

### Horizontal-flip-only 结果：否决

- 固定 600 张、同一个 e30 YOLO26m checkpoint；只新增一次 `1024` 水平翻转推理，16 个通道一起翻转，预测框按 `x1=W-x2, x2=W-x1` 映射回原图。
- 七尺度 supported-full cache control 精确复现 `0.7057081826`，与预期只差 `2.60e-9`，因此对照有效。
- 单尺度 `1024` 原图+flip 有正向变化，但加入当前七尺度最佳方案后，最佳仅为 `fusion_iou=0.65`、`support_gain=0.125` 的 `0.7058779597`，相对 supported-full 只增 `+0.0001697771`。
- 该增益低于预设 `+0.001` 推理门禁，判定否决；不生成正式候选、不做比赛提交。结果文件：`artifacts/inference_mode_validation/m_ablation_flip_only_20260921.json`，SHA-256 `49FA5146E4C1392F71C2D9BFE2B3609F742D3F85B2AAC29D9A855D7A4650F904`。

### YOLO26m 45 轮结果：否决

- 私有 Notebook `zephyrpong/hsi-yolo26m-ablation` 版本 2 正常完成；配置保持 HSI16、固定 2400/600、`imgsz=1024`、batch 8、workers 2、seed 2026，唯一训练变量是 `epochs 30→45`。
- 最佳与最终均为 epoch 45：`mAP50-95=0.69899`、`mAP50=0.95899`。相对同规格 e30，mAP50 提升 `+0.00572`，但 mAP50-95 下降 `-0.00244`；相对全量门槛仍差 `0.00544`。训练 CSV/日志没有 mAP75 字段，不能补写该指标。
- epochs 31–45 的 mAP50-95 均值为 `0.69522`、总体标准差 `0.00318`；`0/15` 轮超过 e30 的 `0.70143`，也没有任何一轮达到 `0.70443`。后期缓慢上升不改变门禁结论。
- 类别层面相对 e30：`e-bike +0.037` 最大，但 `people -0.026`、`orange_plastic -0.020`、`charger_head -0.017` 等退化；粗召回改善没有转化为严格 IoU 总指标提升。
- 1000 图测试 CSV 通过本地结构校验（36,080 detections），但没有上传比赛，也没有 Public 分数。`best.pt` SHA-256 `F7F11CFC1A5AA2A6995B93A76BAAFC70E651F7D258F8D2FB0E76AFF6D33864B0`；`last.pt` SHA-256 `C5E41C051DF775020A7D9C5F9C8D44624A576C97E1784E8DA832D14F37E89AF1`。
- 结论：门禁失败，不启动已预留的 full-data 45 轮训练，不生成或提交该路线的比赛候选。

## 已实现的同 checkpoint 切片推理

- tile：原图坐标 `128×256`；
- stride：`96×192`（约 25% overlap）；
- tile model input：`1024`；
- 每个 tile 有互不重叠的 center-ownership core，只有中心属于该 core 的框能进入融合，减少 overlap 重复框；
- tile-local 坐标先回映到完整图，再与同 checkpoint 的 full-image passes 做同类 box vote；
- 首轮只扫 fusion IoU `0.55 / 0.60 / 0.65`，support gain 固定为 `0`，避免同时改两个变量。

真实权重的一图端到端 smoke 已通过：16 通道 NPY、完整图 + 9 tiles、坐标回映、融合、CSV 生成与独立 submission checker 均成功，无非法框。

旧 `artifacts/ensemble_cache/m.pkl` 没有 manifest。2026-09-20 通过一张验证图重新推理溯源：cache 的 1024 pass 与 fixed-split ablation `last.pt` 在框、类别、置信度上逐项完全一致，而与 full-data `last.pt` 的预测数量不同。后续 tile 对照因此使用 fixed-split checkpoint；仍会要求 full-cache control 复现 `0.704035 ± 0.0002` 才接受结果。

### Tile 结果：否定

完整 600 张 fixed split 的 control 为 `0.7040353524`，相对预期 `0.704035` 只差 `3.52e-7`，通过控制门槛。三档 tile 结果全部显著下降：

| fusion IoU | mAP50-95 | 相对 legacy full | 相对 support-gain full |
|---:|---:|---:|---:|
| 0.55 | 0.695962 | -0.008073 | -0.009480 |
| 0.60 | 0.695275 | -0.008761 | -0.010167 |
| 0.65 | 0.694793 | -0.009242 | -0.010649 |

最优 tile 配置还把预测数从 `71,179` 增至 `100,210`，但 mAP50、mAP75 和原本最弱的 `car / people / e-bike` 都下降。这不是小幅噪声，而是明确的分布不匹配：当前 full-image 训练模型在局部 crop 上产生过多低质量框。结论是：**当前 checkpoint 的 tile TTA 停止，不进入提交候选，不再扫 support gain 或更多 IoU。** 只有 object-crop 训练模型本体先通过 full-val 门禁后，才允许对那个新 checkpoint 重新做一次 tile 对照。

结果文件：`artifacts/inference_mode_validation/m_ablation_tile_grid_20260920.json`，SHA-256 `88072418BB5CE4FDC9700B12D5A2FBF115BB5914DF8BDB105DBB830D6E919670`。

## 门禁

| 阶段 | 通过条件 | 当前状态 | 不通过动作 |
|---|---|---|---|
| zero-init fixed split | 标准 full-val mAP50-95 `>=0.70443` | `0.69690`，否决 | 记录负结果，不跑全量 |
| object-crop fixed split | 标准 600 full-val mAP50-95 `>=0.70443` | `0.69772`，否决 | 停止该训练路线 |
| tile TTA | 相对同 checkpoint、同 evaluator 的 full baseline `>=+0.003`，且 cache control 通过 | `-0.00807`，否决 | 不用于全量/提交 |
| SpectralStem fixed split | 标准 full-val mAP50-95 `>=0.70443` | `0.69930`，否决 | 不跑全量、不提交测试 CSV |
| phase-aware HSI16 fixed split | 标准 full-val mAP50-95 `>=0.70443` | `0.69935`，否决 | 不跑全量、不提交测试 CSV |
| HSI16 `cls_pw=0.25` fixed split | 标准 full-val mAP50-95 `>=0.70443` | 最佳 `0.69863`，否决 | 不跑 full-data、不提交测试 CSV |
| random-affine `scale=0.3` fixed split | 标准 full-val mAP50-95 `>=0.70443` | `0.69992`，否决 | 不跑 full-data、不提交测试 CSV |
| horizontal-flip-only | cache control 通过，且相对 supported-full mAP50-95 `>=+0.001` | `0.70587796`，仅 `+0.00016978`，否决 | 不生成正式候选、不提交 |
| YOLO26m e45 fixed split | 标准 full-val mAP50-95 `>=0.70443` | `0.69899`，否决 | 不跑 full-data e45 |
| full training | status success、权重和日志完整、单 checkpoint | 无新候选 | 不生成正式候选 |
| final CSV | 1000/1000 图、0 非法框、独立校验通过 | f0.74 已通过并提交 | 不上传 |
| Kaggle Submit | 用户在动作时明确确认，且实时额度已复核 | ref `56392305` 已完成 | 不提交 |

## 备选路线及优先级

1. `SpectralStem`、phase-aware HSI16、`cls_pw=0.25`、random-affine `scale=0.3`、horizontal-flip-only 与同规格 e45 均已被固定划分门禁否决，不再投入全量训练或提交。
2. 新 seed 的同规格 YOLO26m：只用于估计固定划分方差，没有正向增益先验；启动前必须重新取得 GPU 授权。
3. `cls_pw=0.5`：机制与已失败的 `0.25` 相同、先验更弱；若仍尝试，必须视为新的单变量 fixed-split 候选并先取得 GPU 授权，不能直接跑 full-data。
4. 同 YOLO family 蒸馏：最终部署学生模型，可能有收益，但 teacher forward 增加显存与时间，且需要先重新核对当前版本的官方支持边界。[Ultralytics knowledge distillation guide](https://docs.ultralytics.com/guides/knowledge-distillation)
5. KD-DETR / RT-DETR：研究上有增益，但检测 query/feature 对齐和 16 通道迁移工作量高。KD-DETR 的贡献不是简单接入任意 teacher logits；RT-DETR 的 COCO 结果也不能外推到本数据。[KD-DETR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_KD-DETR_Knowledge_Distillation_for_Detection_Transformer_with_Consistent_Distillation_Points_CVPR_2024_paper.html)、[RT-DETR paper](https://arxiv.org/abs/2304.08069)、[official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR)

## 可复现命令

准备 fixed-split crop：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_object_crops.py `
  --dataset-root data\processed\hsi16_shared_p005_995 `
  --splits train --tile-height 128 --tile-width 256 `
  --crops-per-image 1 --jitter-fraction 0.15 `
  --minimum-visible 0.9 --seed 2026
```

复用 full-scale cache 验证 tile：

```powershell
.\.venv\Scripts\python.exe scripts\eval_tiled_inference.py `
  --weights kaggle_remote\outputs\ablation\kaggle_ablation_yolo26m_e30\last.pt `
  --data data\processed\hsi16_shared_p005_995\dataset.yaml `
  --full-cache artifacts\ensemble_cache\m.pkl `
  --tile-cache artifacts\tile_cache\m_ablation_h128w256_s96x192_i1024.pkl `
  --output artifacts\inference_mode_validation\m_ablation_tile_grid_20260920.json `
  --tile-size 128 256 --tile-stride 96 192 --tile-imgsz 1024 `
  --tile-fusion-ious 0.55 0.60 0.65 --tile-support-gains 0 `
  --expected-full-map 0.704035 --control-tolerance 0.0002 --minimum-gain 0.003
```

复用七尺度 full cache 验证 horizontal-flip-only：

```powershell
.\.venv\Scripts\python.exe scripts\eval_flip_tta.py `
  --weights kaggle_remote\outputs\ablation\kaggle_ablation_yolo26m_e30\last.pt `
  --data data\processed\hsi16_shared_p005_995\dataset.yaml `
  --full-cache artifacts\ensemble_cache\m.pkl `
  --flip-cache artifacts\flip_tta_cache\m_ablation_hflip_i1024.pkl `
  --output artifacts\inference_mode_validation\m_ablation_flip_only_20260921.json `
  --imgsz 1024 --batch 1 --device 0 `
  --full-fusion-iou 0.74 --full-support-gain 0.125 `
  --flip-fusion-ious 0.55 0.60 0.65 0.70 0.74 0.78 0.82 `
  --flip-support-gains 0 0.125 `
  --expected-full-map 0.70570818 --control-tolerance 0.0002 --minimum-gain 0.001
```

这些是受控实验，不是达到 `0.66` 的保证。任何增益都必须先由固定划分、同 evaluator 和单 checkpoint 证据支持，再决定是否消耗全量训练与提交额度。

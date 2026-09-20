# 0.66 冲刺路线：单模型、可复现、门禁式推进（2026-09-20）

## 结论先行

当前奖项合规的 Public 最佳是 `0.63066`，目标 `0.66` 仍差 `0.02934`。这是一次显著跃升，不能承诺一定达到。四天内最合理的主线不是继续微调 NMS，也不是重写检测器，而是只保留一个 YOLO26m HSI16 checkpoint，直接修复已经量化出的主要误差：小目标的高 IoU 定位。

执行顺序：

1. 等待已经在跑的 zero-init 固定划分消融，不重复训练；
2. 已验证并否定当前 fixed-split checkpoint 的重叠切片推理；
3. 训练 `2400 full + 2376 object crop` 的单变量 fixed-split 模型；
4. object-crop 模型只有标准完整图验证达到 `0.70443` 才进入 3000 张全量训练；tile 是独立可选门禁，只有对新 checkpoint 相对自身基线增加至少 `0.003` 才启用；
5. 最终 Kaggle Submit 仍需单独确认，不由训练或生成 CSV 自动触发。

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
- Public：`0.63066`，ref `56379896`；
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

本项目已有实测负结果：`imgsz=1280`、训练期 multi-scale、P2 head、close-mosaic=20、box loss=10、P1–P99、伪标签、YOLO26l。它们不应在截止前重复消耗 GPU。详见 [HANDOFF.md](../HANDOFF.md) 与 [experiments.csv](experiments.csv)。

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

| 阶段 | 通过条件 | 不通过动作 |
|---|---|---|
| zero-init fixed split | 标准 full-val mAP50-95 `>=0.70443` | 记录负结果，不跑全量 |
| object-crop fixed split | 标准 600 full-val mAP50-95 `>=0.70443` | 停止该训练路线 |
| tile TTA | 相对同 checkpoint、同 evaluator 的 full baseline `>=+0.003`，且 cache control 通过 | 不用于全量/提交 |
| full training | status success、权重和日志完整、单 checkpoint | 不生成正式候选 |
| final CSV | 1000/1000 图、0 非法框、独立校验通过 | 不上传 |
| Kaggle Submit | 用户在动作时明确确认，且实时额度已复核 | 不提交 |

## 备选路线及优先级

1. `SpectralStem 16→3→YOLO26m`：identity 初始化到当前 5/8/13 三通道，再学习全 16 波段线性组合。潜力高于继续手选三波段，但需要自定义模块/YAML/权重重载测试，作为 crop 失败后的 Plan B。
2. 温和 `cls_pw=0.25/0.5`：Ultralytics 8.4.147 原生支持，工程风险小，但只影响分类 BCE，未直接解决定位主因。
3. 同 YOLO family 蒸馏：官方 Ultralytics 只支持相同 YOLO family，最终部署学生模型；可能有收益，但 teacher forward 增加显存与时间。[Ultralytics knowledge distillation guide](https://docs.ultralytics.com/guides/knowledge-distillation)
4. KD-DETR / RT-DETR：研究上有增益，但检测 query/feature 对齐和 16 通道迁移工作量高。KD-DETR 的贡献不是简单接入任意 teacher logits；RT-DETR 的 COCO 结果也不能外推到本数据。[KD-DETR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_KD-DETR_Knowledge_Distillation_for_Detection_Transformer_with_Consistent_Distillation_Points_CVPR_2024_paper.html)、[RT-DETR paper](https://arxiv.org/abs/2304.08069)、[official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR)

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

这些是受控实验，不是达到 `0.66` 的保证。任何增益都必须先由固定划分、同 evaluator 和单 checkpoint 证据支持，再决定是否消耗全量训练与提交额度。

# 单模型合规恢复与支持票候选（2026-09-20）

## 规则纠偏

赛事当前规则只允许一个 detection model。主办方明确说明：

- 同一个训练模型的 TTA / multi-scale inference 可以合并；
- 多个不同训练模型的 voting、weighted fusion、post-NMS fusion 禁止；
- 最终 code-review package 必须证明只使用了一个训练模型。

官方页面：

- Rules：<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/rules>
- 主办方澄清：<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863>

因此，历史 `0.63917` 至 `0.65091` 的多模型提交只保留为审计记录，不再生成、上传或选择为最终提交。当前已验证的合规 Public 基线是单个 YOLO26m HSI16 checkpoint 的七尺度推理：`0.62953`，Kaggle ref `56303572`。

## 同一 checkpoint 七尺度支持票快筛

唯一模型：

`kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt`

固定条件：16 通道 P0.5–P99.5、七尺度 `832/896/960/1024/1088/1152/1216`、NMS IoU `0.70`、fusion IoU `0.70`、`conf=0.0001`、`max_det=300`。每个尺度只是同一 checkpoint 的输入变换，符合主办方的单模型多尺度许可。

600 张固定留出集，复用 `artifacts/ensemble_cache/m.pkl`：

| support gain | mAP50-95 | 相对 gain=0 |
|---:|---:|---:|
| 0 | 0.70404 | — |
| 0.025 | 0.70429 | +0.00025 |
| 0.05 | 0.70451 | +0.00047 |
| 0.075 | 0.70459 | +0.00055 |
| 0.10 | 0.70485 | +0.00081 |
| **0.125** | **0.70544** | **+0.00140** |
| 0.15 | 0.70422 | +0.00018 |
| 0.20 | 0.69668 | -0.00736 |
| 0.25 | 0.69286 | -0.01118 |

峰值窄而清楚，0.15 以后迅速退化，因此选择 `0.125`，不继续盲扫更大值。

fusion IoU 交叉检查：

| fusion IoU | gain=0 | gain=0.125 | 差值 |
|---:|---:|---:|---:|
| 0.65 | 0.70426 | 0.70439 | +0.00013 |
| **0.70** | **0.70404** | **0.70544** | **+0.00140** |
| 0.75 | 0.70407 | 0.70512 | +0.00105 |

原始结果：

- `artifacts/ensemble/single_m_support_gain_20260920.json`
- `artifacts/ensemble/single_m_support_gain_iou_grid_20260920.json`

## 正式合规候选

- CSV：`submissions/submission_single_m_hsi16_ms7_f070_sg0125.csv`
- 检测数：94,094；覆盖 1000/1000；无效框 0
- SHA-256：`E4ED7BBCD5FD9D7E7BB7765850746A7F8B5A50034BA226B44A03D23E3233D266`
- 同缓存 gain=0 回归文件 SHA-256：`31BC2BACF376866A5D066BB1A5793972BDD0F4E0B4B12CA8A05D1F6192EFA3B8`
- 与 gain=0：93,998 个框坐标完全一致，双方各有 96 个 top-300 边界替换，共同框平均置信度变化 `+0.00316898`
- 独立 `check_submission.py`：通过
- Kaggle：2026-09-20 04:11:50 UTC（北京时间 12:11:50）完成；Public `0.63066`；ref `56379896`

远端原始单模型 CSV 有 94,117 条、SHA-256 `EBA65AB84F6FFD99FD06812D2AD14E90690CF39F035F8031781BCC73F3597422`。本候选使用本机缓存重放，和远端 Ultralytics/CUDA 环境存在 23 条检测的微小差异。实际 Public 从 ref `56303572` 的 `0.62953` 提升到 `0.63066`，增量 `+0.00113`；留出预测的 `+0.00140` 方向正确但幅度不能当作精确换算关系。

## zero-init 固定划分消融

为寻找更强的合规单模型，新增训练变量：将预训练 RGB 首层之外的 13 个输入通道从随机初始化改为零初始化；通道仍可正常反向传播。其他配置与旧 YOLO26m 固定划分基准保持一致。

- 私有 Notebook：`zephyrpong/hsi-yolo26m-xczero-ablation`
- version：1
- run：`kaggle_ablation_yolo26m_hsi16_xczero_e30`
- 数据：固定 2400/600，HSI16 P0.5–P99.5
- 模型：YOLO26m，1024，30 epoch，seed 2026，batch 降级序列 `8/6/4`
- 2026-09-20 12:14（北京时间）实时状态：`RUNNING`
- 不自动提交比赛

门槛：旧同规格固定划分 `0.70143`，zero-init 必须至少达到 `0.70443`（+0.003）才训练全量 3000 张；否则记录为负结果并停止。

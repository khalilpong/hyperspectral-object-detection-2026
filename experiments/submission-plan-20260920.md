# 2026-09-20 提交清单（单模型合规版）

## 规则红线

官方规则只允许一个 detection model。主办方明确允许同一 checkpoint 的 TTA / multi-scale inference，但禁止不同训练模型的 voting、weighted fusion 和 post-NMS fusion：

<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863>

因此：

- 历史 `0.65091`（ref `56378610`）来自八模型融合，视为不合规历史记录，不得选为最终提交；
- `submission_ens*.csv`、P1 第九成员、B/D 权重候选均不得再上传；
- 后续只允许一个训练 checkpoint 的单尺度、多尺度或 TTA 结果。

## 已完成的合规提交

- 文件：`submissions/submission_single_m_hsi16_ms7_f070_sg0125.csv`
- 唯一模型：`kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt`
- 推理：同一 checkpoint 七尺度 `832/896/960/1024/1088/1152/1216`，fusion IoU `0.70`，support gain `0.125`
- 留出证据：`0.70404 -> 0.70544`，+`0.00140`
- 尺度交叉检查：IoU `0.65/0.70/0.75` 中，`0.70` 仍最佳
- 规模：94,094 detections，1000/1000 images，invalid 0
- SHA-256：`E4ED7BBCD5FD9D7E7BB7765850746A7F8B5A50034BA226B44A03D23E3233D266`
- 相对同缓存 gain=0：93,998 个框完全一致，只替换 96 个 top-300 边界框
- 当前状态：`Success/Complete`；Public `0.63066`；Kaggle ref `56379896`
- 提交时间：2026-09-20 04:11:50 UTC（北京时间 12:11:50）
- 相对合规基线 ref `56303572`：`0.63066 - 0.62953 = +0.00113`

该候选已经把已验证的合规 Public 基线 `0.62953`（ref `56303572`）提高到 `0.63066`，没有追认不合规的 `0.65091`。

## 训练侧并行路线

私有 Notebook `zephyrpong/hsi-yolo26m-xczero-ablation` version 1 已推送，2026-09-20 12:14 实时状态仍为 RUNNING。它只改变 16 通道首层新增 13 通道的初始化：`random -> zero`；固定 2400/600，不自动提交。

- 旧同规格基准：`0.70143`
- 全量训练门槛：`>= 0.70443`
- 达标：生成全量 zero-init 单模型并做同 checkpoint 七尺度推理
- 未达标：停止，不浪费全量 GPU 和提交额度

## 操作门禁

1. 本次提交后页面实时显示今天还剩 1 次，约 20 小时后重置；下一次操作仍以当时页面为准。
2. 对确切 CSV 重新执行 `scripts/check_submission.py`、SHA-256、大小和 1000/1000 覆盖检查。
3. 上传后，在最终 Submit 前向用户展示确切文件、规则合规性、额度和说明，并获得当时确认。
4. 等待状态为 `Success/Complete`，读取 Public Score；排队、本地检查或上传成功都不等于正式完成。
5. 已回写 ref `56379896`、Public `0.63066`、时间和哈希。
6. 比赛结束前手工确认最终选择的是合规单模型提交，不能让 Kaggle 自动选择历史最高的多模型文件。
7. 是否向主办方主动说明/询问历史多模型提交，由用户单独决定；不得擅自发帖或发送消息。

# 2026-09-20 提交清单

## 第一提交：已完成

- 文件：`submissions/submission_ens8_b368_b0715_ms7_f070_sg0125.csv`
- SHA-256：`A4158EF71A57AC38C49F6F11EB9BD7279065E35610D5ED8EE03617644CB0C410`
- 规模：287,232 detections，1000/1000 images，invalid 0
- 唯一变量：在当前 Public 0.64831 的八成员、七尺度、原权重和 fusion IoU 0.70 上，把融合排序改为 `support_gain=0.125`
- 证据：2/3/4 模型固定留出组合均比 legacy 正向；默认 `gain=0` 已逐字节回归到当前最佳文件
- 建议说明：`Ensemble8 ms7 f070 support_gain 0.125; same members and weights as public 0.64831`

2026-09-20 11:21（UTC+08）经用户当时确认后提交，Kaggle 状态 `Complete`：

- 提交编号：`56378610`
- Public Score：**0.65091**，比原最佳 `0.64831` 提升 **+0.00260**
- 公开榜即时排名：**19**
- 提交前页面显示 3 次额度，提交后剩余 2 次

## 第二提交如何选择

先等待第一份达到 Kaggle `Success` 并记录 Public Score，不要连续盲交。

### P1–P99 留出实验：负向，已淘汰

固定 2400/600 消融已完成：P1–P99 YOLO26m 的最佳/末轮 mAP50-95 为 `0.69668`，低于同规格旧 P0.5–P99.5 基准 `0.70143`，差值 `-0.00475`。因此以下两份均不得提交：

- `submission_ens9_p010990_w020_sg0125_ms7_f070.csv`
- `submission_ens9_p010990_w040_sg0125_ms7_f070.csv`

### 第二提交首选

第二顺位保持为已有权重坐标候选：

- 文件：`submissions/submission_ens8_b075_d070_sg0125_ms7_f070.csv`
- SHA-256：`4FFE084D97EBD4A9404986BB022DF2BFC725EDF5966D086786D9F532235C6E28`
- 固定留出：0.71180，对比同 support gain 的旧权重 0.71089
- 风险：只与已提交 support 候选共享 205,687 个精确框，改动明显大于纯 support 方案
- 2026-09-20 11:24 再次预检：29,144,005 bytes，SHA-256 匹配；287,231 detections、1000/1000 images，独立检查通过
- 仍需在最终 Submit 前得到用户当时确认；不要因第一份已提分而自动提交第二份

## 其他备用（不优先）

- `submission_ens8_b368_b0715_ms7_f070_sg0100.csv`：support gain 0.1，SHA-256 `2A3F955EBDE2807C6B7DD63D7162DBC595E18C5DAFC2F005EA23FF9878550AC5`；作为更保守备用，不优先占额度。
- `submission_ens8_b368w04_b0715_ms7_f070.csv`：G 单变量降权；没有 support 候选那样的跨组合稳定证据。

## 后续操作门禁

1. 先实时确认 Kaggle 当日剩余提交次数；历史页面显示通常北京时间 08:00 重置，但以当时页面为准。
2. 对将提交的确切文件重新执行 `scripts/check_submission.py`、SHA-256 和大小检查。
3. 只上传本清单中的确切文件，不上传两图 smoke、远端单模型 CSV 或未完成产物。
4. 文件上传后，最终点击 Kaggle Submit 前向用户做当时确认。
5. 等待 Kaggle 状态为 `Success` 并读取 Public Score；失败、排队或只有本地校验都不算提交完成。
6. 立即把 submission ref、Public Score、时间、文件哈希回写 `experiments/experiments.csv` 和相关实验文档。

当前最佳为 Public **0.65091**，提交编号 `56378610`；今日仍有 2 次额度。第二候选尚未上传。

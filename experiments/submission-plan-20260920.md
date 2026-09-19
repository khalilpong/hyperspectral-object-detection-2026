# 2026-09-20 提交清单

## 已经可以提交的首选

- 文件：`submissions/submission_ens8_b368_b0715_ms7_f070_sg0125.csv`
- SHA-256：`A4158EF71A57AC38C49F6F11EB9BD7279065E35610D5ED8EE03617644CB0C410`
- 规模：287,232 detections，1000/1000 images，invalid 0
- 唯一变量：在当前 Public 0.64831 的八成员、七尺度、原权重和 fusion IoU 0.70 上，把融合排序改为 `support_gain=0.125`
- 证据：2/3/4 模型固定留出组合均比 legacy 正向；默认 `gain=0` 已逐字节回归到当前最佳文件
- 建议说明：`Ensemble8 ms7 f070 support_gain 0.125; same members and weights as public 0.64831`

这是明天额度重置后的第一提交顺位。上传和最终 Submit 前仍需用户当时确认。

## 第二提交如何选择

先等待第一份达到 Kaggle `Success` 并记录 Public Score，不要连续盲交。

### P1–P99 留出实验明确正向时

优先使用较保守的第九成员版本：

- 文件：`submissions/submission_ens9_p010990_w020_sg0125_ms7_f070.csv`
- SHA-256：`5402F4A8DCD172E01C4C0C03FA7B8067ED9958C47082BB542E4B95BF3C46109E`
- 规模：289,452 detections，1000/1000 images，invalid 0
- 相对首选 support 候选保留 250,995 个精确框（87.38%）
- 建议说明：`Ensemble9 adds HSI16 P1-P99 YOLO26m at 0.2; ms7 f070 support_gain 0.125`

只有固定 2400/600 的 `zephyrpong/hsi-yolo26m-p010-990-ablation` 相比旧 P0.5–P99.5 同规格模型有明确增益，并且融合留出检查不退化时，才提升它的提交顺位。

### P1–P99 证据不足或负向时

不提交两个 P1 文件。第二顺位保持为已有权重坐标候选：

- 文件：`submissions/submission_ens8_b075_d070_sg0125_ms7_f070.csv`
- SHA-256：`4FFE084D97EBD4A9404986BB022DF2BFC725EDF5966D086786D9F532235C6E28`
- 固定留出：0.71180，对比同 support gain 的旧权重 0.71089
- 风险：只与首选 support 候选共享 205,687 个精确框，改动明显大于纯 support 方案

## 暂不排入前两位

- `submission_ens9_p010990_w040_sg0125_ms7_f070.csv`：P1 权重 0.4，SHA-256 `514CE3995BB0935D409E5177FC5A95F39A7E3C45228F0E691B816C15EDAE8952`；相对首选只保留 85.93% 精确框，需比 0.2 版本更强的留出证据。
- `submission_ens8_b368_b0715_ms7_f070_sg0100.csv`：support gain 0.1，SHA-256 `2A3F955EBDE2807C6B7DD63D7162DBC595E18C5DAFC2F005EA23FF9878550AC5`；作为更保守备用，不优先占额度。
- `submission_ens8_b368w04_b0715_ms7_f070.csv`：G 单变量降权；没有 support 候选那样的跨组合稳定证据。

## 明天操作门禁

1. 先实时确认 Kaggle 当日剩余提交次数；历史页面显示通常北京时间 08:00 重置，但以当时页面为准。
2. 对将提交的确切文件重新执行 `scripts/check_submission.py`、SHA-256 和大小检查。
3. 只上传本清单中的确切文件，不上传两图 smoke、远端单模型 CSV 或未完成产物。
4. 文件上传后，最终点击 Kaggle Submit 前向用户做当时确认。
5. 等待 Kaggle 状态为 `Success` 并读取 Public Score；失败、排队或只有本地校验都不算提交完成。
6. 立即把 submission ref、Public Score、时间、文件哈希回写 `experiments/experiments.csv` 和相关实验文档。

当前最佳仍为 Public 0.64831，提交编号 56346325。新候选尚未上传，不消耗明日额度。

# 新对话提示词：Phase 2 最后两次提交机会

请在 `D:\kaggle\hyperspectral-object-detection-2026` 接手 Kaggle 比赛 `hyperspectral-object-detection-challenge-2026` 的最后收尾。请用中文交流，尽量少打扰我，但任何不可逆外部动作仍须按下面边界确认。

第一步必须完整阅读：

1. `PHASE2_LAST_TWO_SUBMISSIONS_HANDOFF_20260926.md`（本次唯一首要交接）；
2. `experiments/phase2-completion-20260926.md`；
3. 即将修改的确切代码与测试。

当前权威基线：Phase 2 ref `56568811` 已 `COMPLETE`，Public test-reference `0.62717`，ranking private score未公布；最终选择仍为 `0/2`。baseline CSV SHA-256 是 `1744A3545E89F0E695EE106B177C582C7158D686D675940CFCAD3A72956C5E23`，覆盖 2,000 图/198,063 行。Phase 1 frozen 合规最佳是 ref `56455800` / `0.63546`。历史 `0.65091` 是违规多模型 ensemble，绝不能提交或最终选择。

我的目标：在不训练、不适配 ranking、不使用第二 checkpoint 的前提下，最多使用当天剩余 2 次提交机会尝试提升。先现场核验确有 2 次额度、截止时间、baseline 状态和 final `0/2`，不要先点最终选择。

只允许两个已经固定、已有轻微正向验证证据的候选：在 incumbent 七尺度预测上，各图额外加入一次 `imgsz=1024` horizontal-flip 推理，16 通道一起翻转并把框映射回原图：

- 候选 A：`flip fusion_iou=0.65`, `support_gain=0.125`，固定验证相对 incumbent `+0.0001697771`；
- 候选 B：`flip fusion_iou=0.82`, `support_gain=0.125`，固定验证相对 incumbent `+0.0001039762`。

两者增益都很小，不保证私榜提升。不要扩展参数网格，也不要重开已经否决的 max_det、support-count、scale 权重、boxscale、tile、训练或多模型方向。必须复用唯一 checkpoint SHA-256 `8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3`。test1000 和 ranking1000 的 flip 原始推理各只跑一次并缓存，两个候选只重复融合，不重复模型推理。

为每个候选生成独立 raw/calibrated/combined CSV 与 manifest，保留 boxscale101，严格验证 2,000 图、schema、geometry、连续 row id、test/ranking overlap 0、checkpoint/输入/输出 SHA-256、exact commands。先比较候选与 baseline 的框替换率、类别/尺寸分布和风险；如果没有保留价值，可以不浪费第二次机会。

每次 Kaggle 操作必须这样做：先把 exact 文件、SHA-256、行数/图数、参数、描述和实时剩余额度报告给我；停在最终 Submit 前，等我对该候选单独明确确认。确认不能跨候选复用。Submit 后必须等到 `COMPLETE` 并记录 ref/score，Pending 不算完成。

最终 leaderboard selection 是另一项独立不可逆动作：候选结束后先提出准确的两个合规 refs，等我确认后再勾选并验证 `2/2`。绝不能选择任何 `submission_ens*.csv`。

完成后更新 HANDOFF、实验记录和本次交接，只把代码、测试、文档安全 commit/push 到私有 `origin/main`；数据、权重、CSV、cache、artifacts、manifest 和凭证不得进 Git。

现在先复述：当前 baseline、两个候选、红线、你准备做的第一项只读核验、以及下一次需要我确认的动作。然后开始执行，直到停在第一个真正需要我确认的 Kaggle Submit 前。

# Phase 2 最后两次提交机会交接（2026-09-26）

## 执行后更新（2026-09-26 13:10，北京时间；覆盖下方原始计划）

- A 已 `COMPLETE`：ref `56569707`，Public test-reference `0.62865`，较 baseline `+0.00148`。
- B 已 `COMPLETE`：ref `56569768`，Public test-reference `0.62767`，较 baseline `+0.00050`。
- 两者 private score仍隐藏；均为完整 2,000 图，分别 196,609 / 241,891 行，严格校验和审计已完成。
- test 七尺度 cache成功复用；ranking 原始逐尺度 cache确实不存在，按用户缺缓存时可新增窄生产路径的要求补建，并精确复现 baseline。两个 split 的 flip各仅推理一次，A/B 构建全为 0 新推理。
- 用户在新任务中明确取消逐次 Submit 确认，由代理自行决定，已经依此使用两次额度；**最终勾选与 Save 仍需独立确认**。
- 推荐最终选择 **`56569707 + 56569768`**。当前仍 **0/2，未勾选/未 Save**。旧 Phase 1-only ref无需占最终名额，其 ranking部分为0；此结论已按官方补充通知现场复核。
- 完整哈希、命令、测试、风险与回执见 [phase2-flip-finalization-20260926.md](experiments/phase2-flip-finalization-20260926.md)。全套 223 pytest通过。本次 Git 范围仅限代码、测试与 Markdown文档；数据/CSV/cache/manifest/权重均不进 Git。

## 以下为执行前原始交接（保留历史依据）

> 这是新对话的首要交接文档。目标很窄：保留现有合规 baseline，在不训练、不适配 ranking、不引入第二 checkpoint 的前提下，最多使用当天剩余 2 次 Kaggle 提交机会尝试同模型 horizontal-flip TTA 候选。

## 30 秒摘要

- 比赛：`hyperspectral-object-detection-challenge-2026`，账号 `zephyrpong`。
- 截止：2026-09-27 16:00（北京时间）。
- Phase 2 baseline 已完成：ref `56568811`，状态 `COMPLETE`，Public test-reference `0.62717`，ranking private score暂不公开。
- baseline 文件：`submission_phase2_single_m_hsi16_ms7_f074_sg0125_boxscale101.csv`。
- baseline SHA-256：`1744A3545E89F0E695EE106B177C582C7158D686D675940CFCAD3A72956C5E23`。
- baseline 规模：198,063 行、2,000 张图、test/ranking overlap 0、严格校验无问题。
- Phase 1 frozen 合规最佳：ref `56455800`，frozen Public `0.63546`。
- 最终榜单选择当前仍为 `0/2`；两个候选结束前不要勾选。
- 用户称当天还剩最后 2 次提交机会；任何使用前都要在 Kaggle UI 现场复核。
- GitHub 私有仓库：`https://github.com/khalilpong/hyperspectral-object-detection-2026.git`；本交接创建前 HEAD 为 `9d6719f`。

## 用户当前意图与动作权限

用户要在一个新对话中，用最后两次机会尝试提高 Phase 2 成绩。可以自主完成以下可逆工作：

- 读取代码、缓存、manifest 和实验记录；
- 做零 GPU cache 审计与本机无状态推理；
- 为同一个 checkpoint 生成 horizontal-flip 预测；
- 生成、校验、验哈希并暂存候选 CSV；
- 更新代码、测试、实验文档并安全推送 GitHub。

下列动作没有被概括性授权，必须在动作发生前单独确认：

- 每一次 Kaggle 最终 `Submit`；
- 最终 leaderboard submission checkbox 的勾选与保存；
- 任何训练、付费算力、数据外传或删除。

用户希望少消耗额度：两次是上限，不是必须耗尽。没有正向且可复现证据时应保留机会。

## 不可改变的合规边界

1. 只允许 checkpoint：
   `kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt`
   SHA-256 `8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3`。
2. ranking1000 只能做无状态推理。禁止训练、微调、伪标签、蒸馏、BN 统计更新、test-time adaptation 或任何模型状态写回。
3. 允许同一 checkpoint 的 scale/flip TTA 和预测合并；禁止不同 checkpoint、不同模型或不同架构的 voting、WBF、weighted fusion、post-NMS fusion。
4. 历史 `submission_ens*.csv` 全部不合规，不得上传或最终选择。
5. test1000 与 ranking1000 都必须完整包含在每个 Phase 2 CSV 中；image ID 集必须零重叠，schema/geometry/连续 row id 必须通过严格校验。
6. 原始数据、权重、CSV、ignored artifacts/manifests 不得加入 GitHub。

## baseline 的不可变血缘

### Ranking 数据与预处理

- 官方 ranking PNG：精确 1,000 张 16-bit grayscale，官方总字节 `3,893,334,217`。
- ranking-only ZIP SHA-256：`C6734C0DECF4B9772F698F02B20A56A62679C067DA05049C7D5F46AD01F1D590`。
- band order：`[5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15]`。
- 编码：16-channel uint8，per-image shared P0.5/P99.5。
- 已准备目录：`data/processed/phase2_ranking/images`（1,000 NPY + 1,000 preview PNG）。

### Baseline 推理

- NPY 输入，batch 1，FP16，`conf=0.0001`，NMS IoU `0.7`，`max_det=300`。
- 七尺度：`832 896 960 1024 1088 1152 1216`。
- baseline fusion：`fusion_iou=0.74`，`support_gain=0.125`。
- 全局框校准：width/height `×1.01`，中心不移动；audit SHA-256 `056C8DD8A9274B3A4F49B8D5A9160BC4B42B021EA54EAD09A44E40F6B87844E2`。
- frozen Phase 1 test CSV SHA-256：`C01214E6B250075C7DA522F2E9E6F89C829BF94AE1EB2F57CF69618FDC995A86`。

## 已有零 GPU 证据：只剩两个正向 flip 配置

证据文件：

- `artifacts/inference_mode_validation/m_ablation_flip_only_20260921.json`
- SHA-256 `49FA5146E4C1392F71C2D9BFE2B3609F742D3F85B2AAC29D9A855D7A4650F904`
- 账本：`experiments/experiments.csv` 的 `heldout-single-m-ms7-horizontal-flip`
- 说明：`experiments/roadmap-to-066-single-model-20260920.md` 的 Horizontal-flip-only 小节

固定 600 张验证集、同一个 YOLO26m checkpoint；baseline 七尺度 cache control 精确复现 `0.7057081826`。只新增一次 `imgsz=1024` 水平翻转：16 通道一起翻转，框映射回原图 `x1=W-x2, x2=W-x1`。

14 个 `flip_fusion_iou × support_gain` 组合中只有两个相对 baseline 为正：

| 候选 | 固定验证 mAP50-95 | 相对 baseline | 预测框数 | 角色 |
|---|---:|---:|---:|---|
| `flip_iou065_sg0125` | `0.7058779597` | `+0.0001697771` | 73,918 | 第一候选 |
| `flip_iou082_sg0125` | `0.7058121588` | `+0.0001039762` | 91,336 | 第二候选/高召回差异化 |

两项都低于旧的 `+0.001` 推理门禁，所以此前正确地判为 NO-GO。用户现在明确开放最后两次机会后，它们只是“低置信度但已有正向证据”的最后候选，绝不能写成保证涨分。不要再扩展 IoU、support、boxscale、max_det、scale 权重或类别自适应网格。

## 新对话的执行顺序

1. **先读，不动作。** 完整读本文件，再读 `experiments/phase2-completion-20260926.md`、`scripts/eval_flip_tta.py`、`scripts/predict_submission.py`、`scripts/merge_phase2_submission.py` 和相关测试。即将修改的代码必须主代理亲自完整阅读。
2. **确认现场。** 检查 `git status -sb`、HEAD/remote；只读核验 Kaggle baseline ref `56568811` 仍 `COMPLETE`、final selection `0/2`、确有 2 次剩余提交、截止时间未变。不要点击 selection。
3. **复核旧证据。** 验证 flip artifact hash、cache control、两项正向数值和 checkpoint lineage。若不匹配，停止并报告。
4. **只实现一个可复现生产路径。** 最小扩展现有推理代码或新增窄脚本，使一次额外的 1024 hflip 预测能与现有七尺度来源按精确 IoU/sg 参数融合；写单元测试覆盖 16 通道翻转、inverse box transform、单 checkpoint guard、manifest 和 deterministic output。
5. **避免重复 GPU。** test1000 和 ranking1000 的 1024 hflip 原始预测各只跑一次并缓存；两个候选只做不同融合，不重复模型推理。RTX4060 够用，不租服务器、不启 Kaggle GPU。
6. **两套都完整生成。** 对 test1000 和 ranking1000 使用相同候选参数，保留 boxscale101。每套都生成独立 raw/calibrated/combined CSV 和 manifest，记录 exact command、checkpoint/cache/input/output SHA-256、行数、图数、ID overlap 和校验结果。
7. **先比再花额度。** 与 baseline 做行数、框替换率、类别/尺寸分布和 test Public proxy 风险审计。若第二候选没有保留价值，不必消耗第二次机会。
8. **逐次提交。** 每次先上传/暂存，报告精确文件名、SHA-256、2,000 图/行数、候选参数、剩余额度和描述；然后停在最终 Submit 前请求用户明确确认。确认不能跨候选复用。
9. **权威完成证据。** Submit 后等待 `COMPLETE`，记录 ref、Public test-reference、private score是否仍隐藏。Pending 不是完成。
10. **最终选择另行确认。** 候选结束后，结合离线证据和 Public test-reference 提议准确的两个 refs；先现场核对文件名和当前 selection，再请求用户确认后才保存。绝不选择 ensemble。
11. **文档与 GitHub。** 更新本文件、`HANDOFF.md`、实验账本和候选 manifest 摘要；只 commit/push 代码、测试、文档。大 CSV、cache、权重、数据保持 ignored。

## 已否决且不得重开的方向

- 不训练 `reg_max16`、order1385、YOLO11m/26x、EIoU、AdamW 或任何新模型。
- 不使用多 checkpoint、伪 RGB model、RT-DETR、ensemble、pseudo-label。
- 不重开 scale 子集/权重、support-count、`max_det`、size-conditioned calibration、更多 boxscale、tile TTA 或新参数网格。
- 不重跑 baseline 七尺度模型推理；优先复用已有预测/CSV/cache。
- 不让 Kaggle 自动选择高分历史 ensemble。

## 完成定义

只有同时满足以下条件才算这次收尾完成：

- 最多两个合规 flip 候选均被生成、严格验证并完整记录，或因证据不足明确保留额度；
- 每个实际 Submit 都有独立用户确认和 `COMPLETE` 证据；
- 最终 leaderboard 只选择两个明确合规的 refs，并有用户独立确认与 `2/2` 现场证据；
- private score若仍隐藏，就如实写“未公布”，不猜测；
- GitHub `main` 与本地 HEAD 一致，且没有数据、权重、CSV、cache、artifact 或凭证泄漏。

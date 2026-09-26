# 交接文档：从这里开始

> 接手本项目的人**先读完这一页**，再按需跳转到详细文档。
> 最后更新：2026-09-26 12:28（北京时间）
> 🤖 **要把项目交给另一个 AI 接手？** 直接把 [PROMPT_FOR_NEXT_AGENT.md](PROMPT_FOR_NEXT_AGENT.md) 里的提示词发给它。

## 一句话现状

**Phase 2 合规基线已经完成；用户现决定另开对话，用当天最后 2 次提交机会做一次严格受限的 inference-only 提升冲刺。** 基线 ref `56568811` 于 2026-09-26 12:13（北京时间）完成，Public test-reference 为 `0.62717`，ranking private score仍由 Kaggle 隐藏。它沿用 Phase 1 合规单 YOLO26m checkpoint：冻结的 test1000 预测与 inference-only ranking1000 预测合并，共 2,000 图、198,063 行，最终 CSV SHA-256 `1744A354...56C5E23`。最终选择仍为 `0/2`；在两个候选完成前不要勾选。Phase 1 合规最佳仍是 ref `56455800` / frozen Public `0.63546`。历史 `0.65091` 来自违规八模型融合，绝不能提交或最终选择。

- 官方 Rules 要求只能使用一个 detection model；主办方进一步明确：同一训练模型的 TTA/多尺度允许，不同模型的 voting、weighted fusion、post-NMS fusion 禁止。官方澄清：<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863>。
- `0.63917` 到 `0.65091` 的历史多模型成绩仅保留作研究记录；**禁止再提交任何 ensemble 文件，也不要把它们选为最终提交**。
- 第二个合规单模型候选已于 2026-09-20 19:27（北京时间）提交成功：同一个 YOLO26m checkpoint、七尺度、`fusion_iou=0.74`、`support_gain=0.125`。Public `0.63072`，较上一版 `0.63066` 仅提升 `+0.00006`；CSV 98,556 条、1000/1000 图、无效框 0，SHA-256 `E6F5BC18...2CB48E`，ref `56392305`。这证明细扫方向为正，但幅度已不足以承担 0.66 冲刺。
- 同 checkpoint 框校准候选已于 2026-09-22 14:30（北京时间）提交并 `Success`：在上述 `f0.74/support_gain=0.125` 的框上保持中心不变、宽高统一放大 1%。三折 held-out 增益均为正，合并 `0.70570818→0.70750647`（`+0.00179829`）；Public `0.63546`，相对原合规最佳 `0.63072` 提升 `+0.00474`。CSV 仍为 98,556 条/1000 图，SHA-256 `C01214E6...995A86`，ref `56455800`；这是新的合规安全线。
- Phase 2 ranking 数据已按官方清单选择性取得：1,000 张 16-bit grayscale PNG，官方总字节数 `3,893,334,217`；ranking-only ZIP SHA-256 `C6734C0D...1D590`。预处理严格复用 HSI16 band order `[5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15]` 和 per-image shared P0.5/P99.5 uint8 合同，无训练、伪标签、BN 更新或其他模型状态适配。
- Phase 2 ranking 推理只加载既有 checkpoint SHA-256 `8E4BFBF7...4254C3`，使用七尺度 `832/896/960/1024/1088/1152/1216`、`fusion_iou=0.74`、`support_gain=0.125`、`max_det=300`，再应用已审计的全局 boxscale101。ranking CSV 为 99,507 行/1,000 图，SHA-256 `1F9947E0...95B42`。
- 严格合并器核验 frozen Phase 1 CSV SHA-256 `C01214E6...995A86`、ranking CSV、checkpoint 与 lineage manifest；test/ranking ID overlap 为 0，最终 2,000 图覆盖完整、行 ID 连续、无 geometry/schema 问题。Kaggle ref `56568811` 状态 `COMPLETE`；最终选择仍为 `0/2`，必须在动作前确认后只勾选 refs `56568811` 与 `56455800`。
- 2026-09-26 12:28 用户重新开放**最多两次** Phase 2 提交用于提升，但只允许同一 checkpoint 的无状态推理变化。当前首选是复用七尺度 cache，再各图新增一次 `1024` horizontal-flip 推理；固定验证里仅 `flip fusion_iou=0.65/support_gain=0.125`（`+0.00016978`）和 `0.82/0.125`（`+0.00010398`）相对 incumbent 为正。两者都低于旧 `+0.001` 门禁，属于最后机会的低置信度候选，不得宣传为确定提升。完整交接见 `PHASE2_LAST_TWO_SUBMISSIONS_HANDOFF_20260926.md`。
- zero-init 私有消融已完成并否决：最佳 `0.69690`，低于旧同规格 `0.70143` 和门槛 `0.70443`；不跑全量、不提交。
- object-crop 私有消融已完成并否决：标准 full-val `0.69772`，低于旧同规格 `0.70143` 和门槛 `0.70443`；不跑全量、不提交，也不扩展 tile。
- P1–P99 已否决：固定划分 `0.69668 < 0.70143`；不要重复训练或提交其第九成员候选。
- SpectralStem 私有消融已完成并否决：标准 full-val 最佳与最终均为 epoch 30 的 `0.69930`，低于旧同规格 `0.70143` 和门槛 `0.70443`；不跑全量、不提交。生成的测试 CSV 仅通过本地结构校验，未上传比赛。
- phase-aware bilinear v1 私有消融已完成并否决：固定 2400/600，唯一训练变量是保留 4×4 mosaic 的物理 row/column phase，并按宽高比重建到长边 1024；最佳/最终 epoch 30 为 `0.69935`，比普通同规格 e30 的 `0.70143` 低 `0.00208`，比门槛 `0.70443` 低 `0.00508`。测试 CSV 的 38,304 条检测仅通过本地结构校验，未上传比赛。
- 显式 horizontal-flip-only 在原 `+0.001` 门禁下曾被正确否决：cache control 精确通过，七尺度 + flip 最高 `0.70587796`，只比 supported-full 高 `+0.00016978`。2026-09-26 用户为最后两次机会重新开放了该精确候选及第二个微正配置；这不是推翻旧证据，也不允许扩展扫参。
- 同规格 YOLO26m 45 轮 fixed split 已完成并否决：最佳/最终均为 epoch 45 的 `0.69899`，比 e30 的 `0.70143` 低 `0.00244`，比全量门槛低 `0.00544`；不启动 full-data 45 轮训练，测试 CSV 仅通过本地结构校验，未上传比赛。
- `cls_pw=0.25` 私有 fixed-split 消融已完成并否决：固定 2400/600，唯一训练变量为分类频次权重；最佳 epoch 27 为 `0.69863`（mAP50 `0.95662`），最终 epoch 30 为 `0.69862`，低于普通同规格 e30 `0.70143` 和 `0.70443` 门槛。训练 return code 0、batch 8/workers 2/device 0，无 CUDA OOM 或 shared-memory 错误；测试 CSV 的 36,784 条检测仅通过本地 checker，未上传比赛、无 Public 分数。
- random-affine `scale=0.3` 私有 fixed-split 消融已完成并否决：固定 2400/600，已记录的训练与数据参数中唯一差异为 `scale 0.5→0.3`；最佳/最终 epoch 30 为 `0.69992`，较普通同规格 e30 `0.70143` 低 `0.00151`，低于 `0.70443` 门槛 `0.00451`。训练 return code 0，无 CUDA OOM 或 shared-memory 错误；测试 CSV 的 31,848 条检测仅通过本地 checker，未上传比赛、无 Public 分数。
- 同 checkpoint 七尺度来源消融已完成并否决：七项 leave-one-out、中心 3/5 尺度和两组固定对称权重共 11 个候选均低于 `0.70570818` control；最好是删除 `1216`，仍低 `0.00044541`。保留全部七尺度统一权重，不生成或提交新候选。
- `dfl=2.0` 与 `dfl=2.5` 两个彼此独立的单 YOLO26m fixed-split 私有作业均已完成并否决：最佳/最终分别为 `0.70165` 和 `0.70186`，虽比普通 e30 `0.70143` 高 `+0.00022/+0.00043`，但仍比 `0.70443` 门禁低 `0.00278/0.00257`；不跑对应 full-data、不生成正式候选、不提交，且绝不融合二者。
- RT-DETR-L HSI16 random-extra fixed 作业已完成并否决：私有 `zephyrpong/hsi-rtdetr-l-ablation` version 1 成功跑完 30 轮、batch 2/workers 2/device 0，最佳 epoch 29 `mAP50-95=0.69309`、最终 `0.69294`，比 YOLO26m 基线低 `0.00834`、比门槛低 `0.01134`。fresh reload 证实 16-channel stem、18 类、`num_denoising=100`；300,000 行测试 CSV 结构有效但不得提交。
- RT-DETR-L 条件式 full-data 私有包 `kaggle_remote/kernel_full_rtdetr` 仍只在本地；由于 random fixed 已失败，**永不推送该 full-data 包**。RT-DETR 分支只剩 zero-extra fixed，再失败才轮到 `num_denoising=200`，二者均须等 GPU 配额真实恢复。
- YOLO26m `degrees=5` fixed-split 单变量作业已完成并否决：合同审计通过，首个 batch 8/workers 2 attempt 正常完成；最佳 epoch 29 `mAP50-95=0.68169`，最终 `0.68092`，比普通 e30 `0.70143` 低 `0.01974`，比 `0.70443` 门槛低 `0.02274`。不跑 full-data，不上传其测试 CSV，无 Competition Submit。
- Phase 2/ranking set 已发布并完成独立清单、下载、哈希、预处理、推理和合并审计；官方 ranking 精确 1,000 张，与 test1000 的 image ID 集合无重叠。不得再重复下载、推理或提交同一文件。
- 公开 `YOLO11m + [13,8,5]` Notebook 显示的 `0.7394` 已证实存在严重重复泄漏：其双 `**` glob 把 3,000 张训练 PNG 列成 6,000 条，实际 844 张 val 中有 788 张（93.36%）也在 train，test 也从 1,000 重复为 2,000 条。该分数不得与 fixed 2400/600 门禁比较，也不据此启动 full-data；完整证据见 `experiments/latest-public-strategy-research-20260922.md`。
- `[13,8,5]` 伪 RGB 后备 fixed 实验已预注册并生成独立私有本地包 `kernel_ablation_prgb1385`；当前全仓 `135 passed`，生成 runner 与主 runner 只差预期 CONFIG。该包**未上传、未运行**，只有 degrees=5 与 EIoU 审计后仍值得投入时才考虑推送，门禁仍为 `0.70443`。
- RT-DETR-L 的 `zero` 额外通道初始化也已预注册为独立 fixed 单变量后备：私有本地包 `kernel_ablation_rtdetr_xczero`、远端 slug `zephyrpong/hsi-rtdetr-l-xczero-ablation`，精确 CONFIG 已检查；架构感知门禁新增了显式 `--expected-extra-channel-init zero` 校验，当前全仓 `135 passed`。它**未上传、未运行**。只有当前 random-extra-channel RT-DETR fixed 门禁失败时才允许推送一次，YOLO26m zero-init 的既有负结果不能替代这项不同架构的测量。
- YOLO26 扩容路线已复核：YOLO26l 早已在同类 HSI16/1024/30-epoch fixed 实验中以 `0.69628` 失败，P2 与 1280 也有明确负信号，不重复；唯一未测的 YOLO26x 已预注册为最低优先级 fixed 后备 `kernel_ablation_yolo26x`（slug `zephyrpong/hsi-yolo26x-ablation`），只改 `yolo26m.pt -> yolo26x.pt`，保守单卡 attempts `2:2,1:2,1:0`。它**未上传、未运行**，排在 degrees=5、EIoU、`adamw_lr001` 与 `[13,8,5]` 之后。
- RT-DETR 类名别名已否决：`people -> person` 与 `e-bike -> bicycle` 没有官方语义等价证据，不能把当前 4/18 精确 COCO 行迁移当作 bug。现成的 `num_denoising=200` 单变量包 `kernel_ablation_rtdetr_nd200` 已预注册（slug `zephyrpong/hsi-rtdetr-l-nd200-ablation`），它**未上传、未运行**，只在 random 与 zero extra-channel 两项 RT-DETR fixed 都失败后排队。
- YOLO26m EIoU 已在 AutoDL 重庆 743 单机完成严格 fixed：30/30、batch 8/workers 2，最佳 epoch 29 `mAP50-95=0.70128`、最终 `0.70106`，比 `0.70443` 门禁低 `0.00315`，因此明确 NO-GO，不跑 full-data。正式 wrapper 在训练后因远端缺少 `pandas` 停于测试推理，留下完整权重/合同/失败归档；这不改变 fixed 精度结论，也不得以“补依赖”为由推进 full-data。
- YOLO26m 有效 AdamW `lr0=0.001` 本机 fixed 已完成并否决：严格 2400/600、30 轮、batch 2/workers 0，最佳/最终 epoch 30 `mAP50-95=0.67233`，比普通 fixed `0.70143` 低 `0.02910`、比门禁 `0.70443` 低 `0.03210`。进程正常退出、stderr 为空，runtime optimizer/model contracts 与 fresh `best.pt` reload 均通过；不跑 full-data、不生成正式 CSV、不提交。详见 `experiments/local-adamw-lr001-run-20260923.md`。
- 用户仍禁止任何训练和广泛调参，但于 2026-09-26 重新开放一个最多两次提交的 inference-only 收尾冲刺。不得启动 `reg_max=16`、通道顺序、YOLO11m/26x、fixed/full-data 或新模型；只允许交接文档里固定的同 checkpoint horizontal-flip 候选，且每次 Submit 与最终选择都要动作前确认。一次因上下文重置误启动的 `reg_max=16` 本机进程在第 1 轮完成前已按精确 PID 树停止，没有 `results.csv` 或权重，不能当作实验结果。
- YOLO26m `reg_max=16` 已完成可审计 fixed 预注册：它把当前 `reg_max=1` 的归一化 L1 路径改为真正 16-bin DFL，其他训练合同不变。真实权重审计显示可精确复用源 checkpoint 的 `99.4216%` 参数，regmax 专属 shape mismatch 仅 1,560 个源参数；真实 16 通道 forward/loss/backward 与 4 图 Trainer/contract/checkpoint smoke 均通过。一次上下文重置后的本机误启动在第 1 轮完成前即按用户“停止优化”决定中止，没有 `results.csv` 或权重，故仍无 fixed 精度结论，绝不能续跑或引用成正负结果。
- 普通 HSI16 `[13,8,5,...]` 通道顺序候选已完成可审计 fixed 预注册：它仍是同一 16 个物理 band、同一 shared P0.5-P99.5 uint8、同一标签/manifest，只把 pretrained RGB 三个数组槽位的物理映射从 `[5,8,13]` 改为 `[13,8,5]`。新 `preparation_config/report/dataset/manifest` 合同与哈希门禁、独立数据目录会拒绝静默复用 baseline NPY；合成集端到端测试证明数组只发生预期 permutation，标签、manifest 与 normalization bounds 不变。全仓 `177 passed`；本地私有包 `kernel_ablation_order1385`（slug `zephyrpong/hsi-yolo26m-order1385-ablation`）和 code-dataset staging 已就绪但**未上传、未运行**，详见 `experiments/yolo26m-hsi16-order1385-prereg-20260922.md`。
- YOLO11m HSI16 checkpoint-native 候选也已完成可审计 fixed 预注册：官方 Ultralytics Assets `v8.4.0` 权重已下载并验 SHA-256 `D5FFC1A6...305B95`（40,684,120 bytes）；真实架构是 `reg_max=16`、true DFL、非 end-to-end Detect。HSI16/18 类目标可精确复用源 checkpoint 的 `99.6848%` 参数；真实 forward/loss/backward、4 图 Trainer、model/optimizer contract、保存后 fresh reload 与全仓 `184 passed` 均通过。本地私有包 `kernel_ablation_yolo11m_rm16`（slug `zephyrpong/hsi-yolo11m-rm16-ablation`）及 code-dataset staging 已就绪但**未上传、未运行**；来源种类、权重大小和 SHA-256 都进入 status/gate，详见 `experiments/yolo11m-hsi16-prereg-20260922.md`。
- AutoDL 重庆 A 区 743 实例 `xbprde6fuz-ac3c0e3d` 曾在用户逐步确认后创建、私有上传并完成 EIoU fixed；随后已关机。2026-09-23 最后一次开机尝试被平台以余额 `-1.01` 元、需支付 `6.46` 元拒绝，没有新增计费。完赛不需要恢复这条失败路线：不充值、不再开机、不安装远端依赖、不做恢复推理或新训练。
- AutoDL EIoU 固定链及原始证据保留：私有 bundle 41,035,836 bytes、SHA-256 `849E82D90865FBC42510AEE11E4382AC2EE48FE625964554CE2672C10981D243`；原始比赛 ZIP 6,694,200,518 bytes、SHA-256 `C99BE7F2F930813E0846F64931851466F8DEB3720E35AB0F5800EB0F02E0B8AE`。远端训练失败归档 81,695,952 bytes、SHA-256 `E7E094A860A37C59CBA51A532DBB486EEB8FE3627503B2FA5429C269A9A1090F`；失败点是训练后 `ModuleNotFoundError: No module named 'pandas'`，不是训练失败。当前只保留材料，不再恢复远端作业。
- 面积分层框缩放已零 GPU 审计并否决：75 个 threshold/small-scale/large-scale 候选的三折 OOF 为 `0.70686997`，比 identity 高 `+0.00116178`，但比已经提交的全局 `×1.01` 低 `0.00063651`；full-val 最优本身就是全局 `×1.01`，折间选择也不一致。保留全局校准，停止继续扫框后处理。
- distinct-scale support-count 排序已零 GPU 审计并否决：固定单 checkpoint 七尺度、`fusion_iou=0.74`、全局框 `×1.01`，只用 `gamma` 奖励独立尺度票数；三折与 full 均保留现有 confidence-mass `support_gain=0.125`。最佳 count-only `gamma=0.08` 仅 `0.70590884`，比 incumbent `0.70750647` 低 `0.00159764`；不生成 CSV、不提交，停止继续置信度排序扫参。
- 当前 incumbent 的最终 `max_det` 已按预注册网格 `{200,300,400,500}` 做零 GPU OOF 审计并否决：三折与 full 均保留 `300`，OOF 仍为 `0.70750647`；`500/200/400` 分别低 `0.00058980/0.00105322/0.00114808`。不生成 CSV、不提交，不做自适应 cap 扩展。
- 主办方于 2026-09-21 16:02（北京时间）发布官方帖子“【最终提醒】参赛团队信息收集即将截止”：所有团队须在 **9 月 23 日前**把团队信息发往所属赛道指定邮箱；邮件主题格式为“赛道名称－团队名称－队长姓名”，正文需列出全部队员姓名、所在单位及指导教师（如有）。逾期可能影响成绩认定和奖励。该提醒正文**没有给出目标检测赛道的指定收件邮箱**，只给出疑问联系人 `1522859637@qq.com`，不得把疑问邮箱擅自当作材料收件箱；也不能从 Kaggle 页面证明用户是否已经发送。官方帖：<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742296>。
- 2026-09-22 官方补充通知 Discussion `742487` 覆盖冲突 Rules：Phase 1 于 **2026-09-25 16:00 北京时间**冻结；Phase 2 至 **2026-09-27 16:00**。Phase 2 每个 CSV 必须同时覆盖 test 1000 + ranking 1000；最终分数为冻结 Phase 1 与 Phase 2 各 50%，且最多手工指定 2 个最终提交。ranking 数据只能无状态推理，不能训练、伪标签、BN 统计更新或自适应。本项目已按该合同完成 ref `56568811`；私榜仍隐藏，最终勾选待用户确认。

## 👉 接手后第一件事

1. random-affine `scale=0.3` fixed split 已以 `0.69992 < 0.70443` 结束；不要启动其 full-data 训练，也不要上传它生成的测试 CSV。
2. `cls_pw=0.25` fixed split 已以最佳 `0.69863 < 0.70443` 结束；不要启动其 full-data 训练，也不要上传它生成的测试 CSV。
3. phase-aware fixed split 已以 `0.69935 < 0.70443` 结束；不要启动其 full-data 训练，也不要上传它生成的测试 CSV。
4. e45 fixed split 已以 `0.69899 < 0.70443` 结束；不要启动其全量 3000 张训练。Horizontal-flip-only 低于旧 `+0.001` 门禁；现在仅按专用交接精确复用两个微正配置，不得重新扫参。
5. `dfl=2.0/2.5` 都已门禁失败；不要启动其 full-data，也不要上传两条作业各自产生的测试 CSV。
6. 禁止训练与开放式扫参。Phase 2 baseline 已完成；新的窄范围任务只允许同一 checkpoint、一次额外 `1024` horizontal flip 的两个预先固定融合配置，最多消耗两次提交。先做代码/验证/哈希/2000图合并，逐次停在 Submit 前等确认。
7. RT-DETR random fixed 已以 `0.69309 < 0.70443` 结束并完整审计；禁止推它的 full-data 包、提交其测试 CSV或继续 zero/nd200 分支。
8. Kaggle GPU 配额最后一次登录态刷新为 `38:03 / 30 hrs`；AutoDL 743 实例已关机且余额最后为 `-1.01` 元。完赛不需要继续试探配额、开机、充值或恢复远端依赖。
9. 私有 bundle、原始 ZIP、远端训练权重/合同与失败归档都保留作学习证据；不得公开分享或清理。Phase 2 只复用已验证的 incumbent 权重与本地推理链。
10. 单 checkpoint `boxscale101` Phase 1 ref `56455800` / frozen Public `0.63546`，Phase 2 baseline ref `56568811` / Public test-reference `0.62717` 均已 `COMPLETE`。不要重复提交这两个 CSV；只可提交带明确新 manifest/hash 的 flip 候选。最终选择动作仍等待单独确认。
11. 立即向用户确认团队信息邮件是否已经发送；若未发送，先找到“目标检测赛道指定邮箱”，再由用户提供/确认团队名称、队长姓名、队员、单位和指导教师信息。发送邮件会对外传输个人信息，不能凭公告内容擅自代发。

### 成绩与合规状态

| 配方 | Public | 合规状态 |
|---|---:|---|
| 单模型 s（16 波段，同 checkpoint 七尺度） | 0.62651 | 合规 |
| 单模型 m（16 波段，同 checkpoint 七尺度） | 0.62953 | 合规基线，ref `56303572` |
| 单模型 m + 七尺度支持票，`f0.70/gain0.125` | 0.63066 | 合规；ref `56379896` |
| 单模型 m + 七尺度支持票，`f0.74/gain0.125` | 0.63072 | 原合规最佳；ref `56392305` |
| **单模型 m + 上述七尺度 + 全局框宽高 `×1.01`** | **0.63546** | **当前已验证的合规最佳；ref `56455800`；较前者 `+0.00474`** |
| **Phase 2：冻结 test1000 + inference-only ranking1000，同 checkpoint 配方** | **0.62717（Public test-reference）** | **`COMPLETE`；ref `56568811`；ranking private score 尚未公布；最终选择待确认** |
| 两到八模型融合 | 0.63917 ~ 0.65091 | **不合规，历史记录，不得再提交/最终选择** |

### 若未来明确重启时的候选（当前全部停止）

1. Phase 1 `boxscale101` ref `56455800` 与 Phase 2 baseline ref `56568811` 均已 `COMPLETE`，不要重复提交。当前最终榜单仍是 `0/2`；先完成最多两个受限 flip 候选，再根据合规性、Public test-reference 和离线证据提议最终两条。任何勾选都需用户单独确认，绝不能让 Kaggle 自动选中历史不合规 ensemble。
2. EIoU fixed 已以 `0.70128` 失败，AdamW `lr0=0.001` 本机 fixed 已以 `0.67233` 失败。不要续跑或转 full-data。
3. 冻结的研究队列下一项原为 `reg_max=16`，再考虑普通 HSI16 `[13,8,5,...]` 通道顺序；当前用户已停止优化，两项都不得启动。公开 Notebook 的 `0.7394` 有 93.36% 验证泄漏，不能作为收益证据。
4. RT-DETR zero 若也失败、仍有额度，再只跑一次 `num_denoising=200` fixed；不要添加 COCO 类名别名。
5. `reg_max=16`、普通 HSI16 `[13,8,5,...]` 与 YOLO11m HSI16 都仍只有“值得一次 fixed 测量”的资格，绝不能把本地 smoke/合同、权重可迁移率或既有 `dfl=2.0/2.5` 写成精度收益。YOLO11m 已下载官方 checkpoint、完成哈希/血缘审计并打包，排在顺序候选之后。
6. 只有 EIoU、`adamw_lr001`、`reg_max=16`、普通 HSI16 通道顺序与 YOLO11m 等更直接候选都失败且额度仍足，才运行已预注册的 YOLO26x fixed。YOLO26l、P2、1280 不得重复。
7. `dfl=2.0/2.5`、YOLO26m zero-init、object-crop、SpectralStem、phase-aware HSI16、`cls_pw=0.25`、random-affine `scale=0.3`、e45、tile TTA、尺度来源/权重、support-count 排序、最终 `max_det` 网格和 RT-DETR random-init 均已否决。Horizontal-flip-only 仍属旧门禁 NO-GO，但用户仅为最后两次机会例外开放专用交接中的两个已有微正配置。框校准只保留全局 `1.01×1.01`，不继续扫参。
8. 不再扫多模型权重、WBF、成员组合或类别融合。历史 ensemble cache 仅供离线研究，不得生成比赛提交。

> Phase 2 baseline ref `56568811` 已完成。用户声称当天还剩 2 次提交机会；接手者必须现场复核额度后再使用。最多提交两个交接文档中固定的同-checkpoint flip 候选，每次都要在最终 Submit 前重新确认。最终勾选同样单独确认，绝不能让 Kaggle 自动选中历史不合规 ensemble。

### 常用命令（Git Bash，项目根目录）

生成并推送一个 Kaggle 训练（例：伪RGB 数据、全量、7 尺度推理）：

```bash
(cd kaggle_remote && ../.venv/Scripts/python.exe make_kernel.py --mode full --model yolo26m.pt --epochs 30 --data pseudo_rgb --multiscale)
```

```bash
(cd kaggle_remote/kernel_full_prgb && kaggle kernels push -p .)
```

查看状态 / 下载结果（把 `<slug>` 换成上面打印的 Notebook 名，如 `hsi-yolo26m-prgb-full`）：

```bash
kaggle kernels status zephyrpong/<slug>
```

```bash
(mkdir -p kaggle_remote/outputs/新目录 && cd kaggle_remote/outputs/新目录 && kaggle kernels output zephyrpong/<slug> -p .)
```

手动提交一个已经生成好的文件：

```bash
(cd submissions && kaggle competitions submit hyperspectral-object-detection-challenge-2026 -f 文件名.csv -m "说明")
```

> ⚠️ Kaggle 登录（OAuth）**约 12 小时过期**，报 `Permission 'kernels.get' was denied` 其实是掉登录了，
> 用 `kaggle auth login` 重新登录（会开浏览器，需要人操作）。

### git 状态

**核心代码、文档和实验记录持续提交到本地仓库**（最新提交和远端同步状态以 `git log -1 --oneline`、`git status -sb` 为准）。2026-09-26 用户已明确授权把最终安全收尾推送到 GitHub；推送时只允许代码、测试和文档，不得包含数据、权重、CSV 或 ignored artifacts。
远程是 `origin = https://github.com/khalilpong/hyperspectral-object-detection-2026.git`。
**push 之前先确认这个 GitHub 仓库是私有的**：文档里有比赛方法和成绩细节，比赛期间不宜公开。改完东西记得再 `git status` 看一眼，别让新文件漏提交。

以下内容**已经配置为不进 git**，不要手动加进去：

| 路径 | 原因 |
|---|---|
| `exports/` | 私有恢复备份包（每个约 145MB），**比赛期间不得外传** |
| `kaggle_remote/code_dataset/`、`raw_dataset/`、`outputs/` | 含模型权重、6GB 原始数据、训练产物 |
| `data/`、`runs/`、`*.pt`、`submission*.csv` | 大文件 |

### Kaggle 上的现有资源

| 资源 | 说明 |
|---|---|
| 私有数据集 `zephyrpong/hsi-detection-code` | 2026-09-22 07:38 新版本已实时核验为 `ready`；远端清单新增 `hsi_detection.box_loss.py` 并含更新后的训练入口/runner，同时保留 RT-DETR trainer/HGStem 迁移、SpectralStem、phase-aware、划分清单与 yolo26m/s 权重；`rtdetr-l.pt` 不放进该 CC0 staging，Kernel 联网从官方 Ultralytics Assets 获取 |
| 私有数据集 `zephyrpong/hsi-competition-raw` | 原始比赛 zip（比赛数据无法直接挂载进 Notebook，见 Kaggle 文档第 5 条坑） |
| Notebook `hsi-yolo26m-{smoke,ablation,full}`、`hsi-yolo26l-ablation` | 16 波段 m / l 的各次训练，产物在 `kaggle_remote/outputs/` 对应目录 |
| Notebook `hsi-yolo26m-p010-990-{full,ablation}` | P1–P99 归一化的新成员；full/ablation 均完成并下载；ablation `0.69668` 低于旧基准 `0.70143`，该方向已否决 |
| Notebook `hsi-yolo26m-prgb-{ablation,full}`、`prgb368-full`、`prgb0715-full` | 伪RGB（5/8/13、3/6/8、0/7/15）的训练，产物同上 |
| `hsi-yolo26m-stem-ablation` | 私有 Kernel 版本 1 `COMPLETE`；SpectralStem fixed 2400/600，epoch 30 最佳/最终 `0.69930`，门禁失败；不跑全量、不提交 |
| `hsi-yolo26m-ablation` | 私有 Kernel 版本 2 `COMPLETE`；普通 HSI16 YOLO26m fixed 2400/600 延长到 45 轮，最佳/最终 `0.69899`，门禁失败；不跑 full-data e45、不提交 |
| `hsi-yolo26m-phase-ablation` | 私有 Kernel 版本 1 `COMPLETE`；phase-aware HSI16 fixed 2400/600，epoch 30 最佳/最终 `0.69935`，门禁失败；不跑 full-data、不提交 |
| `hsi-yolo26m-clspw025-ablation` | 私有 Kernel 版本 1 `COMPLETE`；普通 HSI16 fixed 2400/600，`cls_pw=0.25`，最佳 epoch 27 `0.69863`，门禁失败；不跑 full-data、不提交 |
| `hsi-yolo26m-scale030-ablation` | 私有 Kernel 版本 1 `COMPLETE`；普通 HSI16 fixed 2400/600，random-affine `scale=0.3`，最佳/最终 epoch 30 `0.69992`，门禁失败；不跑 full-data、不提交 |
| `hsi-yolo26m-dfl200-ablation` / `dfl250-ablation` | 私有 Kernel 版本 1 均 `COMPLETE`；最佳/最终 `0.70165/0.70186`，均未过 `0.70443`；不跑 full-data、不提交 |
| `hsi-rtdetr-l-smoke` | 私有 Kernel 版本 1 `COMPLETE`；单 T4 batch 2、1 epoch、1024、HSI16，预训练/16 通道迁移/训练/重载/NPY 推理/checker 全通过；仅是远程全链路门禁，不是 fixed-split 成绩 |
| `hsi-rtdetr-l-ablation` | 私有 Kernel version 1 `COMPLETE`；单 RT-DETR-L checkpoint、fixed 2400/600、30 epoch、1024、random extra init、`num_denoising=100`；首个 batch2/workers2/device0 attempt 成功，最佳 epoch29 `0.69309 < 0.70443`，已否决，无 Competition Submit |
| `hsi-yolo26m-deg5-ablation` | 私有 Kernel 版本 1 `COMPLETE`；精确合同通过，最佳 epoch 29 `0.68169`、最终 `0.68092`，显著低于普通 e30 `0.70143` 与门槛 `0.70443`；已否决，不跑 full-data、不提交 |
| `hsi-yolo26m-prgb1385-ablation` | 本地私有包已生成，`pseudo_rgb:13,8,5`、YOLO26m、fixed 2400/600、e30；尚未 `kaggle kernels push`，不是远端作业；预注册见 `experiments/yolo26m-pseudo-rgb-1385-prereg-20260922.md` |
| EIoU AutoDL fixed（非 Kaggle Kernel） | 重庆 743 已完整训练 30 轮；最佳 `0.70128 < 0.70443`，NO-GO。训练后测试推理因缺 `pandas` 中止；保留权重/合同/失败归档，不恢复、不跑 full-data |
| `hsi-yolo26m-lr001-ablation` / 本机 fixed | 本机 30/30 完成，最佳/最终 `0.67233 < 0.70443`；合同与 fresh reload 通过但精度明显下降，NO-GO；Kaggle 私有包未上传，不再运行 |
| `hsi-yolo26m-rm16-ablation` | 本地私有包与 smoke 已完成；一次本机误启动在首轮完成前停止，没有结果/权重；当前用户停止优化，禁止续跑或上传 |
| `hsi-yolo11m-rm16-ablation` | 本地私有包已生成；checkpoint-native 官方 `yolo11m.pt` SHA-256 `D5FFC1A6...305B95`，HSI16/18 类精确形状迁移率 `99.6848%`，true-DFL forward/loss/backward、真实 Trainer model/optimizer contract、fresh reload 和全仓 `184 passed`；本地 code-dataset staging 已同步，但 dataset 新版本与 Kernel 都未上传、未运行；预注册见 `experiments/yolo11m-hsi16-prereg-20260922.md` |
| `submission_single_m_hsi16_ms7_f074_sg0125_boxscale101.csv` | 同一 full-data YOLO26m checkpoint 的七尺度合规 CSV，仅将每框宽高 `×1.01`；三折 OOF `+0.00179829`，SHA-256 `C01214E6...995A86`；2026-09-22 14:30 `Success`，Public `0.63546`，ref `56455800`；旧“今日余 2 次”快照已过期，不得复用 |
| `submission_phase2_single_m_hsi16_ms7_f074_sg0125_boxscale101.csv` | Phase 2 合并 CSV；冻结 test1000 + inference-only ranking1000，单 checkpoint，198,063 行/2,000 图，SHA-256 `1744A354...56C5E23`；2026-09-26 `COMPLETE`，Public test-reference `0.62717`，ref `56568811`；CSV 本体 ignored，不得公开推送 |
| Notebook `hsi-yolo26-smoke` | ❌ 第一次失败的旧版本，可忽略或删除 |
| ⚠️ 已训好的模型不要重训 | 权重都已下载在 `kaggle_remote/outputs/*/…/last.pt`，重训只会白耗额度 |

## 比赛与成绩

| 项目 | 内容 |
|---|---|
| 比赛 | [Hyperspectral Object Detection Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026) |
| 任务 | 高光谱图像目标检测，18 类，评分指标 mAP@[0.5:0.95] |
| 阶段截止 | 官方补充通知 `742487`：Phase 1 **2026-09-25 16:00** 冻结；Phase 2 排名集随后发布并于 **2026-09-27 16:00** 截止（北京时间）；冲突处以该通知为准 |
| 当前公开榜前三（2026-09-23 登录态快照） | **0.68118 / 0.67803 / 0.67795**；方法与单模型血缘未公开 |
| 本项目账号历史显示最高 | **0.65091**（八模型融合，提交编号 56378610；当前规则下不合规，不得作为最终提交） |
| 已验证合规最佳 | **0.63546**（单 YOLO26m checkpoint + 七尺度支持票 + 全局框宽高 `×1.01`，提交编号 56455800；较原合规最佳 ref 56392305 提升 `0.00474`，较原始 ref 56303572 提升 `0.00593`） |
| Phase 2 提交 | **COMPLETE**，ref `56568811`，Public test-reference `0.62717`；ranking private score 尚未公布；最终选择待确认 |
| 历史显示排名 | 第 23（2026-09-23 登录态快照，仍以不合规 ensemble `0.65091` 计；不代表安全名次） |
| 提交额度 | **每天 3 次**（09-18 实测：第 4 次报 400 Bad Request），**UTC 零点重置（北京时间 08:00）**；Kaggle 保留历史最佳，提交差的不会掉排名 |
| Kaggle 账号 | `zephyrpong` |

> Kaggle 命令行显示的时间都是 UTC。Kaggle **保留历史最佳成绩**，提交一个更差的结果不会降低排名。

## 历史不合规提交是怎么做出来的（仅供审计，禁止再提交）

> **规则红线：** 下列 A~H 是八个不同训练模型。官方明确禁止这种 voting/weighted fusion；保留本节是为了复盘已经发生的提交，不是复现建议。

**八个模型 × 7 个尺度 = 56 路预测投票融合**（脚本 `scripts/eval_ensemble.py`，一键封装 `scripts/build_ensemble_submission.sh`）：

| 成员 | 模型 | 数据 | 训练 | 权重文件 | 置信度系数 |
|---|---|---|---|---|---|
| A | YOLO26m | 16 波段 | Kaggle T4，batch 8，30 轮，全部 3000 张 | `kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt` | 1.0 |
| B | YOLO26s | 16 波段 | 本机，batch 4，30 轮，全部 3000 张 | `runs/final_s1024_all3000_hsi16_shared_p005_995_r1/weights/last.pt` | 0.6 |
| C | YOLO26m | **伪RGB（波段 5/8/13）** | Kaggle T4，batch 8，30 轮，全部 3000 张 | `kaggle_remote/outputs/prgb-full/kaggle_full_yolo26m_prgb_e30/last.pt` | 0.8 |
| D | YOLO26s | **伪RGB** | 本机，batch 4，30 轮，全部 3000 张 | `runs/final_s1024_all3000_b5-8-13_r1/weights/last.pt` | 0.6 |
| E | YOLO26m | 16 波段 | 本机，**batch 2**，30 轮（单模型只有 0.60553） | `runs/final_m1024_all3000_hsi16_shared_p005_995_r1/weights/last.pt` | 0.5 |
| F | YOLO26s | 16 波段 + **伪标签** | 本机，batch 4，30 轮（单模型 0.61709） | `runs/final_s1024_all3000_hsi16_pseudo722_r1/weights/last.pt` | 0.5 |
| G | YOLO26m | 伪RGB **波段 3/6/8** | Kaggle T4，batch 8，30 轮，全部 3000 张 | `kaggle_remote/outputs/prgb368-full/kaggle_full_yolo26m_prgb368_e30/last.pt`（图片 `data/processed/pseudo_rgb_b3-6-8`） | 0.8 |
| H | YOLO26m | 伪RGB **波段 0/7/15** | Kaggle T4，batch 8，30 轮，全部 3000 张 | `kaggle_remote/outputs/prgb0715-full/kaggle_full_yolo26m_prgb0715_e30/last.pt`（图片 `data/processed/pseudo_rgb_b0-7-15`） | 0.8 |

> 只用 A~D 四个成员是 0.64744，A~F 六个是 0.64766，A~H 八个用旧排序是 0.64831；相同 A~H、只加入 `support_gain=0.125` 后为 0.65091。
> 想省时间就用 A~D。

推理：每个成员各跑 7 个尺度（832~1216），NMS iou 0.70，conf 0.0001，max_det 300；
所有结果按置信度加权投票融合（fusion IoU 0.70），融合排序使用 `support_gain=0.125`。

仅供离线复盘（Git Bash，项目根目录；不得上传所得文件）：

```bash
SUPPORT_GAIN=0.125 bash scripts/build_ensemble_submission.sh submissions/repro.csv
```

有缓存时只做融合，约 10 分钟；缓存被删了要重新跑 GPU 推理，8GB 显卡约 1 小时。上述命令会复现提交过的最佳文件
`submissions/submission_ens8_b368_b0715_ms7_f070_sg0125.csv`（SHA-256 `A4158EF7...0C410`）。

> ⚠️ `eval_ensemble.py` 按 `--model` 的**标签**缓存每个尺度的预测到 `artifacts/ensemble_cache/<标签>.pkl`。
> **换了权重或图片目录就必须换标签**（或删掉对应 pkl），否则会读到旧缓存。
> `#目录` 表示这个成员用自己的数据目录（伪RGB 是 3 通道 PNG），`@数字` 是它的置信度系数。

### 历史融合的四条经验（都有数据支撑，但不得用于比赛提交）

1. **成员之间"训练数据不同"才有用**。加伪RGB 模型 +0.0025（留出集），加 yolo26l（同样数据只是模型更大）只 +0.0008，
   加 60 轮的同数据模型 **0**（0.70824 vs 0.70830）。
2. **别只挑单独分数高的成员**。Kaggle 训的伪RGB-m 单独 0.70297 > 本机的 0.70183，但因为它和成员 A 的超参完全相同
   （30 轮 batch 8 同种子），错误相关，融合反而更低（0.70875 vs 0.71079）。
3. ⚠️ **600 张留出集对融合完全不可信**，三次都错：两模型留出 +0.0043 → 榜上 +0.0096；三模型留出 +0.0005 → 榜上 +0.0046；
   四模型留出 **−0.0012（说会变差）** → 榜上 **+0.0037**。
   **但反过来也会骗人**：留出集上最好的成员（本机 batch3/60轮 伪RGB-m）对应的全量模型加进去，榜上反而 **−0.0099**（七模型 0.63772）。
   **历史结论仅供研究：** 留出集无法可靠排序多模型组合；当前规则又明确禁止它们，因此不再生成或提交任何多模型候选。单模型训练改动仍应先走固定留出门槛。

4. **弱成员堆着没用**：加 batch2 的 m（0.60553）和伪标签 s（0.61709）两个弱成员只换来 +0.0002。
   涨分靠的是少数几个**训练数据不同的强成员**。

**单模型对照基准**（600 张留出集，7 尺度融合后）：s 0.70009 / m 0.70404 / 伪RGB-m（本机 batch3 e60）0.70183 /
伪RGB-m（Kaggle batch8 e30）0.70297 / yolo26l 0.70231。两模型 m+s×0.6 = 0.70830，三模型最好 0.71079。

## 已经证明没用的方向（不要重复做）

每一条都有完整数据，见 `experiments/experiments.csv`。

| 方向 | 结果 | 原因 |
|---|---|---|
| yolo26l（Kaggle 两张 T4 DDP，batch 8） | 单模型 0.69628（比 m 低 0.005）；当第 3 个融合成员 +0.00076 | 每卡 BN 只有 4 张图可能拖累；融合 2 个模型后已接近饱和 |
| 训练 60 轮 | +0.0009 | 低于轮间标准差 0.0016，噪声 |
| imgsz 1280 | −0.0075 | 原图只有 248×494，1024 之上已无真实细节 |
| 训练时多尺度增强 | 持续落后，中止 | — |
| 延长关闭 mosaic（close_mosaic 20） | −0.0121 | mosaic 对这个任务是正收益 |
| 按波段独立归一化 | 未训练，统计证否 | 各波段动态范围只差 1.7 倍，无精度浪费 |
| NMS / fusion IoU / max_det 微调 | `f0.70→0.74` 留出 +0.00027、Public 仅 +0.00006；其他旧扫参有负结果 | 推理侧已接近榨干 |
| 全分辨率灰度图（pan） | 落后 0.07~0.19 | 抹掉了光谱；真鸡蛋/塑料鸡蛋/木鸡蛋形状相同，只能靠光谱区分 |
| pan + 光谱 17 通道融合 | −0.0089 | 前期打平，后期精修阶段 pan 通道成了干扰 |
| SpectralStem 16→3 可学习投影 | `0.69930`，比旧同规格 `0.70143` 低 `0.00213` | mAP50 提升 `0.00392`，但 mAP50-95 下降；没有改善高 IoU 定位，`stone_block` 等类别退化 |
| Phase-aware HSI16 输入重建 | `0.69935`，比普通同规格 e30 `0.70143` 低 `0.00208` | 保留 4×4 物理 phase 的 bilinear v1 重建提高了 mAP50，但未改善严格 IoU 定位；低于 `0.70443` 门禁，不跑全量、不提交 |
| Horizontal-flip-only TTA | 七尺度+flip 最佳 `0.70587796`，只比同 evaluator supported-full 高 `+0.00016978` | 低于旧 `+0.001` 门禁；仅因最后两次机会按专用交接例外开放两个既有微正配置，不扩展搜索 |
| YOLO26m 延长到 45 轮 | `0.69899`，比 e30 `0.70143` 低 `0.00244` | epochs 31–45 无一轮超过 e30；mAP50 上升但严格 IoU 总指标下降，不跑全量 |
| `cls_pw=0.25` | fixed split 最佳 `0.69863`，比普通同规格 e30 `0.70143` 低 `0.00280` | 分类频次加权提高 mAP50，但没有改善严格 IoU 定位；低于 `0.70443` 门禁，不跑全量、不提交 |
| Random-affine `scale=0.3` | fixed split `0.69992`，比普通同规格 e30 `0.70143` 低 `0.00151` | 温和缩放提高 mAP50，但没有改善总体严格 IoU 定位；低于 `0.70443` 门禁，不跑全量、不提交 |
| 伪标签（给测试图自动打标签再训练） | Kaggle **−0.0094** | 丢了低分框长尾的召回；伪框继承了教师模型的定位误差 |
| yolo26m（本机） | Kaggle 0.60553 | **被 8GB 显存逼到 batch=2**，不是模型不行 |
| P2 小目标检测头 / box loss 权重 10 | 更差 | — |
| RT-DETR-L random-extra, `num_denoising=100` | `0.69309`，比 YOLO26m baseline 低 `0.00834` | 训练/迁移/重载均正常但严格 IoU 明显落后；不跑同配方 full-data，后续变量必须独立过门禁 |
| 面积分层框缩放 | OOF `0.70686997`，比全局 `×1.01` 低 `0.00063651` | full-val 最优仍是全局 `×1.01`，折间选择不一致；不再继续框校准扫参 |
| 独立尺度票数排序 | 最佳 count-only `gamma=0.08` 为 `0.70590884`，比 incumbent `0.70750647` 低 `0.00159764` | 三折与 full 均保留现有 confidence-mass `support_gain=0.125`；不生成 CSV，不继续置信度排序扫参 |
| 最终 `max_det` 网格 | `300` 仍为 `0.70750647`；`500/200/400` 全部更低 | 三折 fit 均保留 300，门禁 false；不生成 CSV，不做图像/类别自适应 cap |
| EIoU（AutoDL batch 8） | fixed split 最佳 `0.70128`、最终 `0.70106` | 比 `0.70443` 门禁低 `0.00315`；训练后推理因缺 pandas 中止，但 fixed NO-GO 已成立 |
| AdamW `lr0=0.001`（本机 batch 2） | fixed split 最佳/最终 `0.67233`，比普通 fixed 低 `0.02910` | 30/30 正常完成且合同/重载通过，但明显低于 `0.70443` 门禁；不跑 full-data、不提交 |

## 训练算力：Kaggle 免费 GPU（已验证够用）

所有训练都在 Kaggle 上跑，脚本 `kaggle_remote/run_hsi_yolo26.py` + `make_kernel.py`，
详见 [docs/KAGGLE_REMOTE_TRAINING.md](docs/KAGGLE_REMOTE_TRAINING.md)。

| 事实 | 数据 |
|---|---|
| 硬件 | 2×T4 15GB、4 核、31GB 内存，单次最长 12 小时，每周约 30 小时 |
| yolo26m + 16 波段 | batch 8，每轮 4.5~4.8 分钟，30 轮全量约 2.6 小时 |
| yolo26m + 伪RGB | batch 8，每轮 4.2 分钟，30 轮全量约 2.2 小时（数据准备只要 2.5 分钟，比 16 波段快得多） |
| yolo26m + phase-aware HSI16 fixed split | 数据生成 21.8 分钟、训练 2.09 小时、推理 83 秒，总流程约 2.50 小时；临时 NPY 约 33.65 GB |
| yolo26m + HSI16 `cls_pw=0.25` fixed split | 从启动到数据准备完成约 8.9 分钟、训练 2.12 小时、单尺度推理 59 秒，总流程约 2.29 小时；batch 8/workers 2/device 0；36,784 detections 本地 checker 通过；无比赛 Submit |
| yolo26m + HSI16 random-affine `scale=0.3` fixed split | 数据准备约 5.5 分钟、训练约 1.95 小时、单尺度推理 54 秒，总流程约 2.06 小时；batch 8/workers 2/device 0；31,848 detections 本地 checker 通过；无比赛 Submit/Public |
| 两张卡 DDP | `--attempts 8:2:0+1` 可用，已实测跑通（yolo26l 每轮 3.1 分钟） |
| 额度消耗 | 09-17~09-18 两天共用掉约 20 小时（每周约 30 小时，重置日以 Kaggle 页面为准） |

Kaggle 周额度最后实测为 `38:03 / 30 hrs`。AutoDL 重庆 743 曾成功运行 EIoU fixed，后已关机；最后余额 `-1.01` 元，开机请求被拒且未计费。用户已明确停止比赛优化，因此当前不存在任何待启动的算力任务。自审计 bundle、wrapper 和远端失败证据只作为学习材料保留，见 [docs/CLOUD_SERVER_TRAINING.md](docs/CLOUD_SERVER_TRAINING.md)。

## 环境与操作的坑（本机）

| 坑 | 正确做法 |
|---|---|
| 用系统 Python 跑 | 必须用 `.venv\Scripts\python.exe`，否则缺 torch/ultralytics |
| 16 通道数据开多个数据加载进程 | 本机 31.7GB 内存只能 `workers=1`；≥2 会因 mosaic 画布内存崩溃 |
| 看"空闲内存"判断会不会爆 | 要看**已提交内存**；空闲内存下降可能只是文件缓存，可回收 |
| 长训练被中断 | Ultralytics 每轮写 `last.pt`，用 `--resume` 续跑，见 `docs/REBOOT_RECOVERY_HSI_E60.md` |
| 续跑时改了 imgsz/batch | **会让实验作废**（有前车之鉴），续跑不要改超参 |
| D 盘空间 | 失败实验的派生数据集要及时删；删前确认最佳模型的数据 `data/processed/hsi16_shared_p005_995` 不在删除列表里 |
| Kaggle 命令行在 Windows 上 | 带 `/` 的相对路径会出错，**先 cd 进目录再用 `-p .`** |

## 文档地图

| 文档 | 内容 |
|---|---|
| **HANDOFF.md**（本页） | 现状、成绩、已证否方向、下一步 |
| [PROMPT_FOR_NEXT_AGENT.md](PROMPT_FOR_NEXT_AGENT.md) | 交给另一个 AI 接手时直接发给它的提示词 |
| [PHASE2_LAST_TWO_SUBMISSIONS_HANDOFF_20260926.md](PHASE2_LAST_TWO_SUBMISSIONS_HANDOFF_20260926.md) | 最后两次提交机会的精确状态、候选、红线与执行顺序 |
| [PROMPT_PHASE2_LAST_TWO_SUBMISSIONS.md](PROMPT_PHASE2_LAST_TWO_SUBMISSIONS.md) | 新对话可直接复制的短提示词 |
| `scripts/build_ensemble_submission.sh` | 一键生成八成员融合提交，可用环境变量调权重 |
| [docs/KAGGLE_REMOTE_TRAINING.md](docs/KAGGLE_REMOTE_TRAINING.md) | Kaggle 免费 GPU 远程训练：原理、命令、踩坑、运行记录 |
| [docs/CLOUD_SERVER_TRAINING.md](docs/CLOUD_SERVER_TRAINING.md) | 云服务器训练：配置选型、从零到提交的完整步骤 |
| [docs/AUTONOMOUS_RUN_2026-09-17.md](docs/AUTONOMOUS_RUN_2026-09-17.md) | 9/17 通宵托管的完整决策日志（pan、融合、伪标签为何失败） |
| [experiments/experiments.csv](experiments/experiments.csv) | **所有实验的结果总表**，每行有配置、分数和结论 |
| [experiments/phase2-completion-20260926.md](experiments/phase2-completion-20260926.md) | Phase 2 数据、推理、合并、提交 ref 与哈希证据 |
| [docs/learning_path.md](docs/learning_path.md) | 训练/验证/测试等基础概念（新手先看） |
| [docs/REBOOT_RECOVERY_HSI_E60.md](docs/REBOOT_RECOVERY_HSI_E60.md) | 长训练中断后如何安全续跑 |
| `kaggle_remote/run_hsi_yolo26.py` | 远程训练流水线主脚本（Kaggle 和云服务器通用） |
| `kaggle_remote/make_kernel.py` | 从主脚本生成 Kaggle Notebook |
| `scripts/build_autodl_eiou_bundle.py` | 生成只含 EIoU fixed 所需代码/权重的逐文件验哈希 AutoDL 包 |
| `scripts/run_autodl_eiou.py` | AutoDL 资源/数据预检、exact fixed、门禁和结果归档；不跑 full、不提交 |

## 工作纪律（这个项目用教训换来的）

1. **一次只改一个变量**，否则分数变化无法归因。
2. **先留出验证，再全量提交**。跳过验证的伪标签实验让 Kaggle 分数掉了 0.0094。
3. **单模型改动**：留出集增益 < 0.003 不值得全量重训，< 0.001 不值得花提交额度。**融合类改动例外**：留出集不可信，直接提交验证。
4. **看分数，不看"看起来更好"的中间指标**。伪标签模型的置信度分布看着更干净，实际 mAP 更差。
5. **每个实验都登记进 `experiments/experiments.csv`**，包括失败的——失败记录能帮后来者少走弯路。

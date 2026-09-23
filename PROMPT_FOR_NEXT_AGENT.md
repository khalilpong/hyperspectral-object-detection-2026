# 交给另一个 AI 接手：提示词

> 使用方法：在项目根目录 `D:\kaggle\hyperspectral-object-detection-2026` 打开支持读写文件和执行命令的 AI 编程工具（如 Codex），
> 把下面"提示词"整段复制发送即可。所有细节都在 [HANDOFF.md](HANDOFF.md)，提示词只负责让它先读对文件、守住红线。
>
> 如果用的是**不能执行命令的纯聊天窗口**：把 HANDOFF.md 全文一起贴给它，让它把要运行的命令写出来，由你在 Git Bash 里执行并把输出贴回去。

---

## 提示词（从这里开始复制）

你要接手一个正在进行的 Kaggle 比赛项目：**Hyperspectral Object Detection Challenge 2026**（高光谱目标检测，18 类，指标 mAP@[0.5:0.95]）。
项目目录是 `D:\kaggle\hyperspectral-object-detection-2026`（Windows，用 Git Bash 执行命令；项目自带 `.venv`，必须用 `.venv\Scripts\python.exe`）。
我是新手，**请用中文交流**，解释要通俗，不要默认我懂术语。

### 第一步：先读，再动手

1. 完整读 `HANDOFF.md`（现状、成绩、已证否的方向、下一步、复现命令、环境坑）。
2. 读 `experiments/experiments.csv`（所有实验的结果，包括失败的）。
3. 需要跑 Kaggle 训练时读 `docs/KAGGLE_REMOTE_TRAINING.md`（尤其"踩过的坑"9 条）。

读完后先用 5 句话向我复述：合规 Public 基线、排行榜显示但不合规的历史分数、最近完成的实验、你打算先做什么、需要我做什么。**等我确认再开始花新的 Kaggle GPU 额度或提交。**

### 当前状态（以 HANDOFF.md 为准，这里是摘要）

- **最高优先级现状：用户已于 2026-09-23 明确决定停止本比赛的继续优化，把仓库和原始证据保留为后续 AI 学习资料，当前精力转向 FPGA。不要自动训练、推理生成、上传、Submit、最终榜单选择、AutoDL/Kaggle GPU 作业、充值/开机、删除或清理。只有用户以后明确重启比赛优化，才重新评估。**
- Kaggle 账号 `zephyrpong`。排行榜显示最高 `0.65091` 来自八模型融合，**违反官方单模型规则，只能视为历史不合规记录**；Kaggle 会给格式有效的 CSV 评分，但这不代表通过最终规则/代码审核。当前已验证合规的 Public 最佳是单 YOLO26m checkpoint 七尺度支持票加全局框宽高 `×1.01` 的 `0.63546`（ref `56455800`）。官方补充通知 `742487` 规定 Phase 1 于 **2026-09-25 16:00** 冻结，Phase 2 于 **2026-09-27 16:00** 截止（北京时间）；但当前停止决定意味着不得自动参与后续动作。
- 官方允许同一训练模型的 TTA / multi-scale inference，禁止不同训练模型的 voting、weighted fusion 和 post-NMS fusion。任何 `submission_ens*.csv` 都不得再上传或选为最终提交。
- 原合规最佳 `submissions/submission_single_m_hsi16_ms7_f074_sg0125.csv` 是同一 YOLO26m checkpoint 七尺度、`fusion_iou=0.74`、`support_gain=0.125`，Public `0.63072`；ref `56392305`，98,556 detections，SHA-256 `E6F5BC18...2CB48E`。
- 新合规最佳 `boxscale101` 只在上述 CSV 上保持中心不变、把框宽高统一放大 1%。三个 400-fit/200-held-out folds 全部独立选择同一参数，held-out 增益 `+0.00251177/+0.00133919/+0.00198724`，合并 OOF `0.70570818→0.70750647`（`+0.00179829`）。CSV 仍为 98,556 行/1000 图，SHA-256 `C01214E6...995A86`；2026-09-22 14:30 提交 `Success`，Public `0.63546`，ref `56455800`，较 `0.63072` 提升 `+0.00474`。旧“今日余 2 次”只是过期快照，不得复用。
- zero-init fixed split 最佳 `0.69690`，object-crop fixed split `0.69772`；两者均低于旧同规格 `0.70143` 和门槛 `0.70443`，已否决，不跑全量、不提交。
- SpectralStem 本地链路门禁已通过，私有 fixed 2400/600 消融也已正常完成，但标准 full-val 最佳/最终仅 `0.69930`，低于旧同规格 `0.70143` 和门槛 `0.70443`。该路线已否决：不跑全量、不上传其测试 CSV。
- phase-aware bilinear v1 私有 Kernel `zephyrpong/hsi-yolo26m-phase-ablation` version 1 已完成：固定 2400/600，唯一变量是保留 4×4 物理 phase 并按宽高比重建到长边 1024；最佳/最终 epoch 30 为 `0.69935`，低于普通同规格 `0.70143` 和门槛 `0.70443`，已否决。38,304 条测试检测仅通过本地 checker，无比赛 Submit。
- 显式 horizontal-flip-only 已完成同 checkpoint 固定划分验证：cache control 通过，但最佳七尺度+flip 仅 `0.70587796`，相对 supported-full 只增 `+0.00016978 < +0.001`，已否决，不生成或提交候选。
- 普通 HSI16 YOLO26m 延长到 45 轮的 fixed 2400/600 消融已完成：最佳/最终 epoch 45 仅 `0.69899`，比 e30 的 `0.70143` 低 `0.00244`，比 `0.70443` 门槛低 `0.00544`。不启动 full-data e45，不上传其测试 CSV。
- `cls_pw=0.25` 私有 fixed 2400/600 消融已完成并否决：唯一训练变量为分类频次权重；最佳 epoch 27 为 `0.69863`，最终 epoch 30 为 `0.69862`，低于普通同规格 e30 `0.70143` 和门槛 `0.70443`。36,784 条测试检测仅通过本地 checker，无比赛 Submit/Public；不要启动其 full-data 训练或上传该 CSV。
- random-affine `scale=0.3` 私有 fixed 2400/600 消融已完成并否决：已记录的训练与数据参数中唯一差异为 `scale 0.5→0.3`；最佳/最终 epoch 30 为 `0.69992`，较普通同规格 e30 `0.70143` 低 `0.00151`，低于门槛 `0.70443`。31,848 条测试检测仅通过本地 checker，无比赛 Submit/Public；不要启动其 full-data 训练或上传该 CSV。
- `dfl=2.0/2.5` 两个独立单模型 fixed 2400/600 作业均已完成并否决：最佳/最终 `0.70165/0.70186`，虽略高于普通 e30 `0.70143`，但仍低于 `0.70443` 门禁；不要启动对应 full-data 或上传其测试 CSV，二者绝不能融合。
- RT-DETR-L HSI16 random-extra fixed 已完成并否决：私有 `zephyrpong/hsi-rtdetr-l-ablation` version 1 成功跑完 fixed 2400/600、30 epoch、batch2/workers2/device0；最佳 epoch29 `0.69309`、最终 `0.69294`，比 YOLO26m baseline 低 `0.00834`，比 `0.70443` 门禁低 `0.01134`。fresh reload 证实 stem16、18 类、`num_denoising=100`；300,000 行测试 CSV 有效但不得提交。
- RT-DETR-L 条件式 full-data 包 `kaggle_remote/kernel_full_rtdetr` 因 fixed 失败而永久禁止推送。RT-DETR 后续只剩一次 zero-extra fixed；若也失败，才允许一次已预注册的 `num_denoising=200` fixed，均不可与其他 checkpoint 融合。
- YOLO26m 小角度旋转单变量已完成并否决：私有 `zephyrpong/hsi-yolo26m-deg5-ablation` version 1 的普通 HSI16 fixed 2400/600 合同审计通过，唯一变量为 `degrees 0→5`；最佳 epoch 29 `mAP50-95=0.68169`，最终 `0.68092`，比普通 e30 `0.70143` 低 `0.01974`，比门槛 `0.70443` 低 `0.02274`。不跑 full-data、不提交测试 CSV。
- 两个互不混用的后备 fixed 包已在本地生成且全仓 `135 passed`，都未上传：YOLO26m `pseudo_rgb:13,8,5` 为 `kernel_ablation_prgb1385`；RT-DETR-L `extra_channel_init=zero` 为 `kernel_ablation_rtdetr_xczero`，门禁器可用 `--expected-extra-channel-init zero` 精确校验。EIoU 失败时才考虑前者，random-init RT-DETR 失败时才考虑后者；各自仍受 `0.70443` 门禁约束。
- YOLO26l/P2/1280 已有负结果，不重复；唯一未测的容量候选 YOLO26x 已生成最低优先级私有 fixed 包 `kernel_ablation_yolo26x`（slug `zephyrpong/hsi-yolo26x-ablation`），只改模型 m→x，attempts `2:2,1:2,1:0`，未上传。只有 EIoU、`adamw_lr001`、`reg_max=16`、普通 HSI16 `[13,8,5,...]` 与 YOLO11m 均失败且额度仍足时才运行一次。
- RT-DETR 的 COCO 类名别名没有足够语义依据，不做；已有 `num_denoising=200` 单变量私有 fixed 包 `kernel_ablation_rtdetr_nd200`（slug `zephyrpong/hsi-rtdetr-l-nd200-ablation`），未上传。只有 random 与 zero extra-channel RT-DETR 都失败后才运行一次。
- YOLO26m EIoU 已在 AutoDL 重庆 743 完成 30 轮 fixed：最佳 `0.70128`、最终 `0.70106`，低于 `0.70443` 门禁，NO-GO。训练后测试推理因远端缺 `pandas` 中止，但不改变 fixed 失败结论；保留权重/合同/失败归档，不恢复、不跑 full-data。
- YOLO26m `adamw_lr001` 本机 fixed 已完成 30/30：最佳/最终 `0.67233`，低于普通 fixed `0.70143` 和门禁 `0.70443`；optimizer/model contract 与 fresh reload 通过，仍明确 NO-GO，不跑 full-data、不提交。
- YOLO26m `reg_max=16` 只有预注册与 smoke 证据。一次上下文重置后的误启动在第 1 轮完成前按用户停止决定精确终止，没有 `results.csv` 或权重，不得续跑，也不得引用为精度正负结果。
- 普通 HSI16 `[13,8,5,...]` 通道顺序已预注册为后续 fixed 单变量：仍使用相同 16-band 集合、shared P0.5-P99.5、标签和 seed-2026 manifest，只把 pretrained RGB 三个位置从物理 `[5,8,13]` 映射改为 `[13,8,5]`；没有官方波长证据，不能宣称是真 RGB。独立数据身份和 preparation config/report/dataset/manifest 哈希门禁会拒绝 baseline NPY 混用；全仓 `177 passed`，本地包 `kernel_ablation_order1385` 与 code-dataset staging 已就绪但未上传、未运行。
- YOLO11m HSI16 checkpoint-native fixed 候选已完成本地预注册：官方 Ultralytics Assets `v8.4.0` 的 `yolo11m.pt` 为 40,684,120 bytes，SHA-256 `D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95`；真实 live 架构为 `reg_max=16`、true DFL、非 end-to-end Detect。HSI16/18 类可精确复用 `99.6848%` 源参数，真实 forward/loss/backward、4 图 Trainer、model/optimizer contract、fresh reload 与全仓 `184 passed` 均通过。本地私有包 `kernel_ablation_yolo11m_rm16` 和 code-dataset staging 已就绪但未上传、未运行；future gate 必须核验 source kind、权重大小/hash 和 runtime model contract，不能引用公开泄漏 Notebook 的 `0.7394`。
- 面积分层框缩放已否决：75 个候选的三折 OOF `0.70686997`，比 identity 高但比已提交的全局 `×1.01` 低 `0.00063651`；full-val 最优仍等价于全局 `×1.01`，折间选择不一致。不要继续扫框校准。
- distinct-scale support-count 排序已否决：固定一个 checkpoint、七尺度、`fusion_iou=0.74` 与全局框 `×1.01`，三折和 full 都保留现有 confidence-mass `support_gain=0.125`；最佳 count-only `gamma=0.08` 为 `0.70590884`，比 incumbent `0.70750647` 低 `0.00159764`。不要生成 CSV，也不要继续扫置信度排序。
- 最终 `max_det={200,300,400,500}` 零 GPU OOF 审计已否决：三折和 full 都保留 incumbent `300`；500/200/400 分别比 `0.70750647` 低 `0.00058980/0.00105322/0.00114808`。不要生成 CSV或继续扩展自适应 cap。
- 已经证明没用、**不要重复**的方向见 HANDOFF.md "已经证明没用的方向"表格。
- 提交额度官方为每天 3 次；任何历史余量均已过期。最后一次 Kaggle GPU 页面为 `38:03 / 30 hrs`，但当前禁止继续试探、训练或提交。
- AutoDL 重庆 743 实例已关机；最后开机尝试因余额 `-1.01` 元被拒，未新增计费。不要充值、开机、补依赖、恢复推理或启动新候选。
- EIoU 私有 bundle、原始比赛 ZIP、远端权重/合同和失败归档全部保留为学习证据。不要重新上传、公开分享、恢复执行或清理。
- 主办方官方帖子要求所有团队在 **2026-09-23 前**发送团队信息，逾期可能影响成绩认定与奖励；邮件主题为“赛道名称－团队名称－队长姓名”，正文需列出全部队员、单位及指导教师（如有）。帖子没有写明目标检测赛道的指定收件邮箱，只给了疑问联系人 `1522859637@qq.com`，不得擅自把它当材料收件箱或代发个人信息。先让用户确认是否已经发送，并取得准确收件邮箱与具体个人信息。官方帖：<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742296>。

### 当前接手行为（停止状态）

1. 只读核对 `HANDOFF.md`、实验账本和原始日志；不要启动任何训练、推理生成、远端作业或 Kaggle 操作。
2. 不要恢复部分 `reg_max=16` 本机目录；它没有完整 epoch、结果或权重，只是一次已停止的操作记录。
3. 保留 `data/`、`runs/`、`kaggle_remote/outputs/`、AutoDL bundle/归档、提交 CSV 与全部哈希证据；不要删除、搬移或公开分享。
4. 不要重复提交 `boxscale101`，不要选择历史 ensemble，也不要自动做 Phase 2/final-selection。
5. 只有用户明确说要重启本比赛优化时，才先重新核验当前日期、官方 Discussion/Rules、ranking 数据、额度、算力和合规边界；旧确认、旧余量和旧阶段状态都不能复用。
6. 团队信息通知的 Track 1 指定收件邮箱仍未由官方公开确认；不要猜邮箱或代发个人信息。

**很重要的方法论（这是这个项目用分数换来的）**：
- 比赛提交始终只用一个训练 checkpoint；同一 checkpoint 的多尺度/TTA可以融合，不同 checkpoint 的输出不能合并。
- 单模型训练改动先用固定 600 张留出集验证，增益 ≥ +0.003 才值得全量重训。
- 同 checkpoint 推理排序可以用小幅正向留出证据筛选，但要记录 top-300 替换数，不能承诺 Public 必涨。
- 一次只改一个变量，否则分数变化没法归因。
- **每个实验都登记进 `experiments/experiments.csv`，包括失败的**；重要结论同步更新 `HANDOFF.md`。
- 看真实提交分数，不要相信"看起来更好"的中间指标。

### 红线（必须遵守，违反过的都有教训）

1. **Kaggle 登录**：`kaggle` 命令行的登录约 12 小时过期，症状是 `Authentication required` 或 `Permission 'kernels.get' was denied`。
   遇到时**告诉我，我自己在终端跑 `kaggle auth login`**。**绝不要向我索要 API token/密码，也不要把任何凭证写进文件、日志、git 或提示词。**
2. **数据与模型保密**：所有 Kaggle 数据集/Notebook 保持**私有**（`is_private: true`），不要设为公开。AutoDL 实例创建不自动授权上传；用户还必须针对所选实例私有路径 `/root/hsi/input` 与 `/root/autodl-tmp/hsi` 单独确认，绝不能使用 AutoDL“公开数据”入口。除此以外不要把比赛数据、权重、`exports/` 目录传到第三方；`exports/` 比赛期间不得外传。
3. **不要擅自停止或重启正在跑的训练**。指令有歧义时先问我（这个项目有过误杀训练 24/30 轮的教训）。
4. **不要重训已有的模型**：所有成员的权重都已下载在 `kaggle_remote/outputs/*/…/last.pt` 和 `runs/*/weights/last.pt`，重训只会白耗额度。
5. **D 盘空间**：目前剩约 140GB。生成新数据集前先看剩余空间；**删除任何文件前先告诉我**，特别是 `data/raw/`（6GB 原始比赛 zip）、
   `data/processed/hsi16_shared_p005_995`（最佳模型的数据）、`runs/`、`kaggle_remote/outputs/`（权重）——这些都不能删。
6. **git**：核心代码和文档持续提交到本地仓库（最新提交用 `git log -1 --oneline` 查看，**没有 push**）。之后你改了东西可以 `git commit`（提交前 `git status` 确认
   没有把 `exports/`、`data/`、`runs/`、`*.pt`、`submission*.csv`、`kaggle_remote/outputs|code_dataset|raw_dataset` 加进去，它们已在 .gitignore，不要强加）。
   **`git push` 必须先问我**（我还没确认远程 GitHub 仓库是私有的）。
7. **花提交额度和 Kaggle GPU 额度前先经我同意**（提交是不可撤回的；GPU 每周额度有限）。最后一天不要留到最后才试。
   AutoDL 创建/充值/开机属于付费动作，即使已有 500 元总预算，也要在选定具体主机、即时单价和预计最长时长后做一次操作时确认；不得超过 500 元。
8. **比赛规则**：最终提交必须只有一个训练模型；同模型多尺度/TTA允许，多模型 voting/weighted fusion/post-NMS fusion 禁止。使用任何外部数据或非公开预训练权重前，先让我确认。目前只用了官方数据和 Ultralytics 的公开 YOLO26/YOLO11 预训练权重；YOLO11 尚未远程训练或提交。
9. 改动 Kaggle 会从 `hsi-detection-code` 数据集恢复的 `scripts.*.py` / `hsi_detection.*.py` 后，必须同步 `kaggle_remote/code_dataset/` 并创建新的私有 dataset version；只改直接随 Notebook 上传的 `run_hsi_yolo26.py` 时，重新生成并推送 Kernel 即可。
10. 本机 PowerShell 5.1 不支持 `&&`，命令一律用 **Git Bash**；Windows 上 Kaggle 命令带 `/` 的相对路径会出错，**先 `cd` 进目录再用 `-p .`**。

### 汇报方式

- 每完成一步，用中文告诉我：做了什么、结果（数字）、下一步、需要我做什么。
- 出提交文件后给我**可以直接复制的命令**（单独一个代码块，一条命令一个块），由我确认后提交，或经我同意后你来提交。
- 结果出来后，更新 `experiments/experiments.csv` 和 `HANDOFF.md`，让下一个接手的人不用问我。
- 如果卡住或不确定，直接问我，不要猜。

## 提示词（到这里结束）

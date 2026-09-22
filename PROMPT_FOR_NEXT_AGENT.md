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

- Kaggle 账号 `zephyrpong`。排行榜显示最高 `0.65091` 来自八模型融合，**违反官方单模型规则，只能视为历史不合规记录**；Kaggle 会给格式有效的 CSV 评分，但这不代表通过最终规则/代码审核。当前已验证合规的 Public 最佳是单 YOLO26m checkpoint 七尺度支持票加全局框宽高 `×1.01` 的 `0.63546`（ref `56455800`）。2026-09-22 登录态 Submissions 页面显示截止为 **2026-09-27 16:00（北京时间）**，但此前 Rules 阶段日期与之不一致，其他阶段/团队材料截止不能自动顺延。
- 官方允许同一训练模型的 TTA / multi-scale inference，禁止不同训练模型的 voting、weighted fusion 和 post-NMS fusion。任何 `submission_ens*.csv` 都不得再上传或选为最终提交。
- 原合规最佳 `submissions/submission_single_m_hsi16_ms7_f074_sg0125.csv` 是同一 YOLO26m checkpoint 七尺度、`fusion_iou=0.74`、`support_gain=0.125`，Public `0.63072`；ref `56392305`，98,556 detections，SHA-256 `E6F5BC18...2CB48E`。
- 新合规最佳 `boxscale101` 只在上述 CSV 上保持中心不变、把框宽高统一放大 1%。三个 400-fit/200-held-out folds 全部独立选择同一参数，held-out 增益 `+0.00251177/+0.00133919/+0.00198724`，合并 OOF `0.70570818→0.70750647`（`+0.00179829`）。CSV 仍为 98,556 行/1000 图，SHA-256 `C01214E6...995A86`；2026-09-22 14:30 提交 `Success`，Public `0.63546`，ref `56455800`，较 `0.63072` 提升 `+0.00474`。本次后今日余 2 次。
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
- YOLO26m EIoU 已预注册为更直接的严格定位后备：只替换 `BboxLoss` 的 CIoU，不改 assigner、DFL/L1、模型或推理；真实 16 通道 YOLO26m 单 batch forward/loss/backward finite。私有代码数据集新版本已 `ready` 且远端清单含 EIoU 源码；本地 Kernel 包 `kernel_ablation_eiou` 的一次 push 在创建前因周 GPU 配额超限拒绝，远端没有运行版本。2026-09-22 15:34 实时刷新后仍为 `38:03 / 30 hrs`；额度真实恢复后只重试一次，按 `0.70443` 门禁。
- YOLO26m `adamw_lr001` 已预注册为 EIoU 后的本地后备：两条真实 16 通道 smoke 的 runtime contract 证明基线和新配方都使用 AdamW、beta1 `0.9`、warmup-bias `0`，唯一有效差异为初始 LR `0.000455→0.001`；全仓 `141 passed`。本地包 `kernel_ablation_lr001` 未上传、未运行，排在 `[13,8,5]` 与 YOLO26x 前；必须随下载结果审计 `optimizer_contract.json` 并继续使用 `0.70443` 门禁。
- YOLO26m `reg_max=16` 已预注册为后续直接定位架构候选：真实权重审计显示精确复用 `99.4216%` 的源 checkpoint 参数，regmax 专属 mismatch 只有 1,560 个源参数；真实 16 通道 true-DFL forward/loss/backward、4 图 Trainer runtime contract/checkpoint smoke 和全仓 `163 passed`。本地包 `kernel_ablation_rm16` 与 code-dataset staging 已就绪但未上传、未运行；未来结果必须同时审计 `model_contract.json`，固定门禁仍为 `0.70443`。
- 普通 HSI16 `[13,8,5,...]` 通道顺序已预注册为后续 fixed 单变量：仍使用相同 16-band 集合、shared P0.5-P99.5、标签和 seed-2026 manifest，只把 pretrained RGB 三个位置从物理 `[5,8,13]` 映射改为 `[13,8,5]`；没有官方波长证据，不能宣称是真 RGB。独立数据身份和 preparation config/report/dataset/manifest 哈希门禁会拒绝 baseline NPY 混用；全仓 `177 passed`，本地包 `kernel_ablation_order1385` 与 code-dataset staging 已就绪但未上传、未运行。
- YOLO11m HSI16 checkpoint-native fixed 候选已完成本地预注册：官方 Ultralytics Assets `v8.4.0` 的 `yolo11m.pt` 为 40,684,120 bytes，SHA-256 `D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95`；真实 live 架构为 `reg_max=16`、true DFL、非 end-to-end Detect。HSI16/18 类可精确复用 `99.6848%` 源参数，真实 forward/loss/backward、4 图 Trainer、model/optimizer contract、fresh reload 与全仓 `184 passed` 均通过。本地私有包 `kernel_ablation_yolo11m_rm16` 和 code-dataset staging 已就绪但未上传、未运行；future gate 必须核验 source kind、权重大小/hash 和 runtime model contract，不能引用公开泄漏 Notebook 的 `0.7394`。
- 面积分层框缩放已否决：75 个候选的三折 OOF `0.70686997`，比 identity 高但比已提交的全局 `×1.01` 低 `0.00063651`；full-val 最优仍等价于全局 `×1.01`，折间选择不一致。不要继续扫框校准。
- 已经证明没用、**不要重复**的方向见 HANDOFF.md "已经证明没用的方向"表格。
- 提交额度：**每天 3 次**，北京时间 08:00 重置；`boxscale101` 成功后今日余 2 次。Kaggle 只保留历史最佳，提交更差的文件不会降低排名。2026-09-22 15:34 实时 GPU 周额度为 `38:03 / 30 hrs`，不得用重复 push 试探后备作业。
- 用户已明确批准 AutoDL、总预算上限 500 元。2026-09-22 官方公开首页实时显示 RTX 4090 24GB `1.88 元/小时`、会员 95 折；4 小时 fixed 普通价约 `7.52 元`，fixed+full 共 8 小时约 `15.04 元`。控制台目前仍在登录页，未购买/开机/上传；用户必须自己登录。登录后只读核对具体 1×4090、≥8 vCPU、≥64GB RAM、80GB 数据盘和即时按量价，列出精确最坏成本，再停在创建付费实例前取得操作时确认。
- AutoDL EIoU wrapper/bundle 已准备并全仓 `194 passed`：`scripts/build_autodl_eiou_bundle.py` 生成的 41,035,826-byte bundle SHA-256 为 `C3D3EA9FE14AF537F77537499C8510BF39D175900BAE570205834EC4D4544804`；`scripts/run_autodl_eiou.py` 锁死 fixed 合同、做硬件/逐文件/raw-zip 安全预检、自动 gate 和结果归档，并拒绝清单漏列必需文件或出现未声明顶层文件，且永不启动 full 或 Submit。bundle 在 ignored `artifacts/autodl/`，尚未上传。具体流程见 `docs/CLOUD_SERVER_TRAINING.md`。
- Kaggle GPU：免费账号每周约 30 小时；e45 fixed split 使用约 3.1 小时，phase-aware fixed split 使用约 2.5 小时，`cls_pw=0.25` fixed split 使用约 2.29 小时，`scale=0.3` fixed split 使用约 2.06 小时。启动任何新训练前先实时复核余额。
- 主办方官方帖子要求所有团队在 **2026-09-23 前**发送团队信息，逾期可能影响成绩认定与奖励；邮件主题为“赛道名称－团队名称－队长姓名”，正文需列出全部队员、单位及指导教师（如有）。帖子没有写明目标检测赛道的指定收件邮箱，只给了疑问联系人 `1522859637@qq.com`，不得擅自把它当材料收件箱或代发个人信息。先让用户确认是否已经发送，并取得准确收件邮箱与具体个人信息。官方帖：<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742296>。

### 建议你做的事（按优先级，每一步先告诉我再做）

1. `boxscale101` 已 `Success`，Public `0.63546`、ref `56455800`，不得重复提交。最终榜单目前仍是 `0/2` 手工选择；截止前经用户单独确认后至少选择 ref `56455800`，第二个安全候选可选 ref `56392305`，绝不能让自动选择落到历史 ensemble。
2. RT-DETR random fixed 已按 `0.70443` 门禁失败，禁止推送 `kernel_full_rtdetr` 或上传其测试 CSV。GPU 配额恢复后，RT-DETR 分支只允许一次 zero-extra fixed；再失败才一次 `num_denoising=200`，不要加类名别名。
3. 先完成 AutoDL 登录态只读询价并在付费创建前确认；实例创建后先上传私有 EIoU bundle/raw 到私有数据盘并跑 `--preflight-only`，只有 `PREFLIGHT_OK` 才启动一次 EIoU fixed，精确审计 `--expected-box-iou-loss eiou`。若不租 AutoDL，则等 Kaggle 配额恢复且确认远端不存在 EIoU 版本后只推送一次。失败后排 `adamw_lr001`（核验 runtime optimizer contract），再排已预注册 `reg_max=16`（核验 runtime model contract），之后是已打包的普通 HSI16 `[13,8,5,...]`（核验四份 preparation artifacts 与哈希），再后才是 checkpoint-native YOLO11m（核验 exact source SHA 与 model contract）。
4. YOLO11m HSI16 尚无合法 fixed 分数，但官方 checkpoint 的下载、SHA-256、runtime 架构和本地私有包均已审计完成。它只有一次 fixed 测量资格；不得引用公开泄漏的 `0.7394`，也不得把 smoke 或 `99.6848%` 迁移率写成精度收益。
5. `dfl=2.0/2.5`、scale、`cls_pw`、phase-aware、e45、horizontal-flip、YOLO26m zero-init、object-crop、SpectralStem、tile TTA、尺度来源/权重、RT-DETR random 和继续扫框校准均已否决，不重复投入。
6. 更直接候选都失败且额度仍足时，才运行一次已预注册 YOLO26x fixed；门禁器显式使用 `--model yolo26x.pt`。不要重跑 YOLO26l、P2 或 1280。
7. 对任何新候选重新核对哈希、格式和额度；每次最终 Competition Submit 前都让用户确认。
8. 检查 9 月 23 日团队信息邮件是否已由用户发送；未发送时先确认目标检测赛道指定邮箱和全部个人信息，不能仅凭公告猜收件人。

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
2. **数据与模型保密**：所有 Kaggle 数据集/Notebook 保持**私有**（`is_private: true`），不要设为公开。用户当前只授权把本次 EIoU 所需 bundle 和原始比赛包传到所选 AutoDL 实例的**私有数据盘**；上传前仍要复核具体目标，绝不能使用 AutoDL“公开数据”入口。除此以外不要把比赛数据、权重、`exports/` 目录传到第三方；`exports/` 比赛期间不得外传。
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

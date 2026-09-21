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

- Kaggle 账号 `zephyrpong`。排行榜显示最高 `0.65091` 来自八模型融合，**违反官方单模型规则，只能视为历史不合规记录**；Kaggle 会给格式有效的 CSV 评分，但这不代表通过最终规则/代码审核。当前已验证合规的 Public 最佳是单 YOLO26m checkpoint 七尺度支持票 `0.63072`（ref `56392305`）。截止 **2026-09-24 16:00 UTC（北京时间 9 月 25 日 00:00）**。
- 官方允许同一训练模型的 TTA / multi-scale inference，禁止不同训练模型的 voting、weighted fusion 和 post-NMS fusion。任何 `submission_ens*.csv` 都不得再上传或选为最终提交。
- 当前合规最佳 `submissions/submission_single_m_hsi16_ms7_f074_sg0125.csv` 已成功提交：同一 YOLO26m checkpoint 七尺度，`fusion_iou=0.74`、`support_gain=0.125`，Public `0.63072`；ref `56392305`，98,556 detections，SHA-256 `E6F5BC18...2CB48E`。它只比前一版 `f0.70` 的 `0.63066` 高 `+0.00006`，说明当前 checkpoint 后处理已接近饱和。
- zero-init fixed split 最佳 `0.69690`，object-crop fixed split `0.69772`；两者均低于旧同规格 `0.70143` 和门槛 `0.70443`，已否决，不跑全量、不提交。
- SpectralStem 本地链路门禁已通过，私有 fixed 2400/600 消融也已正常完成，但标准 full-val 最佳/最终仅 `0.69930`，低于旧同规格 `0.70143` 和门槛 `0.70443`。该路线已否决：不跑全量、不上传其测试 CSV。
- phase-aware bilinear v1 私有 Kernel `zephyrpong/hsi-yolo26m-phase-ablation` version 1 已完成：固定 2400/600，唯一变量是保留 4×4 物理 phase 并按宽高比重建到长边 1024；最佳/最终 epoch 30 为 `0.69935`，低于普通同规格 `0.70143` 和门槛 `0.70443`，已否决。38,304 条测试检测仅通过本地 checker，无比赛 Submit。
- 显式 horizontal-flip-only 已完成同 checkpoint 固定划分验证：cache control 通过，但最佳七尺度+flip 仅 `0.70587796`，相对 supported-full 只增 `+0.00016978 < +0.001`，已否决，不生成或提交候选。
- 普通 HSI16 YOLO26m 延长到 45 轮的 fixed 2400/600 消融已完成：最佳/最终 epoch 45 仅 `0.69899`，比 e30 的 `0.70143` 低 `0.00244`，比 `0.70443` 门槛低 `0.00544`。不启动 full-data e45，不上传其测试 CSV。
- `cls_pw=0.25` 私有 fixed 2400/600 消融已完成并否决：唯一训练变量为分类频次权重；最佳 epoch 27 为 `0.69863`，最终 epoch 30 为 `0.69862`，低于普通同规格 e30 `0.70143` 和门槛 `0.70443`。36,784 条测试检测仅通过本地 checker，无比赛 Submit/Public；不要启动其 full-data 训练或上传该 CSV。
- random-affine `scale=0.3` 私有 fixed 2400/600 消融已完成并否决：已记录的训练与数据参数中唯一差异为 `scale 0.5→0.3`；最佳/最终 epoch 30 为 `0.69992`，较普通同规格 e30 `0.70143` 低 `0.00151`，低于门槛 `0.70443`。31,848 条测试检测仅通过本地 checker，无比赛 Submit/Public；不要启动其 full-data 训练或上传该 CSV。
- `dfl=2.0/2.5` 两个独立单模型 fixed 2400/600 作业均已完成并否决：最佳/最终 `0.70165/0.70186`，虽略高于普通 e30 `0.70143`，但仍低于 `0.70443` 门禁；不要启动对应 full-data 或上传其测试 CSV，二者绝不能融合。
- RT-DETR-L HSI16 本地全链 vertical slice、远程架构分支和当前全仓 `122` 项测试均已通过；私有代码数据集新版本已 `ready`。独立 `zephyrpong/hsi-rtdetr-l-smoke` version 1 已于 04:29 `COMPLETE`：单 T4 batch 2、官方预训练和 16 通道迁移、1 epoch、重载、NPY 推理与 checker 全通过。fixed 2400/600 的 `zephyrpong/hsi-rtdetr-l-ablation` version 1 已于 04:43 推送，04:44 状态为 `RUNNING`；仍不是正式候选或 Public 成绩。
- YOLO26m 小角度旋转单变量已进入 fixed 门禁：训练脚本、远程 runner、生成器和门禁器均已支持并验证 `degrees`，全仓 `122` 项测试通过；私有 `zephyrpong/hsi-yolo26m-deg5-ablation` version 1 于 05:29 推送并实时核验为 `RUNNING`。普通 HSI16、fixed 2400/600、30 epochs、1024、batch 8 fallback、seed 2026、`scale=0.5`、`dfl=1.5` 不变，唯一变量为 `degrees 0→5`；门槛仍为 `0.70443`，无 Competition Submit。
- 两个互不混用的后备 fixed 包已在本地生成且全仓 `124 passed`，都未上传：YOLO26m `pseudo_rgb:13,8,5` 为 `kernel_ablation_prgb1385`；RT-DETR-L `extra_channel_init=zero` 为 `kernel_ablation_rtdetr_xczero`，门禁器可用 `--expected-extra-channel-init zero` 精确校验。degrees=5 失败时才考虑前者，random-init RT-DETR 失败时才考虑后者；各自仍受 `0.70443` 门禁约束。
- 已经证明没用、**不要重复**的方向见 HANDOFF.md "已经证明没用的方向"表格。
- 提交额度：**每天 3 次**，北京时间 08:00 重置。Kaggle 只保留历史最佳，提交更差的文件不会降低排名。
- Kaggle GPU：免费账号每周约 30 小时；e45 fixed split 使用约 3.1 小时，phase-aware fixed split 使用约 2.5 小时，`cls_pw=0.25` fixed split 使用约 2.29 小时，`scale=0.3` fixed split 使用约 2.06 小时。启动任何新训练前先实时复核余额。

### 建议你做的事（按优先级，每一步先告诉我再做）

1. 只监控当前 `hsi-rtdetr-l-ablation` version 1 和 `hsi-yolo26m-deg5-ablation` version 1，不要重启或重复推送；完成后分别下载到独立目录，用架构感知门禁器核验精确配置、30 个 epoch、训练状态和哈希。
2. 两个 fixed 均按 `0.70443` 门禁；RT-DETR 审计 `num_denoising=100`，YOLO 审计 `dfl=1.5/degrees=5`。低于门禁立即否决，达到门禁才允许生成对应单 checkpoint full-data 作业，且两个 checkpoint 绝不能融合。
3. 若 degrees=5 失败，可在空出的远端槽位推送一次已预注册的 `[13,8,5]` fixed；若 random-init RT-DETR 失败，可推送一次已预注册的 RT-DETR zero-init fixed。不要把后者与已否决的 YOLO26m zero-init 混为一谈，也不要并行重复同一 slug。
4. `dfl=2.0/2.5`、scale、`cls_pw`、phase-aware、e45、horizontal-flip、YOLO26m zero-init、object-crop、SpectralStem、tile TTA、尺度来源/权重和继续细扫后处理均已否决，不重复投入。
5. 对任何新候选重新核对哈希、格式和额度；最终 Competition Submit 前让我确认。

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
2. **数据与模型保密**：所有 Kaggle 数据集/Notebook 保持**私有**（`is_private: true`），不要设为公开，不要把比赛数据、权重、`exports/` 目录上传到任何第三方。
   `exports/` 是私有恢复备份，比赛期间不得外传。
3. **不要擅自停止或重启正在跑的训练**。指令有歧义时先问我（这个项目有过误杀训练 24/30 轮的教训）。
4. **不要重训已有的模型**：所有成员的权重都已下载在 `kaggle_remote/outputs/*/…/last.pt` 和 `runs/*/weights/last.pt`，重训只会白耗额度。
5. **D 盘空间**：目前剩约 140GB。生成新数据集前先看剩余空间；**删除任何文件前先告诉我**，特别是 `data/raw/`（6GB 原始比赛 zip）、
   `data/processed/hsi16_shared_p005_995`（最佳模型的数据）、`runs/`、`kaggle_remote/outputs/`（权重）——这些都不能删。
6. **git**：核心代码和文档持续提交到本地仓库（最新提交用 `git log -1 --oneline` 查看，**没有 push**）。之后你改了东西可以 `git commit`（提交前 `git status` 确认
   没有把 `exports/`、`data/`、`runs/`、`*.pt`、`submission*.csv`、`kaggle_remote/outputs|code_dataset|raw_dataset` 加进去，它们已在 .gitignore，不要强加）。
   **`git push` 必须先问我**（我还没确认远程 GitHub 仓库是私有的）。
7. **花提交额度和 Kaggle GPU 额度前先经我同意**（提交是不可撤回的；GPU 每周额度有限）。最后一天不要留到最后才试。
8. **比赛规则**：最终提交必须只有一个训练模型；同模型多尺度/TTA允许，多模型 voting/weighted fusion/post-NMS fusion 禁止。使用任何外部数据或非公开预训练权重前，先让我确认。目前只用了官方数据和 Ultralytics 的公开 yolo26 预训练权重。
9. 改动 Kaggle 会从 `hsi-detection-code` 数据集恢复的 `scripts.*.py` / `hsi_detection.*.py` 后，必须同步 `kaggle_remote/code_dataset/` 并创建新的私有 dataset version；只改直接随 Notebook 上传的 `run_hsi_yolo26.py` 时，重新生成并推送 Kernel 即可。
10. 本机 PowerShell 5.1 不支持 `&&`，命令一律用 **Git Bash**；Windows 上 Kaggle 命令带 `/` 的相对路径会出错，**先 `cd` 进目录再用 `-p .`**。

### 汇报方式

- 每完成一步，用中文告诉我：做了什么、结果（数字）、下一步、需要我做什么。
- 出提交文件后给我**可以直接复制的命令**（单独一个代码块，一条命令一个块），由我确认后提交，或经我同意后你来提交。
- 结果出来后，更新 `experiments/experiments.csv` 和 `HANDOFF.md`，让下一个接手的人不用问我。
- 如果卡住或不确定，直接问我，不要猜。

## 提示词（到这里结束）

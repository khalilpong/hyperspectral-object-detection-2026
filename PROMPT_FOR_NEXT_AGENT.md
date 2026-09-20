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

读完后先用 5 句话向我复述：合规 Public 基线、排行榜显示但不合规的历史分数、正在运行的实验、你打算先做什么、需要我做什么。**等我确认再开始花新的 Kaggle GPU 额度或提交。**

### 当前状态（以 HANDOFF.md 为准，这里是摘要）

- Kaggle 账号 `zephyrpong`。排行榜显示最高 `0.65091` 来自八模型融合，**违反官方单模型规则，只能视为历史不合规记录**；Kaggle 会给格式有效的 CSV 评分，但这不代表通过最终规则/代码审核。当前已验证合规的 Public 最佳是单 YOLO26m checkpoint 七尺度支持票 `0.63072`（ref `56392305`）。截止 **2026-09-24 16:00 UTC（北京时间 9 月 25 日 00:00）**。
- 官方允许同一训练模型的 TTA / multi-scale inference，禁止不同训练模型的 voting、weighted fusion 和 post-NMS fusion。任何 `submission_ens*.csv` 都不得再上传或选为最终提交。
- 当前合规最佳 `submissions/submission_single_m_hsi16_ms7_f074_sg0125.csv` 已成功提交：同一 YOLO26m checkpoint 七尺度，`fusion_iou=0.74`、`support_gain=0.125`，Public `0.63072`；ref `56392305`，98,556 detections，SHA-256 `E6F5BC18...2CB48E`。它只比前一版 `f0.70` 的 `0.63066` 高 `+0.00006`，说明当前 checkpoint 后处理已接近饱和。
- zero-init fixed split 最佳 `0.69690`，object-crop fixed split `0.69772`；两者均低于旧同规格 `0.70143` 和门槛 `0.70443`，已否决，不跑全量、不提交。
- 已经证明没用、**不要重复**的方向见 HANDOFF.md "已经证明没用的方向"表格。
- 提交额度：**每天 3 次**，北京时间 08:00 重置。Kaggle 只保留历史最佳，提交更差的文件不会降低排名。
- Kaggle GPU：免费账号每周约 30 小时，本周已用约 20 小时。训一个 yolo26m 全量模型约 2.5 小时。

### 建议你做的事（按优先级，每一步先告诉我再做）

1. 下一主线是 `SpectralStem`：一个 checkpoint 内用可学习光谱投影处理 16 通道，再进入 YOLO26m。先做本地构造、预训练迁移、反向传播、checkpoint 新进程恢复和 16 通道预测 smoke。
2. 当前 HSI16 NPY 的物理波段顺序是 `5,8,13,0,1,...`，所以 5/8/13 identity 初始化应接数组通道索引 `(0,1,2)`；把 `(5,8,13)` 当数组索引会静默接错。
3. smoke 全通过后，先报告准确配置、预计 GPU 时长和门禁，再经我确认运行 2400/600 fixed split；只有 `>=0.70443` 才训练全量。
4. 同规格新 seed 只作为方差对照，排在 SpectralStem 后。继续扫 fusion/NMS、zero-init、object-crop、旧 checkpoint tile TTA 均不再投入。
5. 对任何新候选重新核对哈希、格式和额度；最终 Submit 前让我确认。

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
6. **git**：核心代码和文档已在 09-19 提交到本地仓库（**没有 push**）。之后你改了东西可以 `git commit`（提交前 `git status` 确认
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

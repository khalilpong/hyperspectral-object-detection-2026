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

读完后先用 5 句话向我复述：当前分数、排名、已经试过什么、你打算先做什么、需要我做什么。**等我确认再开始花 Kaggle GPU 额度或提交。**

### 当前状态（以 HANDOFF.md 为准，这里是摘要）

- Kaggle 账号 `zephyrpong`。**当前最佳 0.64831，排名 21/243**，榜首 0.67943。截止 **2026-09-24 16:00 UTC（北京时间 9 月 25 日 00:00）**。
- 最佳做法：**8 个模型 × 7 个尺度的框投票融合**（4 种输入数据：16 波段，以及伪RGB 的波段 5/8/13、3/6/8、0/7/15）。
  一条命令复现：`bash scripts/build_ensemble_submission.sh submissions/xxx.csv`（有缓存，约 10 分钟，不用 GPU）。
- 单个模型已经榨干（0.6265）。融合在第 4 个成员之后进入饱和区：再加成员每次只有 ±0.001，和噪声一样。
- 已经证明没用、**不要重复**的方向见 HANDOFF.md "已经证明没用的方向"表格。
- 提交额度：**每天 3 次**，北京时间 08:00 重置。Kaggle 只保留历史最佳，提交更差的文件不会降低排名。
- Kaggle GPU：免费账号每周约 30 小时，本周已用约 20 小时。训一个 yolo26m 全量模型约 2.5 小时。

### 建议你做的事（按优先级，每一步先告诉我再做）

1. **调融合成员的权重**（零 GPU）：用 `W_B`~`W_H` 环境变量一次只改一个系数出文件，我来确认后再提交。
2. **换融合算法**（零 GPU）：现在是按置信度加权的框投票（`scripts/predict_submission.py` 的 `_box_vote`）。可试 WBF、多数票加成等。
3. **再训一个强的 16 波段成员**（约 2.6 小时 Kaggle GPU）：换归一化方式或随机种子。
4. 你有更好的想法也可以提，但要说明依据。

**很重要的方法论（这是这个项目用分数换来的）**：
- 融合类改动，**600 张留出验证集完全不可信**（判断错了四次，有两次方向都反了）。融合的组合只能靠**提交验证**。
- 单模型类改动，仍然先用留出集验证，增益 ≥ +0.003 才值得全量重训。
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
8. **比赛规则**：使用任何外部数据或非预训练权重前，先让我确认比赛规则允许。目前只用了官方数据和 Ultralytics 的 yolo26 预训练权重。
9. 主脚本 `kaggle_remote/run_hsi_yolo26.py` 改动后，必须同步到 `kaggle_remote/code_dataset/` 并 `kaggle datasets version`，
   否则 Kaggle 上跑的还是旧版（踩过的坑 7）。
10. 本机 PowerShell 5.1 不支持 `&&`，命令一律用 **Git Bash**；Windows 上 Kaggle 命令带 `/` 的相对路径会出错，**先 `cd` 进目录再用 `-p .`**。

### 汇报方式

- 每完成一步，用中文告诉我：做了什么、结果（数字）、下一步、需要我做什么。
- 出提交文件后给我**可以直接复制的命令**（单独一个代码块，一条命令一个块），由我确认后提交，或经我同意后你来提交。
- 结果出来后，更新 `experiments/experiments.csv` 和 `HANDOFF.md`，让下一个接手的人不用问我。
- 如果卡住或不确定，直接问我，不要猜。

## 提示词（到这里结束）

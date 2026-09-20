# 交接文档：从这里开始

> 接手本项目的人**先读完这一页**，再按需跳转到详细文档。
> 最后更新：2026-09-20 20:15（北京时间）
> 🤖 **要把项目交给另一个 AI 接手？** 直接把 [PROMPT_FOR_NEXT_AGENT.md](PROMPT_FOR_NEXT_AGENT.md) 里的提示词发给它。

## 一句话现状

**排行榜显示的最高分是 0.65091，但它来自八模型融合，违反当前比赛的单模型规则，不能作为安全的最终成绩。** Kaggle 上传时只按 CSV 评分，不会自动检查训练/推理用了几个模型；资格风险会留到规则与获奖代码审核。当前已验证合规的 Public 最佳是 **0.63072**：单个 YOLO26m HSI16 checkpoint 的七尺度支持票推理（Kaggle ref `56392305`）。

- 官方 Rules 要求只能使用一个 detection model；主办方进一步明确：同一训练模型的 TTA/多尺度允许，不同模型的 voting、weighted fusion、post-NMS fusion 禁止。官方澄清：<https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863>。
- `0.63917` 到 `0.65091` 的历史多模型成绩仅保留作研究记录；**禁止再提交任何 ensemble 文件，也不要把它们选为最终提交**。
- 第二个合规单模型候选已于 2026-09-20 19:27（北京时间）提交成功：同一个 YOLO26m checkpoint、七尺度、`fusion_iou=0.74`、`support_gain=0.125`。Public `0.63072`，较上一版 `0.63066` 仅提升 `+0.00006`；CSV 98,556 条、1000/1000 图、无效框 0，SHA-256 `E6F5BC18...2CB48E`，ref `56392305`。这证明细扫方向为正，但幅度已不足以承担 0.66 冲刺。
- zero-init 私有消融已完成并否决：最佳 `0.69690`，低于旧同规格 `0.70143` 和门槛 `0.70443`；不跑全量、不提交。
- object-crop 私有消融已完成并否决：标准 full-val `0.69772`，低于旧同规格 `0.70143` 和门槛 `0.70443`；不跑全量、不提交，也不扩展 tile。
- P1–P99 已否决：固定划分 `0.69668 < 0.70143`；不要重复训练或提交其第九成员候选。
- SpectralStem 本地门禁已通过：标准模块 checkpoint、optimizer 实际更新、受限安全加载、新进程恢复、16 通道真实 NPY 预测和完整 pytest 均通过；私有 code dataset / Kernel 仅在本地 staged，尚未上传或运行。

## 👉 接手后第一件事

1. SpectralStem 本地门禁已经完成；先取得用户授权，再为私有 `hsi-detection-code` 创建新版本并 push 私有 `zephyrpong/hsi-yolo26m-stem-ablation` fixed-split Kernel。
2. 运行后下载并核验 `status.json/results.csv/best.pt/last.pt`；fixed-split 门槛仍是旧 YOLO26m 基准 `0.70143 + 0.003 = 0.70443`。
3. 新结构达标才申请全量 3000 张训练；未达标就停止。全量训练和任何 Submit 均需再次确认。

### 成绩与合规状态

| 配方 | Public | 合规状态 |
|---|---:|---|
| 单模型 s（16 波段，同 checkpoint 七尺度） | 0.62651 | 合规 |
| 单模型 m（16 波段，同 checkpoint 七尺度） | 0.62953 | 合规基线，ref `56303572` |
| 单模型 m + 七尺度支持票，`f0.70/gain0.125` | 0.63066 | 合规；ref `56379896` |
| **单模型 m + 七尺度支持票，`f0.74/gain0.125`** | **0.63072** | **当前已验证的合规最佳；ref `56392305`** |
| 两到八模型融合 | 0.63917 ~ 0.65091 | **不合规，历史记录，不得再提交/最终选择** |

### 下一步可以试的（按优先级）

1. 经用户确认后启动已 staged 的 `SpectralStem` fixed split；当前 HSI16 NPY 的物理波段 5/8/13 已排列为数组通道索引 0/1/2，identity 初始化接 `(0,1,2)`。
2. 预计约 2 小时 Kaggle T4；只在标准 full-val `mAP50-95 >= 0.70443` 时申请全量训练。
3. 新 seed 的同规格 YOLO26m 只作为方差对照，没有正向先验；排在 SpectralStem 后。
4. zero-init、object-crop、当前 checkpoint tile TTA 和继续细扫后处理均已否决，不重复消耗截止前时间。
5. 不再扫多模型权重、WBF、成员组合或类别融合。历史 ensemble cache 仅供离线研究，不得生成比赛提交。

> 提交额度每天 3 次，北京时间 08:00 重置。19:27 已用完当日最后一次，下一次提交须等重置后再实时复核。最终提交必须手工选中合规的 ref `56392305`（可选 ref `56379896` 作为第二项），不能让 Kaggle 自动按最高 Public 选择历史 ensemble。

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

**核心代码、文档、实验记录已在 2026-09-19 提交到本地仓库**（提交 `a8789d8` 代码、`4442f0d` 文档），**没有 push**。
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
| 私有数据集 `zephyrpong/hsi-detection-code` | 代码（平铺）+ 划分清单 + yolo26m/s 预训练权重 + 主脚本 |
| 私有数据集 `zephyrpong/hsi-competition-raw` | 原始比赛 zip（比赛数据无法直接挂载进 Notebook，见 Kaggle 文档第 5 条坑） |
| Notebook `hsi-yolo26m-{smoke,ablation,full}`、`hsi-yolo26l-ablation` | 16 波段 m / l 的各次训练，产物在 `kaggle_remote/outputs/` 对应目录 |
| Notebook `hsi-yolo26m-p010-990-{full,ablation}` | P1–P99 归一化的新成员；full/ablation 均完成并下载；ablation `0.69668` 低于旧基准 `0.70143`，该方向已否决 |
| Notebook `hsi-yolo26m-prgb-{ablation,full}`、`prgb368-full`、`prgb0715-full` | 伪RGB（5/8/13、3/6/8、0/7/15）的训练，产物同上 |
| `hsi-yolo26m-stem-ablation` | **仅本地 staging，尚未 push**；SpectralStem fixed 2400/600，需用户授权后启动 |
| Notebook `hsi-yolo26-smoke` | ❌ 第一次失败的旧版本，可忽略或删除 |
| ⚠️ 已训好的模型不要重训 | 权重都已下载在 `kaggle_remote/outputs/*/…/last.pt`，重训只会白耗额度 |

## 比赛与成绩

| 项目 | 内容 |
|---|---|
| 比赛 | [Hyperspectral Object Detection Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026) |
| 任务 | 高光谱图像目标检测，18 类，评分指标 mAP@[0.5:0.95] |
| 截止 | **2026-09-24 16:00 UTC（北京时间 9 月 25 日 00:00）** |
| 排行榜显示最高 | **0.65091**（八模型融合，提交编号 56378610；当前规则下不合规，不得作为最终提交） |
| 已验证合规最佳 | **0.63072**（单 YOLO26m checkpoint + 七尺度支持票，提交编号 56392305；较上一合规 ref 56379896 提升 0.00006，较原始 ref 56303572 提升 0.00119） |
| 历史显示排名 | 第 19（09-20 以不合规 ensemble 计，仅作页面快照，不代表安全名次） |
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
| 伪标签（给测试图自动打标签再训练） | Kaggle **−0.0094** | 丢了低分框长尾的召回；伪框继承了教师模型的定位误差 |
| yolo26m（本机） | Kaggle 0.60553 | **被 8GB 显存逼到 batch=2**，不是模型不行 |
| P2 小目标检测头 / box loss 权重 10 | 更差 | — |

## 训练算力：Kaggle 免费 GPU（已验证够用）

所有训练都在 Kaggle 上跑，脚本 `kaggle_remote/run_hsi_yolo26.py` + `make_kernel.py`，
详见 [docs/KAGGLE_REMOTE_TRAINING.md](docs/KAGGLE_REMOTE_TRAINING.md)。

| 事实 | 数据 |
|---|---|
| 硬件 | 2×T4 15GB、4 核、31GB 内存，单次最长 12 小时，每周约 30 小时 |
| yolo26m + 16 波段 | batch 8，每轮 4.5~4.8 分钟，30 轮全量约 2.6 小时 |
| yolo26m + 伪RGB | batch 8，每轮 4.2 分钟，30 轮全量约 2.2 小时（数据准备只要 2.5 分钟，比 16 波段快得多） |
| 两张卡 DDP | `--attempts 8:2:0+1` 可用，已实测跑通（yolo26l 每轮 3.1 分钟） |
| 额度消耗 | 09-17~09-18 两天共用掉约 20 小时（每周约 30 小时，重置日以 Kaggle 页面为准） |

**不需要租云服务器**（[docs/CLOUD_SERVER_TRAINING.md](docs/CLOUD_SERVER_TRAINING.md) 保留备用：Kaggle 额度用完时启用）。

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
| `scripts/build_ensemble_submission.sh` | 一键生成八成员融合提交，可用环境变量调权重 |
| [docs/KAGGLE_REMOTE_TRAINING.md](docs/KAGGLE_REMOTE_TRAINING.md) | Kaggle 免费 GPU 远程训练：原理、命令、踩坑、运行记录 |
| [docs/CLOUD_SERVER_TRAINING.md](docs/CLOUD_SERVER_TRAINING.md) | 云服务器训练：配置选型、从零到提交的完整步骤 |
| [docs/AUTONOMOUS_RUN_2026-09-17.md](docs/AUTONOMOUS_RUN_2026-09-17.md) | 9/17 通宵托管的完整决策日志（pan、融合、伪标签为何失败） |
| [experiments/experiments.csv](experiments/experiments.csv) | **所有实验的结果总表**，每行有配置、分数和结论 |
| [docs/learning_path.md](docs/learning_path.md) | 训练/验证/测试等基础概念（新手先看） |
| [docs/REBOOT_RECOVERY_HSI_E60.md](docs/REBOOT_RECOVERY_HSI_E60.md) | 长训练中断后如何安全续跑 |
| `kaggle_remote/run_hsi_yolo26.py` | 远程训练流水线主脚本（Kaggle 和云服务器通用） |
| `kaggle_remote/make_kernel.py` | 从主脚本生成 Kaggle Notebook |

## 工作纪律（这个项目用教训换来的）

1. **一次只改一个变量**，否则分数变化无法归因。
2. **先留出验证，再全量提交**。跳过验证的伪标签实验让 Kaggle 分数掉了 0.0094。
3. **单模型改动**：留出集增益 < 0.003 不值得全量重训，< 0.001 不值得花提交额度。**融合类改动例外**：留出集不可信，直接提交验证。
4. **看分数，不看"看起来更好"的中间指标**。伪标签模型的置信度分布看着更干净，实际 mAP 更差。
5. **每个实验都登记进 `experiments/experiments.csv`**，包括失败的——失败记录能帮后来者少走弯路。

# Hyperspectral Object Detection Challenge 2026

基于 **16 波段输入、单个 YOLO26m checkpoint、多尺度与水平翻转推理**的高光谱目标检测项目。

**项目已完成提交并归档。** 比赛提交截止时间为 **2026-09-27 16:00（北京时间）**。赛后现场核验确认：最终 A/B 均为 `Complete`，选择已锁定为 **2/2**，页面仍未显示 ranking private score。以下公开分数不是最终私榜成绩。

- [比赛主页](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026)
- [提交记录](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/submissions)
- [完整资料归档与恢复说明（公开 Release）](https://github.com/khalilpong/hyperspectral-object-detection-2026/releases/tag/archive-20260927)

## 最终结果

| 候选 | Submission ref | Public test-reference | 融合 IoU | CSV 行数 | 图像数 |
|---|---:|---:|---:|---:|---:|
| A | **56569707** | **0.62865** | 0.65 | 196,609 | 2,000 |
| B | **56569768** | **0.62767** | 0.82 | 241,891 | 2,000 |

两份文件都覆盖 `test1000 + ranking1000`，最终只选择了这两个合规候选。历史多 checkpoint ensemble 实验不属于最终方案，其分数不能作为本项目的合规最终成绩。

最终提交文件与 SHA-256：

```text
A: submission_phase2_single_m_hsi16_ms7_hflip1024_f065_sg0125_boxscale101.csv
   94A3DF3CCAE9BDE34419CFAFA4BE908ED9E13468EAF7CA017E4F927C6F9E529C

B: submission_phase2_single_m_hsi16_ms7_hflip1024_f082_sg0125_boxscale101.csv
   EF2F8BB1521E5980310F420EFE1561B6F069570E0A4672A9A290D7527BAF53F7
```

## 数据与最终方法

数据包含 3,000 张有标注训练图、1,000 张 test 图和 Phase 2 新增的 1,000 张 ranking 图，共 18 个类别。原始图像为 16 位单通道 PNG，以 4×4 马赛克排列 16 个光谱波段；标注坐标对应解包后的空间尺寸。评估指标为 mAP@[0.5:0.95]。

| 环节 | 最终配置 |
|---|---|
| 模型 | 单个 YOLO26m，16 通道输入 |
| 输入编码 | 每图共享 P0.5/P99.5 归一化，16 通道 uint8 NPY |
| 波段顺序 | `5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15` |
| 七尺度 | `832,896,960,1024,1088,1152,1216` |
| 额外 TTA | 一次 `imgsz=1024` 水平翻转，16 个通道同时翻转 |
| 原始预测 | batch 1、FP16、conf 0.0001、NMS IoU 0.7、max_det 300 |
| 融合 | A/B 分别使用 IoU 0.65/0.82，support_gain 均为 0.125 |
| 全局框校准 | 框宽、高 ×1.01，中心保持不变 |

唯一生产 checkpoint：

```text
kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt
SHA-256: 8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3
```

test 与 ranking 使用相同的推理和后处理参数。ranking 仅作推理，没有用于训练、微调、伪标签适配或统计更新。A/B 复用相同的模型和原始预测缓存，只改变融合 IoU。

## 仓库结构

| 路径 | 内容 |
|---|---|
| [`src/hsi_detection/`](src/hsi_detection/) | 光谱解包、标注、框校准、提交格式等基础模块 |
| [`scripts/`](scripts/) | 数据处理、训练、推理、审计与验证脚本 |
| [`tests/`](tests/) | 自动化测试 |
| [`kaggle_remote/`](kaggle_remote/) | Kaggle 远程运行的代码与配置 |
| [`experiments/`](experiments/) | 实验账本、候选比较及最终提交证据 |
| [`docs/`](docs/) | 数据、模型来源与历史运行说明 |

最终生产路径的主要入口：

- [`produce_phase2_flip.py`](scripts/produce_phase2_flip.py)：准备缓存并构建 Phase 2 A/B 候选。
- [`calibrate_submission_boxes.py`](scripts/calibrate_submission_boxes.py)：应用已审计的全局框校准。
- [`merge_phase2_submission.py`](scripts/merge_phase2_submission.py)：合并 test/ranking 并验证完整覆盖。
- [`check_submission.py`](scripts/check_submission.py)、[`audit_phase2_flip.py`](scripts/audit_phase2_flip.py)：格式、几何与候选差异检查。

精确运行命令和参数见[最终生产与提交记录](experiments/phase2-flip-finalization-20260926.md)，原始 manifest 和缓存位于归档中。历史训练说明保留作技术参考，不代表仍有待执行任务。

## 获取代码与恢复完整资料

本仓库保留代码、测试与文档。数据、模型、提交 CSV、预测缓存和原始实验日志存放在[公开归档 Release](https://github.com/khalilpong/hyperspectral-object-detection-2026/releases/tag/archive-20260927)，代码与归档资料均可公开访问。

只查看代码：

```powershell
git clone https://github.com/khalilpong/hyperspectral-object-detection-2026.git
```

完整归档包含 **73,419 个原始文件、约 62.37 GiB 资料**，经内容去重与压缩后为 **35 个分卷、约 34.98 GiB**。另附文件清单、SHA-256 清单、恢复脚本、环境版本和 Git history bundle。所有 **42 项远端文件**均已核对 SHA-256 与大小，整包测试、关键模型/CSV 解压和远端回下载抽检通过。

在新的工作目录中恢复（需要 GitHub CLI、7-Zip 和 Python 3.11）：

```powershell
gh release download archive-20260927 --repo khalilpong/hyperspectral-object-detection-2026 --dir .\hsi-archive
7z t .\hsi-archive\project-materials.7z.001
7z x .\hsi-archive\project-materials.7z.001 -o.\hsi-extracted
py -3.11 .\hsi-extracted\restore.py --verify-only
py -3.11 .\hsi-extracted\restore.py --output .\hsi-project
```

下载时必须保留全部 35 卷，并按 `SHA256SUMS.txt` 校验。`hsi-project` 必须是不存在的新目录；恢复脚本会按 manifest 重建原始路径，不覆盖已有目录。完整恢复约需 63 GiB 输出空间，此外还需容纳下载分卷和解包后的内容库。

归档没有保留可重建的 `.venv`、字节码/测试缓存、本机 Ultralytics 设置和代理工作便笺。运行环境使用 Python 3.11，可按 [`pyproject.toml`](pyproject.toml) 与归档中的 `environment-freeze.txt` 重建。完整 Git 历史另有 `project-history.bundle`。归档内文档保留生成时状态（含旧的私有访问说明）；当前代码与完整归档已公开，最新说明以本 README 为准。

## 结果与实验记录

- [最终 A/B 生产、校验、提交与选择](experiments/phase2-flip-finalization-20260926.md)
- [截止日提交状态核验](experiments/phase2-final-day-verification-20260927.md)
- [最后一轮验证及停止扩展实验的依据](experiments/phase2-last-day-20260927.md)
- [公开方法核查与收尾决定](experiments/final-public-methods-review-20260926.md)
- [完整实验账本](experiments/experiments.csv)
- [模型来源与许可记录](docs/model_provenance.md)

最后一次代码测试记录为 **238 passed**；本次归档整理仅修改文档。最终私榜、资格及主办方代码审查结果应以官方后续公布为准。

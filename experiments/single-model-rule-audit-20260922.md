# 单模型规则与当前链路审计（2026-09-22）

## 官方规则快照

2026-09-22 通过已登录的 Kaggle 比赛页面重新只读核验：

- [Competition Rules](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/rules) 的 Prohibited Conduct 明确要求只允许一个 detection model；多个模型的 voting、weighted fusion、post-NMS fusion 等任何 ensemble 均禁止。
- [主办方单模型澄清](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863) 明确：同一个 single model 的 flip/scaled/cropped TTA 与 multi-scale prediction merging 不算 ensemble；禁止的是多个不同模型的输出组合。代码审核包必须证明最终 CSV 只由一个训练模型生成。
- [主办方推理与格式澄清](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/729747) 再次明确：必须是 single checkpoint；公开 ImageNet/COCO 预训练权重允许，但 README 要声明模型、来源和许可证。
- 正式 CSV 列必须是 `id,image_id,class_id,confidence,x1,y1,x2,y2`；`id` 必须存在且逐行唯一。
- 每队每日最多 3 次提交。Phase 2 开始后，正式 CSV 必须同时覆盖 test 与 ranking 两个集合；仅含 test 的文件会使 ranking 图像计零分。
- 禁止手工伪造标注、使用未声明外部预训练数据、私下向队外分享 competition code/data，以及对评测系统发动攻击。

## 当前合规最佳的模型血缘

正式候选只加载一个训练 checkpoint：

`kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt`

- SHA-256：`8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3`
- 模型：单个 YOLO26m detection model，16 通道 HSI16 输入。
- 推理：同一 checkpoint 的 7 个输入尺度 `832/896/960/1024/1088/1152/1216`。
- 允许的合并范围：只合并上述同一 checkpoint 的重复尺度观测；不加载第二个 `.pt`。
- 当前已验证合规 Public 最佳：ref `56392305`，`0.63072`。

fixed-val 参数选择缓存只来自另一个、单独的一模型 fixed-split checkpoint：

`kaggle_remote/outputs/ablation/kaggle_ablation_yolo26m_e30/last.pt`

- checkpoint SHA-256：`A4E7B10B93CADC241F6C6E10BEB9DFEE4E0887F96C265EC022323AE82C4C7657`
- 7 尺度缓存：`artifacts/ensemble_cache/m.pkl`
- cache SHA-256：`B63C2E81D683316DC5308DFF522D5BAE9AEEDE2BE7F4D0F1F826CCDF62731FD1`
- 该 cache 仅用于 600 张 fixed-val 的规则选择，不会和 full-data checkpoint 的预测在同一 CSV 中合并。

公开预训练初始化：

- `yolo26m.pt` SHA-256：`401CEA9AB23AD19246FF7744859816BC599F350E93C9DD30367B6F0A0745D0B7`
- 来源：Ultralytics Assets `v8.4.0` release；许可证声明见 `docs/model_provenance.md`。
- 它只初始化上述单个 YOLO26m；最终推理不同时加载预训练文件与训练后 checkpoint。

## 代码防误用措施

- `scripts/predict_submission.py` 的正式入口只接受一个 `--weights`。
- `scripts/eval_ensemble.py` 虽保留历史多模型离线研究能力，但一旦传 `--submission`，现在强制要求恰好一个 `--model`；两个或更多 checkpoint 会立即报错，不生成 CSV。
- 所有历史 `0.63917–0.65091` 多模型文件仅作审计记录，不得上传、不得选为最终提交。
- 任何新训练先过同一 2400/600 fixed split 门禁 `0.70443`；未过门禁不训练 full-data、不生成正式候选、不提交。
- 最终 Kaggle Submit 仍需用户在动作前单独确认。

## 2026-09-22 新实验状态

- 按类别 fusion-IoU：OOF `0.70518531`，相对统一 `f0.74` 基线 `-0.00052288`，否决。
- 10 个几何聚类/坐标估计变体：无一达到 `+0.001` 稳定性门禁；最好 `conf^2` 坐标权重仅 `+0.00009068`，否决。
- 11 个尺度来源候选（七项 leave-one-out、中心 3/5 尺度、两组固定对称权重）：全部低于七尺度统一权重 control；最好是删除 `1216`，仍为 `-0.00044541`，否决。
- 上述实验均只重放 `m.pkl`，不加载第二个模型、不开 GPU、不生成测试 CSV、不提交。
- 私有 fixed-split `dfl=2.0` 与 `dfl=2.5` 作业均已在 Kaggle 运行；远端拉回的实际源码已核验为两个彼此独立的单 YOLO26m 实验，二者绝不做预测融合。
- 两条作业均使用同一数据、2400/600 划分、seed 2026、30 epochs、1024；相对基线分别只有 DFL gain `1.5 -> 2.0` 或 `1.5 -> 2.5` 一个训练变量。
- 两条结果各自独立按 `0.70443` 门禁判断；未过门禁的不训练 full-data、不提交。即使两条都过门禁，也只能分别生成单 checkpoint 候选，禁止把二者输出合并。
- full-data 模板已在本地预生成但尚未推送：各自只加载一个训练后的 `last.pt`，并使用已验证的同 checkpoint 七尺度 `fusion_iou=0.74`、`support_gain=0.125`；只有对应 fixed-split 作业通过门禁后才允许推送。

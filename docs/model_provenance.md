# 模型与预训练权重来源

## 当前 YOLO26m HSI16 竞赛路线

- 模型：单个 Ultralytics YOLO26m object detection model；每份比赛 CSV 只能由一个训练 checkpoint 生成。
- 公开预训练初始化：`yolo26m.pt`，仅用于初始化这一个 detection model，不作为第二个推理模型。
- 本地与 Kaggle 私有 code dataset 中的初始化权重 SHA-256：`401CEA9AB23AD19246FF7744859816BC599F350E93C9DD30367B6F0A0745D0B7`。
- 自动下载地址：Ultralytics Assets 的 `v8.4.0` GitHub Release。
- 模型说明：https://docs.ultralytics.com/models/yolo26/
- 权重仓库：https://github.com/ultralytics/assets/
- 本次安装版本：`ultralytics==8.4.147`。
- 许可证：Ultralytics 文档标注代码、模型和文档提供 AGPL-3.0 与 Enterprise 两种许可。本学习项目按公开的 AGPL-3.0 路径使用；若后续用于闭源商业产品，需要重新确认许可要求。

主办方要求在 code-review package 的 README 中声明公开预训练权重的模型、来源和许可证；本文件即为该声明的源记录。训练完成后的 `last.pt`/`best.pt` 是同一个 YOLO26m 的不同 checkpoint，单次推理与提交只能选其中一个，不得同时合并。

同一个训练 checkpoint 的 flip/TTA/多尺度输入变换可以合并；不同 checkpoint、不同训练模型或不同架构的 voting、weighted fusion、post-NMS fusion 均不得用于比赛提交。每次提交所用的唯一权重、代码提交、波段、推理尺度与验证分数都记录在 `experiments/experiments.csv` 及对应实验记录中。

## 早期 YOLO26n 基线

- 模型：Ultralytics YOLO26n object detection model。
- 权重文件：`yolo26n.pt`，来源和许可证与上面的 Ultralytics Assets 记录相同。
- 该模型仅属于早期基线记录，不与当前 YOLO26m 输出合并。

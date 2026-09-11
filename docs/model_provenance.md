# 模型与预训练权重来源

## YOLO26n 基线

- 模型：Ultralytics YOLO26n object detection model。
- 权重文件：`yolo26n.pt`。
- 自动下载地址：Ultralytics Assets 的 `v8.4.0` GitHub Release。
- 模型说明：https://docs.ultralytics.com/models/yolo26/
- 权重仓库：https://github.com/ultralytics/assets/
- 本次安装版本：`ultralytics==8.4.147`。
- 许可证：Ultralytics 文档标注代码、模型和文档提供 AGPL-3.0 与 Enterprise 两种许可。本学习项目按公开的 AGPL-3.0 路径使用；若后续用于闭源商业产品，需要重新确认许可要求。

预训练模型只作为一个单模型的初始化，不与其他模型做集成。每次提交所用的权重、代码提交、波段与验证分数都记录在 `experiments/experiments.csv`。

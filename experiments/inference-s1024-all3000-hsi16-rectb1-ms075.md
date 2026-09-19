# HSI16 矩形单图推理与三尺度框投票

## 目的

修正 16 通道 NumPy 推理与 Ultralytics 验证几何不一致的问题，并只用一个已训练 checkpoint 测试允许的多尺度推理。原提交使用 `batch=2`；由于验证图像尺寸几乎都不同，NumPy 列表批次会被统一补成 `1024 x 1024`。`batch=1` 会启用按 stride 对齐的矩形 letterbox，更接近训练和验证的长宽比。

## 固定条件

- 留出权重：`runs/ablation_s1024_b4_hsi16_shared_p005_995_r1/weights/best.pt`
- 全量权重：`runs/final_s1024_all3000_hsi16_shared_p005_995_r1/weights/last.pt`
- 数据：HSI16，共享单图 P0.5-P99.5 缩放
- `conf=0.0001`、NMS `iou=0.70`、`max_det=300`、FP16
- 多尺度：`960, 1024, 1088`
- 融合：同类别、同一 checkpoint、置信度加权坐标投票；同一尺度的框不会互相投票

赛事主办方明确说明，同一个模型的 TTA 或多尺度推理不属于多模型集成：

https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863

## 固定 600 张留出集结果

以下 `custom` 指标使用与最终 CSV 相同的 NumPy `model.predict()` 路径和同一个 Ultralytics `DetMetrics` 实现。原生 `model.val()` 使用按长宽比分组的数据加载器，因此只作为参考锚点，不能替代提交路径的相对 A/B。

| 推理路径 | mAP50-95 | 相对原 batch=2 单尺度 | 改善类别数 |
|---|---:|---:|---:|
| batch=2，1024 单尺度 | 0.692355 | 0 | - |
| batch=1，1024 矩形单尺度 | 0.694989 | +0.002633 | 10/18 |
| batch=1，三尺度，fusion IoU 0.55 | 0.698691 | +0.006336 | - |
| batch=1，三尺度，fusion IoU 0.65 | 0.698869 | +0.006514 | - |
| **batch=1，三尺度，fusion IoU 0.75** | **0.698923** | **+0.006568** | **相对 batch=1 为 12/18** |

`fusion IoU=0.75` 在原 batch=2 和新 batch=1 两次独立执行几何中都取得三种融合阈值里的最高 mAP，因而选为全量候选。TTA 仍为负向：原生留出 mAP 从 0.698346 降到 0.695643。

原生 validator 与提交路径的绝对差值在 batch=2 为 -0.006380，在 batch=1 缩小到 -0.003357。这个差值反映执行几何不同，不应再把所有提交路径的相对改进一票否决。

## 全量模型候选

- CSV：`submissions/submission_final_s1024_all3000_hsi16_shared_p005_995_r1_rectb1_ms960-1024-1088_f075_conf0001.csv`
- 覆盖：1000/1000 张测试图
- 检测数：99,583
- 达到 `max_det=300` 的图像：87
- `confidence >= 0.001`：27,636
- `confidence >= 0.01`：9,627
- `confidence >= 0.10`：4,582
- 无效框：0
- `check_submission.py`：通过
- SHA-256：`97F78DC2D03840D1F865B2B8253D6B1EAE62DA18D3757E9E0F2B491AA66FD933`
- Kaggle Public Score：**0.62608**
- 相对原最佳 `0.61766`：**+0.00842**
- 提交完成时排行榜位置：**第 34 名**（动态排名，仅记录当时页面）

生成命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\predict_submission.py `
  --weights .\runs\final_s1024_all3000_hsi16_shared_p005_995_r1\weights\last.pt `
  --images .\data\processed\hsi16_shared_p005_995\images\test `
  --output .\submissions\submission_final_s1024_all3000_hsi16_shared_p005_995_r1_rectb1_ms960-1024-1088_f075_conf0001.csv `
  --input-format npy --batch 1 --device 0 --half `
  --conf 0.0001 --iou 0.7 --max-det 300 `
  --multi-scale 960 1024 1088 --fusion-iou 0.75
```

## 结论

Kaggle 已验证该推理路径有效：Public Score 从 `0.61766` 提升到 **`0.62608`**。提升来自同一 checkpoint 的几何修正与三尺度框投票，不是新训练模型。后续通过留出集门槛的 HSI16 模型应优先使用 `batch=1 + 960/1024/1088 + fusion_iou=0.75` 生成第一候选，同时保留标准单尺度作为回退和 A/B 对照。

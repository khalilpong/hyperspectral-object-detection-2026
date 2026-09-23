# Single-checkpoint final max-det cap preregistration

Date: 2026-09-23 (Asia/Singapore)

Status: completed cache-only audit; **rejected**. No GPU, test CSV, Kaggle
upload, Competition Submit, or checkpoint fusion was used. The contract below
was preregistered before reading candidate metrics.

## Question and immutable contract

The current compliant incumbent uses one YOLO26m fixed-split checkpoint for
parameter selection, seven cached scales `832/896/960/1024/1088/1152/1216`,
`fusion_iou=0.74`, confidence-mass `support_gain=0.125`, and the already
accepted global width/height calibration `x1.01`. Its expected fixed-validation
mAP50-95 is `0.7075064748` at `max_det=300`.

This audit changes exactly one value: the final number of fused detections kept
per image. The predeclared grid is `max_det={200,300,400,500}`. It does not
change per-scale inference, checkpoint, boxes, classes, fusion clustering,
confidence scoring, scale weights, or calibration.

Input cache:

- `artifacts/ensemble_cache/m.pkl`
- SHA-256 `B63C2E81D683316DC5308DFF522D5BAE9AEEDE2BE7F4D0F1F826CCDF62731FD1`
- 600 fixed validation images from one YOLO26m checkpoint

## Selection and stop gate

Use numeric-image-order modulo-three folds. On each fold, choose a non-300 cap
on the 400-image fit portion only if it improves `max_det=300` by at least
`0.001`; otherwise retain 300. Measure that fixed choice on the untouched 200
images.

A test candidate is eligible only when all conditions hold:

1. guarded OOF gain over the incumbent is at least `+0.001` mAP50-95;
2. at least two held-out fold deltas are positive;
3. the worst held-out delta is at least `-0.0005`;
4. all three folds choose the same non-300 cap;
5. the full 600-image choice matches every fold.

If any condition fails, record NO-GO and stop. Do not broaden the grid, add
image/class-adaptive caps, generate a test CSV, upload, or submit.

## Results

All three 400-image fit folds retained the `max_det=300` incumbent, so guarded
OOF exactly matched the incumbent at `0.70750647` and the gate is false. The
full 600-image results were:

| max_det | mAP50-95 | Delta vs 300 | Predictions | Images at cap |
|---:|---:|---:|---:|---:|
| 300 | **0.70750647** | - | 74,572 | 79 |
| 500 | 0.70691668 | -0.00058980 | 84,893 | 34 |
| 200 | 0.70645325 | -0.00105322 | 65,097 | 127 |
| 400 | 0.70635840 | -0.00114808 | 80,808 | 49 |

Audit artifact:
`artifacts/ensemble/single_m_max_det_oof_20260923.json`, SHA-256
`6C1595599F7A420B3B670319F6D408F5D6EF514B6FD799CC4C782BA804C71D3E`.

## Decision

**NO-GO.** Keep `max_det=300`. The cap is active for 79/600 validation images,
but both lowering and raising it reduced full-validation mAP50-95, and no fold
fit cleared the preregistered `+0.001` selection threshold. Do not generate a
test CSV or spend a submission on this direction.

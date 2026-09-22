# Single-checkpoint distinct-support-count scoring audit

Date: 2026-09-22 (Asia/Singapore)

Status: completed cache-only audit; **rejected**. No GPU, test CSV, Kaggle
upload, Competition Submit, or checkpoint fusion was used.

## Question and fixed contract

The production incumbent ranks fused boxes with confidence-mass support:
`max_conf + 0.125 * (sum_conf - max_conf) / 7`. This audit asked whether a
lower-freedom score based only on how many distinct image scales support a box
could improve that incumbent:

`min(1, max_conf + gamma * (distinct_sources - 1) / 6)`.

Everything else remained fixed: one YOLO26m checkpoint, the seven cached scales
`832/896/960/1024/1088/1152/1216`, `fusion_iou=0.74`, `max_det=300`, the
accepted global width/height calibration `x1.01`, and the fixed 600-image
validation set. The preregistered grid was
`gamma={0,0.01,0.02,0.03,0.04,0.05,0.06,0.08}`.

Selection used numeric-order modulo-three folds. Each fold selected on 400
images only when a count candidate improved the incumbent by at least `0.001`,
then measured that fixed choice on the held-out 200 images. The final gate also
required OOF gain `>=0.001`, at least two positive held-out folds, worst-fold
delta `>=-0.0005`, and unanimous fold/full selection of a non-incumbent rule.

## Results

All three fit folds and the full 600-image selection retained the existing
confidence-mass incumbent. Consequently, the guarded OOF result exactly equals
the incumbent:

| Rule | mAP50-95 | Delta vs incumbent |
|---|---:|---:|
| confidence-mass incumbent (`support_gain=0.125`) | 0.70750647 | - |
| guarded OOF selection | 0.70750647 | 0.00000000 |
| best count-only candidate (`gamma=0.08`) | 0.70590884 | **-0.00159764** |

The best count-only candidate also reduced mAP50 from `0.95713993` to
`0.95637212` and mAP75 from `0.85427778` to `0.85279711`. The stability gate is
false.

Audit artifact:
`artifacts/ensemble/single_m_support_count_scoring_oof_20260922.json`, SHA-256
`2E2A61A7F42EC82441A301BB548A7D596EB79DB8D3DE0E3E31D2574FFD19F575`.

## Decision

**NO-GO.** Keep `support_gain=0.125`; do not generate a test CSV or submission
from distinct-support-count scoring. This audit reuses the same heavily studied
600-image validation set, so even a small positive result would have remained
exploratory. The observed result is negative enough that no further confidence
ranking sweep is justified; paid GPU time should go to the preregistered EIoU
fixed experiment instead.

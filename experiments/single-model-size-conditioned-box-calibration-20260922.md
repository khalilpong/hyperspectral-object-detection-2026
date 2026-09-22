# Single-checkpoint size-conditioned box calibration audit

Date: 2026-09-22 (Asia/Singapore)

Status: completed cache-only audit; **rejected**. No GPU, test CSV, Kaggle
upload, Competition Submit, or checkpoint fusion was used.

## Question and contract

Could the already validated global `width/height x1.01` box calibration be
improved by applying different center-preserving width/height scales to small
and large predicted boxes?

The audit reused only the same YOLO26m checkpoint's seven cached scales
`832/896/960/1024/1088/1152/1216`, `fusion_iou=0.74`, and
`support_gain=0.125` on the fixed 600-image validation set. Predicted box area
was normalized by image area. Thresholds were `0.005/0.01/0.02`; each side of
the threshold used a scale from `0.99/1.00/1.01/1.02/1.03`, for 75 fixed
candidates. Confidence, class, center, fusion, and ranking were unchanged.

Selection used the same deterministic three folds as the global calibration:
fit each candidate on 400 images and evaluate the selected candidate on the
held-out 200 images.

## Results

| Comparison | mAP50-95 | Delta |
|---|---:|---:|
| identity control | 0.70570818 | - |
| existing global `x1.01` | 0.70750647 | +0.00179829 vs identity |
| size-conditioned OOF | 0.70686997 | +0.00116178 vs identity |
| size-conditioned OOF vs global `x1.01` | - | **-0.00063651** |

Fold selections and held-out deltas versus identity were:

- fold 0: threshold `0.02`, small `1.01`, large `1.01`, `+0.00251177`;
- fold 1: threshold `0.02`, small `1.01`, large `1.01`, `+0.00133919`;
- fold 2: threshold `0.01`, small `1.01`, large `1.00`, `+0.00002526`.

The full-validation optimum was threshold `0.02`, small `1.01`, large `1.01`,
which is exactly the existing global `x1.01` transform. Relative to global
`x1.01`, folds 0/1 tied and fold 2 lost `0.00196199`; the three fold choices
were not unanimous.

## Decision

**NO-GO.** The conditional model clears the old identity-relative numeric
threshold but does not improve the actual incumbent. It is less stable across
folds and loses `0.00063651` OOF versus global `x1.01`. Retain only the already
submitted global calibration and stop additional box-calibration sweeps.

# Single-checkpoint global box calibration — 2026-09-22

## Decision

**PASS the inference gate and the guarded Public transfer check.** Every
out-of-fold split independently selected the same global transform: keep each
box center fixed and multiply both width and height by `1.01`. This is a pure
post-process of one YOLO26m checkpoint's permitted seven-scale inference; it
does not add a model, checkpoint, class remap, confidence change, or detection.

## Pre-registered cache-only contract

- Source: `artifacts/ensemble_cache/m.pkl`, previously traced to one fixed-split
  YOLO26m HSI16 checkpoint.
- 600 fixed validation images; scales `832, 896, 960, 1024, 1088, 1152, 1216`.
- Production fusion held fixed at `fusion_iou=0.74`, `support_gain=0.125`,
  `max_det=300`.
- 73 bounded global candidates:
  - width/height scale grid `0.97..1.03` in 0.01 steps (49 pairs), and
  - x/y center-shift grid `-0.02..0.02` in 0.01 steps (25 pairs),
  - with identity counted once and scale/shift never fitted jointly.
- Three folds by numeric validation order modulo 3. Each fold selected on 400
  images and was measured only on its 200 held-out images.
- A fold accepted a non-identity transform only with training gain `>=0.001`.
- Final gate: combined OOF gain `>=0.001`, at least 2/3 positive folds, worst
  fold `>=-0.0005`, unanimous fold choice, and the full-data choice matching
  every fold.

## Results

All folds selected `width_scale=1.01`, `height_scale=1.01`, zero center shift.

| Fold | fit gain | held-out delta mAP50-95 |
|---:|---:|---:|
| 0 | +0.00179136 | +0.00251177 |
| 1 | +0.00202965 | +0.00133919 |
| 2 | +0.00169239 | +0.00198724 |

| Metric | Identity | OOF calibrated | Delta |
|---|---:|---:|---:|
| mAP50-95 | 0.70570818 | 0.70750647 | +0.00179829 |
| mAP50 | 0.95706411 | 0.95713993 | +0.00007582 |
| mAP75 | 0.85383964 | 0.85427778 | +0.00043814 |

The stability gate passed. This is stronger evidence than the rejected
box-vote geometry and classwise fusion-IoU sweeps because the exact candidate
was selected identically in all three independent fit folds and improved every
held-out fold.

## Test candidate and provenance

The audited transform was applied to the already submitted compliant source:

- Source: `submission_single_m_hsi16_ms7_f074_sg0125.csv`
- Source SHA-256:
  `E6F5BC1892988A08FF4D5F97F2F976CC094CA770CFAB377A86708C71482CB48E`
- Candidate:
  `submission_single_m_hsi16_ms7_f074_sg0125_boxscale101.csv`
- Candidate SHA-256:
  `C01214E6B250075C7DA522F2E9E6F89C829BF94AE1EB2F57CF69618FDC995A86`
- 98,556 rows, 1000/1000 images, zero invalid boxes.
- `id`, `image_id`, `class_id`, confidence, row order, and row count are
  unchanged; only the four box coordinates changed.
- Audit artifact:
  `artifacts/ensemble/single_m_box_calibration_oof_20260922.json`, SHA-256
  `056C8DD8A9274B3A4F49B8D5A9160BC4B42B021EA54EAD09A44E40F6B87844E2`.
- Manifest:
  `artifacts/submission_manifests/single_m_hsi16_ms7_f074_sg0125_boxscale101.json`.

`calibrate_submission_boxes.py` now requires the passing audit JSON and derives
the transform from it. It rejects non-unanimous folds and records the audit hash
in the manifest, so arbitrary hand-entered calibration parameters cannot be
presented as audited.

## Public result

After the user explicitly confirmed the final action, the exact audited CSV was
submitted once at 2026-09-22 14:30:14 (Singapore/Beijing time). Kaggle reported
`Success` with Public `0.63546`, ref `56455800`. This is `+0.00474` over the
previous compliant best `0.63072`, so the OOF direction transferred positively
to the public test partition and `0.63546` is now the compliant safety line.
Two daily submissions remained after this action. The final-leaderboard checkbox
was deliberately left untouched pending a separate user decision.

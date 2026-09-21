# YOLO26m EIoU fixed-split preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: local private Kernel package generated and tested; the synchronized private
`hsi-detection-code` version is remotely `ready`; the Kernel itself is **not uploaded
and not running**.

## Question

Does replacing only YOLO26m's CIoU localization term with Extended IoU (EIoU) improve
strict fixed-validation box localization enough to pass the existing full-data gate?

The fixed baseline has high coarse detection quality (`mAP50` about `0.953-0.956`) but
a much lower `mAP50-95` (`0.70143` ordinary single-scale; about `0.70404` under the
seven-scale control). Several input/augmentation variants raised mAP50 while lowering
strict mAP. EIoU therefore targets the measured high-IoU bottleneck more directly than
another input representation or model-size increase.

## Exact implementation boundary

- `hsi_detection.box_loss.bbox_eiou` replaces only
  `ultralytics.utils.loss.bbox_iou`, the symbol used by `BboxLoss`;
- `ultralytics.utils.tal.bbox_iou` remains unchanged, so TaskAlignedAssigner is not
  modified;
- DFL/L1 supervision, model architecture, state-dict shapes, pretrained transfer,
  inference, and post-processing remain unchanged;
- CIoU remains the default, and checkpoints record `box_iou_loss` in model YAML metadata
  so a resume cannot silently switch the recipe;
- RT-DETR rejects this YOLO-only option.

## Fixed contract

- one YOLO26m checkpoint initialized from `yolo26m.pt`;
- unchanged HSI16 shared P0.5-P99.5 data and exact seed-2026 manifest: 2,400 train /
  600 val / 1,000 test;
- 30 epochs, image size 1024, seed 2026, attempts `8:2,6:2,4:2,4:0`;
- only effective variable: `BOX_IOU_LOSS ciou -> eiou`;
- unchanged random extra-channel initialization, `degrees=0`, `scale=0.5`, `dfl=1.5`,
  `cls_pw=0`, no train multiscale, object crops, tile inference, or spectral stem;
- no second checkpoint, fusion, pseudo-labeling, external data, or Competition Submit.

The private slug is `zephyrpong/hsi-yolo26m-eiou-ablation`; the tracked package is
`kaggle_remote/kernel_ablation_eiou`.

## Verification and gate

- formula tests cover identical boxes, finite nonzero gradients, degenerate boxes, and
  rejection of unintended GIoU/DIoU/plain-IoU call signatures;
- a real YOLO26m 16-channel/18-class, 64-pixel single-batch forward/loss/backward smoke
  produced finite total loss `4.7768364`, finite box/class/L1 parts, and a finite nonzero
  first-convolution gradient; report:
  `artifacts/smoke_yolo26m_eiou_20260922.json`;
- the repository suite passes `135 passed`;
- flattened `hsi_detection.box_loss.py`, `scripts.train_baseline.py`, and
  `run_hsi_yolo26.py` copies have exact matching SHA-256 hashes with their sources.
- the new private code-dataset version reached `ready`, and the authoritative remote
  file list includes `hsi_detection.box_loss.py` (4,754 bytes), the updated
  `scripts.train_baseline.py` (11,293 bytes), and runner (33,522 bytes).

Audit a downloaded fixed result with:

```text
scripts/check_fixed_split_gate.py --architecture yolo --expected-dfl 1.5 --expected-degrees 0 --expected-box-iou-loss eiou --gate 0.70443
```

Require 30 complete epochs, exact config/manifest, one successful non-OOM/non-SHM
attempt, fresh reload, 1,000-image NPY inference, checker success, and hashes. Only a
fixed best mAP50-95 `>= 0.70443` may advance the same single-checkpoint recipe to full
data.

## Scheduling decision

Do not push while the existing degrees=5 YOLO fixed job is running. If that job fails
the gate, EIoU is the first YOLO fallback because it directly targets the repeated
strict-localization failure pattern. The `[13,8,5]` pseudo-RGB and YOLO26x packages move
behind EIoU. Push at most once after checking that the remote slug has no RUNNING or
COMPLETE version and that Kaggle GPU quota remains.

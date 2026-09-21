# YOLO26x HSI16 fixed-split preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: local private Kernel package generated and syntax-checked; **not uploaded and
not running**.

## Question

Does changing only the detector scale from YOLO26m to YOLO26x improve strict
fixed-validation localization enough to pass the existing full-data gate?

YOLO26l is not a candidate: the existing HSI16/1024/30-epoch fixed run reached only
`0.69628`, below YOLO26m `0.70143` and the `0.70443` gate. YOLO26x is the only
unmeasured YOLO26 scale, so this is a low-priority capacity hypothesis rather than a
claim that larger models have shown a positive trend.

## Fixed contract

- single model: `yolo26x.pt`, official Ultralytics YOLO26 asset;
- only variable versus the ordinary YOLO26m fixed baseline: model scale m -> x;
- data: unchanged HSI16 shared P0.5-P99.5 NPY representation;
- split: exact seed-2026 manifest, 2,400 train / 600 val;
- training: 30 epochs, image size 1024, seed 2026;
- conservative single-GPU attempts: `2:2,1:2,1:0`;
- unchanged: random extra-channel init, `scale=0.5`, `degrees=0`, `dfl=1.5`,
  `cls_pw=0`, no multiscale/crop/tile during the fixed gate;
- no second checkpoint, fusion, pseudo-labeling, external data, or Competition Submit.

The private slug is `zephyrpong/hsi-yolo26x-ablation`; the dedicated local package is
`kaggle_remote/kernel_ablation_yolo26x`. The package enables Internet because the
official `yolo26x.pt` is supported by Ultralytics 8.4.147 but is not cached locally.

## Gate and scheduling

The package is queued only after the higher-priority degrees=5 and physically ordered
`[13,8,5]` pseudo-RGB fixed candidates have failed. Push at most once after confirming
the remote slug is not already running or complete and that Kaggle GPU time remains.

Admission requires fixed best mAP50-95 `>= 0.70443`, 30 complete epochs, exact
manifest, successful return code, fresh checkpoint reload, 1,000-image NPY inference,
and submission checker success. Audit with `scripts/check_fixed_split_gate.py` using
`--architecture yolo --model yolo26x.pt --expected-dfl 1.5 --gate 0.70443`.
Failure stops this route without full-data training or submission.

Local verification: private metadata and exact CONFIG inspected; generated script
compiles; repository suite was `124 passed` before package generation and the package
contains no source change.

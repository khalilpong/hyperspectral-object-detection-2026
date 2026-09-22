# YOLO26m `reg_max=16` fixed-split preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: implementation, model/loss smoke, real Trainer callback smoke, tests,
private local Kernel package, and local code-dataset staging are complete. The
code-dataset version and Kernel are **not uploaded and not running**.

## Question

Does changing only YOLO26m box-distance parameterization from `reg_max=1` to
`reg_max=16` improve strict fixed-validation localization enough to pass the
existing full-data gate?

This is not equivalent to the rejected `dfl=2.0/2.5` gain experiments. In
Ultralytics 8.4.147, YOLO26 `reg_max=1` uses the normalized-L1 fallback in
`BboxLoss`; `reg_max=16` activates the true 16-bin Distribution Focal Loss and
its softmax/projection head. The retained CIoU term and requested DFL gain remain
unchanged.

## Exact architecture and transfer boundary

- architecture source: official Ultralytics 8.4.147 `yolo26.yaml`;
- only architecture value changed: `reg_max: 1 -> 16`;
- model scale remains `m`, input remains 16-channel HSI, and output remains the
  same 18 competition classes;
- pretrained source remains the single official `yolo26m.pt` checkpoint;
- TaskAlignedAssigner, CIoU, backbone, neck, classification heads, augmentation,
  optimizer recipe, inference, and post-processing are unchanged;
- the resulting run still produces exactly one detection checkpoint and cannot
  be fused with any other checkpoint.

The real weight-shape audit found:

| Item | Value |
|---|---:|
| source checkpoint parameters | 21,896,248 |
| target HSI16/18-class/regmax16 parameters | 21,831,548 |
| exact-shape reusable source parameters | 21,769,600 (99.4216%) |
| exact-shape coverage of target parameters | 99.7162% |
| source parameters mismatched specifically by `reg_max` box heads | 1,560 |

The other expected mismatches are the RGB-to-HSI16 first convolution and the
COCO-80-to-competition-18 classification outputs. This directly disproves the
earlier rough claim that switching to `reg_max=16` would discard about 60% of
the box-head weights.

## Runtime evidence

`scripts/smoke_yolo_regmax.py` built the real YOLO26m target, transferred
`yolo26m.pt`, and completed a 16-channel/18-class forward, loss, and backward
pass. Evidence includes:

- `head.reg_max == 16`;
- one-to-many and one-to-one box outputs are `[64, 64, 64]`;
- `head.dfl` is `DFL`, not `Identity`;
- the pretrained RGB stem prefix is copied exactly and the 13 extra input
  channels are finite/nonzero random initialization;
- total loss, box/class/DFL parts, stem gradient, and box-head gradients are
  finite, with nonzero stem and box gradients;
- report: `artifacts/smoke/yolo26m_regmax16_cpu_20260922.json`, SHA-256
  `86E95CE46616A000859F2DC498D2C7F2D4721C674ECBFE50E2CAEC5E5C4BC318`.

A second smoke used the real `scripts/train_baseline.py` Trainer path on four
HSI16 images for one CPU batch. Training, validation, and checkpoint
serialization completed. Its runtime `model_contract.json` confirms 16 input
channels, 18 classes, `reg_max=16`, true DFL, and both 64-channel box branches;
SHA-256 `DAC17569A2F06A4CD30C0B2621C405682E6BD11A178FD5DF5F7CF6B676B3694C`.
The effective optimizer contract remained the ordinary baseline AdamW
`lr=0.000455`, beta1 `0.9`, warmup-bias `0`; contract SHA-256
`FAAEC6EBCBF7E7645B3AF6BD004BF1744B885CC317A764529DC16A600421728B`.

The full repository suite passes `163 passed`; targeted regmax/runner/gate tests
pass inside that suite, and all changed/generated Python files compile.

## Fixed contract and gate

- one YAML-defined YOLO26m checkpoint initialized only from `yolo26m.pt`;
- canonical HSI16 shared P0.5-P99.5 seed-2026 manifest, exactly 2,400 train /
  600 val / 1,000 test;
- 30 epochs, image size 1024, seed 2026, attempts
  `8:2,6:2,4:2,4:0`;
- unique variable: `REG_MAX 1 -> 16`;
- unchanged random extra-channel initialization, `degrees=0`, `scale=0.5`,
  `dfl=1.5`, `cls_pw=0`, CIoU, auto optimizer, no train multiscale, crop,
  tile, or SpectralStem;
- no full-data run, test submission, or Competition Submit before the fixed gate.

Audit a future downloaded run with:

```text
scripts/check_fixed_split_gate.py --architecture yolo --model yolo26m.pt --expected-model-yaml yolo26m-regmax16.yaml --expected-reg-max 16 --expected-dfl 1.5 --model-contract <run>/model_contract.json --gate 0.70443
```

The gate requires the exact config, one successful non-OOM/non-SHM attempt,
30 complete epochs, matching runtime model contract, fresh checkpoint reload,
1,000-test inference/checker success, and best fixed mAP50-95 `>=0.70443`.
Failure rejects the route without full-data training or submission.

## Package and scheduling

- local package: `kaggle_remote/kernel_ablation_rm16`;
- intended private slug: `zephyrpong/hsi-yolo26m-rm16-ablation`;
- model YAML SHA-256:
  `B448F7A729718439F5BD6EC37C6F0091B8C637C017628F87A3B49F5A040C5412`;
- generated runner SHA-256:
  `7A1A3E291EC098088DC8A5C130233F585FDD88E4C0B7276DEA57F965E87F10FE`;
- metadata SHA-256:
  `474BA6D15A787F1E750D0CB18EE5E7D2BFFCE06774A79B45B9ECD573DD03AFF3`;
- private metadata remains `is_private: true`.

Do not upload while the live weekly GPU meter remains above 30 hours. EIoU is
still the first YOLO localization candidate and `adamw_lr001` remains second.
This regmax16 candidate is the next direct localization-architecture fallback,
ahead of broader architecture/capacity changes. Before any future push, create
and verify the synchronized private code-dataset version, confirm the remote
slug does not already exist, and push at most once.

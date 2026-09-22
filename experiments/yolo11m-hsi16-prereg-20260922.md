# YOLO11m HSI16 checkpoint-native fixed-split preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: official checkpoint download and hash audit, architecture/transfer smoke,
real Trainer callback and checkpoint-reload smoke, source-lineage gate, full
repository tests, private local Kernel package, and local code-dataset staging
are complete. The code-dataset version and Kernel are **not uploaded and not
running**.

## Question

Does replacing only the YOLO26m checkpoint-native architecture with official
YOLO11m improve strict fixed-validation localization enough to pass the existing
full-data gate?

This candidate is motivated by architecture, not by the public Notebook that
reported `0.7394`. That Notebook used duplicate recursive paths and leaked
788/844 effective validation images into training, so its score is invalid for
this fixed-split comparison and supplies no expected-gain evidence.

## Exact architecture and checkpoint lineage

- source: official Ultralytics GitHub Assets release `v8.4.0`, file
  `yolo11m.pt`;
- checkpoint bytes: `40,684,120`;
- checkpoint SHA-256:
  `D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95`;
- source model: `DetectionModel`, RGB input, 80 COCO classes,
  `reg_max=16`, true `DFL`, non-end-to-end Detect head, and box outputs
  `[64,64,64]`;
- target model: the checkpoint's own YOLO11m architecture widened to 16 input
  channels and changed to the same 18 competition classes;
- source mode: `checkpoint_native`; no custom YAML is applied;
- the run still produces exactly one detection checkpoint and cannot be fused
  with YOLO26m or any other checkpoint.

The real shape audit found:

| Item | Value |
|---|---:|
| source checkpoint parameters | 20,114,688 |
| target HSI16/18-class parameters | 20,074,374 |
| exact-shape reusable source parameters | 20,051,280 (99.6848%) |
| exact-shape coverage of target parameters | 99.8850% |

The only shape mismatches are the expected RGB-to-HSI16 first convolution and
the three COCO-80-to-competition-18 class-output layers. The pretrained RGB stem
prefix is copied exactly; the 13 extra input channels retain finite, nonzero
random initialization.

## Runtime evidence

`scripts/smoke_yolo_checkpoint_native.py` loaded the exact checkpoint, built the
real HSI16/18-class target, and completed forward, loss, and backward on CPU.
It confirmed:

- source and target `reg_max == 16` with real `DFL`;
- target box outputs `[64,64,64]` and no one-to-one branch;
- finite box/class/DFL loss parts;
- finite, nonzero stem and box-head gradients;
- report:
  `artifacts/smoke/yolo11m_hsi16_checkpoint_native_cpu_20260922.json`,
  SHA-256
  `41352F214D34DBFFB3DD205787161C5E25EC6CF14B4CA6472BD5DFFBADC394BC`.

A second smoke used the real `scripts/train_baseline.py` Trainer path on four
HSI16 images for one CPU batch. Training, validation, serialization, and fresh
`last.pt` reload completed. The runtime `model_contract.json` confirms 16 input
channels, 18 classes, `reg_max=16`, true DFL, non-end-to-end YOLO11 Detect, and
three 64-channel box outputs; SHA-256
`0978CAE62A313140C0135F3AE48DD00F4F29B33A272FF17C2C6CAFFA58E6EB64`.
The optimizer contract remains the ordinary baseline AdamW with effective
initial LR `0.000455`, beta1 `0.9`, and warmup-bias LR `0`; SHA-256
`FAAEC6EBCBF7E7645B3AF6BD004BF1744B885CC317A764529DC16A600421728B`.
Fresh-reloaded `last.pt` SHA-256 is
`6C037F96D1F984BA89571A702065285A98BC13478559921CA03F9D44A8DC603F`.

The full repository suite passes `184 passed`; the source-kind, runner, gate,
generator, preparation, and prior-candidate tests are included.

## Fixed contract and gate

- one checkpoint-native YOLO11m model initialized only from the exact checkpoint
  hash above;
- canonical ordinary HSI16 shared P0.5-P99.5 seed-2026 manifest, exactly 2,400
  train / 600 val / 1,000 test;
- 30 epochs, image size 1024, seed 2026, attempts
  `8:2,6:2,4:2,4:0`;
- unique experimental variable versus the ordinary fixed baseline: model/checkpoint
  architecture `yolo26m.pt -> yolo11m.pt`;
- unchanged physical band order, random extra-channel initialization,
  `degrees=0`, `scale=0.5`, `dfl=1.5`, `cls_pw=0`, CIoU, auto optimizer, no
  train multiscale, crop, tile, SpectralStem, EIoU, or optimizer change;
- no full-data run, test submission, or Competition Submit before the fixed gate.

Audit a future downloaded run with:

```text
scripts/check_fixed_split_gate.py --architecture yolo --model yolo11m.pt --expected-model-source-kind checkpoint_native --expected-pretrained-weights-sha256 D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95 --expected-reg-max 16 --expected-dfl 1.5 --model-contract <run>/model_contract.json --gate 0.70443
```

The gate requires the exact config, checkpoint source kind/hash/size, one
successful non-OOM/non-SHM attempt, 30 complete epochs, matching runtime model
contract, fresh checkpoint reload, 1,000-test inference/checker success, and
best fixed mAP50-95 `>=0.70443`. Failure rejects the route without full-data
training or submission.

## Package and scheduling

- local package: `kaggle_remote/kernel_ablation_yolo11m_rm16`;
- intended private slug: `zephyrpong/hsi-yolo11m-rm16-ablation`;
- generated runner SHA-256:
  `8F266CD714753208CD765F898F0D2FC0E41B1E8509130B2AA4E71E637EED3054`;
- metadata SHA-256:
  `1A7892BD252366857D1D66C8E3DEEC604BCB81D103FD3E1E06C89E2771C27931`;
- synchronized main/code-dataset runner SHA-256:
  `F5D50B4C210D68226F96B1EF724A0ADF3AAB1049891C242033998052AF037864`;
- generated runner is byte-equivalent to the main runner outside the expected
  CONFIG block; metadata remains `is_private: true`.

Do not upload while the live weekly Kaggle GPU meter remains above 30 hours.
The queue remains EIoU, then `adamw_lr001`, `reg_max=16`, ordinary HSI16
`[13,8,5,...]`, then this YOLO11m candidate. Before a future push, publish and
verify a synchronized **private** code-dataset version containing the exact
checkpoint hash, confirm the remote slug does not already exist, and push at
most once.

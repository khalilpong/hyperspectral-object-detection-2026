# YOLO26m ordinary-HSI16 `[13,8,5,...]` order preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: implementation, preparation-contract audit, synthetic encoding/label
integration test, full repository tests, private local Kernel package, and local
code-dataset staging are complete. The code-dataset version and Kernel are
**not uploaded and not running**.

## Question

Does changing only the array position of the ordinary HSI16 physical bands so
that the pretrained RGB slots receive physical bands `[13,8,5]` instead of
`[5,8,13]` improve strict fixed-validation localization enough to pass the
existing full-data gate?

This is a 16-channel ordinary-HSI16 experiment, not the three-channel
`pseudo_rgb:13,8,5` experiment. All 16 physical bands remain present, and the
same per-image shared P0.5-P99.5 affine normalization is retained. The only data
value change is the channel permutation:

```text
baseline  = 5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15
candidate = 13,8,5,0,1,2,3,4,6,7,9,10,11,12,14,15
```

Ultralytics transfers pretrained RGB convolution weights by array position, so
the first three physical-band mappings change from `R/G/B -> 5/8/13` to
`R/G/B -> 13/8/5`. There is no repository or organizer evidence assigning
official wavelengths or RGB semantics to bands 5/8/13; this is therefore a
low-cost positional-bias hypothesis, not a claim that `[13,8,5]` is physically
true RGB.

## Isolation and data evidence

- `scripts/prepare_multispectral.py` now accepts an explicit, validated
  `--band-order` that must be a permutation of `0..15`; its default remains the
  established baseline order.
- Every ordinary-HSI16 output gets an atomic `preparation_config.json` covering
  method version, band order, channels, shared-scale encoding, percentiles,
  manifest hash, and split counts. Existing NPY files without that contract are
  rejected, and a changed contract must use a new output directory.
- The candidate data identity is isolated as
  `hsi16_order_13-8-5-0-1-2-3-4-6-7-9-10-11-12-14-15_p005_995`; it cannot reuse
  `hsi16_shared_p005_995`.
- The remote runner verifies the preparation config, report, dataset YAML,
  manifest, counts, 16 channels, 18 classes, band order, shared P0.5-P99.5
  uint8 encoding, and fixed scope before training. It copies all four evidence
  files into `data_contract/` and records their SHA-256 hashes in `status.json`.
- The fixed gate requires the exact expected order and the four downloaded
  artifacts; it recalculates hashes, parses the 2,400/600 manifest, and rejects
  baseline-order, missing, or drifted evidence.
- A synthetic end-to-end preparation test generated baseline and candidate
  datasets from identical mosaics. Candidate NPY arrays were exactly the
  expected channel permutation; shared low/high summaries, label bytes, and
  manifest bytes were unchanged. Unsafe resume and contract drift were rejected.
- Full repository suite: `177 passed`; changed/generated Python files compile.

## Fixed contract and gate

- one YOLO26m checkpoint initialized only from `yolo26m.pt`;
- ordinary 16-channel HSI, target order shown above, shared P0.5-P99.5 uint8;
- exact seed-2026 manifest: 2,400 train / 600 val / 1,000 test;
- 30 epochs, image size 1024, seed 2026, attempts
  `8:2,6:2,4:2,4:0`;
- unique variable: ordinary-HSI16 channel order;
- unchanged `reg_max=1`, random extra-channel initialization, `degrees=0`,
  `scale=0.5`, `dfl=1.5`, `cls_pw=0`, CIoU, auto optimizer, no train
  multiscale, crop, tile, SpectralStem, EIoU, or optimizer change;
- no full-data run, test submission, or Competition Submit before the fixed gate.

Audit a future downloaded run with:

```text
scripts/check_fixed_split_gate.py --architecture yolo --model yolo26m.pt --expected-model-source-kind checkpoint_native --expected-pretrained-weights-sha256 401CEA9AB23AD19246FF7744859816BC599F350E93C9DD30367B6F0A0745D0B7 --expected-reg-max 1 --expected-dfl 1.5 --expected-band-order 13,8,5,0,1,2,3,4,6,7,9,10,11,12,14,15 --preparation-config <run>/data_contract/preparation_config.json --preparation-report <run>/data_contract/preparation_report.json --dataset-yaml <run>/data_contract/dataset.yaml --split-manifest <run>/data_contract/split_manifest.csv --gate 0.70443
```

The gate also requires one successful non-OOM/non-SHM attempt, 30 complete
epochs, fresh checkpoint reload, 1,000-test inference/checker success, and best
fixed mAP50-95 `>=0.70443`. Failure rejects the route without full-data training
or submission.

## Package and scheduling

- local package: `kaggle_remote/kernel_ablation_order1385`;
- intended private slug: `zephyrpong/hsi-yolo26m-order1385-ablation`;
- generated runner SHA-256:
  `10E5E2136D606A929861C1E3FA3ABD2F5316F6B95C836E9A80107C9A6B58AAA7`;
- metadata SHA-256:
  `6FBE3C7236C4AF9A3392AD792308661C696ED3BE2213FD2D24876ADBF518EFBD`;
- staged preparation script SHA-256:
  `BCE78EAD7118B01BA85FB350E1CF6CEC90086E50F8C4C654CBC8F99299AB4C1A`;
- staged main runner SHA-256:
  `F5D50B4C210D68226F96B1EF724A0ADF3AAB1049891C242033998052AF037864`;
- generated runner is byte-equivalent to the main runner outside the expected
  CONFIG block; metadata is private.

Do not upload while the live weekly GPU meter remains above 30 hours. The YOLO
queue remains EIoU, then `adamw_lr001`, then `reg_max=16`; this channel-order
candidate follows those more direct localization candidates and precedes the
now-packaged YOLO11m and YOLO26x fallbacks. Before any future push, create and verify
the synchronized private code-dataset version, confirm the slug has no existing
RUNNING/COMPLETE version, and push at most once.

# YOLO26m effective AdamW LR `0.001` fixed-split preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: implementation, runtime-contract smoke, tests, private local Kernel package,
and local code-dataset staging are complete. Nothing in this experiment has been
uploaded or run on Kaggle.

## Why this is a distinct single-variable question

The ordinary YOLO26m HSI16 baseline requests Ultralytics `optimizer=auto`.
Ultralytics 8.4.147 ignores the displayed `lr0=0.01` and `momentum=0.937`, then
actually builds `AdamW(lr=0.000455, beta1=0.9)` and changes warmup-bias LR to
`0.0`. Therefore, simply passing `optimizer=AdamW --lr0 0.001` would also change
beta1 and warmup-bias LR unless they are explicitly locked.

This preregistered recipe keeps the effective optimizer algorithm, beta1,
warmup-bias LR, weight decay, warmup duration, final LR factor, model, data,
augmentation, and loss unchanged. Its only effective optimizer change is:

`initial AdamW LR: 0.000455 -> 0.001`.

The leaked public YOLO11m pseudo-RGB Notebook also requested AdamW `0.001`, but
it changed many other variables and its validation split contains 93.36% overlap.
Its displayed metric is not evidence that this LR will improve the current fixed
HSI16 model; the new run must stand on the canonical 2400/600 gate.

## Auditable implementation

- `scripts/train_baseline.py` now accepts validated `--momentum` and
  `--warmup-bias-lr` overrides and writes `optimizer_contract.json` after
  Ultralytics has actually constructed the optimizer and scheduler.
- Runtime contract fields include the request-time optimizer/LR/momentum and the
  effective optimizer class, initial/current group LRs, beta1, warmup-bias LR,
  warmup epochs, `lrf`, and weight decay.
- `kaggle_remote/run_hsi_yolo26.py` exposes one bounded recipe name:
  `OPTIMIZER_RECIPE=adamw_lr001`. It expands only to:
  `AdamW`, `lr0=0.001`, `momentum=0.9`, `warmup_bias_lr=0.0`.
- RT-DETR rejects this YOLO-only recipe.
- `scripts/check_fixed_split_gate.py` requires both the exact recipe config and
  a matching runtime `optimizer_contract.json`; it rejects missing or drifted
  contracts.

## Runtime smoke evidence

Two real YOLO26n, 16-channel, one-epoch CPU smokes used the same one-image train
and validation data, seed, augmentation defaults, and Ultralytics 8.4.147:

| Field | ordinary `auto` | `adamw_lr001` |
|---|---:|---:|
| effective optimizer | AdamW | AdamW |
| effective initial LR | 0.000455 | 0.001 |
| beta1 | 0.9 | 0.9 |
| effective warmup-bias LR | 0.0 | 0.0 |
| warmup epochs | 3.0 | 3.0 |
| `lrf` | 0.01 | 0.01 |
| requested weight decay | 0.0005 | 0.0005 |

Both smokes completed training, validation, and checkpoint serialization. Contract
hashes:

- auto: `FAAEC6EBCBF7E7645B3AF6BD004BF1744B885CC317A764529DC16A600421728B`
- lr001: `799D675DFC6AA935A4A4FF01CC23C1EEB234635E7E221DCBB007AE614DC9C16F`

Targeted tests pass `60 passed`; the complete repository suite passes
`141 passed`.

## Fixed contract and gate

- one `yolo26m.pt` checkpoint;
- canonical HSI16 shared P0.5-P99.5 seed-2026 manifest, exactly 2400 train / 600
  val / 1000 test;
- 30 epochs, 1024, attempts `8:2,6:2,4:2,4:0`;
- random extra-channel initialization, CIoU, `dfl=1.5`, `scale=0.5`,
  `degrees=0`, `cls_pw=0`, no train multiscale/crop/tile/stem;
- only effective variable: initial AdamW LR `0.000455 -> 0.001`;
- full-data gate: fixed best mAP50-95 `>= 0.70443` with a matching runtime
  optimizer contract. Otherwise reject without full-data training or submission.

Audit command after a future download:

```text
scripts/check_fixed_split_gate.py --architecture yolo --expected-dfl 1.5 --expected-optimizer-recipe adamw_lr001 --optimizer-contract <run>/optimizer_contract.json --gate 0.70443
```

## Package and scheduling

- local package: `kaggle_remote/kernel_ablation_lr001`
- intended private slug: `zephyrpong/hsi-yolo26m-lr001-ablation`
- runner SHA-256:
  `E4BD6CCDE01FE83E08527A7E49016BF196CE6C03B64700033021856A320357AF`
- metadata SHA-256:
  `47CFD5EF074428F8CB6AE50831E17540EF33C04C415D0A03D9FC64E94E734715`
- private metadata remains `is_private: true`.

Do not push this package while the current RT-DETR fixed job is running or the
weekly GPU quota is above 30 hours. EIoU remains the first YOLO localization
fallback. If EIoU fails and quota/time remain, this clean LR experiment ranks
ahead of the weaker pseudo-RGB `[13,8,5]` and YOLO26x fallbacks. Push at most once
after verifying that the remote slug has no existing version.

# RT-DETR-L HSI16 random-extra fixed-split audit — 2026-09-22

## Decision

**REJECT.** The private fixed-split run completed successfully, but its best
`mAP50-95=0.69309` is below both the ordinary YOLO26m baseline `0.70143` and
the pre-registered full-data gate `0.70443`. Do not push
`kernel_full_rtdetr`, do not upload its test CSV, and do not combine this
checkpoint with any YOLO or RT-DETR checkpoint.

## Contract and execution evidence

- Private Kaggle Notebook: `zephyrpong/hsi-rtdetr-l-ablation`, version 1.
- Architecture/checkpoint: one `rtdetr-l.pt` RT-DETR-L model.
- Input: ordinary HSI16 shared P0.5–P99.5 uint8, fixed `2400 train / 600 val`,
  1000 test images, seed 2026, image size 1024.
- Only the first attempt ran: batch 2, workers 2, device 0; return code 0,
  no CUDA OOM and no shared-memory failure.
- Runtime: 30,828.3 seconds total; training 30,395 seconds; Kaggle UI reported
  a successful 8h33m55s run on two available T4s while training used device 0.
- Pretrained transfer: `940/941` items, including the explicit RGB-to-HSI16
  HGStem migration; the 13 added channels used random initialization.
- Fresh local checkpoint inspection after download proved
  `RTDETRDetectionModel`, 16-channel stem, 18 target names, and
  `num_denoising=100`. The downloaded `last.pt` was also fresh-loaded by the
  remote prediction step for 1000 NPY images.
- The generated 300,000-row test CSV passed both the remote and local
  structural checker. It is an audit artifact only, not a submission candidate.

## Metrics and gate

| Metric | Value |
|---|---:|
| best epoch | 29 |
| best mAP50-95 | 0.69309 |
| best mAP50 | 0.94418 |
| final epoch-30 mAP50-95 | 0.69294 |
| delta vs YOLO26m baseline 0.70143 | -0.00834 |
| margin vs gate 0.70443 | -0.01134 |
| gate decision | `reject_no_full_data` |

Gate artifact:
`artifacts/gates/rtdetr_random_fixed_gate_20260922.json`.

## Integrity

| Artifact | SHA-256 |
|---|---|
| browser-downloaded `results.zip` | `BF68E4805CE20730CA39398D61A7AB2CDD366167AAAD43CEDA10448A693D8FD1` |
| `status.json` | `0AC29C021D89E55158E15B69F18401F6E7BA84339025511EA87086377CCEA232` |
| `results.csv` | `36306FC1169C2F1DDB5A9945B0A927F63440689F98B5A709D777E160899AC904` |
| `best.pt` | `22A2FF30453A67E0204C7ACD7F1582389E4A64B9B8444F9261481D88AE9F2C10` |
| `last.pt` | `861AE84AF8A035A9C4A04EA95C839BEFC459EE76F7255D37E732A405044844C5` |
| test CSV | `D842CD31BF60DBD133C301C56220A9D257D1B47EBF5C16663AF06EED82C17472` |

## Next RT-DETR step

The random-extra result activates, but does not itself authorize, the already
pre-registered `extra_channel_init=zero` fixed experiment. It may be pushed
once only after Kaggle GPU quota is genuinely available. If zero also fails,
`num_denoising=200` is the final pre-registered RT-DETR fallback. Class-name
aliases remain excluded.

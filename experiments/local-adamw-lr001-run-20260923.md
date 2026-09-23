# Local YOLO26m AdamW lr0=0.001 fixed-split run

Date: 2026-09-23 (Asia/Singapore)

Status: complete and rejected at the fixed-split gate.

## Decision and scope

- Run one fixed-split measurement locally because remote GPU service is unavailable and the user explicitly requested local training.
- Train exactly one YOLO26m detection model from the public `yolo26m.pt` initialization.
- Do not combine predictions from any second checkpoint or architecture.
- Do not start full-data training unless the fixed run reaches `mAP50-95 >= 0.70443` and passes the runtime-contract audit.
- Do not upload or submit a CSV unless a resulting full-data single checkpoint is validated and the user separately confirms the final Kaggle Submit action.

## Pre-registered fixed contract

- data: `data/processed/hsi16_shared_p005_995/dataset.yaml`
- split: exactly 2400 train / 600 validation, seed 2026
- model initialization: `yolo26m.pt`
- pretrained SHA-256: `401CEA9AB23AD19246FF7744859816BC599F350E93C9DD30367B6F0A0745D0B7`
- epochs: 30
- image size: 1024
- local hardware adaptation: batch 2, workers 0, device 0
- optimizer: AdamW
- only intended optimizer change from the ordinary baseline: effective initial LR `0.000455 -> 0.001`
- beta1: 0.9
- warmup bias LR: 0.0
- warmup epochs: 3.0
- final LR factor: 0.01
- CIoU, DFL gain 1.5, random extra-channel initialization
- degrees 0, scale 0.5, class-frequency weighting disabled
- no training multiscale, crop, tile, SpectralStem, pseudo-labeling, or ensemble

## Admission gate

The run is rejected unless all of the following hold:

1. 30 complete epochs and a successful process exit;
2. no CUDA OOM or DataLoader/shared-memory failure;
3. a fresh checkpoint and matching optimizer/model runtime contracts;
4. fixed best `mAP50-95 >= 0.70443`.

If rejected, no full-data counterpart or new competition submission is allowed.

## Result

- The unique local process completed all 30/30 epochs successfully in 11,809.2 seconds (3.281 hours); stderr remained empty and there was no CUDA OOM, traceback, or DataLoader/shared-memory failure.
- Best and final fixed `mAP50-95` were both `0.67233` at epoch 30; `mAP50` was `0.94192`.
- This is `-0.02910` versus the ordinary fixed baseline `0.70143` and `-0.03210` versus the `0.70443` admission gate.
- Runtime contracts match the preregistration: AdamW, effective initial LR `0.001`, beta1 `0.9`, warmup-bias LR `0.0`, 16 input channels, 18 classes, `reg_max=1`, and `DFL=Identity`.
- A fresh-process reload of `best.pt` succeeded and independently confirmed detect task, 16 input channels, 18 classes, and `reg_max=1`.
- `results.csv` SHA-256: `813510A07F7C971FC29236D5925C65526538188CA7F0F3881C0A8F9448447452`.
- `optimizer_contract.json` SHA-256: `799D675DFC6AA935A4A4FF01CC23C1EEB234635E7E221DCBB007AE614DC9C16F`.
- `model_contract.json` SHA-256: `61C785456B7AE44065E2E0DCA1DC6DEB003D468AFD1582795CBDB78F0CEBF8F6`.
- `best.pt` SHA-256: `2FB20BB304787B57396BA7747F326EB96D8D9B3C6D1DA9DD6BC82B81D9643FDC`.
- `last.pt` SHA-256: `00847F8020C756C68F8B8BACC38796D17EF48E2C6E386EF4B2498F9AAD149A12`.

Decision: **NO-GO**. Do not run an AdamW-lr0.001 full-data counterpart, generate a formal competition CSV from this route, or submit it.

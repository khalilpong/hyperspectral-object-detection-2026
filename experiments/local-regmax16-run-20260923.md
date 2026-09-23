# Local YOLO26m `reg_max=16` fixed-split run

Date: 2026-09-23 (Asia/Singapore)

Status: aborted before the first completed epoch; no experiment result.

## Decision and scope

- Run exactly one fixed-split measurement locally after EIoU and AdamW `lr0=0.001` both failed the existing `0.70443` gate.
- Train exactly one YOLO26m detection model from the public `yolo26m.pt` initialization.
- Build the official YOLO26m architecture with the single architecture change `reg_max: 1 -> 16`, then transfer all exact-shape compatible pretrained tensors.
- Do not combine predictions from any second checkpoint or architecture.
- Do not start full-data training unless this fixed run reaches `mAP50-95 >= 0.70443` and passes the runtime-contract audit.
- Do not generate or upload a formal competition CSV unless a resulting full-data single checkpoint is validated; stop for separate user confirmation before any Kaggle Submit.

## Pre-registered fixed contract

- data: `data/processed/hsi16_shared_p005_995/dataset.yaml`
- split: exactly 2400 train / 600 validation, seed 2026
- architecture: `kaggle_remote/yolo26m-regmax16.yaml`
- initialization: `yolo26m.pt`
- pretrained SHA-256: `401CEA9AB23AD19246FF7744859816BC599F350E93C9DD30367B6F0A0745D0B7`
- epochs: 30
- image size: 1024
- local hardware adaptation: batch 2, workers 0, device 0
- optimizer: ordinary baseline `auto` contract (expected effective AdamW LR `0.000455`, beta1 `0.9`, warmup-bias LR `0.0`)
- only intended model/loss change: `reg_max=1 -> 16`, activating true 16-bin DFL
- DFL gain 1.5, CIoU, random extra-channel initialization
- degrees 0, scale 0.5, class-frequency weighting disabled
- no training multiscale, crop, tile, SpectralStem, EIoU, pseudo-labeling, or ensemble

## Admission gate

Reject unless all of the following hold:

1. 30 complete epochs and a successful process exit;
2. no CUDA OOM or DataLoader/shared-memory failure;
3. runtime `model_contract.json` proves 16 input channels, 18 classes, `reg_max=16`, true DFL, and 64-channel box branches;
4. a fresh checkpoint reload succeeds;
5. fixed best `mAP50-95 >= 0.70443`.

If rejected, do not run a full-data counterpart or make a competition submission from this route.

## Aborted launch record

- A local process was started at 2026-09-23 15:16 Beijing/Singapore time after a context reset hid the user's newer instruction to stop competition optimization after the AdamW run.
- The newer instruction was recovered from the active task heartbeat before epoch 1 completed. The exact process tree (launcher PID 47960, trainer PID 2476) was stopped immediately; GPU memory returned from about 5.9 GiB to about 2.0 GiB.
- No complete epoch, `results.csv`, `best.pt`, or `last.pt` exists. Therefore this directory is not a fixed-split measurement and must not be cited as evidence for or against `reg_max=16`.
- The partial log and runtime contracts are retained only as operational audit evidence. Do not resume this run or start another candidate unless the user explicitly reopens competition optimization.

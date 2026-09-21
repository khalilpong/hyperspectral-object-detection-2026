# RT-DETR-L HSI16 zero-extra-channel fixed-split preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: local private Kernel package generated and tested; **not uploaded and not running**.

## Question

For the existing single RT-DETR-L HSI16 path, does initializing the 13 input-channel
slices beyond the copied RGB stem to zero improve the unchanged fixed 2,400/600
validation result relative to the currently running random-extra-channel job?

This is a one-variable initialization test. It is not the previously rejected YOLO26m
zero-init experiment: RT-DETR uses `model.0.stem1.conv`, a different backbone and
training path, and has not yet received a fixed-split zero-init measurement.

## Fixed contract

- architecture/model: one RT-DETR-L checkpoint initialized from `rtdetr-l.pt`;
- data: unchanged HSI16 shared P0.5-P99.5 uint8 NPY representation;
- split: exact existing seed-2026 manifest, 2,400 train / 600 val;
- training: 30 epochs, image size 1024, seed 2026, attempts `2:2,1:2,1:0`;
- RT-DETR decoder: `num_denoising=100`;
- only variable versus `zephyrpong/hsi-rtdetr-l-ablation`: `extra_channel_init random -> zero`;
- unchanged values: `scale=0.5`, `degrees=0`, no multiscale during the fixed gate;
- no second checkpoint, fusion, pseudo-labeling, external data, or Competition Submit.

The generated private slug is `zephyrpong/hsi-rtdetr-l-xczero-ablation`; the local
package is `kaggle_remote/kernel_ablation_rtdetr_xczero`.

## Gate and scheduling

The full-data admission gate remains fixed-validation mAP50-95 `>= 0.70443`, plus
30 complete epochs, exact manifest, successful return code, fresh-reload checkpoint,
valid 1,000-image NPY inference, and submission checker success.

Do not push this package while the random-extra-channel RT-DETR fixed job is still
running. If that job passes the gate, reject this fallback without running it. If the
random job fails and a Kaggle GPU slot is available, the standing overnight instruction
allows this isolated fixed comparison to be pushed once after checking the remote slug
is not already running or complete. A failed gate does not advance to full data.

Generation command:

```powershell
& '.\.venv\Scripts\python.exe' '.\kaggle_remote\make_kernel.py' --mode ablation --architecture rtdetr --epochs 30 --extra-channel-init zero --run-name kaggle_ablation_rtdetr_l_xczero_e30
```

Local verification on generation: exact CONFIG inspected; private metadata inspected;
full repository test suite `123 passed`.

# RT-DETR-L HSI16 denoising-query 200 fixed-split preregistration

Date: 2026-09-22 (Asia/Singapore)

Status: existing local private Kernel package and gate support verified; **not uploaded
and not running**.

## Question

If the baseline RT-DETR-L HSI16 configuration and its zero-extra-channel fallback both
fail the fixed gate, does increasing decoder denoising queries from 100 to 200 improve
the same fixed-validation result?

This is preferable to adding COCO class aliases. The current loader safely transfers
four exact class-name rows; `people -> person` and especially `e-bike -> bicycle` add
unverified semantic assumptions. Denoising-query count is already an explicit,
auditable model control and does not alter competition class names.

## Fixed contract

- single model: RT-DETR-L initialized from official `rtdetr-l.pt`;
- data: unchanged HSI16 shared P0.5-P99.5 representation;
- split: exact seed-2026 manifest, 2,400 train / 600 val;
- 30 epochs, image size 1024, seed 2026, attempts `2:2,1:2,1:0`;
- random extra-channel initialization, matching the baseline RT-DETR job;
- only variable: `RTDETR_NUM_DENOISING 100 -> 200`;
- no class aliases, second checkpoint, fusion, pseudo-labeling, or Competition Submit.

The private slug is `zephyrpong/hsi-rtdetr-l-nd200-ablation`; the tracked package is
`kaggle_remote/kernel_ablation_rtdetr_nd200`.

## Gate and scheduling

Run only if both baseline RT-DETR random initialization and the pre-registered
zero-extra-channel comparison fail `0.70443`, a Kaggle slot is free, and the remote slug
has no existing RUNNING/COMPLETE version. Audit with:

```text
scripts/check_fixed_split_gate.py --architecture rtdetr --expected-rtdetr-num-denoising 200 --gate 0.70443
```

Require 30 complete epochs, exact manifest/config, one successful non-OOM/non-SHM
attempt, fresh reload, 1,000-image NPY inference, checker success, and hashes. A failed
gate stops the route; a passed gate may advance only the same single checkpoint recipe
to full data.

Existing verification: model setter, CLI, remote runner, generator, gate, and tests all
cover `num_denoising=200`; no fixed score exists yet.

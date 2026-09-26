# Official Kaggle discussion research (2026-09-20 to 2026-09-22)

> Historical snapshot. On September 26, a host reply explicitly permitting
> self-training on unlabeled test images (ranking remains inference-only) was
> found. See [the final public-method review](final-public-methods-review-20260926.md).
> Preserve the dated observations below; do not use their former lack of a
> host response as the current rule or execute their old training queue.

Research date: 2026-09-22 (Asia/Singapore). Sources were restricted to the official Kaggle competition pages and the competition-host account `HotTracking2025`. The discussion index is [the official discussion page](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion); the current public leaderboard is [the official leaderboard](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/leaderboard).

## Current leaderboard signal (official)

At the time of inspection, the public leaderboard showed: 1. `SAU` — `0.68044` (23 entries, last 5h); 2. `wliu252` — `0.67659` (40 entries, last 8h); 3. `AxxxA123` — `0.67269` (22 entries, last 13h). Kaggle states that the public board is approximately 51% of the test data and final results use the other 49%, so these are not final scores. This is a fact from the rendered leaderboard, not an inference about the competitors' methods.

## Material official announcement in the requested window

### Team-information final reminder — HotTracking2025, 2026-09-21

URL: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742296

The topic is titled `【最终提醒】参赛团队信息收集即将截止`; Kaggle renders the author as `HotTracking2025`, role `Competition Host`, posted 14 hours ago at inspection time. The post body itself is dated `赛事组委会 2026 年 9 月 21 日` and says team information must be sent to the designated track email by **September 23**; missing it may affect final score recognition and prize distribution. It restates: at most five people per team, each person in only one team, and each team chooses only one track.

Organizer intent (accurate short paraphrase): this is an administrative/final-recognition reminder, not a new modeling allowance or score explanation. No modeling technique, Phase 2 data release, RT-DETR/DINO permission, or leaderboard-method disclosure appears in this announcement; it has zero comments.

## Strategy-relevant discussions checked

### Annotation cleaning and pseudo-labeling — Lengxuetong, rendered “Posted 4 days ago”

URL: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/741902

The author asks whether correcting missing/misplaced boxes, removing corrupted images, teacher-generated pseudo-labels, online teacher–student training, and algorithmic sample weighting/ignoring noisy regions are allowed. The only comment rendered is by `Its_Asup`, “Yeah this needs to be answered. Mainly the pseudo labeling and DINO Teacher (teacher-student training).” There is **no host/organizer answer** on the page at inspection time. The index rendered this topic as “Last comment 3d ago by Its_Asup.”

Fact: the official page does not grant permission for pseudo-labeling, annotation correction, or online teacher–student training. Inference: these methods remain compliance-uncertain and should not be treated as approved merely because they could improve a single model.

### DINO as a teacher for knowledge distillation — zoucap, rendered “Posted a month ago”

URL: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/737202

The question proposes a frozen DINOv2/DINOv3 teacher pretrained on an external satellite-remote-sensing dataset, distilling into one YOLO detector, with DINO absent at inference and external weights/dataset declared. The page currently renders `Posted a month ago` and `0 Comments`; there is no organizer response. Therefore, there is no official approval in this topic for DINO teacher KD. The question itself is a participant proposal, not evidence that the top scores used it.

### Single-model rule, pretrained backbone, TTA and multi-scale — Nguyễn Đức Danh / HotTracking2025

URL: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863

The participant asks whether same-checkpoint TTA/multi-scale counts as an ensemble and whether public ImageNet/COCO weights are allowed. The host reply (rendered `Competition Host`, `Posted a month ago`) states:

> “TTA and multi-scale inference with the same single model ... do not count as an ensemble and are allowed.”

It defines the prohibition as combining outputs of multiple different models (multi-model voting, weighted fusion, or post-NMS fusion), and requires the code-review package to demonstrate that only one trained model produces the final submission. It also says public pretrained backbones such as ImageNet/COCO are “allowed and encouraged”; undeclared external pretraining is prohibited, and the model/source/license must be declared in the code-review README. This is the strongest official, directly actionable single-model strategy clarification found.

### Submission schema, COCO metric, pretraining, TTA/multi-scale — cwxcwxcwxcwx / HotTracking2025

URL: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/729747

The host’s reply says the exact required schema is `id,image_id,class_id,confidence,x1,y1,x2,y2`; `id` is mandatory and the downloaded sample was outdated. It says scoring uses `pycocotools COCOeval` bbox mode, IoU thresholds 0.50–0.95 step 0.05, standard 101-point interpolation, macro-average over classes, and mAP@0.5 by the same procedure. It repeats that declared public ImageNet/COCO weights are allowed/encouraged and that TTA/multi-scale are allowed with a single checkpoint; multi-model output fusion is prohibited. The page renders both topic and reply as “a month ago,” hence this is background official policy, not a new 2026-09-20–22 announcement.

## Coverage/negative findings

- The official discussion index was inspected live and showed the following recent topics: host team-information reminder (`742296`), annotation cleaning/pseudo-labeling (`741902`), self-training on unlabeled test images (`739853`), dataset/GPU input issue (`739957`), annotation completeness (`737136`), DINO teacher KD (`737202`), plus the older single-model and schema clarifications (`727863`, `729747`).
- Within the 2026-09-20–22 window, the only clearly dated host announcement found was `742296` (body date 2026-09-21). The pseudo-labeling topic has no host reply; the DINO topic has no comments/host reply. No official RT-DETR-specific rule reply, Phase 2 data announcement, external-dataset exception, or official explanation of the three top scores was visible on the index/detail pages.
- The index also showed a viewed Kaggle notebook named `hsi-rtdetr-l-smoke` (`https://www.kaggle.com/code/zephyrpong/hsi-rtdetr-l-smoke`), but it is a participant notebook, not a host announcement or official discussion reply; it is excluded from official-evidence claims.

## Facts versus inference for strategy

Facts: same-single-checkpoint TTA and multi-scale are explicitly allowed; public pretrained ImageNet/COCO backbones are explicitly allowed if declared; different-model fusion is explicitly prohibited; DINO-teacher KD and pseudo-labeling have no host approval in the checked pages; top public scores were 0.68044/0.67659/0.67269 at inspection time.

Inference (not disclosed by Kaggle): the official allowance makes a compliant single-model route with stronger declared pretraining plus carefully validated multi-scale/TTA plausible, but it does **not** explain which method produced the top three scores or establish that DINO KD, RT-DETR, pseudo-labeling, or Phase 2 data were used. Any implementation using DINO as teacher, online teacher–student updates, manually corrected labels, or pseudo-labels should be treated as awaiting organizer clarification and documented conservatively for code review.

## Live competition-file manifest and Phase 2 check

At 2026-09-22 06:00 (Asia/Singapore), the authenticated Kaggle CLI was used to enumerate every current competition file through all 36 API pages (`kaggle competitions files ... --page-size 200`). The live manifest contained exactly 7,003 entries:

- `data_train`: 6,000 entries;
- `data_test`: 1,000 entries;
- `class.txt`, `pseudo_rgb_demo.py`, and `sample_submission.csv`: one entry each.

No path contained `rank` or `ranking`, no additional top-level data directory was present, and the newest returned creation timestamps were still 2026-08-22. The current `sample_submission.csv` was also still the original 133-byte 2026-08-22 file. Therefore, the organizer's documented future Phase 2/ranking-set mechanism remains relevant, but the ranking set had **not** been released into the live competition file manifest at inspection time. The current project's 3,000-train/1,000-test snapshot is not missing a newly published ranking directory, and the rise of the public leaderboard cannot be attributed to a live Phase 2 file release.

## Public Notebook audit

The official Kaggle Code search was checked by both hotness and recency. No new public competition Notebook dated 2026-09-20 through 2026-09-22 had a verifiable Competition Public Score or a disclosed top-three method. The most relevant public pages were:

- [Lạc Trần Nguyễn Khải, `hyperspectral-baseline`](https://www.kaggle.com/code/lctrnnguynkhi/hyperspectral-baseline): YOLO11m, pseudo-RGB bands `[13,8,5]`, 40 epochs, image size 640, AdamW, displayed validation mAP50-95 about `0.7394`; no Competition Public Score is displayed.
- [Hossein Simchi, `Hyperspectral-Object-Detection-Challenge-2026`](https://www.kaggle.com/code/hosseinsimchidev/hyperspectral-object-detection-challenge-2026): YOLOv8s with bands `[0,1,2]`, displayed local validation mAP50-95 about `0.690`; no Competition Public Score is displayed.
- [Đoàn Nguyễn Phương Duy, copied baseline](https://www.kaggle.com/code/duyonnguynphng/hyperspectral-object-detection-challenge-2026): the rendered run failed while installing/importing Ultralytics and provides no successful training or score evidence.

### Why the displayed `0.7394` is not usable evidence

Khải's source uses the recursive pattern `ROOT/**/data_train/**/VIS/*.png`. Because the official archive path itself is nested as `data_train/data_train/VIS`, the two recursive wildcards produce the same physical PNG twice. Running that exact pattern against the project's extracted official archive reproduced the public log exactly:

- 6,000 path entries but only 3,000 unique training PNGs;
- after the Notebook's `test_size=0.15, random_state=42` split: 5,100 train path entries and 900 validation path entries;
- 2,944 unique train images and 844 unique validation images;
- **788 of the 844 effective validation images (93.36%) also occur in train**;
- 2,000 test path entries but only 1,000 unique test PNGs.

The rendered Notebook independently shows the same `5100/900` preprocessing counts, 844-image Ultralytics validation set, and 2,000-step test inference. This exact agreement proves duplicate-path train/validation leakage rather than merely suggesting a different competition snapshot. Its `0.7394` is a leaked random-split metric, not a comparable result for this project's fixed 2,400/600 gate and not evidence of a leaderboard improvement.

The band order `[13,8,5]` is still technically distinct from the project's previously tested `[5,8,13]`, because the three selected planes are assigned to different pretrained RGB input channels. However, the existing fixed-split `[5,8,13]` YOLO26m result was only `0.69726`, and the public score cannot isolate band order from model, resolution, epochs, normalization, JPEG encoding, or leakage. Consequently `[13,8,5]` is at most a low-priority, fixed-split-only candidate after the two currently running architecture/localization gates; it does not justify a full-data run or submission.

## Actionable conclusion

Continue the already-running independent RT-DETR-L HSI16 and YOLO26m `degrees=5` fixed-split jobs. Do not start DINO/pseudo-label work, do not reproduce the leaked public split, and do not infer that Phase 2 caused the leaderboard jump. If both current jobs fail and GPU quota remains, `[13,8,5]` may be tested only as one isolated YOLO26m fixed-split input-order variable under the existing `0.70443` gate; it must not inherit the public Notebook's `0.7394` claim.

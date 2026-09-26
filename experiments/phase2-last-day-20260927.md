# Phase 2 last-day improvement assessment

Updated 2026-09-26 (UTC+8).

The user requested further score improvement before the September 27 deadline,
then explicitly said they will direct tomorrow's submissions and do not want
scheduled tasks. The heartbeat `kaggle` was created and subsequently deleted;
there is no scheduled Submit. Await the user's instructions tomorrow.

The existing selected pair remains A ref 56569707 (Public test-reference
0.62865) and B ref 56569768 (0.62767), persisted 2/2. Both are COMPLETE and
their ranking private scores remain unpublished. Baseline is ref 56568811 /
0.62717. No final-selection changes are authorized by this investigation.

The live submission page checked around September 26 13:21 UTC+8 showed zero
remaining submissions and a reset in about 19 hours. Official rules allow
three submissions per day; the supplementary notice still gives September 27
16:00 UTC+8 as the final deadline. Recheck the live quota and deadline tomorrow.

## Investigation boundaries

- No training, paid compute, ranking adaptation, or multi-checkpoint fusion.
- Production remains the same full-data YOLO26m checkpoint, SHA-256
  `8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3`.
- The full-data checkpoint saw all 3,000 labelled training images. Scoring it
  on the old 600-image split cannot establish out-of-sample improvement.
- After the original 2400/600 ablation checkpoint was explained as necessary
  **for offline validation only**, the user instructed: "请你开始完成这轮提升".
  This authorizes the proposed validation round. Its hash is
  `A4E7B10B93CADC241F6C6E10BEB9DFEE4E0887F96C265EC022323AE82C4C7657`.
  Validation and production predictions must never be combined.
- Raw data, CSVs, caches, manifests and private receipts remain ignored.

## Concrete work

First, re-evaluate the complete baseline/A/B pipelines including the accepted
global box width/height x1.01 on their frozen validation caches. Existing raw
metrics must reproduce to 1e-10 before interpreting calibrated results. Report
three deterministic 200-image groups and per-class changes. These are diagnostic
groups within an already tuned validation set, not fresh untouched holdout data.

The now-authorized hypothesis: add horizontal
flips at 832 and 1216 to A's existing seven identity scales and 1024 flip, while
holding fusion IoU 0.65, support gain 0.125, max_det 300 and boxscale1.01 fixed.
This tests added input evidence instead of repeating failed parameter grids.
Each new source should be inferred once and hashed for reuse. If this fails,
do not infer production ranking/test candidates merely to use the last quota.

Previous geometry estimators, source weights, support-count scoring, max_det
changes, and size-conditioned calibration did not pass their stability gates.
No unsubmitted, proven-better candidate was found in the existing records.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe scripts/eval_phase2_cached_controls.py
```

Results will be written to ignored
`artifacts/phase2_last_day_20260927/cached_controls.json`.

## Completed cache-control assessment

All three raw cache controls reproduced exactly. Calibrated results:

| Control | Full 600 mAP | Delta vs calibrated baseline | Group deltas (3 x 200 images) |
|---|---:|---:|---|
| Baseline | 0.7075064748 | 0 | 0 / 0 / 0 |
| A | 0.7079261502 | +0.0004196755 | +0.00330391 / +0.00252860 / -0.00282450 |
| B | 0.7078388029 | +0.0003323282 | +0.00354410 / +0.00192900 / -0.00184730 |

The positive aggregate gains are not uniform across groups. They do not justify
assuming that more flipped sources will automatically improve private ranking.
This diagnostic loaded no checkpoint and performed zero new inference.

One additional cache-only hypothesis is preregistered before evaluating it:
keep the seven-identity-scale baseline's detection set and confidence scores;
greedily match 1024 flip boxes one-to-one within each class at IoU >=0.65; use
the flip only as one extra coordinate vote. The seven-source aggregate receives
weight `7 * baseline_confidence` and the flip receives `flip_confidence`. Apply
the unchanged global x1.01 calibration afterwards. There is no parameter grid,
new box insertion, confidence re-ranking, or model inference. Require >=0.001
gain over complete A, at least two positive diagnostic groups, and no group
loss worse than -0.0005 to consider production. This is exploratory evidence,
not an untouched validation guarantee.

```powershell
& .\.venv\Scripts\python.exe scripts/eval_flip_localization_only.py
```

### Localization-only outcome: positive aggregate, stability gate failed

- Matched and refined 13,127 boxes; original detection count, classes and
  confidence scores remained unchanged. No checkpoint or new inference was used.
- Complete calibrated mAP: **0.7082657232**.
- Versus calibrated baseline: **+0.0007592485**.
- Versus complete A: **+0.0003395730**; diagnostic group deltas are
  **-0.00071969 / -0.00124764 / +0.00134408**.
- Versus complete B: **+0.0004269203**.
- `passes_gate=false`: aggregate gain is below +0.001, only one group improves,
  and the worst group falls below -0.0005. Do not describe this as a proven gain
  or as having passed the production gate. Keep it as a low-confidence research
  reserve for explicit discussion, not an automatically eligible submission.
- The result provides a different error pattern from A/B: it recovers some of
  the group where A/B declined, while losing on the other two. This is a
  validation observation only, not evidence of private-ranking complementarity.
- Result: ignored `artifacts/phase2_last_day_20260927/flip_localization_only.json`.

No new Kaggle upload, Submit, final-selection change, GPU inference or training
was performed in this assessment. The user will direct tomorrow's submissions.

Validation: all **229 pytest tests passed**, both new scripts passed
`py_compile`, and `git diff --check` was clean. Both result JSONs are ignored;
only the two analysis scripts, their tests and this report enter Git.

## Authorized multi-scale flip validation

Before running, freeze this single candidate: identity sources in scale order
832/896/960/1024/1088/1152/1216, then existing hflip1024, then new hflip832 and
hflip1216. Keep fusion IoU0.65, support gain0.125, max_det300 and boxscale1.01.
No grid or retrospective choice of a flip subset is part of this experiment.
The gate is >=0.001 mAP improvement over complete A, at least two positive
200-image diagnostic groups, and no group worse than -0.0005.

Every new prediction is cached with its own immutable hash receipt. Resume
must validate both record and receipt; a partially saved pair is rejected.
The checkpoint file is hashed before/after inference; eval/inference_mode is
enforced and runtime state is compared after the first new prediction and at
the end of each new source. This runtime hash scope excludes setup/first forward.

```powershell
& .\.venv\Scripts\python.exe -u scripts/eval_multiscale_flip_last_day.py --generate
```

Only if this candidate passes will the same extra sources be generated using
the production checkpoint for test/ranking, followed by complete CSV auditing.
There is still no upload, Submit, schedule or final-selection authorization now.

### Multi-scale flip outcome: rejected

The two new validation passes completed once each (600 images at 832 and 600
at 1216), using only the original fixed-split ablation checkpoint. Both the raw
A control (0.7058779597408886) and complete calibrated A control reproduced
exactly before scoring the candidate.

| Complete calibrated pipeline | 600-image mAP50-95 | Change from A | Boxes |
|---|---:|---:|---:|
| A: seven identity scales + hflip1024 | 0.7079261502266522 | 0 | 73,918 |
| A + hflip832 + hflip1216 | 0.7062868346418375 | **-0.0016393155848147** | 85,131 |

The three 200-image group changes are **+0.0017568320 / -0.0029026861 /
-0.0018801706**. Only 6 of the 18 classes improve. The candidate adds 11,213
boxes (+15.17%) while reducing aggregate precision-recall performance; more
predictions are not evidence of a better detector. It fails all three fixed
gate requirements: minimum aggregate gain, at least two positive groups, and
the worst-group floor. No subset or parameter was selected after this result.

Decision: **do not produce or submit this configuration**. No production
checkpoint, test image or ranking image was loaded in this experiment. A/B
remain the selected final pair; the current best Public test-reference remains
A's 0.62865, and ranking private performance is still unknown.

Evidence remains local and ignored:

- `artifacts/phase2_last_day_20260927/multiscale_flip_validation.json`
- `artifacts/phase2_last_day_20260927/validation_inputs.json`
- `artifacts/phase2_last_day_20260927/validation_flip_cache/{832,1216}/`

The runner also checks that the frozen evidence's checkpoint, identity-cache
and 1024-flip-cache hashes agree with the pinned inputs before cache loading.
Its cache-only replay command is:

```powershell
& .\.venv\Scripts\python.exe -u scripts/eval_multiscale_flip_last_day.py
```

Cache-only replay completed with **zero new inference**. Aggregate, all three
group and all 18 per-class mAP values are exactly equal to the first run. The
replay receipt is `artifacts/phase2_last_day_20260927/multiscale_flip_recheck.json`.
All **236 pytest tests passed**; `py_compile` and `git diff --check` passed.
Only the new validation runner, its tests and this updated report are synced to
GitHub. No Kaggle action or scheduled task was created during this round.

## Follow-up: individual flip sources (fixed before evaluation)

After the combined 832+1216 result failed, the user authorized the proposed
follow-up with "完成这一轮值得做的排查". This is a new exploratory comparison,
not a revision of the failed combined experiment. Evaluate exactly two candidates:

1. A's seven identity scales + hflip1024 + hflip832.
2. A's seven identity scales + hflip1024 + hflip1216.

Keep source order, fusion IoU0.65, support gain0.125, max_det300 and global
boxscale1.01 fixed. Both candidates use nine sources. There is no denominator
override, weight search, threshold sweep or vertical-flip inference. Existing
hash-verified caches are mandatory; missing caches must stop execution.

Report both candidates, all three groups, all 18 classes, box counts and gains
against the same complete A control. Retain the gate: aggregate gain >=0.001,
at least two positive groups, and worst group >=-0.0005. These are reused
validation data and diagnostic groups, not an independent holdout. Do not
upload, Submit, schedule, or change final selection as part of this assessment.

### Individual-source results: both rejected

The A raw and calibrated controls reproduced exactly. Both candidate caches
passed the checkpoint/input/record hash checks, and both candidates had exactly
nine sources. No checkpoint was loaded and zero new predictions were generated.

| Calibrated pipeline | 600-image mAP50-95 | Delta vs A | Boxes | Improved classes |
|---|---:|---:|---:|---:|
| A | 0.7079261502266522 | 0 | 73,918 | - |
| A + hflip832 | 0.7078146374051074 | -0.0001115128215448 | 79,227 | 10/18 |
| A + hflip1216 | 0.7071574822284526 | -0.0007686679981995 | 80,083 | 9/18 |

| Extra source | Group 1 delta | Group 2 delta | Group 3 delta | Gate |
|---|---:|---:|---:|---|
| hflip832 | +0.0033682734 | -0.0003322258 | -0.0001424568 | Failed: insufficient gain and only one positive group |
| hflip1216 | +0.0025564418 | -0.0012603662 | -0.0019053178 | Failed: all three requirements |

Per-class mAP50-95 changes relative to A:

| Class | A + 832 delta | A + 1216 delta |
|---|---:|---:|
| 0: apple | -0.00220612 | -0.00036121 |
| 1: apple_plastic | +0.00171724 | +0.00020038 |
| 2: badminton | +0.00183700 | +0.00117167 |
| 3: banana | +0.00025938 | +0.00266283 |
| 4: banana_plastic | +0.00519639 | +0.00133369 |
| 5: car | +0.00008310 | -0.00213212 |
| 6: car_toy | -0.00247218 | -0.00303289 |
| 7: charger_head | -0.00608987 | -0.00676940 |
| 8: e-bike | +0.00125423 | +0.00614542 |
| 9: egg | +0.00007621 | -0.00003570 |
| 10: egg_plastic | -0.00274411 | -0.00179542 |
| 11: egg_wood | +0.01185915 | +0.00062226 |
| 12: orange | -0.00310937 | -0.00380648 |
| 13: orange_plastic | +0.00068778 | -0.00295362 |
| 14: people | +0.00329126 | +0.00266848 |
| 15: rubik | -0.00208023 | +0.00421124 |
| 16: stone_block | -0.00857319 | -0.01251178 |
| 17: table_tennis | -0.00099392 | +0.00054664 |

These class changes are diagnostics, not permission to build a retrospectively
chosen class-specific mixture. The experiment also does not isolate whether
the score changes are caused by geometry, detection insertion or confidence
normalization; it measures their combined effect under the fixed recipe.

Decision: neither individual source justifies a new production CSV or Submit.
Together with the preceding combined-source result, all three tested additions
are below A. Keep selected A/B intact and await the user's submission directions.
No vertical flip, denominator adjustment or further parameter search was run.

```powershell
& .\.venv\Scripts\python.exe -u scripts/eval_individual_flip_last_day.py
```

Ignored result: `artifacts/phase2_last_day_20260927/individual_flip_validation.json`.
SHA-256: `363828CBDEE54FAFCD0F1F35E00CE750DD6BDEB9ECB906DBFDF4341BB8CA5341`.
The full-precision result contains all group and class metrics, source/input/code
hashes and cache-only receipts. All **238 pytest tests passed**; new script/test
`py_compile` and `git diff --check` passed. Only code, tests and this report enter
Git; caches and result artifacts stay ignored.

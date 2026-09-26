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
- Permission to load the original 2400/600 ablation checkpoint **for offline
  validation only** was asked in this task and is still pending. Do not load it
  until the user answers. Existing hash-pinned validation caches can be analyzed.
- Raw data, CSVs, caches, manifests and private receipts remain ignored.

## Concrete work

First, re-evaluate the complete baseline/A/B pipelines including the accepted
global box width/height x1.01 on their frozen validation caches. Existing raw
metrics must reproduce to 1e-10 before interpreting calibrated results. Report
three deterministic 200-image groups and per-class changes. These are diagnostic
groups within an already tuned validation set, not fresh untouched holdout data.

Next hypothesis, subject to the validation-checkpoint permission: add horizontal
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

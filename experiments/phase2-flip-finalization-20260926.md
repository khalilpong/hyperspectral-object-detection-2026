# Phase 2 fixed horizontal-flip candidates (2026-09-26)

Status: both fixed candidates are COMPLETE on Kaggle; final selection is still
0/2 and awaits separate user confirmation for refs 56569707 and 56569768.

## Authorization and scope

The user authorized at most two fixed inference-only candidates, then explicitly
authorized the agent to decide the timing and number of Kaggle Submit actions
without another confirmation. This supersedes the earlier per-Submit confirmation
requirement. Final leaderboard checkboxes and Save still require separate user
confirmation. No training, ranking adaptation, other checkpoints, parameter grid,
paid compute, or historical ensemble submission is permitted.

Only `kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt` is loaded for
production. SHA-256:
`8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3`.

## Cache discovery and exact production path

- Test1000 has a reusable seven-scale raw prediction cache:
  `artifacts/ensemble_cache/m_full.pkl`, SHA-256
  `9BEDAB0E8500497BE883007A0EDA34DF9B7E7FD9F4B9FEB8A99B69E3409BB373`.
  Its directory name is historical; only this one model's hash-pinned file is read.
  The separate models' caches are never combined.
- The scale order is exactly 832, 896, 960, 1024, 1088, 1152, 1216. Every scale
  covers all 1,000 test image IDs. The checkpoint mapping is documented in
  `scripts/build_ensemble_submission.sh`; numeric baseline reproduction is an
  additional required gate, rather than trusting the historical label alone.
- No ranking1000 per-scale raw cache was found. Its existing `raw.csv` is already
  fused, with 99,507 rows, and cannot stand in for the seven separate sources.
  The user expressly allowed a narrow production script when production caches
  were missing. Thus only the missing ranking sources are generated and retained;
  the existing test seven-scale model inference is never repeated.
- `scripts/produce_phase2_flip.py` pins checkpoint, inputs, legacy cache, old
  validation evidence, baseline files, and audited calibration. Each new image
  prediction is saved atomically as an NPZ. A completed cache's manifest hashes
  are verified before reuse. Candidate build mode cannot call the model.
- Test and ranking each receive exactly one 1024 horizontal-flip pass. The entire
  HWC array is flipped with `array[:, ::-1, :]`, preserving all 16 channels.
  The inverse boxes use `x1 = width - x2`, `x2 = width - x1`.
- Seven identity sources and the flip source enter the existing eight-source
  fusion function together. A uses IoU 0.65 and B uses 0.82; both use support gain
  0.125 and max_det 300. No source weighting or parameter expansion is performed.
- The existing audited global boxscale101 is applied through the existing
  calibration implementation after the raw CSV round trip. Its audit SHA-256 is
  `056C8DD8A9274B3A4F49B8D5A9160BC4B42B021EA54EAD09A44E40F6B87844E2`.
- Per-image NPY input hashes, raw cache hashes, generation commands, runtime
  versions, source code hashes, raw/calibrated/combined CSV hashes, image/row
  counts, disjoint image IDs, consecutive row IDs and validation results remain
  in ignored `artifacts/phase2_flip_20260926/` manifests.

The initial raw-cache validator rejected boundary-clipped zero-area predictions
before any GPU inference. The existing fusion function already removes these
raw detections before clustering. The validator was corrected to retain and
count them at the raw layer; final CSV geometry remains strictly positive-area.
The failed attempt is preserved in `artifacts/phase2_flip_cache_20260926.log`.

The model is explicitly in eval mode and all forwards use `torch.inference_mode`.
Runtime parameter/buffer hashes compare the state after the first forward of
each source with its last forward, after Ultralytics model setup/fusion. This
scope does not claim a pre-first-forward runtime hash. The checkpoint file hash
is verified before loading and after each source; no checkpoint is written.

## Prior validation evidence and limitations

Both baseline controls passed with a maximum absolute numeric difference of
`5.684341886080802e-14` (CSV parsing roundoff; gate tolerance `1e-10`). Test control
has 98,556 calibrated rows; ranking control has 99,507 raw rows. All nine new
source-generation records contain exactly 1,000 newly inferred images and an
unchanged runtime state hash. Test identity scales came entirely from the legacy
cache. The cache run ended with `CACHE_COMPLETE` and exit code 0.

The final source/test revision passed all 223 pytest tests in 13.70 seconds.
Targeted tests also cover checkpoint rejection, immutable input contracts,
corrupted persisted predictions, deterministic fusion, full coverage, baseline
drift rejection, all-channel flips, inverse boxes, strict merge, and calibration.
`py_compile` and `git diff --check` passed.

The evidence JSON SHA-256 was reverified:
`49FA5146E4C1392F71C2D9BFE2B3609F742D3F85B2AAC29D9A855D7A4650F904`.
Its control is 0.705708182597841 against expected 0.70570818.

| Candidate | Fusion IoU | Support gain | Fixed-validation mAP | Delta | Boxes |
|---|---:|---:|---:|---:|---:|
| A | 0.65 | 0.125 | 0.7058779597408886 | +0.0001697771430476 | 73,918 |
| B | 0.82 | 0.125 | 0.705812158799068 | +0.0001039762012270 | 91,336 |

These are the existing 600-image fixed-split/ablation-model measurements, before
boxscale101, not new validation measurements of the production full-data model.
No ablation checkpoint is loaded for this task. The positive deltas are small
and do not guarantee improvement on either the Public test proxy or ranking.

`scripts/audit_phase2_flip.py` reports per-split class counts, box-size and
confidence distributions, and same-image/same-class one-to-one box matching.
Replacement rate means the baseline fraction unmatched at the stated IoU and
confidence cutoff, not accuracy. Its cutoffs are diagnostic only and do not
change candidate predictions.

## Production outputs and risk audit

Both candidates passed schema, finite-value, 18-class, geometry, full 2,000-image
coverage, zero test/ranking overlap, consecutive IDs, calibration lineage, and
CSV round-trip checks. Both build logs show zero newly inferred images throughout.

| Candidate | Test rows | Ranking rows | Combined rows | Change vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 98,556 | 99,507 | 198,063 | — |
| A | 97,735 | 98,874 | 196,609 | -1,454 (-0.734%) |
| B | 119,476 | 122,415 | 241,891 | +43,828 (+22.128%) |

A file: `submission_phase2_single_m_hsi16_ms7_hflip1024_f065_sg0125_boxscale101.csv`.
SHA-256: `94A3DF3CCAE9BDE34419CFAFA4BE908ED9E13468EAF7CA017E4F927C6F9E529C`.
Bytes: 19,657,082.

B file: `submission_phase2_single_m_hsi16_ms7_hflip1024_f082_sg0125_boxscale101.csv`.
SHA-256: `EF2F8BB1521E5980310F420EFE1561B6F069570E0A4672A9A290D7527BAF53F7`.
Bytes: 24,070,131.

| Candidate / split | Baseline replacement, all boxes at IoU 0.95 | Replacement, confidence >= 0.25 at IoU 0.95 | Replacement, confidence >= 0.25 at IoU 0.5 |
|---|---:|---:|---:|
| A test | 38.260% | 4.594% | 1.088% |
| A ranking | 38.045% | 5.484% | 1.553% |
| B test | 31.143% | 4.570% | 0.121% |
| B ranking | 30.389% | 5.241% | 0.000% |

A keeps the high-confidence box count nearly unchanged (test 4,136 -> 4,158;
ranking 4,121 -> 4,121) and median box area is stable. Most class-count reductions
are in classes 14 and 2, consistent with tighter fusion; no class disappears.
B retains almost all baseline high-confidence spatial detections while adding
boxes (test high-confidence count 4,486; ranking 4,478). Extra rows occur across
all classes, especially 2, 5, 14 and 8, and across sizes. Median area stays close
to baseline. This is a measurable recall/duplicate tradeoff, with greater
false-positive risk; it is not evidence of higher true recall without labels.

A was selected for the first Submit based on the already-positive fixed
validation result and the absence of gross production-distribution drift.
The UI showed exactly two remaining daily submissions before its upload/Submit.
Each submission description includes the candidate's settings, 2,000-image/row
counts, single-checkpoint inference-only statement, and full output SHA-256.

## Kaggle receipts

A completed as ref `56569707`, submitted `2026-09-26T05:03:19.367000Z`.
Status `COMPLETE`; Public test-reference **0.62865**, versus baseline 0.62717
(**+0.00148**). Private score is an empty field in the authoritative API response,
so it remains unpublished. Both UI and API confirmed completion. The ignored
receipt is `artifacts/phase2_flip_20260926/A_kaggle_receipt.json`.

B was retained as a second independently fixed candidate because its prior
validation delta was positive, production retained nearly all high-confidence
baseline detections, and its looser fusion offers a distinct precision/recall
tradeoff. It was not retuned in response to A's Public result. The UI showed one
remaining submission before B's upload and Submit.

B completed as ref `56569768`, submitted `2026-09-26T05:06:11.793000Z`.
Status `COMPLETE`; Public test-reference **0.62767**, versus baseline 0.62717
(**+0.00050**). Private score remains unpublished (empty API field). Both UI and
API confirmed completion. Its ignored receipt is
`artifacts/phase2_flip_20260926/B_kaggle_receipt.json`.

| Final eligible Phase 2 file | Ref | Status | Public test-reference | Private |
|---|---:|---|---:|---|
| A: ms7 + hflip1024, fusion 0.65 | 56569707 | COMPLETE | 0.62865 | Unpublished |
| B: ms7 + hflip1024, fusion 0.82 | 56569768 | COMPLETE | 0.62767 | Unpublished |
| Baseline: ms7, fusion 0.74 | 56568811 | COMPLETE | 0.62717 | Unpublished |

The final API inventory was checked, including older single-model refs 56455800,
56392305, 56379896 and 56303572. Those older files cover test only; no additional
full Phase 2 candidate was found. Historical `submission_ens*.csv` files are
excluded regardless of their displayed scores.

Recommendation: select **56569707 (A) and 56569768 (B)**. Both have tiny positive
prior fixed-validation deltas and improved Public test-reference scores; they
offer two distinct fixed fusion outcomes for hidden ranking evaluation. This is
an evidence-based choice under uncertainty, not proof that either private score
exceeds baseline. Retain baseline and all original evidence locally.

No final submission checkbox or Save was clicked. The live counter remained
**0/2** after both candidates completed. The user must confirm the exact pair
before this final action. Two Submit opportunities were used; do not run further
candidates or reuse tomorrow's quota for new experiments.

## Official final-selection interpretation

The live [Phase 2 Supplementary Notice](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742487)
was re-read during this task. The deadline remains 2026-09-27 16:00 UTC+8. Public
scores now concern the full test1000 only and are reference scores. Private
scores concern ranking1000 and remain hidden in the current API responses.
Final Phase 2 scoring uses the best ranking score among manually selected
submissions. Old Phase 1-only files have zero ranking score and do not need a
selection slot to preserve the independently frozen Phase 1 component.

Consequently, the old handoff's suggested pair `56568811 + 56455800` should be
reconsidered after candidate results. Prefer two compliant full Phase 2 files;
never select `submission_ens*.csv`. Ref 56455800's historical Public score was
0.63546; its live rescored Public is 0.62717. This task does not independently
certify an organizer-published frozen Phase 1 score.

## Commands

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -u scripts/produce_phase2_flip.py cache
& .\.venv\Scripts\python.exe -u scripts/produce_phase2_flip.py build --candidate A
& .\.venv\Scripts\python.exe -u scripts/produce_phase2_flip.py build --candidate B
& .\.venv\Scripts\python.exe scripts/audit_phase2_flip.py --candidate A
& .\.venv\Scripts\python.exe scripts/audit_phase2_flip.py --candidate B
& .\.venv\Scripts\python.exe -m pytest -q
```

CSV/cache/manifests stay ignored. Only Python source, tests, and Markdown
documentation may enter the final private GitHub commit.

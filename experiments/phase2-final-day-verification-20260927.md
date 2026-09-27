# Phase 2 final-day verification

Verified 2026-09-27 approximately 12:21 UTC+8 through the authenticated Kaggle
submissions page. The user asked to submit now. There was no new passing
candidate after the completed validation rounds; the intended final files had
already been submitted successfully and selected.

## Live state

| Candidate | Submission ref | Status | Public test-reference | Selected |
|---|---:|---|---:|---|
| A, hflip1024 / fusion0.65 | 56569707 | Success / Complete | 0.62865 | Yes |
| B, hflip1024 / fusion0.82 | 56569768 | Success / Complete | 0.62767 | Yes |

The live page showed **2/2**. Both candidate checkboxes were checked, and their
DOM `name` attributes were exactly `56569707` and `56569768`. Baseline and
historical multi-model submissions were not selected. The page displayed
approximately four hours remaining, with the exact deadline tooltip
**2026-09-27 16:00 UTC+8**. No ranking private score was displayed.

Verified filenames and hashes displayed in their existing submission descriptions:

- A: `submission_phase2_single_m_hsi16_ms7_hflip1024_f065_sg0125_boxscale101.csv`;
  SHA-256 `94A3DF3CCAE9BDE34419CFAFA4BE908ED9E13468EAF7CA017E4F927C6F9E529C`;
  196,609 rows / 2,000 images.
- B: `submission_phase2_single_m_hsi16_ms7_hflip1024_f082_sg0125_boxscale101.csv`;
  SHA-256 `EF2F8BB1521E5980310F420EFE1561B6F069570E0A4672A9A290D7527BAF53F7`;
  241,891 rows / 2,000 images.

This was a live verification of existing submissions, not a fresh local
CSV audit or a new submission. Existing local hashes/validation and the initial
selection receipt are documented in [the flip finalization report](phase2-flip-finalization-20260926.md).

Source: [authenticated Kaggle submissions page](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/submissions).

## Actions and completion boundary

- No duplicate upload or Submit was performed today.
- No final-selection checkbox was changed; the saved pair was already correct.
- No training, inference, experiment expansion, or scheduled task was started.
- Daily remaining quota was not opened or checked because no new file needed
  submission; do not quote yesterday's count as today's quota.
- The live submission page was left available for the user.

The Phase 2 submission and selection workflow is complete. Await the official
ranking/private results. This does not certify final eligibility, prize status,
administrative registration or code-review acceptance. User-directed future
changes remain possible before the deadline, but there is currently no new
passing candidate awaiting submission.

Only documentation changed today; the last code test result remains 238 passed.
Models, datasets, CSVs, caches and private receipts remain excluded from Git.

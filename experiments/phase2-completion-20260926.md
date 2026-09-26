# Phase 2 completion: single-checkpoint HSI16 inference

Date: 2026-09-26 (Asia/Singapore)

Status: Kaggle submission complete; final leaderboard selection pending separate user confirmation.

## Scope and compliance

- Reuse exactly one trained YOLO26m checkpoint from the compliant Phase 1 lineage.
- Treat the 1,000 ranking images as inference-only data.
- Do not train, fine-tune, pseudo-label, update batch-normalization statistics, distill, or otherwise adapt model state on ranking data.
- Do not fuse predictions from a second checkpoint or architecture.
- Combine the frozen Phase 1 test predictions with ranking predictions only after independently validating both 1,000-image sets and proving their image-ID sets are disjoint.

## Ranking data contract

- Official ranking members: 1,000 numeric-ID, 16-bit grayscale PNG files.
- Official member bytes: `3,893,334,217`.
- Ranking-only ZIP bytes: `3,893,491,239`.
- Ranking-only ZIP SHA-256: `C6734C0DECF4B9772F698F02B20A56A62679C067DA05049C7D5F46AD01F1D590`.
- ZIP member-inventory SHA-256: `C5221F4D676558943B5562BB207F2570CADFB0C9941726598A8AB01ABC149A3E`.
- HSI16 output order: `[5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15]`.
- Encoding: 16-channel `uint8`, per-image shared selected-band bounds, P0.5/P99.5.
- Preparation config SHA-256: `C9C6182AB7292CE49E85D43733474BD3CE065542B65006C03120C0FE7BDC1C45`.
- Prepared outputs: 1,000 NPY tensors and 1,000 PNG previews; no resume reuse was involved.

The download, ZIP audit, PNG header/size checks, per-file hashes, and atomic manifest are implemented by `scripts/download_phase2_ranking.py`. Safe unpacking and HSI16 preparation are implemented by `scripts/prepare_phase2_ranking.py`.

## Inference and calibration

- Checkpoint: `kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt`.
- Checkpoint bytes: `44,130,777`.
- Checkpoint SHA-256: `8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3`.
- Input format: 16-channel NPY.
- Scales: `832 896 960 1024 1088 1152 1216`.
- Inference parameters: batch 1, FP16, confidence `0.0001`, NMS IoU `0.7`, `max_det=300`.
- Same-checkpoint scale fusion: `fusion_iou=0.74`, `support_gain=0.125`.
- Raw ranking CSV: 99,507 rows / 1,000 images, SHA-256 `6DC2D55513BAFF1FDD94EC613742721DA2345217DDB0347F79278919CBE1FB67`.
- Calibration: the previously audited global width/height scale `1.01`, zero center shift; audit SHA-256 `056C8DD8A9274B3A4F49B8D5A9160BC4B42B021EA54EAD09A44E40F6B87844E2`.
- Calibrated ranking CSV: 99,507 rows / 1,000 images, SHA-256 `1F9947E02B27CAB3C2BD315AFC4CABAA2FDA762315B5B400BD0AB576ECA95B42`.

## Strict merge

- Frozen Phase 1 CSV: 98,556 rows / 1,000 images, SHA-256 `C01214E6B250075C7DA522F2E9E6F89C829BF94AE1EB2F57CF69618FDC995A86`.
- Phase 1 lineage manifest SHA-256: `84D43FD57E92C407BDE3DB60D8CE58C6A0AB106C3ACE226CFC0AEA4A7F181EEB`.
- Test/ranking image-ID overlap: 0.
- Combined CSV: 198,063 rows / 2,000 images.
- Combined CSV bytes: `19,823,598`.
- Combined CSV SHA-256: `1744A3545E89F0E695EE106B177C582C7158D686D675940CFCAD3A72956C5E23`.
- Output columns exactly match the official schema, row IDs are consecutive, and the strict validator reported no issues.

The merge and lineage checks are implemented by `scripts/merge_phase2_submission.py`. The CSV and machine-specific manifests remain ignored private artifacts; Git contains the code, tests, and this hash ledger only.

## Kaggle result

- Submission reference: `56568811`.
- Submitted at: `2026-09-26T04:13:59.147000Z` (`2026-09-26 12:13:59` Asia/Singapore).
- Status: `COMPLETE`.
- Public test-reference score: `0.62717`.
- Ranking private score: withheld by Kaggle at the time of this record.
- Description recorded the single-checkpoint lineage, ranking inference-only scope, frozen test predictions, seven scales, and absence of model adaptation or ensemble.

The baseline submission itself is finished. The Kaggle final-selection counter remained `0/2`.

## Limited continuation authorized after baseline completion

At 2026-09-26 12:28 Asia/Singapore, the user decided to open a new conversation and use at most the final two daily submission opportunities to seek an inference-only improvement. This does not authorize training, ranking adaptation, additional checkpoints, or multi-model fusion. It also does not authorize an irreversible Kaggle Submit or final-selection save without a fresh action-time confirmation.

The only already-measured unsubmitted same-checkpoint family with positive fixed-validation deltas is one extra `1024` horizontal-flip source fused with the existing seven-scale predictions. Two settings were positive versus the supported-full control:

- `flip fusion_iou=0.65`, `support_gain=0.125`: `+0.0001697771` mAP50-95;
- `flip fusion_iou=0.82`, `support_gain=0.125`: `+0.0001039762` mAP50-95.

Both missed the former `+0.001` inference gate, so they are low-confidence last-chance candidates, not proven improvements. The focused continuation contract is recorded in `PHASE2_LAST_TWO_SUBMISSIONS_HANDOFF_20260926.md`. Keep the final-selection counter at `0/2` until the new candidates are complete or rejected, then request a separate confirmation for the exact two refs to select. Never select the historical multi-checkpoint ensemble.

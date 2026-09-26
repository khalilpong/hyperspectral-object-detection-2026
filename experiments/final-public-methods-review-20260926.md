# Final public-method review and closeout decision

Research date: 2026-09-26, Asia/Singapore. User asked what high-scoring teams
use, requested one last assessment, and otherwise wanted to close work until
tomorrow's final submission decision. Existing no-training/no-paid-compute,
single-production-checkpoint and no-scheduled-task constraints remain in force.

## What is and is not publicly established

A bounded search covered the competition slug, Kaggle discussions/code pages,
GitHub, author articles, and leading team names surfaced in search snapshots
(including SAU, ZAOUI_Yassine, KeSiqi, tuko_tanzwoo and cyfffffff). No attributable,
reproducible solution for those leading entries was found in that search.
This does not prove no such publication exists. In particular, scores alone
do not establish which architecture, training recipe, pseudo-labeling or
postprocessing a team used. [Official leaderboard](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/leaderboard)

Search-index leaderboard snippets may lag the live competition. Direct web
opening returned no readable leaderboard rows in the main review, so this
note deliberately does not publish those snippets as current ranks or scores.

One relatively detailed first-party account is ERS/OptiScan-HSI. The author
describes RF-DETR-Small with 16-channel input, transfer from a pseudo-RGB stage,
and spectral augmentation. The page reports Public 0.62083 and validation
mAP 0.6978; aspect-ratio-preserving, low-learning-rate refinement is described
as still being evaluated. These are author-reported results, not a reproduced
experiment or evidence of the leaders' methods. Its reported Public score is
below our last verified 0.62865, though submission date/scoring-version differences
prevent treating that as a controlled architecture comparison. The same site
also discusses a separate industrial multi-model route; it is not a competition
recipe to copy. [Author's technical account](https://er-flowscan.er-systems.org/optiscan-hsi/learn-architecture)

## Updated official boundary

The host's reply now explicitly permits self-training using model-generated
pseudo-labels on unlabeled **test** images and asks for the procedure to be
documented. **Ranking** images remain inference-only: no training, pseudo-label
adaptation, or parameter/statistics updates. Final output must come from one
trained model, with no modification of supplied annotations or prohibited
external labeled data. This updates the September 22 note that the question
had no host response at that time. It does not establish that a leading team
actually used self-training. [Host clarification](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/739853)

## Comparison with local evidence

- Single-model 16-band input, seven-scale inference, horizontal flip and global
  box calibration are already present in our accepted production recipe.
- Three further source additions have now failed: A+hflip832, A+hflip1216,
  and A+both. The exact results and unchanged gates are in
  [the last-day assessment](phase2-last-day-20260927.md).
- Localization-only voting had a small positive aggregate change but failed
  group stability. Source weighting, support-count and geometry alternatives
  also have negative or insufficient evidence; do not turn diagnostic class
  gains into retrospectively chosen class-specific mixtures.
- Historical test pseudo-label training is not a completely untried remedy.
  `experiments.csv` records YOLO26s trained with 3,000 labeled images plus 722
  pseudo-labeled test images: Public 0.61709 versus the then-reference 0.62651,
  a decline of 0.00942 (ref 56288823). This is a historical observation under an
  older model/scoring snapshot, not proof all pseudo-label recipes fail, nor a
  direct comparison with today's Phase 2 Public score. No held-out validation
  was recorded for that historical run. [Experiment registry](experiments.csv)
- A new RF-DETR, distillation or redesigned spectral-training route would
  require implementation, clean validation, full-data training and production
  inference. The public material does not provide a demonstrated improvement
  that can simply be applied to the fixed production checkpoint.
- Vertical flip remains untested. The actual fixed-split training arguments
  specify `flipud: 0.0`, `fliplr: 0.5`, `degrees: 0.0`
  (`kaggle_remote/outputs/ablation/kaggle_ablation_yolo26m_e30/args.yaml`,
  lines 98-104). This does not prove vertical flip would fail, but supplies no
  training-time support for that invariance. It is not selected for a last-minute
  experiment given the lack of positive evidence and the user's closeout intent.

## Decision and tomorrow's handoff

Stop expanding experiments. No new candidate has passed the agreed gate; this
is not a claim that every compliant technique is impossible or that a competition
rule requires our internal gain threshold. No new inference, training, Kaggle
upload/Submit, final-selection change or scheduled task occurred in this review.

The last verified selected pair remains:

| Candidate | Ref | Public test-reference | Last verified state |
|---|---:|---:|---|
| A | 56569707 | 0.62865 | COMPLETE, selected |
| B | 56569768 | 0.62767 | COMPLETE, selected |

Selection was persisted as 2/2 earlier in this task. Ranking private score was
unpublished at that check; neither selection nor private score was refreshed
in this research-only review.

When the user returns tomorrow, recheck the live deadline, quota, COMPLETE
states and exact selected refs first. The previously verified deadline is
2026-09-27 16:00 UTC+8. There is currently **no new passing file staged for a
final Submit**. Re-uploading identical A cannot test a new modeling hypothesis;
do not imply it is a score improvement. Let the user direct the final submission
decision. Any change of final selected refs requires its own explicit confirmation.
No autonomous wakeup or background polling is scheduled.

Code remains at the previously tested 238-passing-test state. This review changes
documentation only; raw evidence, models, data, predictions and credentials stay
out of Git.

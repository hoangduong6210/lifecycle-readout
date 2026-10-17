---
title: Last.fm Recorded Artist Model Comparison
status: fixed development comparison completed; result owner records negative mean primary
last_updated: 2026-10-16
paper_source: false
---

# Same-information phase comparison on recorded listening histories

`LCR-P-LASTFM-MODEL-001` is a new development comparison after the
[exact-artist input audit](../results/LastFM-Recorded-Artist-Audit.md). It does
not amend the completed negative Wikipedia studies or failed KuaiRand intake.
The [pilot owner](../results/LastFM-Recorded-Artist-Pilot.md) records execution.

## Prospective chronology and conditional target

Train on the fixed sampled users' complete source histories before
`2009-01-01T00:00:00Z`. January events before `2009-02-01T00:00:00Z` form
chronological development validation, with strictly earlier observed validation
events available to subsequent queries. No February-or-later feedback or test
score is parsed/evaluated. Prior training anatomy was inspected; this is not a
pristine confirmatory dataset or official benchmark reproduction.

Only valid user-track returns after at least one hour are scored. A fixed
hash selects one sixteenth of training queries; all supported validation queries
remain. Source histories are never thinned. Each query has one uniform negative
from the same user's historical tracks, matching log-two prior-count and
elapsed-time bins, excluding all current positives at that timestamp. Empty
support is retained as exclusion, never replaced by easy negatives. This is
conditional sampled ranking, without observed relationship-death labels.

A sorted historical catalog implements exact count/time eligibility and uniform
draws. Every global timestamp group queries before updating any source or item
history. Missing tracks still update source counters and user neighborhoods;
they do not share a fictitious item node. Neighbor snapshots use the last 20
strictly earlier event rows, keeping typed user/item namespaces separate.

## Common temporal encoder and exact doses

Every arm shares the package temporal neighborhood encoder, history head and a
learned 32-dimensional lookup for training-known artists. Graph messages contain
only earlier events' artist embeddings and known-track/artist flags. Artists
first seen in validation map to the unknown embedding, while strictly prior
recorded IDs remain available for exact prefix doses. No current event's metadata
can set the target reference or enter its query neighborhood.

Common history channels include elapsed time, own count, prefix-normalized count,
time of day, total intervening activity, same-reference artist dose, unknown
artist activity and reference/alternate availability. Same and unknown doses
plus total activity determine known-other activity where reference is available.
Latest-own artist references are frozen between own groups; missing/conflicting
groups remain flagged. Artist identity matching is not a semantic/causal label.

The common affinity receives exact dose inputs even in the primary control.
The phase branch therefore tests a residual function class rather than an
information advantage. All arms share common initial tensors per seed, including
an artist lookup initialized in an isolated random context so branch constructor
draws cannot change its values.

## Arms, fitting and interpretation

| Arm | Extra branch parameters | Purpose |
|---|---:|---|
| history_only | 0 | Common encoder/history scorer |
| dose_mlp | 13 | Flexible physical-time/dose residual |
| scalar_clock | 10 | Proportional time/event phase rates |
| phase_competition | 13 | Independent phase time/event rates |
| shuffled_phase | 13 | Frozen alternate artist reference |
| activity_mlp | 13 | Mask same-artist dose in head and branch |

The alternate reference is selected only from different artists observed by the
source at its own reset, after the group update, then frozen. It changes reference
availability and dose distributions; it is not a marginal-preserving causal
randomization test. The activity arm masks exact same-artist dose only; common
graph histories can still contain artist information.

All own groups reset the five-phase state to uniform. Positive phase rates and
readout gaps with nonnegative zero-initialized amplitude give a conditionally
nonincreasing residual in physical time and same-artist exposure. The affinity
can depend on both, so no whole-predictor monotonicity is certified. The clock
and exponential phase ingredients are not claimed new.

The primary is validation AP of phase_competition minus dose_mlp across fixed
seeds `101`, `202`, `303`, with identical inputs, common tensors, parameter count,
Adam settings, 10 epochs and earliest-maximum validation AP selection. Controls
and paired accuracy/AUC cannot replace that primary. Training-only normalization
and median positive scales are fixed by the corresponding arm's train inputs.
No retuning or unreported seed follows a result.

## Verification and resource contract

Candidate/history anchors, query thinning, unknown graph events, current/future
metadata invariance and exact negative bins pass synthetic/real scoped checks.
Synthetic tests check primary common tensors, initial predictions, capacity,
conditional derivatives, scalar equivalence and six-arm optimizer/checkpoint
roundtrip. The retained-checkpoint auditor replays every validation prediction
on CPU, reconstructs saved-score metrics, normalization, units, parameter counts
and selection-history consistency. This is inference replay, not optimizer replay.

Original prospective preparation settings remain unchanged. A separate launch
configuration adds transport paths and hashes mechanically, without changing
models, seeds, epochs or scientific settings. Real fitting requires one owned
Colab L4 runtime; retrieve and verify artifacts and stop it in a finally path.
No scientific claim follows merely from input validation or pipeline tests.

## Completed execution scope

The [result owner](../results/LastFM-Recorded-Artist-Pilot.md) now records every
arm/seed, the negative mean primary, exact Linux CPU initialization reconstruction,
retained-checkpoint replay, the preserved initial local failure and environment
limitations. Source/data/optimizer settings were not retuned after outcomes.

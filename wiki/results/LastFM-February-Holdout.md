---
title: Last.fm February Holdout
status: complete frozen-checkpoint test; local superiority not established
last_updated: 2026-10-16
paper_source: false
---

# Does the January localization recur in the following month?

The prespecified local confirmation screen is **NOT_ESTABLISHED**. A small
positive mean does not suffice: one frozen seed loses and the conditional
bootstrap interval includes zero. This is neither proof of equivalence nor
universal inferiority. The [January exploratory signal](LastFM-History-Strata.md)
remains recorded, without promotion to a confirmed operating-region claim.

## Fixed question and complete primary

`LCR-P-LASTFM-FEB-001` was fixed after January subgroup discovery and before
February preparation or scoring. Test feedback covers February only. All
January-selected checkpoints, users, historical-negative rules and original
training artist vocabulary remain fixed. No optimizer, checkpoint selection,
threshold change or alternate winning subgroup replaced the sole primary.

The positive query must return after one through fewer than seven days, with
one through nine strictly intervening events whose recorded artist matches its
previous reference. Both positive and matched negative candidates are retained.
This is a retrospective positive-query conditional ranking population, not a
deployable candidate filter or a causal substitution target.

The primary includes `4,328` queries from `57` active users at each seed and
passes the fixed support gate. AP values and all seed differences follow:

| Frozen seed | Phase AP | Dose-MLP AP | Phase minus MLP AP |
|---|---|---|---|
| `101` | `0.640823252` | `0.641127040` | `-0.000303788` |
| `202` | `0.644175134` | `0.643687583` | `+0.000487551` |
| `303` | `0.637138202` | `0.637052407` | `+0.000085794` |

The mean phase-minus-MLP difference is `+0.000089852726 AP`
(`+0.008985 pp`). Its prespecified conditional percentile
interval is `[-0.001374495246, +0.001248376068] AP`.
Only `2/3` seed differences are positive. Therefore both the all-positive-seed
condition and positive-lower-endpoint condition fail.

## Uncertainty and all secondary outcomes

The paired positive-user-cluster bootstrap uses `2,000` replicates, seed
`20261016`, `numpy.default_rng.PCG64` and linear percentile quantiles.
Each sampled user's multiplicity weights both candidates in all frozen seed
pairs. This interval is conditional on these fixed predictors, retained active
primary users and this month. It excludes training uncertainty, other months
and cross-user graph dependence; it is not a general-population guarantee.

All six arms, all sixteen parent gap/dose cohorts and all five contrasts are
retained. No secondary significance search was performed. Whole-month results
below are descriptive and do not replace the local primary:

| Contrast, whole February | Mean AP difference | Sample SD across frozen seeds |
|---|---|---|
| `primary_shape` | `-0.000060712` | `0.000511603` |
| `phase_specific_rates` | `-0.000589892` | `0.000708103` |
| `profile_alignment` | `-0.000081163` | `0.000893370` |
| `profile_information` | `+0.000346516` | `0.000858231` |
| `residual_vs_history` | `-0.000553108` | `0.000766672` |

## Integrity, replay and resource closure

`LCR-E-LASTFM-FEB-PREP-001` maps the preparation and independent prefix
verification. The raw events and original archive match their pinned identities.
The extension adds `61,965` events to the unchanged `1,825,981` pre-February
rows. All old user/track/artist ID prefixes, feature arrays and `108,297`
candidate-query prefixes at each seed remain exactly equal. The original
training artist vocabulary stays `27,972`; January/February-new artists retain
unknown lookup encoding. Missing-track events remain in source history.
March-or-later identities/feedback are excluded before parsing.

Each seed supplies `41,219` February test queries. Direct full prepared-history
scans check all features, reference/count bins, forbidden current positives and
typed graph snapshots for `32` fixed hash-ranked anchors per seed. These checks
do not claim all-row independent raw-history reconstruction.

`LCR-E-LASTFM-FEB-001` maps the inference, complete local replay and endpoints.
One owned L4 attempt scored all eighteen frozen checkpoints, retrieved outputs
and verified teardown before local CPU replay/bootstrap. All `1,483,884`
retained candidate probabilities were replayed; the maximum native GPU/CPU
absolute discrepancy is `0.000000715256` against the prespecified tolerance
`0.000002`. Independent tie-aware rank metrics check every arm/cohort vector.
Separate weighted sklearn endpoints and independent RNG draws verify all
`6,000` replicate/seed contrasts; maximum numerical disagreement is
`0.000000000000000444089`.

The remote environment reports Python `3.13.15`, Torch `2.11.0+cu130`, NumPy
`2.1.3` and NVIDIA L4. The transported historical dependency lock was not
enforced; this is inference replay, not optimizer or fresh-environment
reproducibility. Saved normalization, time/dose units, weights and selected
epochs are unchanged. No initial-common-weight reconstruction is needed or
claimed for this full-state inference.

## Consequences for the contribution

The earlier January same-sign subgroup observation cannot support a confirmed
local-utility headline for this architecture. A mechanistically readable branch
does not by itself establish better predictions or better whole-model
interpretability. The [branch explanation boundary](LastFM-Mechanistic-Readout.md)
and [fixed-rate structural results](Fixed-Rate-Repair-Validation.md) remain
separate working evidence with their own restrictions.

February is now consumed and cannot be reused as an untouched confirmation
window for later designs. Do not move the cohort threshold or advance through
months until a favorable result appears. The next contribution decision must
address the mechanism/target or the structural theorem's scope, rather than
promote an unconfirmed predictive gain. No scientific claim is admitted and
both publication pointers remain `UNRELEASED`.

Artifacts: `results/audit/lastfm_february_preparation_v1/`,
`results/audit/lastfm_february_preparation_closure_v1/`,
`results/audit/lastfm_february_execution_v1/`,
`results/audit/lastfm_february_audit_v1/` and
`results/audit/lastfm_february_closure_v1/`.
Retrieval: `evidence/indexes/lastfm_february_v1.toml`.

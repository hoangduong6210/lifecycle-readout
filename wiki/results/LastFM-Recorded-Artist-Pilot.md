---
title: Last.fm Recorded Artist Pilot
status: completed development comparison with independent initialization and checkpoint replay; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Recorded-artist controlled model outcome

The [fixed method and controls](../methods/LastFM-Recorded-Artist-Model.md),
`LCR-P-LASTFM-MODEL-001`, complete every declared arm and seed. The primary
phase-versus-equal-input/equal-parameter MLP contrast is negative on average,
with mixed signs across seeds. The phase arm is below common history-only at
every seed. This version supplies no demonstrated phase advantage; it does not
establish equivalence, universal inferiority or the impossibility of other models.

## All declared outcomes

Tables are generated from retained, independently checked records. Cells are
seed means and sample standard deviations, not confidence intervals. Each seed
has 40600 supported January validation queries and 67697 selected training
queries. Fixed users and histories stay unchanged; seed variation combines
candidate draws and fitting rather than representing population uncertainty.
The complete precision and all selected epochs/amplitudes live in
`results/audit/lastfm_model_closure_v1/summary.json`.

| Arm | Validation AP | AUC | Sampled Brier | Paired accuracy | Branch parameters |
|---|---|---|---|---|---|
| `history_only` | `0.720691 ± 0.001200` | `0.697309 ± 0.001806` | `0.214093 ± 0.001487` | `0.637336 ± 0.000903` | `0` |
| `dose_mlp` | `0.720405 ± 0.001147` | `0.696644 ± 0.002173` | `0.214649 ± 0.001376` | `0.635665 ± 0.002543` | `13` |
| `scalar_clock` | `0.720443 ± 0.000816` | `0.697068 ± 0.001334` | `0.214665 ± 0.001701` | `0.636223 ± 0.000215` | `10` |
| `phase_competition` | `0.720326 ± 0.000969` | `0.696541 ± 0.001622` | `0.214513 ± 0.001390` | `0.635936 ± 0.001713` | `13` |
| `shuffled_phase` | `0.720380 ± 0.000379` | `0.697463 ± 0.000654` | `0.213829 ± 0.000695` | `0.638415 ± 0.000898` | `13` |
| `activity_mlp` | `0.720243 ± 0.000414` | `0.696888 ± 0.001341` | `0.214369 ± 0.000849` | `0.640501 ± 0.001903` | `13` |

## Prespecified contrasts

| Fixed contrast | Mean AP difference | Sample SD | Positive / tied / negative seeds |
|---|---|---|---|
| `primary_shape` | `-0.000078712` | `0.000531768` | `2 / 0 / 1` |
| `phase_specific_rates` | `-0.000116885` | `0.000154450` | `1 / 0 / 2` |
| `profile_alignment` | `-0.000053945` | `0.000726368` | `2 / 0 / 1` |
| `profile_information` | `+0.000161546` | `0.000755035` | `2 / 0 / 1` |
| `residual_vs_history` | `-0.000364570` | `0.000271748` | `0 / 0 / 3` |

The primary is `primary_shape`. Every secondary is retained; none replaces it.
In particular, the flexible dose-versus-activity comparison has mixed signs,
and the correct-reference phase does not beat the alternate-reference phase on
average. The alternate changes both availability and dose distribution, so this
is not a causal randomization test. Artist information remains in graph histories
of the activity-masked control. These results do not prove artist inputs useless.

## Completed prospective input preparation

`LCR-E-LASTFM-MODEL-PREP-001` / `LCR-JOB-LASTFM-MODEL-PREP-001` record the
complete preparation. The original before-January sample is unchanged. The
prospective January window contributes 63663 events, including 6563 missing-track
source events and 2081 events whose artist is outside the 27972-artist training
lookup vocabulary. New artists map to the unknown embedding, not fitted future
lookup parameters; their observed history can still enter exact prefix counters.

All three fixed seeds yield 67697 selected supported training queries and 40600
supported January validation queries. Their negative identities differ by seed;
all arms within a seed must share its complete candidate cache. Every original
source event remains available to the chronological executor. Exclusion codes
record missing track, absent own history, short gap, thinned training query and
empty matched-negative pool separately. Each seed excludes 3970 selected
eligible queries with empty negative support; no unmatched fallback is used.

All 96 fixed real query anchors pass independent prepared-history scans of both
candidates' doses and alternate references, matching bins and graph snapshots.
This is scoped anchor verification, not every-row independent raw replay.
Source/input guards and artifact manifests pass. The first synthetic fixture
had no matched validation support and failed correctly; the fixture was expanded
without changing production eligibility, and its failure is retained.

Preparation ownership is `results/audit/lastfm_model_preparation_v1/`, with a
separate closure and retrieval index `evidence/indexes/lastfm_model_preparation_v1.toml`.
No optimizer, predictor/test score or remote runtime was used by preparation.

## Verification, failure recovery and resources

`LCR-E-LASTFM-MODEL-001` and `LCR-JOB-LASTFM-MODEL-001` bind the complete fit,
source snapshot, candidates and audits. The owned L4 attempt is
`results/audit/lcr-lastfm-original-8b520b4f5c/`. All 18 retained validation
checkpoints pass native CPU inference replay; saved-score AP, AUC, sampled Brier,
paired accuracy, candidates, training-only normalization/scales, capacities and
declared earliest-max selection histories reconstruct independently.

The original local audit stopped at exact initial common tensor reconstruction
before inference. Matching the PyTorch version alone on macOS did not resolve it.
A separate initialization-only Linux CPU replay regenerated the exact common
initial tensors of every arm/seed from identical uploaded source. Its hashes
match the original fit reports/checkpoints, and those verified initial states
permit the unchanged hash requirement in checkpoint replay. The original driver
failure remains retained; this is a separate completed recovery audit, not a
retroactive success edit. No optimizer was rerun. The exact differing local
random-generator/kernel has not been isolated.

The remote fit used its recorded preinstalled PyTorch and bounded library ranges;
the transported checked-in dependency lock was not enforced. Actual environments
are retained. This is initialization/inference portability, not fresh-environment
or optimizer reproduction. Input lineage inherits complete prepared-row summaries
and the fixed direct history anchors; it is not every-row independent raw replay.

Both the GPU fit and the separate CPU initialization runtime were stopped, with
backend absence and zero usage rate observed. Execution ownership is
`results/audit/lastfm_model_execution_v1/` and
`results/audit/lastfm_initialization_replay_v1/`; closure ownership is
`results/audit/lastfm_model_closure_v1/`. Retrieval is mapped by
`evidence/indexes/lastfm_model_v1.toml`. Displayed balance is not a compute invoice.

January is development validation used for checkpoint selection. No February-or-
later feedback or test score is evaluated. Recorded listening and matched
historical noncurrent candidates are not measured relationship deaths or causal
substitution. Conditional residual monotonicity does not certify the complete
predictor. No official external baseline is reproduced, no scientific claim is
admitted, both publication pointers remain `UNRELEASED`, and no manuscript is
exported. Earlier negative studies and the failed KuaiRand support gate remain.

## Separate posthoc questions

The [complete history strata](LastFM-History-Strata.md) retain every fixed
silence/dose cohort and a small same-sign local signal without replacing this
primary. The [mechanistic readout](LastFM-Mechanistic-Readout.md) quantifies how
much of the score is transparently explained by the phase branch. Those separate
protocols were fixed after aggregate outcomes; neither changes this fitted study.

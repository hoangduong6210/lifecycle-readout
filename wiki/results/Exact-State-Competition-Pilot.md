---
title: Exact State Competition Pilot
status: completed exploratory validation comparison and full checkpoint replay; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Completed exact-state comparison

The [prospective design](../methods/Exact-State-Competition-Study.md),
`LCR-P-EVENT-COUPLING-001`, completes every declared arm and seed on unchanged
Wikipedia recurrent-support candidates. Source/input pins and preflight checks
precede allocation. The owned Colab runtime is released, artifacts retrieved,
and all retained validation checkpoints independently replayed on CPU.

The primary contrast is negative on average. The specified exact-state phase residual does not beat its equal-parameter dose MLP on the declared primary endpoint. Negative and positive seed outcomes remain in the result rather than being replaced by a secondary contrast.

## All declared arm outcomes

Cells report seed means and sample standard deviations, not confidence intervals.
Tables are generated from retained audited result records; full precision lives
in `results/audit/event_coupling_closure_v1/summary.json`.

| Arm | Validation AP | AUC | Sampled Brier | Paired accuracy | Extra parameters |
|---|---|---|---|---|---|
| `history_only` | `0.602938 ± 0.007860` | `0.595741 ± 0.004100` | `0.243810 ± 0.001038` | `0.616516 ± 0.007919` | `0` |
| `dose_mlp` | `0.602854 ± 0.007888` | `0.595720 ± 0.003597` | `0.243814 ± 0.001006` | `0.614630 ± 0.008183` | `13` |
| `scalar_clock` | `0.602831 ± 0.007683` | `0.595713 ± 0.003680` | `0.243845 ± 0.001070` | `0.616516 ± 0.006298` | `10` |
| `phase_competition` | `0.602805 ± 0.007663` | `0.595683 ± 0.003708` | `0.243862 ± 0.001061` | `0.617647 ± 0.005184` | `13` |
| `shuffled_phase` | `0.605268 ± 0.006985` | `0.597397 ± 0.004700` | `0.243458 ± 0.001223` | `0.624434 ± 0.006298` | `13` |
| `activity_mlp` | `0.601635 ± 0.007123` | `0.595578 ± 0.003988` | `0.243795 ± 0.000979` | `0.624434 ± 0.014442` | `13` |

## Prespecified contrasts

| Prespecified contrast | Mean AP difference | Sample SD | Positive / tied / negative seeds |
|---|---|---|---|
| `primary_shape` | `-0.000049` | `0.000572` | `1 / 0 / 2` |
| `phase_specific_rates` | `-0.000025` | `0.000029` | `0 / 1 / 2` |
| `profile_alignment` | `-0.002463` | `0.002070` | `0 / 0 / 3` |
| `profile_information` | `+0.001220` | `0.000868` | `3 / 0 / 0` |
| `residual_vs_history` | `-0.000133` | `0.000622` | `1 / 0 / 2` |

The primary is `primary_shape`. Phase-specific-rate, alignment, information and
history-only contrasts are secondary; none replaces it. Equal parameter count
applies to the primary comparison. The proportional clock deliberately has fewer
branch parameters, and feature masks alter effective information access. Nominal
capacity equality does not imply equal function classes or optimization.
All same-information primary/scalar arms share training-only time/exposure scales.

## Interpretation and limitations

This tests a uniform own-event reset and constant global learned rates with a
bounded phase residual. It does not test context-conditioned resets, arbitrary
interacting particle models or every possible competition architecture. The
same history-aware affinity can already use exposure/time, so the residual's
conditional direction is not a full-score or chronological certificate.
Behavioral hashes remain unverified semantic substitution, with permitted prior
source contributions and collisions. No latent phase is identified as a true
relationship stage or a measured edge-death outcome.

The dataset and candidates were previously inspected. Checkpoint selection uses
validation, optimization has a fixed short budget, and seed dispersion is not
data-population uncertainty. Sampled Brier/paired accuracy concern generated
candidates rather than natural event prevalence. Earlier negative residual and
cadence studies remain unchanged. No tuning, arm dropping, new test score or
automatic dataset allocation follows from this result.

## Verification and resources

The normalized job `LCR-JOB-EVENT-COUPLING-001` and evidence
`LCR-E-EVENT-COUPLING-001` bind configuration, protocol, inputs and execution.
Source/asset hashes, all candidate/history identities, training-only normalization
and scales, selected epoch, learned-rate reports, parameter counts and complete
retained validation checkpoint probabilities pass the scoped auditor. Metrics
and contrasts are reconstructed independently from retained predictions.
Input-history verification inherits the earlier fixed raw-history/state anchors;
this is not every-real-row independent history replay, optimizer replay or a
fresh-dependency reproduction. No fit failed or was omitted.

Execution/preallocation/audit/resource records live in
`results/audit/event_coupling_execution_v1/`; the owned attempt is
`results/audit/lcr-wikipedia-9035005209/`; summary and source/wiki snapshots live in
`results/audit/event_coupling_closure_v1/`. Retrieval is mapped by
`evidence/indexes/event_coupling_v1.toml`. Resource observations verify backend
absence; displayed balance is not an audited compute invoice. Both publication
pointers remain unreleased, and no manuscript file is exported.

## Secondary mechanism disposition

The flexible model gains from the frozen-reference exposure relative to the
activity-masked control on `3` seeds, with mean AP difference
`+0.001219808`. This is an added-feature observation, not proof of the
constrained lifecycle mechanism. The correctly aligned phase model loses to its
reference-coordinate-shuffled counterpart on `3` seeds, with
mean AP difference `-0.002462851`. Thus the recorded predictive effects
do not support the proposed semantic-alignment interpretation. The shuffle
changes one coordinate alignment rather than removing all behavioral information.

Independent phase rates versus a proportional exposure clock have mean AP
difference `-0.000025320`, with `0` positive,
`1` tied and `2` negative seeds. This supplies
no observed advantage for the extra phase-specific rates under the declared
budget; it is not a statistical equivalence or universal inferiority result.

The selected shuffled-model amplitude is exactly zero on one seed. For that
checkpoint its residual is identically zero and its score is the history scorer.
This computational fact prevents treating every shuffled-arm gain as a lifecycle
effect. It is not a counterfactual retraining decomposition of the primary.

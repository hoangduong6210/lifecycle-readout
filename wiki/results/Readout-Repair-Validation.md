---
title: Readout Repair Validation
status: complete no-refit comparison with ranking cost and scoped structural success; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Outcome and useful contribution boundary

The [nearest-matrix restricted repair](../methods/Readout-Repair-Projection.md)
eliminates declared-threshold scored-path reversals in this complete validation
comparison, but does not preserve the original Reddit model's predictive AP.
The primary projected-minus-original contrast is negative on every Reddit seed
and mixed with a small negative mean on Wikipedia. This is not a successful
new superior predictor or evidence that a certificate guarantees useful ranking.

The controls make a more specific finding possible: most of the observed Reddit
AP decrease occurs in the constant-reference control, before the readout bounds
are applied. The incremental readout bound has a much smaller recorded AP change
relative to the unconstrained gradient-form control. These operational contrasts
do not establish a causal decomposition or a universal cost of monotonicity.
They prevent attributing the whole repair loss to the final bound alone.

`LCR-P-REPAIR-VAL-001` fixed the variants, reference, cohort and endpoints before
repair outcomes. Original path results were already known. No parameters were
trained, tuned, selected or updated in this analysis. The reference here is
the fixed zero clock, unlike the parent diagnostic's matrix frozen at each
candidate's observed clock; those are different controlled paths. This is posthoc validation
assessment on previously inspected data, not independent confirmation.

## Complete comparison

Cells are generated from independently audited native-score arrays. AP uses the
same outer probability clipping as parent controls. The grid has `25` normalized
times from `1e-3` to `1e3`, with an adjacent-increase threshold `1e-5`.
Counts are in seed order `1 / 7 / 42`. Every seed includes `1768` candidate
trajectories on Wikipedia and `20278` on Reddit; none are dropped by outcome.

| Dataset | Variant | Validation AP mean ± sample SD | Grid violations by seed |
|---|---|---|---|
| wikipedia | `original_dynamic` | `0.571050 ± 0.002881` | `0 / 0 / 0` |
| wikipedia | `legacy_reference` | `0.571128 ± 0.002867` | `0 / 0 / 0` |
| wikipedia | `safe_reference` | `0.571128 ± 0.002866` | `0 / 0 / 0` |
| wikipedia | `gradient_unconstrained` | `0.570914 ± 0.003263` | `0 / 0 / 0` |
| wikipedia | `readout_projected` | `0.570914 ± 0.003263` | `0 / 0 / 0` |
| wikipedia | `coordinate_projected` | `0.570207 ± 0.002670` | `0 / 0 / 0` |
| wikipedia | `zero_safe` | `0.570056 ± 0.002645` | `0 / 0 / 0` |
| reddit | `original_dynamic` | `0.553625 ± 0.001655` | `20075 / 20270 / 20278` |
| reddit | `legacy_reference` | `0.526508 ± 0.001734` | `302 / 101 / 35` |
| reddit | `safe_reference` | `0.526508 ± 0.001734` | `302 / 101 / 35` |
| reddit | `gradient_unconstrained` | `0.525917 ± 0.001456` | `0 / 0 / 0` |
| reddit | `readout_projected` | `0.525873 ± 0.001446` | `0 / 0 / 0` |
| reddit | `coordinate_projected` | `0.525644 ± 0.001251` | `0 / 0 / 0` |
| reddit | `zero_safe` | `0.525771 ± 0.001152` | `0 / 0 / 0` |

## Fixed contrasts

| Dataset | Projected minus comparator | AP difference mean ± sample SD | Signs positive / tied / negative |
|---|---|---|---|
| wikipedia | `original_dynamic` | `-0.000136 ± 0.000403` | `2 / 0 / 1` |
| wikipedia | `legacy_reference` | `-0.000214 ± 0.000446` | `2 / 0 / 1` |
| wikipedia | `safe_reference` | `-0.000214 ± 0.000447` | `2 / 0 / 1` |
| wikipedia | `gradient_unconstrained` | `+0.000000 ± 0.000000` | `0 / 3 / 0` |
| wikipedia | `coordinate_projected` | `+0.000707 ± 0.000631` | `3 / 0 / 0` |
| wikipedia | `zero_safe` | `+0.000858 ± 0.000762` | `3 / 0 / 0` |
| reddit | `original_dynamic` | `-0.027753 ± 0.000600` | `0 / 0 / 3` |
| reddit | `legacy_reference` | `-0.000636 ± 0.000575` | `0 / 0 / 3` |
| reddit | `safe_reference` | `-0.000635 ± 0.000575` | `0 / 0 / 3` |
| reddit | `gradient_unconstrained` | `-0.000045 ± 0.000038` | `0 / 0 / 3` |
| reddit | `coordinate_projected` | `+0.000228 ± 0.000201` | `2 / 0 / 1` |
| reddit | `zero_safe` | `+0.000102 ± 0.000307` | `2 / 0 / 1` |

The primary is projected minus `original_dynamic`. Secondary comparisons do not
replace that negative primary. In particular, the unconstrained gradient-form
control also has no grid violations at the learned rates, despite lacking the
all-positive-rate readout guarantee. Grid agreement alone therefore does not
establish the empirical necessity of the uniform theorem bounds.

On Wikipedia, no unconstrained potential requires the readout-bound repair;
projected and unconstrained outputs coincide. Their difference from the stronger
coordinatewise repair is nevertheless informative: certifying only the readout
can retain more of the original reference matrix. On Reddit, the readout
projection changes many candidate potentials, with all per-seed counts retained
in the summary. Projected AP exceeds the coordinatewise repair on both-dataset
means but not every Reddit seed. No significance or noninferiority is inferred
from the three seeds.

## Verification

The projection's matrix-distance optimum was tested against independent
constrained full-matrix numerical optimization on synthetic instances.
The artifact auditor independently checks readout/coordinate feasibility and
KKT conditions on every retained potential. It reconstructs every actual-query
and every grid lifecycle score with separate NumPy/SciPy equations, using frozen
native anchors and original checkpoint tensors, rather than importing the repair
engine. The maximum probability discrepancy is `6.349755312662353e-7`.

All primary/secondary metrics and grid flags are reconstructed from retained
scores, with complete cohort and checkpoint identities checked. Original AP
reconstruction differences and native/float64 flag disagreements are retained.
This is complete lifecycle-score replay from frozen anchors; it does not replay
the full backbone or optimizer. Full-predictor observed-input reconstruction
and raw-history checks are owned by the [parent full-cohort audit](Full-Validation-Scored-Paths.md).

## Retrieval and resource closure

`LCR-E-REPAIR-VAL-001` and `LCR-JOB-REPAIR-VAL-001` map the compact input packing,
source/protocol lock, both sequential Colab executions and independent audits.
Inputs and preparation identity are retained in `results/audit/readout_repair_inputs_v1/`;
execution and audits in `results/audit/readout_repair_execution_v1/`; generated
tables and summary in `results/audit/readout_repair_closure_v1/`.
The attempts are `results/audit/lcr-wikipedia-52a921378f/` and
`results/audit/lcr-reddit-22c09d48fe/`. The tracked retrieval manifest is
`evidence/indexes/readout_repair_validation_v1.toml`.

Both owned runtimes were released. The independent backend inventory is empty;
the CLI observation reports `173.27 CU` and `0.00/hr`. Displayed balance is not
an audited final compute invoice. No test score or Word export was produced.

## What can be claimed now

The project supplies a restricted exact condition, a minimum matrix-distance
constructor within that family, and a complete-cohort measurement of its
structural/ranking tradeoff. Readout-specific certification is less restrictive
than requiring every active phase to decline, and the control comparison measures
what that difference retains in these checkpoints. This is stronger analytical
and diagnostic evidence than the earlier isolated counterexample or pure-flow
check. It does not establish the first such mathematical principle, learned
phase semantics, chronological monotonicity or overall predictive superiority.
The matrix optimum uses the archived zero-diagonal representation; it is not a
minimum prediction-error or ranking-error projection.

## Separate specified-rate follow-up

The [new fixed-rate study](Fixed-Rate-Repair-Validation.md) uses a different,
explicit inverse-tilt metric for both of its comparators. It certifies the
retained raw frozen-reference gradient models at their learned rates without
additional changes. This advances the earlier finite-grid observation to a
scoped all-state certificate; it does not change this original negative primary,
Frobenius optimum or original-dynamic comparison. The new matched AP primary
also has no mean advantage.

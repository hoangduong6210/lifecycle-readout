---
title: Full Validation Scored Paths
status: complete posthoc all-supported-validation diagnostic; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Stronger learned-model coverage

The [anchor finding](Scored-Path-Silence-Diagnostic.md) now extends to every
supported validation query of the retained ordered checkpoints on both datasets.
The learned Reddit scorer frequently reverses its fixed-context silence response,
while its pure-flow diagnostic does not cross the declared threshold. Wikipedia
has no threshold crossing on this complete supported validation cohort. This
strengthens coverage of a checkpoint-specific phenomenon rather than asserting
universal inversion or a newly discovered generic softmax failure.

The expansion was explicitly fixed after anchor outcomes were observed.
`LCR-P-SCORED-FULL-001` and its [protocol](../../protocols/scored_path_full_validation_v1.toml)
retain every dataset/seed, all supported validation rows, both candidates,
the original paths, normalized time grid and thresholds. It is posthoc coverage
expansion, not independent confirmation. No optimization or test scoring occurs.

## Complete cohort table

A trajectory violates when any adjacent native-float32 unclipped score increases
by more than `1e-5` on the fixed `25`-point normalized-time grid from `1e-3`
to `1e3`. Encoder context, initial phase and affinity stay fixed. The dynamic
transform varies its explicit elapsed input; the frozen transform remains at
its observed anchor value. The table is generated from retained arrays and
independently reconstructed flags.

| Dataset | Seed | Candidate trajectories | Ideal flow violations | Implemented flow | Frozen transform | Dynamic transform | Dynamic fraction |
|---|---|---|---|---|---|---|---|
| wikipedia | `1` | `1768` | `0` | `0` | `0` | `0` | `0.0000%` |
| wikipedia | `7` | `1768` | `0` | `0` | `0` | `0` | `0.0000%` |
| wikipedia | `42` | `1768` | `0` | `0` | `0` | `0` | `0.0000%` |
| reddit | `1` | `20278` | `0` | `0` | `18037` | `20075` | `98.9989%` |
| reddit | `7` | `20278` | `0` | `0` | `14102` | `20270` | `99.9605%` |
| reddit | `42` | `20278` | `0` | `0` | `15042` | `20278` | `100.0000%` |

Each Wikipedia seed includes every `884` supported validation query and each
Reddit seed every `10139` supported validation query. Across models this is
`33069` model-query records and `66138` candidate trajectories. Query records
recur across seeds, so these are not independent lifetime observations or a
population sample with a binomial confidence interval.

All dynamic-path native/float64 trajectory violation flags agree. One secondary
frozen-transform trajectory for Reddit seed `7` differs across precisions near
the threshold. Its native flag remains authoritative as prespecified and both
arrays are retained. Tiny positive floating-point changes below the threshold
are also retained; no threshold non-crossing is relabeled exact mathematical
monotonicity. Outer-clipped score counts and derivative flags live in the full
summary, alongside every maximum increase and precision discrepancy.

## What was verified

Exact parent source archives, prepared input bytes and checkpoint identities are
pinned. Direct raw-history reconstruction supplies every validation query's
pre-event neighbors and pair statistics; all saved original pair features and
elapsed times match. Every observed query's frozen lifecycle reconstruction is
compared with the full original predictor in its native precision.

A copied lifecycle scorer lifts only these frozen native outputs to float64.
All pointwise path values are checked against native precision. Reverse-mode
score derivatives and forward-mode state/transition velocities agree with the
exact softmax directional formula, including off-boundary clamps. The independent
artifact audit reconstructs all primary and secondary flag counts, complete
query coverage, normalization and agreement with the earlier anchor arrays.
This audit does not independently rerun the entire optimizer or every full
backbone along the artificial grid; the backbone is held fixed by definition.

The largest recorded native/float64 pointwise score discrepancy is
`6.410917536281424e-7`. Native checkpoint contexts are never recomputed in double
precision, avoiding the known failed whole-backbone conversion in the original
anchor audit. The input and output archives are hash verified before teardown.

## Resources and retrieval

`LCR-E-SCORED-FULL-001` and `LCR-JOB-SCORED-FULL-001` map the two sequential owned
Colab executions and their scoped checks. The declared microbatch limits memory;
no parameter update, new model fitting or additional test endpoint was used.
Both owned runtimes were released and an independent backend inventory is empty.
The final CLI observation reports `173.27 CU` and `0.00/hr`; it is an observation,
not an audited final invoice or exact billed-cost measurement.

Execution, source guard and per-dataset artifact audits are in
`results/audit/scored_path_full_execution_v1/`; generated tables and full-precision
summary are in `results/audit/scored_path_full_closure_v1/`. The retained attempts
are `results/audit/lcr-wikipedia-b45c43f103/` and
`results/audit/lcr-reddit-5b1e973c68/`. The tracked retrieval manifest is
`evidence/indexes/scored_path_full_validation_v1.toml`.

## Permitted interpretation and remaining work

A passing pure-flow diagnostic can coexist with frequent scored-path reversal
in retained trained temporal models, even on the complete supported validation
cohort under this specified fixed-context intervention. Frozen-transform results
show that changing contextual features or transition drift are not required
for many of these examples. Their role is not established as a cause of AP
losses: the parent Reddit ordered model had a modest mean predictive gain,
whereas the parent Wikipedia model lost AP but shows no threshold crossing here.
Structural response and predictive utility therefore remain separate properties.

Complete supported-validation coverage does not make a chronological physical
intervention, whole-stream or first-time-link evaluation, test generalization,
independent confirmation, external causal result or universal guarantee.
The [restricted readout-aware repair](../methods/Readout-Aware-Entropic-Preservation.md)
has a proof and constructor but no demonstrated predictive advantage. A useful
next evaluation would measure whether a fixed repair removes the structural
violation and what ranking distortion it causes under equal information,
retaining zero-transition and unconstrained-projection controls. A protocol
must be recorded before looking at repair outcomes. No Word export or scientific
admission follows from this coverage expansion.

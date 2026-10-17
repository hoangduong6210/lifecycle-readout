---
title: Scored Path Silence Diagnostic
status: complete scoped validation-anchor diagnostic; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Main finding

The archived learned Reddit ordered models can reverse their silence response
through the scored candidate transform while their pure-flow trajectories do
not show a corresponding violation. This is no longer only a hand-constructed
parameter counterexample. The selected Wikipedia anchors show no violations
above the declared threshold, so the result is dataset/checkpoint dependent.
It does not explain the predictive performance differences causally.

`LCR-P-SCORED-PATH-001` fixes the [posthoc diagnostic](../../protocols/scored_path_diagnostic_v1.toml)
before its anchor results: evenly spaced supported validation queries, both
candidates, all retained hard-control seeds, a fixed logarithmic normalized-time
grid and common thresholds. Initial state, normalized context, backbone and
affinity remain fixed; the dynamic path varies the explicit elapsed input to
its candidate network. These are conditional model paths, not evolving physical
histories or the total derivative of every feature with real time.

## Complete declared anchor outcomes

Cells count candidate trajectories with at least one adjacent unclipped-score
increase above `1e-5`; each denominator is `64` candidate trajectories from
`32` query anchors. The grid contains `25` times from `1e-3` to `1e3` in the
checkpoint's normalized time unit. Seeds share aspects of the underlying query
cohort; the entries are not independent samples of real-world edge lifetimes.

| Dataset | Seed | Ideal flow | Implemented flow | Frozen transform | Dynamic transform |
|---|---|---|---|---|---|
| wikipedia | `1` | `0/64` | `0/64` | `0/64` | `0/64` |
| wikipedia | `7` | `0/64` | `0/64` | `0/64` | `0/64` |
| wikipedia | `42` | `0/64` | `0/64` | `0/64` | `0/64` |
| reddit | `1` | `0/64` | `0/64` | `58/64` | `64/64` |
| reddit | `7` | `0/64` | `0/64` | `48/64` | `64/64` |
| reddit | `42` | `0/64` | `0/64` | `55/64` | `64/64` |

The dynamic Reddit increases survive the model's outer probability clipping.
Even when the learned transition matrix is held fixed at its actual anchor
value, many Reddit trajectories reverse. Therefore changing contextual features
or explicit transition drift is not required for those frozen-transform examples.
The largest frozen-transform adjacent increase is `0.33631797582005185`; the
largest dynamic-transform adjacent increase is `0.3040969714021331`.

Wikipedia's absence of threshold crossings does not prove exact monotonicity:
the retained tiny positive changes and slopes remain in the complete arrays.
All dynamic Reddit candidates also have positive derivatives above the secondary
threshold; no such secondary threshold crossing is recorded for Wikipedia.
No claim about the fraction of the complete validation cohort follows from this
anchor sample. Whole-cohort expansion remains a separate possible next audit.

## Native model and analytic checks

Exact archived source is extracted with hash verification and imported in a
fresh process per dataset. Selected pre-event neighbor histories and original
pair statistics are independently reconstructed from allowed raw events and
compared with retained candidate identities. Checkpoint identities are verified.

An initial attempt converted the entire backbone to double precision. Long-time
Bochner phases then changed enough to fail an anchor-score tolerance on Reddit.
That failed directory, the initial source and the failure/correction record are
preserved. The correction freezes native float32 backbone/encoder/context
outputs and lifts only those fixed anchors and the lifecycle scorer. The query
grid, endpoints and tolerances are unchanged. Corrected results for both datasets
are authoritative; the initial Wikipedia output is historical.

A separate native float32 full-model replay then repeats every dynamic candidate
trajectory through the original predictor. Probabilities, derivatives and every
primary violation flag agree within the declared tolerances. The maximum native
probability difference is `6.254878934242036e-7`; maximum derivative difference is
`1.0226461708007228e-7`. The analytic directional formula, including transition
drift and clamp effects, agrees with autodiff to `1.1102230246251565e-16` in
these checks. The formula is the standard softmax chain rule, not an invented
optimization theorem.

A separately labeled posthoc range check finds the Reddit violating intervals
inside the global supported-training clock range. That does not establish
per-query counterfactual support, chronological plausibility or a new primary
endpoint. Original predictive AP values and all completed protocols are unchanged.

## Retrieval and scientific boundary

`LCR-E-SCORED-PATH-001` and `LCR-JOB-SCORED-PATH-001` map the retained arrays,
initial failure, corrected reports, source snapshots and native audits in
`results/audit/scored_path_diagnostic_v1/`. No model fit, GPU allocation, test
prediction or Word export was made. Current host dependencies were used; this
is not fresh-environment or optimization reproduction.

The stronger supported working statement is that a pure-flow diagnostic can
mask conditional scored-path reversal in actual learned checkpoints. Its scope
is the declared archived models, anchors, grid and fixed-context intervention.
It does not establish inevitable reversal, a universal ranking-collapse theorem,
whole-corpus prevalence, external causal effects or predictive improvement from
the [restricted readout-aware repair](../methods/Readout-Aware-Entropic-Preservation.md).

## Complete supported-validation expansion

The [full-validation owner](Full-Validation-Scored-Paths.md) now reports all
supported validation queries for both datasets and all retained ordered models,
under a separate posthoc design that keeps the anchor grid and thresholds.
The anchor tables above remain historical scoped evidence and must not be
substituted for the full-cohort fractions. No fitting or old-protocol change
occurs in this expansion.

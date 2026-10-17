---
title: Synthetic Temporal Response Diagnostics
status: independently audited oracle approximation diagnostic
last_updated: 2026-10-15
paper_source: false
---

# Synthetic Temporal Response Diagnostics

`LCR-P-Q1-SYNTH-001` examines function approximation before another large
real-data study. This is a small CPU oracle diagnostic with known probabilities
and a fixed oracle context-only base. It contains no graph interactions, noisy
binary observations, learned base scorer or identified real-world phases.

## Design and exact evidence

The [protocol](../../protocols/q1_synthetic_v1.toml) and
[configuration](../../configs/q1_synthetic_v1.toml) were saved before execution.
There are four mechanisms, three seeds (`11`, `29`, `47`) and three trained
controls plus the fixed base. All use the same sampled contexts and elapsed
values within each seed. Training sees `512` points and validation `256`;
independent same-support test and tail sets each contain `2048` points. Training
elapsed support is `[0,5]`; tail support is `[5,12]`. Context and elapsed are
independent. Contextual mixture weights receive context only, so the pure-time
monotonicity scope is respected.

Truth uses logarithmic time responses, distinct from the mixture-of-exponentials
family. The flexible MLP sees context and log elapsed. This is not a declaration
that equal step budgets yield equally converged or equally complex models.
All train on oracle probabilities using expected Bernoulli loss, with `300` Adam
steps, no extra regularizers and checkpoint selection on validation alone.

Primary error is expected BCE minus oracle entropy, equivalent to Bernoulli KL,
averaged on independent same-support inputs. Lower is better; this is neither
link-prediction AP nor a graph benchmark score.

## Same-support outcomes

Mean and sample standard deviation across the three development seeds.

| True mechanism | Fixed base | Scalar residual | Mixture residual | Flexible MLP |
|---|---:|---:|---:|---:|
| No temporal effect | `0 ± 0` | `6.71663e-17 ± 2.96e-18` | `6.71663e-17 ± 2.96e-18` | `1.7822e-11 ± 1.19e-11` |
| Common decreasing effect | `0.355627 ± 0.00826` | `0.00335046 ± 0.000214` | `0.00169672 ± 0.000472` | `5.17777e-06 ± 6.51e-07` |
| Context-dependent decreasing effect | `0.288852 ± 0.00517` | `0.0969746 ± 0.00298` | `0.00741494 ± 0.000245` | `0.000547969 ± 0.00015` |
| Increasing effect | `0.063588 ± 0.00157` | `0.063588 ± 0.00157` | `0.063588 ± 0.00157` | `2.33313e-07 ± 1.3e-07` |

## Interpretation

The implementation learns non-neutral attenuation when a decreasing effect
exists. A contextual mixture improves over the scalar under the constructed
context-dependent response. Both remain neutral to numerical precision under
the null, and cannot correct the increasing mechanism because their correction
is constrained nonpositive. The flexible MLP has smaller oracle approximation
error in every non-null mechanism. This diagnostic therefore supports capacity
and a known limitation, not unique latent lifecycle utility or a research method claim.

Tail extrapolation also does not establish an advantage for the mixture. Its
mean tail excess BCE under context-dependent decay is
`0.0488667`, compared with the flexible MLP's
`0.00339609`. Full tail, validation, selected-step and parameter records remain in
the raw result. Many fits select the last allowed step; convergence is not proved.
No settings were changed after inspecting these outcomes.

This experiment cannot assess robustness to finite event-label noise, temporal
dependence, missing observations, base-score errors or real-graph generalization.
The context weights are mixture coefficients, not recovered lifecycle labels.

## Retrieval and verification

- [Normalized execution](../../evidence/jobs/LCR-JOB-Q1-SYNTHETIC-001.toml)
- [Results](../../results/audit/q1_synthetic_v1/run-20261015-001/results.json)
- [Independent audit](../../results/audit/q1_synthetic_v1/run-20261015-001-audit.json)
- [All artifact checksums](../../results/audit/q1_synthetic_v1/run-20261015-001/checksums.sha256)
- [Archived source identities](../../results/audit/q1_synthetic_v1/run-20261015-001/source_manifest.json)
- [Auditor](../../scripts/audit_synthetic_response.py)
- [Archived source replay audit](../../results/audit/q1_synthetic_v1/archived_replay_audit.json)

The auditor verifies all source/artifact hashes, independent generator formulas,
checkpoint reconstruction, validation selection, all `144` saved endpoint arrays
and aggregate metrics. `36` trained models across `12` mechanism-seed conditions
were verified. The original input data, predictions, checkpoints, histories and
source snapshot remain local audit artifacts. A replay from those archived
source bytes reproduces all `192` data/prediction NPZ arrays, selected steps and
aggregate metrics exactly in the same local dependency environment. This is not
a clean-clone or independent-environment reproduction.

## Research consequence

Proceed by defining the [prediction target](../methods/Prediction-Target-and-Sampling.md)
and evaluating strong controls under official candidate protocols. The residual
family passes a capacity check but has not earned an expanded efficacy claim.
The [research program](../questions/Research-Program.md) remains open.

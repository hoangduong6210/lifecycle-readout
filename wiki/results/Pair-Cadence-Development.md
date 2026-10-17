---
title: Pair Cadence Development
status: complete audited validation development; primary clock contrast negative
last_updated: 2026-10-16
paper_source: false
---

# Pair Cadence Development

`LCR-P-CADENCE-DEV-001` is complete on both declared corpora and every declared
seed. `LCR-E-CADENCE-DEV-001` maps the scoped development evidence. Both release
pointers remain `UNRELEASED`; these are exploratory validation outcomes, not
admitted publication claims. No test prediction or test clock was constructed.

## Design and information access

The [sealed protocol](../../protocols/pair_cadence_development_v1.toml) separates
added information from replacing the explicit residual attenuation clock:

| Arm | Historical inputs | Explicit attenuation clock | Models |
|---|---|---|---|
| original | Nine channels; the added three are masked to zero | Absolute elapsed time divided by train median | History base, scalar residual, mixture residual, flexible MLP residual |
| features | Same nine channels; cadence bundle exposed | Same absolute clock | Same four families |
| relative | Exactly the same exposed channels as features | Log-one-plus elapsed time divided by smoothed pair cadence | Scalar and mixture residuals |

The bundle contains log normalized mean pair interval, an availability indicator
and relative freshness. Thus both sides of the primary clock comparison already
receive relative freshness as a contextual feature. The contrast isolates the
explicit attenuation-clock replacement within that information setting; it does
not test whether removing all relative-time information is harmful.

Candidate streams, support, original features, training source, budgets,
objectives and within-family capacities are checked across arms. This is the
same supported recurrent cohort as the earlier hard study, with same-source
count/silence-bin matching and no fallback for unsupported queries. Allowed
unsupported events still update history; held-out training exclusions do not.
Pair cadence uses strictly past events. Missing/zero-span cadence uses the
median of available training-candidate cadences with its availability flag.
Selection is the earliest maximum validation AP within the fixed epoch budget.

## Primary clock comparison

The primary endpoint is paired validation AP for relative minus absolute scalar
residual, with identical exposed features. Values are absolute AP differences;
SD is descriptive sample dispersion across the declared seeds, not a confidence
interval. All tables below are generated from audited artifacts.

| Dataset | Scalar relative − absolute AP, mean ± sample SD | Positive / zero / negative seeds |
|---|---:|---:|
| wikipedia | `-0.000365 ± 0.000397` | `0 / 0 / 3` |
| reddit | `-0.000150 ± 0.000083` | `0 / 0 / 3` |

The primary difference is negative for every seed on both datasets. Under this
fixed-budget development design, replacing the explicit attenuation clock does
not improve AP. Small observed differences do not establish statistical
significance, equivalence or inferiority in every population.

## Secondary comparisons

Feature contrasts subtract original from features; the mixture clock contrast
subtracts features from relative. All remain secondary, including positive
individual seeds.

| Dataset | Contrast | AP difference, mean ± sample SD | Positive / zero / negative seeds |
|---|---|---:|---:|
| wikipedia | feature_base_stats | `+0.000585 ± 0.001401` | `2 / 0 / 1` |
| wikipedia | feature_residual_scalar | `+0.001255 ± 0.001460` | `3 / 0 / 0` |
| wikipedia | feature_residual_mixture | `+0.001467 ± 0.001670` | `3 / 0 / 0` |
| wikipedia | feature_mlp_residual | `+0.000479 ± 0.001102` | `1 / 0 / 2` |
| wikipedia | clock_residual_mixture | `-0.000263 ± 0.000431` | `1 / 0 / 2` |
| reddit | feature_base_stats | `+0.004556 ± 0.004086` | `3 / 0 / 0` |
| reddit | feature_residual_scalar | `+0.004396 ± 0.000814` | `3 / 0 / 0` |
| reddit | feature_residual_mixture | `+0.007800 ± 0.003566` | `3 / 0 / 0` |
| reddit | feature_mlp_residual | `+0.005944 ± 0.002690` | `3 / 0 / 0` |
| reddit | clock_residual_mixture | `-0.000160 ± 0.001365` | `2 / 0 / 1` |

The added bundle improves the scalar and mixture residuals on every recorded
seed, and improves the Reddit history base and MLP residual as well. The smaller
Wikipedia history-base/MLP effects are not uniformly positive. These comparisons
concern a bundle: they do not identify which individual channel helps or show
that a new decay law was discovered. Positive secondary outcomes do not replace
the negative primary clock comparison.

## Support and cadence availability

Unsupported totals below cover all original splits; test-query counts remain
structural metadata and are not scored here. Availability refers to candidate
slots, with both classes counted. Train-derived fallback can differ by seed.

| Dataset | Seed | Train queries | Validation queries | Excluded train events | Unsupported events, all splits | Train cadence slots available / total | Validation cadence slots available / total | Train fallback, seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wikipedia | `101` | `2805` | `884` | `14808` | `138000` | `4242 / 5610` | `1434 / 1768` | `30190.670455` |
| wikipedia | `202` | `2805` | `884` | `14808` | `138000` | `4242 / 5610` | `1434 / 1768` | `29880.568966` |
| wikipedia | `303` | `2805` | `884` | `14808` | `138000` | `4242 / 5610` | `1434 / 1768` | `30087.163462` |
| reddit | `101` | `37206` | `10139` | `75405` | `538695` | `52034 / 74412` | `15202 / 20278` | `76813.499556` |
| reddit | `202` | `37206` | `10139` | `75405` | `538695` | `52035 / 74412` | `15202 / 20278` | `76857.923700` |
| reddit | `303` | `37206` | `10139` | `75405` | `538695` | `52034 / 74412` | `15202 / 20278` | `77010.769795` |

[Detailed AP/AUC/Brier, capacity and fit-time table](../../results/audit/pair_cadence_closure_v1/detailed_metrics.md)
retains every arm and family. Fit seconds describe saved training-loop timings,
not total Colab allocation time or billing. Brier measures calibration on the
sampled candidate cohort, not natural event prevalence.

## Verification and execution

Per-arm checks replay histories/support, independently reconstruct pair clocks,
load selected checkpoints and recompute metrics from saved predictions.
Cross-arm checks enforce shared candidates, exposed features, source, budgets
and within-family capacity. A separate fresh-process CPU audit imports each
verified archived model and reconstructs train/validation neighbors, historical
features, cadence and train normalization before re-scoring every validation
query. All retained checkpoints pass the declared CPU/GPU numerical tolerance.
This is inference replay with existing host dependencies, not optimization or
fresh-environment reproduction.

| Dataset | Arm | Normalized job |
|---|---|---|
| wikipedia | original | [LCR-JOB-CADENCE-WIKIPEDIA-ORIGINAL-001](../../evidence/jobs/LCR-JOB-CADENCE-WIKIPEDIA-ORIGINAL-001.toml) |
| wikipedia | features | [LCR-JOB-CADENCE-WIKIPEDIA-FEATURES-001](../../evidence/jobs/LCR-JOB-CADENCE-WIKIPEDIA-FEATURES-001.toml) |
| wikipedia | relative | [LCR-JOB-CADENCE-WIKIPEDIA-RELATIVE-001](../../evidence/jobs/LCR-JOB-CADENCE-WIKIPEDIA-RELATIVE-001.toml) |
| reddit | original | [LCR-JOB-CADENCE-REDDIT-ORIGINAL-001](../../evidence/jobs/LCR-JOB-CADENCE-REDDIT-ORIGINAL-001.toml) |
| reddit | features | [LCR-JOB-CADENCE-REDDIT-FEATURES-001](../../evidence/jobs/LCR-JOB-CADENCE-REDDIT-FEATURES-001.toml) |
| reddit | relative | [LCR-JOB-CADENCE-REDDIT-RELATIVE-001](../../evidence/jobs/LCR-JOB-CADENCE-REDDIT-RELATIVE-001.toml) |

The [retrieval index](../../evidence/indexes/pair_cadence_v1.toml) pins launch
archives, original manifests, selected checkpoints, predictions, comparison
reports, inference replay and generated tables. All owned runtimes were released
between arms. The final independent
[backend observation](../../results/audit/pair_cadence_execution_v1/resource_final.json)
confirms no active session/assignment and zero consumption rate. Earlier live
observations are historical, not the final resource state.

## Interpretation for the journal article

The [relative-freshness attribution](../methods/Shortcut-Resistant-Gating.md#relative-freshness-attribution)
remains binding: the core pair-normalized clock is established prior work.
The result distinguishes adding contextual information from imposing a different
explicit attenuation clock. It supports an operating-limit analysis, not a new
clock invention or universal model improvement. Context itself contains elapsed
and relative time; the residual's fixed-context derivative guarantee is not a
guarantee for the complete time-varying prediction path.

These validation-selected results come from previously inspected corpora with a
fixed budget. They do not supply fresh test confirmation, significance, causal
lifecycle identification, phase semantics or an official benchmark reproduction.
Retain them alongside the hard-control negative results and bounded sequential
baseline. Contribution review, reproducible source/distribution closure and
scientific admission remain open before assembling a submission snapshot.

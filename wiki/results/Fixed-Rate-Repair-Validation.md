---
title: Fixed Rate Repair Validation
status: complete no-refit comparison; extra certification deformation avoided, no mean AP advantage
last_updated: 2026-10-16
paper_source: false
---

# What sharper certification changes on retained learned models

`LCR-P-RATE-REPAIR-VAL-001` fixed this new comparison after the parent repair
and theorem outcomes were known, before any fixed-rate repair result was read.
It retains all parent Wikipedia/Reddit supported validation candidates and all
three original seeds. The [new repair](../methods/Inverse-Tilt-Projection.md)
and the all-rate comparator use the same inverse-square-root tilt metric.
No existing fitted model, candidate, checkpoint selection or old endpoint changes.

## Main structural observation: no additional repair is needed

Every one of the 66138 retained candidate trajectories passes the fixed-rate
pair certificate. For each frozen-reference gradient-form comparator, the
algorithm returns its raw inverse tilts exactly and its actual/grid scores are
unchanged. The minimum pair margin across all candidates is
`0.1730612788890315`, comfortably separated from the declared numerical boundary.
This is an all-initial-state certificate under that candidate's FIXED tilt,
readout, rate vector and context, supported by the preceding real-arithmetic
proof and finite numerical checks. It is not a formal machine-arithmetic proof.

The matched all-rate repair still changes 46619 Reddit candidate configurations;
it changes none on Wikipedia. Thus the earlier uniform-over-rates restriction
requires deformation that is unnecessary for the specified learned rates.
These are candidate trajectories across six checkpoint/cohorts, not independent
population observations. The real QPs finish without updates because all raw
points are feasible; nontrivial optimizer correctness is tested synthetically.

| Dataset | Seed | Candidates | Fixed-rate changes | All-rate changes | Mean squared x-distance: fixed / all-rate |
|---|---|---|---|---|---|
| `wikipedia` | `1` | `1768` | `0` | `0` | `0.000000000 / 0.000000000` |
| `wikipedia` | `7` | `1768` | `0` | `0` | `0.000000000 / 0.000000000` |
| `wikipedia` | `42` | `1768` | `0` | `0` | `0.000000000 / 0.000000000` |
| `reddit` | `1` | `20278` | `0` | `16474` | `0.000000000 / 0.004947618` |
| `reddit` | `7` | `20278` | `0` | `19193` | `0.000000000 / 0.004263658` |
| `reddit` | `42` | `20278` | `0` | `10952` | `0.000000000 / 0.003521432` |

“Unchanged” is relative to the already frozen-reference gradient-form comparator.
Obtaining that comparator has already altered the original dynamic model.
The new certificate does not restore the original clock-dependent transition or
certify the full changing-context predictor.

## Prespecified predictive comparison

The primary is validation AP of fixed-rate minus the matched all-rate x-projection.
It does not show a mean predictive advantage. The geometric distance guarantee
is retained without relabeling it as an AP guarantee.

| Dataset | Fixed − matched all-rate AP: seeds | Mean ± sample SD | Positive / tied / negative |
|---|---|---|---|
| `wikipedia` | `+0.000000000000 / +0.000000000000 / +0.000000000000` | `+0.000000000000 ± 0.000000000000` | `0 / 3 / 0` |
| `reddit` | `-0.000001395882 / +0.000000039665 / -0.000002346597` | `-0.000001234271 ± 0.000001201312` | `1 / 0 / 2` |

Fixed-rate AP equals the unmodified gradient comparator exactly at every seed.
Relative to the original dynamic reference, mean AP remains lower by about
`0.000135934` on Wikipedia and `0.027708126` on Reddit. The large previously
observed dynamic-versus-frozen loss therefore remains. The prior Frobenius
matrix repair uses another objective and its numerical contrasts must not be
mixed with this same-metric comparison. Complete raw/clipped AP, AUC, paired
accuracy, shifts and every candidate grid/derivative outcome are retained.

## Verification, arithmetic scope and resources

All six cohorts complete, with unchanged input/checkpoint identities and no
omitted or outcome-selected row. Primary scoring uses a declared float64 lift of
raw checkpoint parameters and archived native reference potentials; original
native gradient-score reconstruction errors and AP differences are retained.
This is an explicit arithmetic path, not an additional native GPU run.

The separate auditor reconstructs every halfspace, original pair margin, QP KKT
condition and same-metric distance comparison. It independently reconstructs all
actual/query-grid probabilities and derivatives using quotient differentiation,
and AP/AUC from tie-aware ranks. Every repaired variant has zero declared grid
or derivative violation. The inherited frozen anchors were independently checked
by the parent study; this analysis does not rerun its full backbone or raw history.

Setup source-generation/path errors occurred before any real score was produced
and are retained. Scientific settings, tolerances and primary were unchanged;
no tuned retry, refit, new candidate or unseen test was used. All computation was
bounded local CPU; no remote runtime was allocated. No novelty, causal relationship
semantics, statistical significance or publication admission follows.

`LCR-E-RATE-REPAIR-VAL-001` / `LCR-JOB-RATE-REPAIR-VAL-001` map
`results/audit/rate_repair_validation_v1/` and
`results/audit/rate_repair_closure_v1/`. Full precision is in the closure summary;
retrieval is mapped by `evidence/indexes/rate_repair_v1.toml`. Both publication
pointers remain `UNRELEASED`; the wiki remains the paper source.

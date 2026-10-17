---
title: ML-20M Development Baselines
status: scoped audited count screen and bounded author fit
last_updated: 2026-10-16
paper_source: false
---

# ML-20M Development Baselines

`LCR-P-TGBSEQ-SCREEN-001` completed all 364943 provided validation queries.
The official train/validation masks and upstream identity are retained. Evaluation
uses one newly generated draw of 100 training-catalogue negatives per query,
shared across models, excluding every same-source/timestamp positive. Updates
occur only after the entire timestamp group. These candidates are development
inputs, not a supplied official validation or test candidate file.

## Recorded endpoints

| Independently implemented control | Validation MRR |
|---|---:|
| Binary edge memory | `0.018443` |
| Destination popularity | `0.185964` |
| Pooled previous-context transitions with popularity backoff | `0.292783` |

One candidate seed (42), fixed smoothing, no optimizer or model tuning. These
are point estimates, not uncertainty across training seeds or significance.
The transition-minus-popularity difference is about `0.106818` under this specific
candidate draw. The input audit finds no repeated positive pair in train/validation;
this supports investigating shared sequential information, not positive utility
of the previous lifecycle gate. No new test was scored.

## Verification and retrieval

The [independent audit](../../results/audit/lcr-ML-20M-ba89635570/independent_audit.json)
checks all candidate identities against the prepared input, training-catalogue
membership and simultaneous-positive exclusion; it reconstructs reciprocal ranks
and aggregate MRR for every query. A separately implemented sparse chronological
replay reconstructs scores at 24 deterministic query anchors, including both ends
of validation. It does not replay every score or reproduce candidate RNG draws.
Source bundle, prepared input, remote archive and result file hashes are checked.

[The normalized job](../../evidence/jobs/LCR-JOB-ML20M-SCREEN-001.toml) and
[tracked retrieval index](../../evidence/indexes/ml20m_screen_v1.toml) locate exact
local artifacts. Local availability is not clean-clone or independent-environment
reproduction. The Colab owned endpoint was released; separate inventory shows
no sessions/assignments and zero CU/hour. The displayed balance remained `183.40`
CU at CLI precision; that is not proof of exactly zero billed consumption.

## Scientific disposition

The signal makes pooled transitions a necessary strong control for further work.
It is not a novel contribution: the
[September 2026 smoothed-count preprint](https://arxiv.org/html/2609.35364v1)
already combines pooled transitions, smoothed background probabilities, count
age decay and recency; its equations 1–4 and timestamp sensitivity appendix are
directly relevant. Our one-context, untrained control is a narrower independent
implementation, not a reproduction of that paper. Published scores must not be
compared numerically to this validation screen. CRAFT is also required prior work
and a learned comparator; its
[NeurIPS 2025 paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html)
uses candidate-aware matching against recent interactions.

Before another broad GPU sweep: inspect and pin an author implementation of a
strong sequential comparator, close a common validation adapter, and state what
new mechanism can add beyond these controls. Adding a generic time decay or
renaming shared transition counts as lifecycle is insufficient novelty. A
candidate-common positive multiplicative gate cannot improve within-query ranks
in exact arithmetic, so any proposed new-link correction must change relative
candidate scores. This rank-invariance observation is elementary, not a claimed
new theorem. A distinct candidate-specific temporal hypothesis and targeted
prior-art review remain open. No scientific claim or paper release is admitted.


## Bounded CRAFT comparison on the same candidates

The [author-model fit](CRAFT-Adapter-and-Feasibility.md#completed-full-support-validation-fit)
now passes its scoped independent audit. Input hashes bind it to the exact
cached candidate file above. The table is generated from retained result files;
all entries remain validation development point estimates.

| Model under common validation candidates | Validation MRR |
|---|---:|
| Binary edge memory | `0.018443` |
| Destination popularity | `0.185964` |
| Pooled transition plus popularity | `0.292783` |
| Pinned CRAFT author model, bounded fit | `0.417077` |

The CRAFT checkpoint was selected on these validation queries; count controls
have fixed declared smoothing. Training and selection budgets therefore differ.
This comparison shows the recorded trained comparator against the specified
simple controls, not a fair search-budget isolation, statistical significance,
official benchmark reproduction or novel method. One CRAFT training seed and
one generated candidate draw provide no training-seed uncertainty estimate.
The fit stopped at its runtime cap, with an incomplete final epoch retained;
convergence remains unestablished. It adds no evidence for a lifecycle readout.

Exact generated values and scope are retained in
`results/audit/craft_fit_closure_v1/`; `LCR-JOB-CRAFT-FIT-001` and
[the retrieval index](../../evidence/indexes/craft_fit_v1.toml) map the complete
local artifacts. Runtime release was separately verified. Scientific release,
source-data lineage/rights, clean-clone reproduction and contribution novelty
remain open.


## Posthoc source-session inactivity diagnostic

`LCR-P-SESSION-DIAG-001`, `LCR-E-SESSION-DIAG-001` and
`LCR-JOB-SESSION-DIAG-001` record a CPU-only diagnostic of the brainstorm's
source-memory hypothesis. It adds no training or candidate draw. Inactivity is
the current timestamp minus the source's latest strictly earlier timestamp,
including allowed validation history. Simultaneous events cannot update each
other. This is an action-gap clock, not an identified session boundary.

The intervals below use available TRAIN-gap quartiles, fixed in the plan before
examining validation stratification; equality goes to the higher interval.
Counts, shares and endpoints are generated from the retained rank vectors.
Original overall validation MRR remains the primary endpoint.

| Source inactivity, seconds | Validation queries | Share | Edge memory MRR | Popularity MRR | Transition MRR | Transition − popularity |
|---|---:|---:|---:|---:|---:|---:|
| Unavailable | `0` | `0.0%` | — | — | — | — |
| 0 < gap < 12 | `100099` | `27.4%` | `0.018624` | `0.257629` | `0.421685` | `+0.164056` |
| 12 ≤ gap < 27 | `66017` | `18.1%` | `0.018500` | `0.196461` | `0.328922` | `+0.132461` |
| 27 ≤ gap < 54 | `43362` | `11.9%` | `0.018457` | `0.173291` | `0.282080` | `+0.108788` |
| gap ≥ 54 | `155465` | `42.6%` | `0.018298` | `0.138899` | `0.197426` | `+0.058527` |

The transition-minus-popularity advantage decreases across these intervals,
while remaining positive even in the longest-gap interval. This is consistent
with a source-memory attenuation hypothesis, but does not identify a decay law
or establish that attenuation improves a model. User composition, item mix,
history, activity intensity and calendar time can differ between intervals.
The last interval is broad and heavy-tailed; the train-derived intervals need
not balance validation counts. No session threshold is selected from these
results.

A separately implemented chronological memory replay reproduces the complete
train/validation clock and training cutoffs. All saved validation gap values,
event identities, original reciprocal-rank vectors and stratum metrics pass
independent verification. This relies on the original count-screen score audit;
it is not a new full score-inference replay. The initial audit rejected a test
source hash after an additional fixture was added; the original plan/test
snapshot and failure are retained, and the subsequent audit checks that snapshot
while requiring the executed analyzer and protocol to remain unchanged.

[The retrieval index](../../evidence/indexes/source_session_diagnostic_v1.toml)
pins local analysis, queries, source snapshots and audit. One existing candidate
draw supplies no training-seed uncertainty, significance or causal estimate.
This remains exploratory, non-frozen evidence. Source-action time gating has
[direct prior art](../questions/Contribution-Gap-Review.md#source-session-transition-attenuation);
this diagnostic is not a novelty claim. Finish the declared cadence study before
deciding whether a separately specified attenuation comparison adds useful
evidence. No GPU was allocated or test data read for this diagnostic.

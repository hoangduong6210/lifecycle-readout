---
title: Last.fm Mechanistic Readout
status: exact branch decomposition and bounded influence checked; whole-model interpretability unestablished
last_updated: 2026-10-16
paper_source: false
---

# How much of this predictor can be explained mechanically?

`LCR-P-LASTFM-READOUT-001` inspects only the retained Last.fm
`phase_competition` checkpoints. It does not rewrite the legacy lifecycle model
or infer human interpretability from an architectural label. The complete
[parent comparison](LastFM-Recorded-Artist-Pilot.md) remains unchanged.

## Exact additive explanation and its limits

With the uniform five-phase reset and four active components, the implemented
branch has the exact real-arithmetic form

$$r(t,E)=\sum_{j=1}^{4}b_j\{\exp(-k_jt-h_jE)-1\},\qquad
b_j=\frac{a g_j}{5}\ge0,\quad k_j=\frac{d_j}{u_t}>0,\quad
h_j=\frac{\gamma_j}{u_E}>0.$$

The final logit is the common learned base logit plus this branch. Each component
term is readable, nonpositive and nonincreasing in either input at fixed context.
Time and artist dose interact within its exponential; there is no unique additive
allocation of a component into separate time and dose contributions without an
additional attribution convention. The artist count is an observed history
feature, not a measured causal competition dose.

Simultaneously permuting the four rates, event rates and gap coefficients leaves
the branch unchanged. Component indices are therefore not identified lifecycle
stage names. The mean is a finite exponential mixture already familiar from
established constructions; neither the decomposition nor its elementary bound
is claimed as a new interpretability algorithm.

Let $B=\sum_jb_j$. For every nonnegative time/dose,

$$-B\le r(t,E)\le0,\qquad
0\le\sigma(\ell_{\mathrm{base}})-\sigma(\ell_{\mathrm{base}}+r)\le B/4.$$

The probability bound follows from the sigmoid derivative being at most one
quarter. It is a fixed-base influence bound, not an AP bound, calibrated event
risk, causal explanation or whole-score monotonicity certificate. Probability
rounding is allowed only within the recorded native arithmetic tolerance.

## All retained validation scores and direct influence

The native scorer was replayed for both candidates of every January query at
all three seeds: 243600 candidate scores. Every logit exactly equals its stored
base plus residual in native arithmetic; component sums reconstruct the branch,
final probabilities match the retained fit, and bounds/permutation checks pass.
The table reports percentage points for probability changes, not relative percent.
Zeroing the branch keeps the SAME jointly trained base fixed; it is not retraining
the separate `history_only` arm. Tiny AP differences here do not establish
inferiority or bound ranking effects in other data.

| Seed | Mean probability reduction | Maximum observed | Analytic upper bound | Full − fixed-base AP |
|---|---|---|---|---|
| `101` | `0.041977 pp` | `0.066030 pp` | `0.066027 pp` | `-0.000001222` |
| `202` | `0.087420 pp` | `0.145975 pp` | `0.145967 pp` | `-0.000011320` |
| `303` | `0.068449 pp` | `0.117195 pp` | `0.117190 pp` | `-0.000001179` |

A fixed first-query example at seed `101` has base logit
`-0.054318547`, branch `-0.001698892` and final
logit `-0.056017440`. Its model probability changes from
`48.642367%` to `48.599929%`.
The first/middle/last positive-query examples at every seed and all component
half-lives in physical seconds and recorded-artist event units are retained.
These are mathematical response scales, not observed relationship lifetimes.

## What can and cannot be claimed

This version exposes a faithful additive account of its small constrained
branch. The bound concerns its direct evaluation effect; joint training can also
change the base, so it does not prove the branch was irrelevant during fitting. It does not make the dominant common encoder/history scorer transparent.
No human explanation study, external faithfulness experiment or semantic phase
identification has been performed. A readable branch is useful for inspecting
its behavior, but these outcomes do not demonstrate improved whole-model
interpretability over a comparator. Any future explanation contribution must
evaluate coverage, semantic validity and usefulness beyond this algebraic account.

`LCR-E-LASTFM-READOUT-001` / `LCR-JOB-LASTFM-READOUT-001` map
`results/audit/lastfm_readout_v1/`. Independent NumPy checkpoint-derived formulas,
rank-based metric reconstruction and complete stored-array consistency pass;
parent native inference and source/input audits remain inherited. This is not
optimizer or full independent raw-history reproduction. No fit, unseen test,
remote allocation or publication admission occurred. Retrieval:
`evidence/indexes/lastfm_diagnostics_v1.toml`.

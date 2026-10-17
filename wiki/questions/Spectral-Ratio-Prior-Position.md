---
title: Spectral Ratio Prior Position
status: exact representation and predicate separation checked; priority unresolved
last_updated: 2026-10-16
paper_source: false
---

# What is distinct from established ratio-monotonicity results?

The candidate is the [existing state-uniform absorbing-readout certificate](../methods/Fixed-Rate-Readout-Certificate.md)
and its [inverse-tilt repair](../methods/Inverse-Tilt-Projection.md), not the
general observation that normalized weighting can reverse monotonicity.
This targeted comparison does not settle priority or predictive utility.

## Exact representation

Let $p_i=z_i(0)$. For the absorbing star flow,

$$z_i(t)=p_i e^{-d_it},\qquad z_0(t)=1-\sum_i p_i e^{-d_it},$$

$$G(t)-w_0=
\frac{\sum_i\delta_i c_i p_i e^{-d_it}}
{c_0+\sum_i(c_i-c_0)p_i e^{-d_it}}.$$

The denominator is positive because it is a positive tilt's scalar product
with a simplex state. For distinct positive active rates, the natural expansion
has $\beta_0=c_0$, $\beta_i=(c_i-c_0)p_i$, $\alpha_0=0$ and
$\alpha_i=\delta_i c_i p_i$. If $p_i>0$ and $c_i>c_0$, its denominator
coefficients are positive, with

$$\alpha_i/\beta_i=\delta_i c_i/(c_i-c_0).$$

A tilt below its dormant tilt gives a negative denominator coefficient despite
the positive normalizer. Equal/zero rates require grouping terms, and zero
initial masses require dropping absent terms before coefficient quotients are
formed. These remarks concern this representation, not every alternative kernel.

## Inspected primary results

| Source | Inspected result | Relevance and limit |
|---|---|---|
| [Yang and Tian, preprint](https://arxiv.org/html/1803.02513v1), [published article](https://doi.org/10.1016/j.jmaa.2018.10.034) | Theorem three: continuous Laplace integrands, a positive denominator integrand, a single quotient-direction change, endpoint limits and an endpoint test. | Ratio monotonicity and endpoint tests are established. The statement is not itself a finite constraint over every initial simplex state of the coupled absorbing family. An equivalent specialization is not ruled out. |
| [Mao and Tian](https://arxiv.org/html/2306.04659v1) | Series rules and discrete-Laplace corollaries use coefficient-quotient shape, positivity, convergence and endpoint conditions. | Exponential-sum ratio rules predate this work. A shape predicate alone must not be confused with their complete endpoint-refined statements. |
| [Karp, Vishnyakova and Zhang](https://arxiv.org/html/2408.01755v2), [published article](https://doi.org/10.1007/s00025-025-02431-4) | The series theorem uses a sign-regular kernel, positive denominator coefficients and unimodal coefficient quotients; the integral theorem and following remark address corresponding ratios and endpoint derivatives. | This broad preservation framework is prior art. Its inspected sufficient shape predicate is not necessary for the absorbing family, as the example below shows. This does not subsume or improve the full paper. |

The primary HTML was retained for all three preprints. Inspection covers these
theorem/corollary sections and relevant definitions/proof passages, not every
proof or later publication. Publisher identity was fetched directly for Karp.
Direct publisher requests for the Yang and mixture articles returned access
errors, retained in the log; Yang's preprint remained accessible. Earlier
mixture-weighting priors remain in the [existing review](Contribution-Gap-Review.md#expanded-access-to-weighted-aggregation-priors).

## Exact separation from the inspected sufficient predicates

The following are representation/assumption examples, not counterexamples to
the cited theorems. A sufficient result need not cover every safe case.

### Safe readout with more than one quotient-direction change

Take

$$c=(1,2,11/10,2,11/10),\quad
\delta=d=(1,2,3,4),\quad z(0)=(1,1,1,1,1)/5.$$

All denominator coefficients are positive. Ordered by rate including the
constant term, the coefficient quotients are

$$0,\ 2,\ 22,\ 6,\ 44.$$

Their adjacent signs are positive, positive, negative, positive, so neither
unimodal orientation holds. The absorbing readout is nevertheless
nonincreasing for every initial simplex state. Since $\delta=d$, the existing
derivative identity gives, including zero dormant gap/rate,

$$-\dot G=\operatorname{Var}_q(\delta)
+\mathbb E_q[\delta]\,\mathbb E_q[\delta c_0/c]\ge0.$$

Every exact pair margin is positive. This shows the inspected sufficient
quotient-shape predicate is unnecessary here. It does not separate the result
from every sign-regularity theorem, other kernels or additional hypotheses.

### Identical quotient shape with safe and unsafe rates

Let $c=(1,4,4)$, $\delta=(2,1)$ and $z(0)=(0,1/2,1/2)$. For both
$d=(1,2)$ and $d=(1,10)$, the increasing-rate coefficient quotients are

$$0,\ 8/3,\ 4/3.$$

For the first rate vector, the all-state pair margin is $5/4>0$ and the initial
derivative is $-5/16$. For the second the margin is $-15/4+\sqrt5<0$, and
the initial derivative is $3/16>0$. Quotient ordering alone therefore cannot
decide monotone decrease. Existing endpoint refinements can distinguish
trajectories in this situation; the example does not refute them.

### Safe positive normalizer with a negative spectral coefficient

For $c=(1,1/2,2)$, $\delta=d=(1,2)$ and uniform initial mass, the denominator is

$$1-\frac16 e^{-t}+\frac13 e^{-2t}.$$

It stays at least $1/2$, since it is a positive weighted simplex sum, although
its middle spectral coefficient is negative. Co-ordering certifies the readout
for every initial state. This natural expansion fails positive-denominator-
coefficient hypotheses. No universal nonrepresentability claim follows.

## Precise working contribution and unresolved priority

The working mathematical statement is a finite necessary-and-sufficient test
for the specified absorbing star flow, fixed positive tilt and nonnegative-gap
readout over every initial simplex state at specified rates, with a rate-cone
extension, explicit unsafe witnesses and an exact halfspace repair in stated
coordinates. It handles these safe examples without the inspected coefficient-
shape predicate.

Do not call this the first monotonicity theorem for normalized mixtures,
exponential sums or Laplace ratios. The all-positive-tilt co-ordering corollary
is not a newly discovered generic covariance principle. Existing results might
imply an equivalent specialization after another transformation; that priority
question remains open. The support, binary copositivity and QP tools remain
acknowledged prior mathematics.

The [retained-model repair](../results/Fixed-Rate-Repair-Validation.md) supplies
scoped operating evidence without an AP advantage. The [February holdout](../results/LastFM-February-Holdout.md)
still fails local confirmation. No model, data, checkpoint selection, endpoint
or publication pointer changes from this review.

## Verification and retrieval

LCR-P-SPECTRAL-POSITION-001 records this post-discovery comparison.
LCR-E-SPECTRAL-POSITION-001 / LCR-JOB-SPECTRAL-POSITION-001 retain symbolic
representation/derivative checks in four finite dimensions, exact rational
fixtures, and separate endpoint/stationary-point minimization on all nineteen
simplex edges across the four cases. The all-state implication still uses the
existing support/co-ordering proof; this is not independent peer review.
No floating grid substitutes for proof, and no training or remote runtime ran.
Source-access failures are preserved.

Artifacts: results/audit/spectral_prior_position_v1/.
Retrieval: evidence/indexes/spectral_prior_position_v1.toml.
No scientific claim is admitted.

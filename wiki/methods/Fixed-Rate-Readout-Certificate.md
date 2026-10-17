---
title: Fixed Rate Readout Certificate
status: working exact fixed-rate and conic characterization with untrained ordered-rate prototype
last_updated: 2026-10-16
paper_source: false
---

# A certificate for the rates actually specified

The [earlier condition](Readout-Aware-Entropic-Preservation.md) certifies a fixed
positive tilt and a nonnegative-gap readout uniformly over every positive decay
rate vector. A trained or otherwise specified rate vector is a different
quantifier. This note characterizes its exact all-state feasible set and extends
it to specified nonnegative rate channels. It does not establish predictor utility,
priority or a guarantee for arbitrary context-dependent transformations.

## Domain and exact fixed-rate condition

Let $z$ lie in the simplex, with dormant index $0$ and active indices
$j=1,\ldots,m$. The dormant mass $z_0$ may be any nonnegative value. Fix $c_i>0$, gaps $\delta_j\ge0$ and a
finite rate vector $d_j\ge0$. Define

$$\dot z_j=-d_jz_j,\quad\dot z_0=\sum_jd_jz_j,\qquad
q_i=\frac{c_iz_i}{c^Tz},\qquad G=w_0+\sum_j\delta_jq_j.$$

Tilts, readout and all context remain fixed along a path. Put

$$A_i=\frac{c_0\delta_id_i}{c_i},\qquad
B_{ij}=(\delta_i-\delta_j)(d_i-d_j)
+c_0\left(\frac{\delta_id_j}{c_j}+\frac{\delta_jd_i}{c_i}\right).$$

**Working theorem.** For this fixed rate vector, $G$ is nonincreasing for every
initial simplex state and all nonnegative flow times if and only if

$$\boxed{B_{ij}+2\sqrt{A_iA_j}\ge0\quad\text{for every active pair }i<j.}$$

The diagonal conditions $A_i\ge0$ follow from the domain. With one active
component there is no pair restriction. Zero gaps/rates are included. This is an
all-initial-state result; failure does not imply failure on a particular retained
query or a fixed uniform-reset trajectory.

### Proof: the derivative and a two-support reduction

Set $\delta_0=a_0=e_0=0$ and, for active indices,

$$a_i=\delta_id_i,\qquad e_i=d_i(1-c_0/c_i).$$

Direct quotient differentiation gives

$$-\dot G=F(q)=a^Tq-(\delta^Tq)(e^Tq).$$

The positive reweighting is a bijection of the simplex, with inverse
$z_i=(q_i/c_i)/\sum_k(q_k/c_k)$. Thus an all-state sign condition can be checked
on every $q$ in the simplex.

For any $q$, fix $s=\delta^Tq$ and minimize $F(p)$ over
$\{p\ge0:\mathbf1^Tp=1,\delta^Tp=s\}$. On this nonempty compact slice,
$F(p)=(a-se)^Tp$ is linear. It attains a minimum at an extreme point. Such a
point has at most two positive coordinates: with more than the rank of these
two equalities, a nonzero supported null direction would permit both positive
and negative feasible perturbations, contradicting extremality. If the equalities
are dependent the support bound only improves. Therefore a negative derivative
margin anywhere has a negative margin on some simplex edge.

On the edge $q_i=u,q_j=v,u+v=1$, homogeneous expansion yields

$$F=A_iu^2+B_{ij}uv+A_jv^2.$$

The binary form is nonnegative for all $u,v\ge0$ exactly when its nonnegative
diagonals and $B_{ij}+2\sqrt{A_iA_j}\ge0$ hold. Sufficiency follows by writing it
as $(\sqrt{A_i}u-\sqrt{A_j}v)^2+(B_{ij}+2\sqrt{A_iA_j})uv$.
If both diagonals are positive, a failed inequality is witnessed by
$(u,v)$ proportional to $(\sqrt{A_j},\sqrt{A_i})$. If a diagonal vanishes,
a sufficiently small positive mass on the other component gives a negative
value when the cross coefficient is negative. The implementation handles both
zero-diagonal cases explicitly. Dormant/active edges have $A_0=0$ and
$B_{0j}=a_j\ge0$, so they impose no additional restriction.

Sufficiency applies at every state along the flow. Necessity follows from an
unsafe edge state used as an initial condition. Continuity also supplies a nearby
strictly interior witness when an interior-only state domain is intended. This
proves the arbitrary-dimension real-arithmetic statement.

This is a special structured quadratic, not a general pairwise copositivity
algorithm. For comparison, a matrix with diagonal entries one and all off-diagonal
entries $-3/4$ passes every binary test but is negative at the uniform three-vector.
The two-support reduction above is essential.

## Exact extension to a finite cone of rate channels

Let admissible instantaneous rates be
$d(t)=\sum_{k=1}^r\lambda_k(t)v^{(k)}$, with nonnegative finite generators,
nonnegative locally integrable coefficients and a well-defined absorbing flow.
The readout is nonincreasing for every such schedule and every initial state
if and only if the boxed pair condition holds for each generator separately.
At a fixed state the derivative is linear in the rate vector, so the generator
conditions suffice pointwise. Necessity uses a schedule containing only one
admissible generator. If an executor does not permit such schedules, necessity
for that smaller executor is not asserted.

The same condition certifies an explicitly defined exponential event jump
$z_j^+=z_j^-\exp(-v_jE)$, $E\ge0$: integrate its generator flow for duration $E$.
Thus separate physical-time and recorded-dose generators can both be certified.
This does not equate an exponential jump with a linear Euler/SDE jump, and own-edge
resets or changes of context/tilt/readout remain outside the guarantee.
Checking the generators requires $O(rm^2)$ pair margins in this family.

## Two exact corollaries clarify the design tradeoff

**All positive rates.** Requiring the condition for every positive $d$ recovers
$(\delta_*-\delta_i)c_i\le\delta_*c_0$. Necessity follows by making $d_i$ dominate
$d_j$ in the pair margin; its leading coefficient must satisfy
$\delta_i-\delta_j+\delta_jc_0/c_i\ge0$. Sufficiency is the earlier all-rate result.
The nonnegative orthant can equivalently be checked using its unit generators.

**All positive constant tilts.** For a fixed $d$, every positive tilt is safe
if and only if gaps and rates are co-ordered:

$$(\delta_i-\delta_j)(d_i-d_j)\ge0\quad\text{for every active pair}.$$

Sufficiency is immediate from the boxed condition. Equivalently,
$F(q)=\operatorname{Cov}_q(\delta,d)
+\mathbb E_q[\delta]\,\mathbb E_q[dc_0/c]\ge0$, including the dormant index
with zero gap/rate; the covariance is a sum of co-ordered pair products.
For necessity, an oppositely ordered pair has a negative product. Give its two
active tilts the common value $M$, with $c_0=1$, and take equal $q$ mass on that
pair. Then $F$ tends to that negative product divided by four as $M$ grows.
Some finite positive tilt is therefore unsafe.

These exchange different uniform quantifiers. A co-ordered-rate constructor
restricts rates while freeing tilts; it is not an unrestricted replacement for
every all-rate architecture or dynamic transition network.

## Strict separation and executable prototype

Take $c=(1,4,4)$, $\delta=(1,2)$ and $d=(1,2)$. Co-ordering certifies every
constant tilt, while the earlier all-rate inequality fails for component one.
At $z=(1/10,1/10,4/5)$,

$$\dot q_1=56/1369>0,\qquad \dot G=-1416/1369<0.$$

This configuration is safe for the fixed rates despite lying outside the old
all-rate feasible set, and it shows why requiring every active coordinate to
decline is stronger than certifying the chosen readout.
Changing only the rates to $d=(10,1)$ gives an unsafe example: at
$z=(0,1/2,1/2)$, $\dot G=3/16>0$. A zero-gap example is also retained.

`modeling/fixed_rate_tilt.py` implements pair margins, unsafe-state witnesses,
rate-cone checks and `OrderedRateTilt`. The prototype learns cumulative positive
gaps and cumulative positive physical/event rates with arbitrary fixed active
log tilts. It supports both certified directions and realizes the strict-separation
example. It is untrained and not connected to a completed predictor experiment;
no common affinity, semantic stage label or accuracy improvement is certified.

The first prototype used a masked log-zero reweighting. A boundary derivative
test exposed a missing emerging-mass gradient. It was replaced by direct positive
normalization, and exact boundary/interior derivative tests now pass. Underflowing
scaled tilt weights are rejected rather than assigned an artificial floor.
The earlier untrained `ReadoutAwareTilt` remains unchanged, with the same boundary
gradient limitation explicitly recorded in its method owner. Its theorem and
forward boundary values are separate from that automatic-differentiation issue.

## Verification, sources and remaining claim boundary

`LCR-E-FIXED-RATE-TILT-001` binds the written proof, source, tests and artifacts
in `results/audit/fixed_rate_tilt_v1/`. Exact symbolic identities in several
specified dimensions, rational witnesses and independent fixed-moment LP checks
support the derivation; they are not a proof assistant or peer review of the
arbitrary-dimension argument. Float near-zero margins are reported as boundary
cases, not formal machine-arithmetic certificates. No fit or GPU allocation ran.

The binary quadratic criterion is established mathematics, for example
[Kannike, section two](https://arxiv.org/pdf/1205.3781). The finite LP support
argument is elementary; the broader moment-extreme-point framework predates this
project, as the publisher abstract for
[Winkler](https://pubsonline.informs.org/doi/abs/10.1287/moor.13.4.581) confirms.
Only that abstract/metadata, not Winkler's complete proof, was inspected.
Neither copositivity nor the support principle is claimed new.

The expanded [source comparison](../questions/Contribution-Gap-Review.md#expanded-access-to-weighted-aggregation-priors)
now uses the retrieved original weighting draft and a primary mixture-monotonicity
paper. Their generic normalization/monotonicity principles are prior art.
The candidate contribution is the exact absorbing-readout fixed-rate/conic
specialization, its quantifier tradeoff and executable safe family, together with
existing learned-score-path evidence. Exhaustive priority, external proof review,
trained utility and a whole-predictor certificate remain open. Both publication
pointers stay `UNRELEASED`.

## Exact feasible geometry and no-refit application

The [inverse-tilt development](Inverse-Tilt-Projection.md) now factors the pair
margin into linear lower halfspaces in square-root inverse tilts, with a precise
convex repair objective. The [complete retained-model study](../results/Fixed-Rate-Repair-Validation.md)
shows zero additional modification is needed for the frozen-reference gradient
comparators at the actual learned rates. Its AP primary and dynamic-model
limitations remain explicit; no trained or whole-predictor gain is inferred.


## Position relative to exponential-ratio theorems

The [spectral comparison](../questions/Spectral-Ratio-Prior-Position.md) now makes the exact ratio representation explicit and verifies rational safe configurations outside specific inspected coefficient-shape predicates. It preserves established Laplace/series monotonicity and endpoint tests as prior art. No claim of dominance over those complete theories, or absence of an equivalent specialized result, is made.

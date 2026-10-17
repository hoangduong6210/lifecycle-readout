---
title: Readout Aware Entropic Preservation
status: working exact restricted theorem and implemented constructor; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Why certify the readout rather than every phase?

The earlier [simplex construction](Monotone-Simplex-Development.md) requires every
active transformed coordinate to decline. That is sufficient for any readout
whose active weights exceed the dormant weight, but may be stronger than needed
for a particular learned readout. This note supplies an exact characterization
for a restricted family, plus an executable example in which an active phase
increases while the chosen readout remains certified. It does not supply an
exact characterization for arbitrary dynamic transition networks.

## Family and assumptions

Let $x_j=z_j$ for active $j=1,\ldots,m$, $z_0=1-\sum_jx_j$, with fixed positive
rates $\dot x_j=-d_jx_j$. Fix positive tilt factors $c_i$ and readout gaps
$\delta_j=w_j-w_0\ge0$. Put

$$q_i=\frac{c_iz_i}{\sum_\ell c_\ell z_\ell},\qquad
G=w_0+\sum_{j=1}^m\delta_jq_j,\qquad
\delta_*=\max_j\delta_j.$$

The tilt, readout, initial encoder state, rates and any external affinity are
fixed along the path. Increasing links and a fixed nonnegative affinity preserve
the derivative sign. This family is normalized positive reweighting, not an
invented exponential update.

## Exact readout-specific condition

**Proposition (working analytic result).** The readout $G$ is nonincreasing for
every initial simplex state, every fixed strictly positive rate vector, and
all nonnegative silence times if and only if

$$\boxed{(\delta_*-\delta_j)c_j\le\delta_*c_0\quad\text{for every active }j.}$$

If all gaps vanish the readout is constant and the inequality is vacuous.
The proposition is uniform over rate vectors; it is not a necessary condition
for one particular fixed rate vector or one observed trajectory.

### Proof

Write

$$Z=c_0+\sum_i(c_i-c_0)x_i>0,\qquad N=\sum_i\delta_ic_ix_i.$$

Since $G=w_0+N/Z$,

$$\partial_jG=\frac{\delta_jc_jZ-N(c_j-c_0)}{Z^2}.$$

In the interior, $\dot G=-\sum_jd_jx_j\partial_jG$. It is nonpositive for every
positive rate vector if and only if all partial derivatives are nonnegative.
If a partial is negative, choose that positive rate sufficiently large relative
to the others. The chosen interior state is an admissible initial state, so a
positive derivative at the initial time violates the required global property.

The numerator $H_j=\delta_jc_jZ-N(c_j-c_0)$ is affine in active coordinates.
It is nonnegative on the closed active simplex if and only if it is nonnegative
at all vertices. At the dormant vertex,

$$H_j=\delta_jc_jc_0\ge0.$$

At the active vertex $x=e_i$,

$$H_j=c_i[\delta_jc_j-\delta_i(c_j-c_0)].$$

If $c_j\le c_0$, all these values are nonnegative. If $c_j>c_0$, their minimum
uses a maximal gap, so their nonnegativity is equivalent to
$\delta_jc_j\ge\delta_*(c_j-c_0)$. These cases combine into the boxed inequality.
If it fails, $H_j$ is negative at a vertex and, by continuity, at some interior
point. The rate argument then proves necessity. Sufficiency applies along the
whole flow. The positive denominator and continuity extend sufficiency to
boundary initial states. This completes the real-arithmetic proof.

## What this condition adds

For a single active phase, the condition always holds. If all active readout
gaps are equal, it also always holds: individual active coordinates need not
all decline for their equally weighted sum to decline.

When certifying every nonnegative-gap readout simultaneously and there are
multiple active phases, coordinatewise decline requires $c_j\le c_0$ for every
active coordinate. The readout-specific condition is weaker: phases with a
maximal readout gap have no upper tilt restriction, while lower-gap phases obey
bounds determined by their gaps. This is a separation from a coordinatewise
certificate, not a claim that the entire family contains every earlier nonlinear
simplex architecture.

### Exact strict-separation witness

Choose

$$z=(1/5,1/5,3/5),\quad d=(1/10,5),\quad
c=(1,3/2,4),\quad\delta=(1,2).$$

The boxed inequalities hold. At the initial time,

$$\dot q_1=1308/4205>0,\qquad \dot G=-11868/4205<0.$$

Thus enforcing decline of each active coordinate would exclude this trajectory,
although the chosen readout is certified. The implementation can represent this
witness; a test sets its parameters and compares its output and derivatives.

A necessity witness outside the feasible set uses
$z=(19/100,1/100,4/5)$, $d=(1000,1/10)$, $c=(1,4,1)$ and the same gaps.
Its readout derivative is positive. These examples establish mathematical
possibility, not learned phase semantics or predictive gain.

## Connection to a zero-diagonal transition matrix

Set $b_i=\log c_i$ and $T_{ij}=b_j-b_i$. Then $T$ is skew-symmetric with zero
diagonal, and for simplex states

$$T^Tz=b-(b^Tz)\mathbf1,\qquad
\operatorname{softmax}(\log z+T^Tz)=\operatorname{softmax}(\log z+b).$$

The proposition therefore describes an exact restricted subfamily of the
candidate map already in this project, not a replacement by an unrelated
linear Markov operator. It does not rescue arbitrary skew-symmetric matrices;
the earlier cyclic counterexample remains valid. The softmax row-common gauge
invariance itself is elementary and not claimed as new.

## Implemented trainable interior

`src/lifecycle_readout/modeling/readout_aware_tilt.py` learns cumulative positive
gaps. With the last active gap maximal and $c_0=1$, lower phases use

$$b_j=\log\delta_*-\log(\delta_*-\delta_j)
-\operatorname{softplus}(\rho_j),$$

while the last phase's tilt is free. Positive suffix sums compute the gap
difference without subtracting nearly equal cumulative sums. This parameterizes
a strict interior for nonmaximal phases; the maximal phase needs no bound.
The constructor uses boundary-safe logarithms without a positive state floor.
Finite-precision tests are not a formal machine-arithmetic certificate.
The module has not been trained or substituted into a completed experiment.
No context encoder is implicitly certified, and zero contribution to a base
predictor is not a claimed property of this multiplicative gate prototype.

## Verification and priority boundary

`LCR-E-READOUT-TILT-001` maps source, tests, the written proof and symbolic
verification under `results/audit/readout_aware_tilt_v1/`. Exact symbolic algebra
checks the quotient derivative and all vertex identities at several dimensions;
an exact rational calculation verifies the separation witness. Unit tests cover
unequal rates, phase counts, boundary states, gradients, representability of the
witness and an outside-condition failure. These support the proof; they are not
human peer review, an induction proof produced by software or model efficacy.

The ratio $N/Z$ is an established linear-fractional object; see the official
[Boyd and Vandenberghe slides](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf),
slide `2.16` and the linear-fractional example at `3.40`. Only those definitions
were inspected for this positioning note. The normalized exponential update's
optimization precedent remains with the [method owner](Shortcut-Resistant-Gating.md).
The precise scoped preservation condition and constructor have not received
priority clearance. Permitted working language is an exact restricted
characterization and constructive separation; “first,” a general dynamic-transition
solution, new linear-fractional mathematics or improved ranking remain unsupported.

## Fixed-rate refinement and boundary-gradient scope

The [fixed-rate result](Fixed-Rate-Readout-Certificate.md) now gives the exact
all-state condition for specified rates or finite rate cones. It includes a
safe configuration outside this page's stronger all-rate condition. The original
universal-rate proposition remains unchanged.

A new explicit zero-mass boundary probe finds that this earlier untrained
constructor's masked-log implementation loses the derivative of emerging dormant
mass at exactly zero elapsed time. Forward values remain correct; boundary value
and interior gradient tests did not establish that boundary derivative. This
module is retained unchanged as the earlier prototype. The new ordered-rate
constructor uses direct positive normalization and checks the exact boundary
right derivative. No completed fitted model or reported AP uses either untrained
constructor, and no prior result is refitted or silently replaced.

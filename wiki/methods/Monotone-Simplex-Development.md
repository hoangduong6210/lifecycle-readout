---
title: Monotone Simplex Transition Development
status: exploratory architecture and scoped algebra checks; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Selected development direction

The user authorized choosing and developing one direction from the supplied
brainstorm. We select preservation through the candidate transform: the existing
[method owner](Shortcut-Resistant-Gating.md) already contains a constructive
failure and a conservative sufficient bound. This is a separate exploratory
prototype, not an amendment to any completed experiment or training protocol.

## Screening the alternatives

| Direction | Useful element | Correction needed before stronger claims |
|---|---|---|
| Certified transition | Enforce the derivative condition structurally | Skew symmetry alone is insufficient; a certificate restricts the function class |
| Differential gating | Compare pairwise and pointwise objectives under the same sampler | A query-common linear reference offset cancels from pairwise differences; BCE is not proved to cause the recorded losses |
| Kernel history matching | Study support and residual within-bin imbalance | Whole-corpus unsupported counts are not training-only counts; reweighting needs an explicit estimand and overlap, and is not automatically causal or doubly robust |
| Hierarchical clocks | Adaptive fallback deserves a controlled comparison | Source, destination and pair clocks are not automatically orthogonal; sparse variance estimates and time-unit normalization need definition |
| Synthetic regime shifts | Use known simulated mechanisms to test failure boundaries | A generator can favor the imposed architecture; robustness over an MLP is a hypothesis, not a conclusion to build into the benchmark |

## A counterexample to the spectral shortcut

For the ideal existing candidate map $q=\operatorname{softmax}(\log z+T^Tz)$,
choose

$$
z=(1/8,4/25,143/200),\qquad d=(2,1/4),\qquad
T=\begin{pmatrix}0&5&12\\-5&0&2\\-12&-2&0\end{pmatrix}.
$$

The initial coordinates sum to unity. Along the ordinary absorbing decay flow,
$T=-T^T$ yet the derivative of $q_1+2q_2$ is strictly positive at the initial
state. The exact executable example and derivative are retained in
`tests/test_monotone_simplex.py` and the local audit. The sigmoid gate therefore
also increases. The existing two-phase cancellation does not generalize to
arbitrary skew-symmetric many-phase matrices. This example does not disprove a
particular additional spectral bound that has actually been proved; it refutes
skew symmetry as a standalone certificate.
An additional cyclic skew-symmetric example with zero row and column sums also
reverses the gate; tangent-space mass conservation alone is insufficient. Its
matrix and executable check are retained in the expanded test source.

## Exact local criterion for preservation over all decay directions

Use active coordinates $x\in\mathbb R_+^m$, $\sum x_k\le1$, with the dormant
coordinate supplied by mass conservation. Let $F(x)$ be the transformed active
coordinates and $G(x)=\delta^TF(x)$, where $\delta_k=w_k-w_0\ge0$.
For a differentiable map at an interior state,

$$
\dot G=-\sum_j d_jx_j\partial_jG.
$$

For every strictly positive rate vector, $\dot G\le0$ holds at that state if
and only if $\partial_jG\ge0$ for every active coordinate. Sufficiency follows
term by term. For necessity, a negative partial derivative can dominate by
increasing its positive rate relative to the others. This is a local criterion
for all rate vectors, stronger than certification for a single fixed vector.
For all nonnegative readout gaps, the corresponding criterion is entrywise
nonnegative active Jacobian $J_F$. At a zero coordinate its decay contribution
vanishes; the interior necessity statement must not be applied blindly there.
This is an elementary cone/chain-rule characterization, not a priority claim.

## Implemented sufficient construction

Let $A$ be nonnegative and row-substochastic, obtained as the active columns of
a row-softmax over dormant plus active destinations. For a fixed initial state,
fixed rates and fixed learned parameters, set

$$
x_k(t)=z_k(0)e^{-d_kt},\quad S=\sum_kx_k,\quad
h(S)=\sigma(a+\operatorname{softplus}(b)S),\quad
F(x)=h(S)A^Tx,\quad q_0=1-\mathbf1^TF(x).
$$

There is also a linear control with $h\equiv1$. Nonnegativity and row sums give
$\mathbf1^TF\le S\le1$, so the output lies on the simplex, including its boundary.
Moreover,

$$
\frac{\partial F_k}{\partial x_j}
=h(S)A_{jk}+h'(S)(A^Tx)_k\ge0.
$$

Consequently every active transformed coordinate is nonincreasing along any
positive-rate absorbing decay trajectory. Any gate with nonnegative
active-to-dormant gaps is nonincreasing too. A fixed nonnegative affinity
multiplier or a fixed base logit followed by an increasing link preserves the
same direction. No ordering among active readout weights is required.
This proof covers real arithmetic. Floating-point checks exercise boundary,
long-silence and unequal-rate cases but are not a formal machine-arithmetic proof.

Implementation: `src/lifecycle_readout/modeling/monotone_simplex.py`.
It accepts a simplex initial state, nonnegative elapsed time and positive rates;
it validates shapes, finite values, dtype and device. The module learns mixing
and throttle parameters with ordinary differentiation and needs no projection
step. It does not take logarithms of zero-valued state coordinates.

## Scope and expressivity cost

This is a candidate **readout transform along silence**, not an event-resurrection
operator. A new observed event may reset the state through a separately specified
update. Initial state, transition, rates, readout and affinity must be held fixed
when asserting the silence certificate. A time-dependent retention example in
the tests shows why changing context invalidates the guarantee. Current lifecycle
encoders are not automatically compatible: passing silence-varying features to
an encoder would break the assumptions. No existing trained model has been
replaced and no end-to-end score guarantee has been asserted for it.

The map excludes transformations that increase an active coordinate along
silence. It therefore does sacrifice expressivity relative to an arbitrary
entropic candidate jump. The linear variant also satisfies
$\delta^TA^Tx=(A\delta)^Tx$: for a single scalar gate, its mixing can be absorbed
into effective readout weights. That variant is an essential control, not a
standalone new expressive mechanism. The nonlinear throttle changes this simple
linear reduction but does not by itself establish a new function family relative
to existing monotone networks or mixture models. Phase semantics and
identifiability remain unproved.

## Prior-art boundary

Absorbing Markov models and exponential mixtures are established; the
[author-hosted phase-type tutorial](https://www.cs.cmu.edu/~osogami/thesis/html/node39.html)
explicitly gives absorbing exponential and mixture constructions. Its inspected
examples are background, not a theorem about this exact transform.
[Wang, Gao and Pajic](https://arxiv.org/abs/2006.06417) already study learning
monotone dynamics and Lyapunov constraints; the inspected abstract is enough to
reject a blanket claim that earlier neural work only treats flat positive MLPs,
but not enough to establish the closest architectural overlap. Existing entropic
update precedent remains owned by the [method note](Shortcut-Resistant-Gating.md).
We have not completed a priority search for this parameterization. No “first,”
new spectral theorem, universal Lyapunov guarantee or journal readiness claim is
made. The useful development is a directly checkable repair candidate plus its
expressivity and context boundaries.

## Verification and next gate

`LCR-E-SIMPLEX-DEV-001` and `LCR-JOB-SIMPLEX-DEV-001` map the local source, tests
and verification record under `results/audit/monotone_simplex_development_v1/`.
Checks cover arbitrary nonnegative readout gaps, unequal rates, several phase
counts, boundary states, long silence, parameter gradients, optimizer updates,
the skew-symmetric counterexample, changing-context failure and linear reduction.
No data benchmark, model-quality gain, GPU allocation or new test scoring occurs.

Before a trained comparison, seal a separate validation-only development design
that fixes a silence-invariant encoder context, controls parameter budget and
includes pure flow, linear absorbing mixing, nonlinear throttle, the existing
entropic map and an unconstrained MLP. Compare predictive utility and the
constraint cost; do not optimize only the guaranteed monotonicity rate. Human
proof review and closest-architecture literature review remain open. Existing
negative outcomes and finalization dependencies remain unchanged.

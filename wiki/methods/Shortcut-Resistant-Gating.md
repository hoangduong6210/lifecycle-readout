---
title: Shortcut-Resistant Lifecycle Gating
status: conditional guarantee with completed mixed hard-control evidence
last_updated: 2026-10-16
paper_source: false
---

# Lifecycle Survival Gating

The proposed mechanism aims to improve prediction on inactive relations and
reduce reliance on temporal frequency cues. This motivation is recorded in
`DEC-0002`; resistance to shortcuts remains a research question. Existing
transductive gain over an internal base model is insufficient to answer it.

## Scored path

$$\hat y_{uv}=\sigma(f_{\mathrm{base}}(h_u,h_v))\,
\sigma(w^\top z_{\mathrm{next}}+b).$$

The ordered variant parameterizes weights as:

$$w_0=w_{\mathrm{base}},\qquad
w_k=w_{\mathrm{base}}+\sum_{j=1}^k\operatorname{softplus}(\delta_j).$$

When this gate is applied directly to the exact pure-flow state, positive
active-phase decay toward phase zero yields a non-increasing response at fixed
base score and initial encoder state. This is not automatically a property of
the scored transformed state shown above: even a fixed candidate transformation
can reverse the response, as derived below. The ordering and direction are
designed constraints, although their magnitudes are trained. The finite dormant
gate is $\sigma(w_0+b)$; it is not necessarily zero. Changing affinity, encoder
context or numerical clamp regions also falls outside the pure-flow guarantee.

## Model interventions

The silence trajectory propagates phases through the pure silence flow without
a new event jump. Phase replacement measures changes in the model's score when
its internal phase input is replaced. These are model-level interventions, not
identified treatment effects on observed real-world relationships.

An untrained ordered module can already satisfy monotonicity and increasing
phase replacement scores. These endpoints therefore check implementation
consistency, not self-discovered causal semantics. An unordered-gate comparator
is independently trained on the full training split and selected on validation;
no ablation is fit on test positives.

## Required empirical evidence

[The corrected protocol](../../protocols/lifecycle_tkde_v2.toml) requires
strictly earlier destination sampling, shared candidate history rules,
validation-only selection, complete failure/coverage records, and exact gate
metrics. [The benchmark](../../src/lifecycle_readout/benchmark.py) saves
per-event predictions and recurrence/silence strata for further comparison.

Matched official baselines, frequency-only controls, phase stability,
additional shortcut stress tests, and source/license closure remain necessary
before admitting a shortcut-resistance claim. Historical values are preserved
in [the quarantined audit record](../results/Benchmark-and-Baseline-Results.md).

## Corrected mathematical scope on 2026-10-15

For the exact silence flow, with a fixed finite base logit and strictly positive
active decay rates, differentiating the gate logit gives

$$\frac{d}{dt}(w^T z(t)+b)=-\sum_{k=1}^{K-1}d_k(w_k-w_0)z_k(0)e^{-d_k t}.$$

Thus non-increase for every initial simplex distribution holds if and only if
$w_k\ge w_0$ for every active phase. Necessity follows by choosing $z(0)=e_k$.
Strictly larger active weights give strict decrease when initial active mass
is positive, for finite time in exact arithmetic. Ordering active phases among
themselves is not necessary. Cumulative softplus is a sufficient parameterization,
not a uniquely necessary architecture. Numerical clamping, saturation and event
jumps are outside the exact derivative guarantee.

An unconstrained gate is capable of satisfying the inequalities. There is no
impossibility result that binary event supervision must invert the dormant
weight. At a fixed gated state, a positive example contributes
$\partial L_+/\partial w_0=-(1-\Psi)z_0$, while a negative contributes
$\partial L_-/\partial w_0=p_b\Psi(1-\Psi)z_0/(1-p_b\Psi)$.
The combined update also depends on states, examples, other losses and shared
parameters. Observed inversions support an empirical mechanism hypothesis only.

Gating through the propagated state permits task-gradient paths to decay and
transition parameters; it does not guarantee a nonzero derivative for every
input or parameter. For example, decay of the dormant basis state is constant.
Clamping, cancellation and finite-precision underflow can also eliminate gradients.

## Prospective historical hard-negative controls

`LCR-P-HARD-001` is defined in
[the new protocol](../../protocols/hard_controls_v1.toml) and implemented in
[hard_controls.py](../../src/lifecycle_readout/hard_controls.py).
For each positive pair, eligible negatives share its source, have a strictly
past interaction, are absent from its simultaneous-positive set and occupy the
same bins $\lfloor\log_2 c\rfloor$ and
$\lfloor\log_2(1+\Delta t/1000)\rfloor$. There is no easy-negative fallback.
Real events without a matching negative still enter later history, but do not
contribute supervision. This is a conditional recurrent-support study and
cannot measure first-time-pair generalization.

The controls are the original neighbor-only base, Base plus pair statistics,
scalar exponential decay, flexible MLP gate and ordered lifecycle gate.
All four history-aware controls use the same six statistics and affinity head;
all variants share candidate identities, backbone initialization per seed,
batch order, optimizer, validation selection and epoch budget. Statistics are
normalized using training candidates only. The fixed 1000-second clock applies
to scalar and lifecycle decay and distinguishes this study from earlier
raw-second results. Both positive and negative elapsed times must exceed 1000
seconds to enter its prolonged-silence endpoint. Report support exclusions,
parameter counts, paired seed differences and negative outcomes.


## Baseline nesting residual candidate

The research development amendment implements
[NestedMonotoneResidual](../../src/lifecycle_readout/modeling/monotone_residual.py):

$$\ell(c,\tau)=\ell_b(c)-\sum_k a_k q_k(c)(1-e^{-d_k\tau}),
\qquad a_k\ge0,\quad q(c)\in\Delta,\quad d_k>0.$$

Zero amplitudes recover the supplied baseline logit exactly, including at large
elapsed times. Project amplitudes to nonnegative values after every update;
forward clamping also preserves the sign constraint. At fixed context and base,

$$\partial_\tau\ell=-\sum_k a_kq_k(c)d_ke^{-d_k\tau}\le0.$$

At zero elapsed, the residual vanishes; at infinite silence it approaches a
finite logit offset. This avoids a multiplicative gate's probability ceiling,
but does not guarantee optimization, calibration or generalization improvement.
Global minimum unregularized empirical loss cannot exceed the nested baseline's
minimum over the same parameter domain; practical training need not attain it.

The prototype uses deterministic contextual mixture weights. It has no recurrent
state, event transition or identified phase semantics, and is not a discovered
lifecycle. Encoder and rate gradients are zero at neutral amplitudes; amplitude
gradients permit learning away from neutrality. Changing context, history
statistics or base score invalidates the fixed-context silence guarantee.
The same information reaches all history-aware comparators; the development
study cannot ascribe gains to latent lifecycle discovery.

## A fixed candidate transformation can reverse the response

The pure-flow condition does not automatically extend to the scored state, even
when affinity, initial encoder output and transition matrix are all fixed.
This follows directly from the implemented transformation.

Consider the ideal unclamped two-phase interior with
$z(t)=(1-x(t),x(t))$, $x(t)=x_0e^{-dt}$, $d>0$, and constant matrix

$$T=\begin{pmatrix}0&a\\ b&0\end{pmatrix}.$$

The active mass after the candidate transformation is

$$q_1(t)=\sigma\!\left(\log\frac{x(t)}{1-x(t)}+a-(a+b)x(t)\right).$$

Differentiating gives

$$\frac{dq_1}{dt}=-d x q_1(1-q_1)
\left(\frac{1}{x(1-x)}-(a+b)\right).$$

With fixed positive base probability and $w_1>w_0$, the complete score
$p_b\sigma(w_0+(w_1-w_0)q_1+b_{\rm surv})$ has the same derivative sign.
Since $1/[x(1-x)]\ge4$, non-increase for every interior state is equivalent to
$a+b\le4$ for this restricted constant two-phase map. Necessity follows at
$x=1/2$; sufficiency follows from the derivative everywhere.

A constructive counterexample chooses $a=b=3$, $d=1$, $x_0=5/8$ and
$t=\log(5/4)$. Then $x=q_1=1/2$ and $dq_1/dt=1/4>0$.
The original active mass is decreasing. An ordered final gate can therefore
increase along the complete candidate path while decreasing on the pure flow.
The matrix diagonal is zero, as required by the implementation.

[Executable checks](../../tests/test_silence_scope.py) construct admissible full
modules at both two and five phases with constant encoder, transition and affinity,
and inactive numerical clamps. Autograd agrees with central finite differences;
the two-phase derivative agrees with the expression above. The five-phase case
is a configured-architecture counterexample, not a generalization of the
two-phase threshold theorem. The local record is
`results/audit/silence_scope_v1/`.

For an interior higher-dimensional state, let $v=dz/dt$ and
$q=\operatorname{softmax}(\log z+T^Tz)$. Then

$$\frac{dq}{dt}=(\operatorname{diag}(q)-qq^T)
\left[(\operatorname{diag}(1/z)+T^T)v+\dot T^Tz\right].$$

Even when $\dot T=0$, ordered final weights do not force $w^Tdq/dt\le0$.
A changing affinity adds its product-rule term. A changing encoder invalidates
the fixed-initial-state flow; clamp regions need their piecewise Jacobian.

Legacy source explanations now state this scope. Removing docstrings yields
identical executable ASTs before and after correction; the running CRAFT archive
is untouched. The pure-flow diagnostic remains valid for its declared path.
This result does not measure violations in trained checkpoints, show predictive
utility, or establish novelty over monotonic-network theory.


## Implemented pair-cadence development extension

The user-requested extension is implemented in [pair_clock.py](../../src/lifecycle_readout/pair_clock.py) and the existing control runner, behind opt-in configuration. Prior protocols retain their original feature shape and clocks. The separate [cadence protocol](../../protocols/pair_cadence_development_v1.toml) owns the new design. No sampler or supervision-support change is bundled with it.

For strictly prior pair observations, the arithmetic mean interval is

$$m_{uv}=(t_{uv}^{\rm last}-t_{uv}^{\rm first})/(c_{uv}-1).$$

It is available only when there are enough prior observations and their span is positive. A separate indicator records availability. Otherwise the implementation uses the median available cadence from supported training candidate slots; it fails if training provides no estimable clock. This is not an EMA. Simultaneous events are queried before any event in that group is observed, and held-out-node training exclusions never update the tracker.

The explicit relative clock is

$$\tau_{uv}=\log\left(1+\frac{\Delta t_{uv}}{m_{uv}^{\rm eff}+\tau_0}\right),
\qquad \tau_0>0.$$

The smoothing offset is fixed prospectively in seconds, not estimated from validation. The added context channels are log effective cadence relative to the training fallback, the availability indicator, and the relative clock. All history-aware controls receive the same channels. The capacity control masks those additional channels to zero while retaining the same network dimensions. The clock comparison keeps all channels visible and changes only the scalar/mixture residual's explicit time argument. Neighbor histories, candidates, objectives, initialization and budgets remain shared.

The implementation uses the existing baseline-nesting residual; zero amplitudes still recover the supplied jointly trained affinity score exactly. Conditional monotonicity requires fixed affinity, context, cadence and mixture weights. Because the context contains time-dependent features, this is not a global derivative guarantee for the full model. The bounded logit shift is not asserted to be Cox survival or an identified hazard.

Development constructs training and validation clocks only. Test clock values are not queried or fitted. Tiny integration checks exercise the actual runner and ensure scoring remains validation-only. No real-data efficacy fit has been allocated; the active CRAFT fit is unchanged.


### Real-input cadence verification

The [input auditor](../../scripts/audit_pair_clock_inputs.py) now checks the complete declared training and validation cadence inputs on both corpora and all development seeds. An independent timestamp-array and binary-search implementation reproduces prior counts, available mean intervals, train-only fallback and all added feature values. Test clock values remain unconstructed. Wikipedia candidate identities and original feature channels also match the retained earlier development stream. Raw support, neighbor snapshots and candidate destinations are unchanged by augmentation. Exact arrays, source identities and reports are in `results/audit/pair_cadence_inputs_v1/`.

Mutation checks reject changed clock features and a changed fallback. These are implementation/input checks, not model performance or novel-method evidence. Those comparisons were prospectively specified and unfitted at input-audit preparation. The [completed development owner](../results/Pair-Cadence-Development.md) now records their outcomes and limits.


### Cadence result audit preparation

The [study auditor](../../scripts/audit_pair_cadence_results.py) now validates each completed arm and forms paired development differences only after source, configuration, support, clock, checkpoint, normalization, prediction and metric checks. Across arms it requires identical original candidate inputs, equal exposed features for the clock comparison, common budgets and training source, and equal within-family parameter counts. The original masked-channel control is validated separately from exposed cadence channels.

The tests exercise complete tiny synthetic artifacts from all arms, as well as deliberate budget, candidate, feature, capacity and source mismatches. These local synthetic envelopes do not represent real Colab execution. Source and verification records are retained in `results/audit/pair_cadence_result_auditor_v1/`. The auditor reconstructs metrics from saved predictions and loads checkpoint states to verify architecture and counts; it does not independently reproduce their inference scores or optimization. The real cadence comparison was unrun at this preparation stage. It is now [completed](../results/Pair-Cadence-Development.md), including separate checkpoint inference replay; scientific admission remains open.


### Relative-freshness attribution

The pair-normalized recency statistic is related directly to the relative freshness used by [Navarro and colleagues](https://link.springer.com/article/10.1140/epjds/s13688-017-0127-3), including logarithmic feature transformation in their tie-persistence models. Our positive denominator offset, log-one-plus transform, train-only fallback and online sampled-ranking integration are implementation choices; they do not make the core clock a new invention. The development comparison assesses incremental value in its declared setting and retains no novelty presumption.


## A transition norm alone does not diagnose reversal

For the restricted constant two-phase matrix above, choose a=−M and b=M.
Its Frobenius norm is √2|M|, arbitrarily large, while a+b=0 satisfies the
preservation condition. Both signs are allowed by the unconstrained network.
An additional [full-module test](../../tests/test_silence_scope.py) verifies a
large-norm example with decreasing scores. Retained transition norm summaries
therefore cannot prove checkpoint monotonicity failure, and the two-phase scalar
threshold cannot be applied directly to a higher-dimensional matrix norm.
The counterexample showing reversal is possible remains valid; its occurrence
in trained checkpoints is a separate empirical question.

## A sufficient magnitude-and-drift bound for the ideal many-phase map

This working analytic result addresses the general phase count with a
conservative sufficient condition. It is not an exact higher-dimensional
threshold, a cleared novelty claim or a constraint enforced in the experiments.

Let the ideal interior state satisfy the same pure flow,
$z_k(t)=z_k(0)e^{-d_kt}$ for active $k$, with constant positive rates and
$z_0=1-\sum_{k\ge1}z_k$. Keep the initial state, affinity and final gate weights
fixed. Let $T(t)$ be continuously differentiable along this path, including
the total derivative of any inputs on which it depends. Assume uniform bounds

$$\max_{i,j}|T_{ij}(t)|\le M,\qquad
\max_{i,j}|\dot T_{ij}(t)|\le\eta,\qquad M,\eta\ge0.$$

Define $d_{\min}=\min_{k\ge1}d_k$, $d_{\max}=\max_{k\ge1}d_k$ and

$$C(M)=e^{2M}-e^{-2M}+4M.$$

For any phase count, the sufficient condition

$$d_{\min}\ge d_{\max}C(M)+2\eta$$

ensures that every active coordinate of
$q(t)=\operatorname{softmax}(\log z(t)+T(t)^\top z(t))$ is non-increasing.
Consequently a fixed linear gate with $w_k\ge w_0$ for all active phases has a
non-increasing score after this transformation as well. A fixed positive
affinity multiplier and increasing sigmoid preserve that direction.

To see this, write $v=\dot z$ and
$\Lambda=\sum_{k\ge1}d_kz_k$. Then $v_0=\Lambda$, $v_k=-d_kz_k$,
$\|v\|_1=2\Lambda$ and $\Lambda\le d_{\max}$. Put $r_i=q_i/z_i$.
The magnitude bound implies

$$e^{-2M}\le r_i\le e^{2M},\qquad
\sum_iq_i\frac{v_i}{z_i}
=\sum_{k\ge1}d_kz_k(r_0-r_k)
\ge-\Lambda(e^{2M}-e^{-2M}).$$

For each active coordinate, differentiate its log probability:

$$\frac{\dot q_k}{q_k}
=-d_k-\sum_iq_i\frac{v_i}{z_i}
+(T^\top v)_k-\sum_iq_i(T^\top v)_i
+(\dot T^\top z)_k-\sum_iq_i(\dot T^\top z)_i.$$

The transition difference is at most
$2M\|v\|_1=4M\Lambda$, and the drift difference is at most $2\eta$, since
$\|z\|_1=1$. Combining the inequalities gives

$$\frac{\dot q_k}{q_k}
\le-d_k+\Lambda C(M)+2\eta
\le-d_{\min}+d_{\max}C(M)+2\eta\le0.$$

Finally, $\sum_i\dot q_i=0$ implies
$\partial_t(w^\top q)=\sum_{k\ge1}(w_k-w_0)\dot q_k\le0$.

The constant-transition case has $\eta=0$. This bound is deliberately
conservative and is not necessary: the earlier large-norm preserving example
still applies. If the bounds hold only on a finite time window, the conclusion
has only that window's scope. The rates and drift must use the same time unit.
Numerical clamps, a changing initial encoder output, changing gate weights or
changing affinity remain outside this result.

Drift control cannot simply be omitted. In two phases, take
$z(0)=(1/2,1/2)$, $d=1$ and
$T_{01}(t)=\epsilon\sin(\omega t)$ with other entries zero. At the initial time,

$$\dot q_1(0)=\tfrac14(-2+\epsilon\omega/2).$$

A sufficiently high frequency makes this positive even with arbitrarily small
matrix magnitude. The [executable checks](../../tests/test_softmax_silence_bound.py)
cover this example, zero-transition recovery and active-coordinate derivatives
at several phase counts, including near-boundary interior states. These
numerical checks support the derivation; they are not a proof from samples or
an audit of trained checkpoint violation rates. No current model is asserted
to satisfy the sufficient bound, and no fitting configuration is changed.

## Established exponential update form and the narrower question

The normalized exponential form itself has established optimization precedent.
[Beck and Teboulle](https://www.tau.ac.il/~becka/3.pdf), Section 5, give an
entropic descent update that multiplies current simplex weights by exponentiated
negative subgradient coordinates and normalizes them. It must not be presented
as a newly invented simplex transformation here.

An algebraic interpretation of our ideal map makes the overlap explicit. For
fixed interior z and frozen a=Tᵀz, the unique minimizer over the simplex of

$$\operatorname{KL}(x\Vert z)-a^\top x$$

is $q_i=z_i e^{a_i}/\sum_jz_je^{a_j}$. The Lagrange stationarity equations give
this directly; the entropy term is strictly convex. This is a one-step
entropic proximal interpretation with a frozen linear objective. A general
nonsymmetric state-dependent T need not be the gradient of one common convex
potential, so optimization convergence results cannot simply be transferred.
Our separate question is preservation of a chosen time response as the input
state and transformation vary along the silence path. The sufficient bound
above addresses that restricted question; its priority remains unverified.
Primary-source identity and the scoped overlap review are retained in
`results/audit/simplex_map_prior_art_v1/`.

## Constructive preservation development

The [exploratory simplex development](Monotone-Simplex-Development.md) now tests a
structural alternative to the magnitude/drift bound. It records an exact local
Jacobian criterion for all positive decay directions, an absorbing mixing plus
nonlinear attenuation prototype, and a many-phase skew-symmetric counterexample.
It keeps fixed-context assumptions and expressivity costs explicit. The module
is not installed in completed experiments and establishes no predictive or
novelty claim.


## Pointwise scored-path certificate and learned-model diagnostic

For a differentiable supplied state path $z$ and transition path $T$, let
$a_i=\log z_i+(T^Tz)_i$ on an unclamped interior, and put

$$h_i=\dot z_i/z_i+(T^T\dot z)_i+(\dot T^Tz)_i.$$

Then $\dot q_i=q_i(h_i-\sum_jq_jh_j)$ and, for fixed readout weights,

$$\frac{d}{dt}(w^Tq)=\operatorname{Cov}_q(w,h).$$

This is an exact pointwise sign test for the supplied path; an increasing sigmoid
and fixed positive affinity preserve the sign. It is the standard softmax
Jacobian/chain rule, not a new generic theorem. The implementation includes the
off-boundary derivative of the actual log clamp. Initial-state, feature,
affinity or weight drift must be included explicitly if the intended path has
those changes. Row-common changes $T\mapsto T+a\mathbf1^T$, including their
drift, cancel from both the transformed state and this derivative. A raw matrix
norm is consequently representation dependent as well as an insufficient
failure diagnosis.

The [trained-checkpoint diagnostic](../results/Scored-Path-Silence-Diagnostic.md)
now demonstrates reversal on the selected Reddit validation anchors using exact
archived models, while retaining the absence of declared-threshold crossings on
Wikipedia. Its native-precision replay, scope and initial failed precision
conversion are explicit. This is evidence about learned models under a fixed
context intervention, not a chronological full-context guarantee.

The [readout-aware entropic result](Readout-Aware-Entropic-Preservation.md)
adds a necessary-and-sufficient all-rate condition for a restricted constant
reweighting family. It allows active-coordinate increases when the specified
readout still declines. Its exact scope is narrower than the general dynamic
candidate network; priority and predictive usefulness remain open.

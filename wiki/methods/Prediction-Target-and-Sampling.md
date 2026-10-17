---
title: Prediction Target and Negative Sampling
status: mathematical interpretation boundary for research development
last_updated: 2026-10-15
paper_source: false
---

# Prediction Target and Negative Sampling

A decreasing no-event survival probability is not the same estimand as a
sampled positive-versus-negative label, an instantaneous event density or a
candidate-ranking score. The research program must name its target before interpreting
a gate as physical relation survival. These standard identities are not claimed
as novel theory or admitted empirical results.

## Survival and event likelihood

For a waiting time $T$, survival is $S(t)=P(T>t)$, density is
$f(t)=-S'(t)$ and hazard is $h(t)=f(t)/S(t)$ where survival is positive.
Survival decreases by definition; density and hazard need not decrease.
For example, $S(t)=e^{-t^2}$ gives $f(t)=2te^{-t^2}$, which increases before
$t=1/\sqrt{2}$. A silence constraint on an event score therefore needs a
separate modeling assumption. Neural event models such as
[Neural Hawkes](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
are relevant prior work; applying a survival gate does not inherit their model
or likelihood interpretation.

## Sampled classification target

Fix query context $c$. Let $\pi(v\mid c)$ be the distribution of positive
destinations, and let $q(v\mid c)$ generate independent negative destinations.
With negative loss mass $K$ per unit positive loss mass and positive support,
Bayes' rule gives

$$P_D(Y=1\mid v,c)=\frac{\pi(v\mid c)}{\pi(v\mid c)+Kq(v\mid c)},$$

$$\operatorname{logit}P_D=\log\pi(v\mid c)-\log q(v\mid c)-\log K.$$

These are the familiar density-ratio classification identities underlying
[noise-contrastive estimation](https://proceedings.mlr.press/v9/gutmann10a.html).
The equation uses effective loss weighting; counting negatives alone is
insufficient if the loss rebalances classes. It describes a sampled label
posterior, not population link calibration, event hazard or relation survival.
Recovering the positive distribution from this posterior requires the actual
negative distribution and shared-support assumptions. Such a correction is not
automatically desirable for an official sampled-ranking objective; the evaluator
and scientific estimand must be specified first.

## Positive-conditioned hard negatives

The current hard matcher conditions on properties of the realized positive,
including recurrence/silence bins and simultaneous-positive exclusions.
An independent $q(v\mid c)$ formula cannot simply be applied row by row.
For a declared conditional kernel $Q(v\mid w,c)$, its negative marginal is

$$r(v\mid c)=\sum_w\pi(w\mid c)Q(v\mid w,c).$$

The same classification identity uses $r$, which can depend on the unknown
positive distribution. Empty-support exclusion also changes the positive
population. This is why the hard-control result is conditional discrimination,
not automatically calibrated full-stream prediction or a causal effect.

A finite algebra example takes positive masses `[0.40,0.35,0.20,0.05]` and
swaps negatives within candidate groups `{0,1}` and `{2,3}`. Negative masses
become `[0.35,0.40,0.05,0.20]`. Balanced sampled posteriors are
`[0.533333,0.466667,0.8,0.2]`: candidate `2` has the largest sampled posterior
although candidate `0` has the largest positive mass. Knowing the effective
negative marginal restores the positive distribution algebraically, but that
marginal already depends on the positive law. This is an interpretation example,
not a proposed estimator. Exact arrays are retained in
[the diagnostic](../../results/audit/q1_preflight_v1/sampling_target_example.json).

## Consequences for the next experiment

Keep official candidate identities and their evaluator intact. Record the
training negative law, loss weights, history timing and support coverage.
Separate ranking, sampled-label calibration and time-to-event modeling. Do not
rename decreasing scores as validated survival or automatically alter a benchmark
with a sampling correction. A new objective or sampler requires its own
prospective amendment, and old test outcomes remain preserved.

## What a positive-dependent sampled posterior can identify

This is a finite-catalogue derivation at a fixed prediction context, not a new
scientific claim, an event-survival theorem or a validated correction algorithm.
Let the positive destination distribution be a strictly positive vector $\pi$,
and let the known row-stochastic matrix $Q$ generate a negative conditional on
the positive destination. The negative marginal is $r=Q^T\pi$. With $K$ equally
weighted negative examples per positive and $r_v>0$, the Bayes pointwise sampled
posterior satisfies

$$s_v=\frac{\pi_v}{\pi_v+Kr_v},\qquad
\pi=K\operatorname{diag}\!\left(\frac{s}{1-s}\right)Q^T\pi.$$

For exact posteriors strictly between zero and one, irreducibility of $Q$ is a
sufficient condition for uniqueness of the normalized positive solution: the
matrix on the right is nonnegative and irreducible, and its positive eigenvector
is unique up to scale by Perron–Frobenius. This is an application of established
linear algebra, not a novelty claim. Approximate scores need not admit eigenvalue
one, and no finite-sample recovery or stability guarantee follows from this
exact-posterior observation.

A block-diagonal kernel gives a concrete limitation. Write
$Q=\operatorname{diag}(Q_a,Q_b)$ and
$\pi=(m p_a,(1-m)p_b)$, where $p_a,p_b$ are normalized within-block distributions
and $0<m<1$. The factor $m$ or $1-m$ cancels from every coordinate of $s$.
Consequently, different masses across these closed sampling components can yield
identical sampled posteriors and different population rankings. Recurrence/silence
matching can create such components at a fixed history, but the actual matcher
also drops unsupported positives, so its complete target must be analyzed before
applying this idealization.

This is a limitation of recovery from the sampled posterior alone. It does not
claim that the original positive observations lose all information about component
masses: those observations or additional anchors may supply the missing scale.
Nor does it imply that a conditional matched-pair ranking benchmark is invalid.
The result is not a reason to relabel an internal score as physical survival.

## Related objectives that prevent a novelty shortcut

[Conditional NCE, ICML 2018](https://proceedings.mlr.press/v80/ceylan18a.html)
already uses positive-dependent noise. Its equations 1–12 classify the ordered
pair and its reversal, using both forward and reverse proposal probabilities.
That pairwise objective differs from the pointwise sampled posterior above;
its density-estimation argument cannot simply be transferred to an arbitrary
BCE gate or BPR implementation. [GradientDICE, ICML 2020](https://proceedings.mlr.press/v119/zhang20r.html)
already uses Perron–Frobenius in stationary density-ratio estimation. Its setting
is different, but it rules out treating the spectral tool itself as novel.
A complete comparison to conditional NCE, discrete ratio/score matching and
sampling-correction work is still required before proposing a contribution.
No new sampler, loss or correction is added to the running CRAFT protocol.

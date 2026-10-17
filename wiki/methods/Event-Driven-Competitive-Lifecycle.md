---
title: Event Driven Competitive Lifecycle
status: exact operator and restricted equivalence analysis; no trained efficacy or priority claim
last_updated: 2026-10-16
paper_source: false
---

# Mechanism beyond the residual pilot

The renewed user choice is the cross-edge competition direction. The completed
[residual pilot](../results/Competing-Destination-Pilot.md) is retained, including
its negative primary endpoint. That pilot adjusted logits; it did not propagate
an interacting lifecycle state. This separate development implements an exact
absorbing state operator in `src/lifecycle_readout/competitive_lifecycle.py`.
It is not wired into a graph event executor or a fitted prediction model.
No earlier protocol, comparator, checkpoint or result is changed.

## Correct jump semantics

A literal counting-process equation

$$ dz_j=-d_jz_j\,dt-\gamma_j\kappa\,z_j^-\,dN $$

has jump $z_j^+=(1-\gamma_j\kappa)z_j^-$. It does not have the exponential
jump often used to motivate this mechanism. Without $\gamma_j\kappa\leq1$,
the literal update can produce negative active mass. This is a finite-jump
issue, not an integration precision issue.

We explicitly specify the positivity-preserving alternative

$$ z_j^+=e^{-\gamma_j\kappa}z_j^-,\qquad
z_0^+=z_0^-+\sum_jz_j^-(1-e^{-\gamma_j\kappa}). $$

Its counting-process coefficient is $(e^{-\gamma_j\kappa}-1)z_j^-$.
For small doses the two laws agree to first order, but they are different
models. No claim of an exact solver for the literal linear jump is made.
The dormant coordinate receives the removed mass. This conserves each edge's
simplex mass; it does not conserve a shared attention budget over destinations.

For elapsed physical time $\tau\geq0$ and nonnegative accumulated event exposure
$E=\sum_{r:s_{uv}<t_r<t}\kappa_r$, with rates fixed over this interval,

$$ z_j(\tau,E)=z_j^{\rm init}\exp(-d_j\tau-\gamma_jE),\qquad
z_0=z_0^{\rm init}+\sum_j z_j^{\rm init}
\{1-\exp(-d_j\tau-\gamma_jE)\}. $$

Each kernel is evaluated at its event's strictly prior information. The target
edge's own events are excluded. Queries precede all updates at tied timestamps.
These are caller obligations: the operator accepts doses, not raw events, and
cannot enforce graph-history exclusions. The older pilot uses a query-time
candidate profile against event-time competitor profiles; its exposure therefore
cannot silently be substituted for this event-frozen exposure contract.

## Exact exposure-clock equivalence and its boundary

**Restricted proposition.** Suppose all $d_j>0$, rates are constant, $E>0$,
and each active initial coordinate is positive. The resulting full state can
be reproduced by the original pure physical decay with one scalar effective
time $\tau_{\rm eff}$ if and only if $\gamma_j/d_j=a$ is the same for every
active coordinate. In that case $\tau_{\rm eff}=\tau+aE$.

**Proof.** Equality of each active coordinate gives
$-\log(z_j/z_j^{\rm init})/d_j=\tau+(\gamma_j/d_j)E$.
One common effective time exists exactly when the ratios agree. The dormant
coordinate then agrees by mass conservation. Zero initial coordinates carry no
restriction; when $E=0$ the original time suffices. The result concerns the full
state with the original fixed rates. It does not establish score separation,
identifiable fitted parameters, or exclusion of a more flexible scalar model.

Even when the ratios differ, all these diagonal clock/event operators commute.
The sufficient history summary is $(\tau,E)$, with phase dose vector
$\mathbf d\tau+\boldsymbol\gamma E$. The mechanism cannot distinguish
permutations of competitor events with the same summed event-frozen exposure,
or different destination histories with identical exposure. Nonproportional
rates give a two-direction dose family, not competitor-order sensitivity.
A scalar active phase necessarily satisfies the common-clock condition.
These are interval statements between own-edge updates; no claim is made about
arbitrary intervening state resets, context drift or learned rate changes.

For a fixed linear readout with $w_j\geq w_0$, its exposure derivative is

$$ \frac{\partial (\mathbf w^\top\mathbf z)}{\partial E}
=-\sum_j(w_j-w_0)\gamma_j z_j\leq0. $$

An increasing sigmoid and fixed positive affinity preserve this conditional
direction. Changing the encoder, affinity, context or a subsequent competitive
normalization requires a new end-to-end derivative analysis.

## What this contributes and what it cannot support

The algebra gives a falsifiable architecture distinction: a proportional-rate
version adds no state expressivity beyond an exposure clock. A method paper
cannot count the scalar clock and its event-by-event implementation as separate
contributions. The nonproportional version must be compared against a model
receiving both $\tau$ and $E$ before attributing an effect to lifecycle dynamics.
The completed residual pilot does not test this exact-state model, and its
negative result is not discarded or interpreted as its outcome.

[Gelardi and colleagues](https://arxiv.org/abs/2103.11755) already describe
interaction-driven weakening of other ties and finite attention. The earlier
[competition prior-art review](../questions/Contribution-Gap-Review.md#direct-competition-precedents-found-during-the-pilot)
records the inspected primary source and scope. Thus generic cross-edge
weakening is not claimed new. The exponential absorbing operator and the
restricted equivalence argument are elementary structural checks, not a cleared
novelty claim or observed causal edge death.

## Verification and next discriminating experiment

`LCR-E-COMPETE-FLOW-001` / `LCR-JOB-COMPETE-FLOW-001` retain source hashes,
tests and the local verification record. Tests independently compare the
operator with a continuous-time absorbing Markov generator's matrix exponential,
check event/time composition, verify scalar equivalence and its nonproportional
counterexample, and compare automatic derivatives with the analytic readout
slope. Boundary, extreme-dose and invalid-domain cases are included.
This verifies the operator, not real-data history assembly, training, inference
replay, empirical usefulness or novelty. Human mathematical review remains open.

Before a new allocation, a separate prospective protocol must specify the
strict-past event-frozen kernel, own-edge state initialization/update, missing
history and timestamp groups, then compare proportional and nonproportional
rates with identical-information $(\tau,E)$ controls. It must preserve the
existing negative result and test whether useful candidate-specific exposure
survives activity and shuffled-alignment controls. No data fit, GPU allocation,
new test-set score or manuscript export occurs in this operator development.

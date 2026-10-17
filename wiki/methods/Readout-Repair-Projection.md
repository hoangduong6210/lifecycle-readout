---
title: Readout Repair Projection
status: implemented and evaluated with scoped repair/ranking tradeoff
last_updated: 2026-10-16
paper_source: false
---

# A minimum-distortion restricted repair

The [constant-tilt theorem](Readout-Aware-Entropic-Preservation.md) supplies a
fixed-readout guarantee, but does not establish useful ranking after repair.
`LCR-P-REPAIR-VAL-001` fixes a separate [validation design](../../protocols/readout_repair_validation_v1.toml)
before any repair outcome. Existing checkpoints and frozen validation contexts
are retained; no parameters are fitted or selected.

Evaluate the learned matrix at an algebraic reference clock of zero,
$T_*=T(c,0)$, holding context fixed. This reference is not asserted to be a
historical event state. Removing explicit clock dependence is necessary for
the constant-tilt certificate and is separately controlled in the experiment.

## Nearest gradient-form matrix

For matrices $G(b)_{ij}=b_j-b_i$, unconstrained Frobenius least squares gives

$$\bar b_j=\frac{\sum_iT_{*,ij}-\sum_iT_{*,ji}}{2K},\qquad
\hat b_j=\bar b_j-\bar b_0.$$

This is ordinary least-squares gradient projection, not a newly invented
matrix decomposition. The gauge fixes $b_0=0$. For fixed readout gaps, the
theorem supplies active upper bounds $u_j$; maximal-gap phases have no bound.
The projected repair minimizes the same full matrix distance under those bounds,
not componentwise clipping of the unconstrained potential.

In active coordinates the quadratic metric is $K I-\mathbf1\mathbf1^T$.
The metric is positive definite, with eigenvalues one and the phase count.
The constrained optimum is therefore unique. Its KKT equations give

$$b_j=\min(\hat b_j+\alpha,u_j),\qquad
\alpha=\frac{\sum_{j\in A}(u_j-\hat b_j)}{1+|A|},\qquad
A=\{j:u_j-\hat b_j<\alpha\}.$$

Starting with all violated bounds at zero shift, the active set only shrinks.
At most the phase-count number of passes finds the unique real-arithmetic
solution. The implementation is compared with independent constrained
full-matrix numerical optimization on synthetic examples. A coordinatewise
repair uses all bounds zero and therefore certifies every nonnegative-gap
readout; it has the same matrix-distance objective.

## Flow and controlled variants

The repaired family uses a boundary-safe mass-conserving absorbing flow and
safe logarithms with no artificial positive state floor. An explicit safe-versus-
legacy reference control measures this numerical change separately. All repairs
retain the initial phase, contextual inputs, affinity, readout and trained
weights. The only changes are the declared reference, matrix projection and
flow operations.

The seven variants are original dynamic scoring, legacy constant-reference,
safe constant-reference, unconstrained gradient projection, readout-constrained
gradient projection, coordinate-constrained gradient projection and zero tilt
on the safe flow. All are fixed before results. Predictive AP uses the same
outer clipping as the parent evaluation; raw-score AP and ordering distortion
are retained separately.

The guarantee is conditional on fixed context and affinity. Context includes
time-varying features in the original model, so no full chronological or causal
guarantee is asserted. Elimination of violations is a structural design check,
not discovered phase semantics. The [result owner](../results/Readout-Repair-Validation.md)
will report whether and how much the repair changes ranking. No new benchmark
superiority, novelty clearance or manuscript export is implied by implementation.

## Completed comparison

The [complete validation owner](../results/Readout-Repair-Validation.md) now
records structural success and negative total repair-versus-original Reddit AP.
It separates reference/flow/projection operations and retains the much smaller
incremental readout-bound contrast. The optimum is matrix distance in the
archived zero-diagonal convention, not minimum functional prediction distortion
or a gauge-invariant ranking optimum. No prior fitting protocol changes.

---
title: Inverse Tilt Projection
status: exact feasible geometry and metric-specific projection; synthetic optimizer verification and complete no-refit application
last_updated: 2026-10-16
paper_source: false
---

# A tractable repair for specified rates

The [fixed-rate condition](Fixed-Rate-Readout-Certificate.md) can be expressed as
linear inequalities after a change of coordinates. This gives a precise repair
objective and a comparison with the stronger all-rate condition in the SAME
metric. It does not minimize prediction or ranking error, or the earlier
[Frobenius transition distance](Readout-Repair-Projection.md).

## Exact halfspace geometry

Fix nonnegative gaps $\delta_i$, a nonnegative rate vector $d$, positive tilts,
and $x_i=\sqrt{c_0/c_i}>0$. Set
$K_{ij}=(\delta_i-\delta_j)(d_i-d_j)$. The pair margin factors exactly as

$$B_{ij}+2\sqrt{A_iA_j}
=K_{ij}+\left(\sqrt{\delta_jd_i}\,x_i+
\sqrt{\delta_id_j}\,x_j\right)^2.$$

For $K_{ij}\ge0$ the condition is automatic. For $K_{ij}<0$, nonnegativity
of $x$ and the coefficients makes it equivalent to the halfspace

$$\boxed{\sqrt{\delta_jd_i}\,x_i+
\sqrt{\delta_id_j}\,x_j\ge\sqrt{-K_{ij}}.}$$

Each such row has at least one positive coefficient. A finite rate cone gives
the intersection of these halfspaces over all its generators. Thus the exact
feasible set for positive tilts is $\{x>0:Hx\ge h\}$, with $H\ge0$ and at most
$O(rm^2)$ rows. The inverse map is $c_i=c_0/x_i^2$; the coordinates are unchanged
by common positive rescaling of all tilts. Dynamic readouts/contexts/tilts and
unrestricted transition matrices remain outside this characterization.

## Unique projection without an arbitrary positivity floor

For raw $x^0>0$, consider

$$\min_{x\ge0,\ Hx\ge h}\ \frac12\|x-x^0\|_2^2.$$

The closed feasible set is nonempty: a sufficiently large positive constant
vector satisfies every nonzero nonnegative lower row. The objective is coercive
and strictly convex, so a unique minimizer exists. That minimizer satisfies
$x_i\ge x_i^0$: if a coordinate were smaller, replacing it by $x_i^0$ improves
the objective and cannot hurt any constraint because $H\ge0$. Consequently the
optimum is strictly positive and also solves the original positive-tilt problem.
No artificial positivity floor is needed. The same argument makes explicit
nonnegativity constraints redundant at the optimum of the halfspace QP.

The dual has $\lambda\ge0$, primal relation $x=x^0+H^T\lambda$, and objective

$$D(\lambda)=\lambda^T(h-Hx^0)-\frac12\|H^T\lambda\|_2^2.$$

A coordinate update is
$\lambda_j\leftarrow\max\{0,\lambda_j+(h_j-H_jx)/\|H_j\|^2\}$.
The implementation normalizes rows to unit norm and checks primal feasibility,
nonnegative duals, stationarity and complementarity. These KKT conditions certify
the unique convex-QP optimum in real arithmetic. A fixed numeric budget can fail;
the implementation then raises an error instead of reporting an optimum.
The chosen coordinate metric is an explicit modeling choice, without a claim
that it equals functional distance or a statistical divergence.

## Fair comparison with the all-rate repair

Let $\delta_*=\max_i\delta_i$. The earlier all-rate condition is
$x_i\ge\sqrt{(\delta_*-\delta_i)/\delta_*}$ when the maximum is positive;
zero gaps give no restriction. Its Euclidean projection in these coordinates is
coordinatewise maximum with that lower vector. Its feasible set lies inside
the specified-rate feasible set, so

$$\|x^{\rm fixed}-x^0\|^2\le\|x^{\rm all}-x^0\|^2.$$

This compares two optima under the same $x$ metric. It makes no ordering claim
about their AP, score error or earlier matrix-distance repairs.
For $c=(1,4,4)$, $\delta=(1,2)$, $d=(1,2)$, fixed-rate projection leaves
$x^0=(1/2,1/2)$ unchanged. All-rate projection changes its first coordinate to
$\sqrt{1/2}$ and incurs squared distance $(3-2\sqrt2)/4>0$.

## Implementation, checks and known mathematical tools

`modeling/inverse_tilt_projection.py` implements the halfspaces, the matched
all-rate projection and batched dual coordinate optimization. Independent
exhaustive active-set solutions test nontrivial small synthetic QPs, along with
zero-gap/orthant cases and explicit invalid-domain rejection. Exact symbolic
factorization and the strict-distance witness pass. An initial symbolic test
used expression-tree equality instead of simplification; that setup failure and
its correction are retained. Neither a favorable synthetic case nor KKT alone
establishes useful prediction on real data.

Cyclic dual-coordinate QP methods are established; the inspected author-hosted
[Hsieh and colleagues paper](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)
provides an example of the one-coordinate quadratic update. Its SVM convergence
rate or speed results are not transferred here. The candidate contribution is
the absorbing-readout feasible specialization and its demonstrated operating
consequence, rather than a new generic QP solver or newly invented projection.
Exhaustive priority and external proof review remain open.

`LCR-E-INVERSE-TILT-001` owns the analytical/synthetic record under
`results/audit/inverse_tilt_projection_v1/`. A separate
[complete retained-model application](../results/Fixed-Rate-Repair-Validation.md)
was fixed before its new repair outcomes. No fitted model, old protocol or
publication pointer was changed. Retrieval is mapped by
`evidence/indexes/rate_repair_v1.toml`.

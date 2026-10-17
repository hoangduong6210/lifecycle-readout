---
title: Differentiable Latent Lifecycle Discovery
status: proposed constrained latent lifecycle method
last_updated: 2026-10-15
paper_source: false
---

# Differentiable Latent Lifecycle

The journal direction is recorded in `DEC-0002`. The current implementation is
under corrective protocol `LCR-P-TKDE-002`; efficacy and phase semantics have not
been admitted. It learns phase assignments within a designer-chosen phase count
and an imposed dormant-phase decay structure. This is not unrestricted causal
structure discovery.

## Candidate representation

For candidate pair $(u,v)$ at time $t$, past neighbor representations $h_{uv}$
and pre-event pair statistics $\phi_{uv}$ enter a differentiable encoder:

$$z_{uv}=\operatorname{Softmax}(W_z[h_{uv}\Vert\phi_{uv}]+b_z).$$

Training uses a Gumbel-Softmax relaxation; evaluation uses deterministic
Softmax. Positive and negative candidates follow the same history query rules.
Current event features and future events are excluded. Histories at a shared
timestamp are queried before any event at that timestamp is observed.

## Implemented dynamics

During silence, active phases decay toward designer-designated phase zero:

$$z_k(\Delta t)=z_k(0)e^{-d_k\Delta t},\quad k\ge1,$$
$$z_0(\Delta t)=1-\sum_{k\ge1}z_k(\Delta t),\quad d_k=\operatorname{softplus}(\theta_k).$$

This is the exact solution of this restricted silence flow. The event update
uses a learned zero-diagonal matrix $T_\theta(\phi,\log(1+\Delta t))$:

$$z_{\mathrm{next}}=\operatorname{Softmax}\left(\log z_{\mathrm{decayed}}+
 z_{\mathrm{decayed}}^\top T_\theta\right).$$

The implementation clamps small phase probabilities for numerical stability.
This event jump is not a claimed closed-form integration of an arbitrary
learned matrix ODE, nor a stochastic diffusion process. The realized transition
matrix Frobenius norm is audited separately from parameter norms.

## Regularization and interpretation

Individual entropy minimization, marginal phase-diversity maximization, and
transition-magnitude regularization are optimized with the link prediction loss.
These are entropy/sparsity penalties; their presence alone does not establish
an information-bottleneck bound, phase identifiability, or non-collapse.

Phase population counts, entropy, historical recurrence, and elapsed-time
profiles are retained with each attempt. Independent unordered-gate training
uses the same train/validation/test inputs. Cross-seed phase alignment and
external semantic validity remain open scientific requirements.

## Implementation and evidence

- [Latent module](../../src/lifecycle_readout/modeling/latent_lifecycle.py)
- [Causal candidate builder](../../src/lifecycle_readout/causal_stream.py)
- [Matched benchmark](../../src/lifecycle_readout/benchmark.py)
- [Audit disposition](../results/Benchmark-and-Baseline-Results.md)

No current result is paper eligible; the claim registry owns admission.

## Manuscript correction boundary

The component equations of the simplex-preserving flow are
$\dot z_k=-d_k z_k$ for $k\ge1$ and
$\dot z_0=\sum_{k\ge1}d_k z_k$; writing $\dot z=-Dz$ for the
entire vector with a diagonal $D$ omits the dormant inflow.
The active experimental runner does not supply future-state targets and therefore
does not optimize the optional transition-consistency loss. The historical
runner uses default temperature, not an implemented annealing schedule.
See [the gating method](Shortcut-Resistant-Gating.md) for the corrected
monotonicity condition, gradient-connectivity scope and hard-control design.


## Backbone equation correspondence

The implementation uses a masked token mixer followed by a masked channel
mixer. Token mixing acts on the neighbor dimension after transposition;
channel mixing acts on the hidden feature dimension. Pooling divides the
masked feature sum by the masked neighbor count clamped to at least one,
then adds the learned empty-neighborhood vector only for zero count. The
empty vector is outside the pooling fraction.

For one-based harmonic index $r$ and even time dimension $d_t$, the implemented
frequency is $10^{-12(r-1)/d_t}$, since code indexes the even coordinates before
multiplying by six. The Word backbone equations were reconciled to this
implementation; this is a documentation correction, not a change to trained
model behavior.

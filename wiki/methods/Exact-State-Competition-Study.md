---
title: Exact State Competition Study
status: implemented with completed fixed validation comparison; scientific scope remains exploratory
last_updated: 2026-10-16
paper_source: false
---

# Discriminating comparison after input validation

`LCR-P-EVENT-COUPLING-001` is a new exploratory Wikipedia model comparison,
following the [negative residual pilot](../results/Competing-Destination-Pilot.md)
and [frozen-reference input audit](../results/Competitive-Event-History-Audit.md).
It does not amend either completed study. Exact settings live in
`configs/event_coupling_v1.toml`; reusable code is
`src/lifecycle_readout/event_coupling_study.py`.

## State and own-edge reset

Every allowed own-edge timestamp group resets the state to a uniform simplex
with dormant phase and active phases. The query's latest-own reference and
cumulative dose follow the [frozen-reference contract](Frozen-Reference-Competition-History.md).
Constant global learned clock rates $d_j>0$ and event rates $\gamma_j>0$ give

$$ z_j=\tfrac{1}{5}\exp(-d_j\tau/s_t-\gamma_jE/s_E). $$

Removed active mass goes to dormant through the exact absorbing operator.
Training-only median positive physical time and exposure define $s_t,s_E$.
The masked-exposure control has unit exposure scale when no positive value
remains. Neither validation nor test outcomes choose scales.

For positive learned readout gaps $\delta_j$ and nonnegative amplitude $a$,
the logit residual is

$$ r=a\sum_j\delta_j(z_j-\tfrac{1}{5}). $$

It is zero at the reset and at initialization ($a=0$). It is nonpositive and
conditionally nonincreasing in time/exposure. All models use the same
history-aware affinity architecture receiving the original history channels and
new event-dose/count/availability inputs. That affinity may change with time or
exposure, so the whole score is not certified monotone.
The fixed reset tests this bounded basis construction. It is not a learned
resurrection rule, phase semantic discovery or point-process likelihood.

## Arms and primary comparison

| Arm | Added branch | Extra trainable parameters |
|---|---|---|
| `history_only` | No residual | `0` |
| `dose_mlp` | Normalized time/exposure, log transform, small GELU MLP and signed amplitude | `13` |
| `scalar_clock` | Exact state with $\gamma_j=\alpha d_j$ | `10` |
| `phase_competition` | Independent phase clock/event rates and positive gaps | `13` |
| `shuffled_phase` | Same phase branch; permute target-reference coordinates before dose | `13` |
| `activity_mlp` | Same MLP; exposure masked in affinity and residual | `13` |

The primary is validation AP of `phase_competition` minus `dose_mlp`.
They share exactly the same inputs, total parameter count, backbone/history
initial tensors, initial predictions, loss, optimizer, epoch budget and
checkpoint-selection rule. Their function classes and optimization trajectories
are intentionally different; nominal capacity equality is not functional or
optimization equivalence. Feature-masked/shuffled controls change inputs and
need not have identical initial predictions after arm-specific normalization.
The simpler proportional clock intentionally has fewer parameters.

Secondary contrasts compare phase-specific rates, profile alignment, added
profile information and a residual-free history scorer. None replaces the
primary. All declared seeds and arms are retained, regardless of results.
The protocol fixes the earlier matched historical candidates, training budget,
validation selection and no test scoring. There is no automatic expansion or
retuning after these results.

## Verification and resource contract

Preflight tests verify primary branch capacity, shared backbone/history tensors,
zero-amplitude baseline recovery, proportional-state equivalence, conditional
derivatives, training-only units and a guarded synthetic validation pipeline.
The auditor verifies source identity, all retained validation checkpoint
predictions, metrics, earliest-maximum selection, input identity, units,
parameter counts and reported learned rates. Raw-history verification inherits
the previous declared input anchors; it is not every-real-row independent
reconstruction or optimizer replay.

One bounded owned Colab runtime is retrieved and released in a finally path.
The [result owner](../results/Exact-State-Competition-Pilot.md) records execution,
all outcomes and limitations. No superiority, calibrated survival, causal death,
semantic substitution, priority clearance or publication admission follows from
the design. Phase-type exponential bases and generic cross-edge weakening are
existing ingredients; any contribution must rest on the controlled evidence.

## Completed comparison

The [result owner](../results/Exact-State-Competition-Pilot.md) now records all
declared arms, primary and secondary outcomes, full validation inference audit
and owned-runtime release. The original protocol and configuration remain
unchanged. The empirical outcome does not enlarge the conditional guarantee or
establish novelty of exponential phase bases or competing ties.

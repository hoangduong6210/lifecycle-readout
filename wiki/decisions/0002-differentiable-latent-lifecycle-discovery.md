---
title: Decision 0002 - Differentiable Latent Lifecycle Discovery for the research study
status: accepted
date: 2026-10-13
paper_source: false
---

# DEC-0002 — Differentiable Latent Lifecycle Discovery for the research study

## Context

The legacy lifecycle readout evaluated in historical drafts suffered from four critical limitations that prevented admission to the research study:
1. **Designer-imposed state space and heuristics:** The 5 discrete states (`IDLE`, `BIRTH`, `REINFORCE`, `DECAY`, `DEATH`), the tridiagonal admissibility band matrix $C \in \{0,1\}^{5\times5}$, and the prior initialization weights were hand-crafted. This created circular reasoning where interventions merely reflected designer initialization rather than discovered causal dynamics.
2. **The detachment utility paradox:** By isolating the FSM behind a stop-gradient wall (`h.detach()`), the lifecycle readout provided zero utility to link prediction and failed to act as an inductive bias against temporal shortcut overfitting.
3. **Catastrophic confidence failure:** The walked-chain causal coherence signal $c_t$ failed as an error predictor (with historical AUC near chance level across seeds).
4. **Single-dataset scope:** The model was tuned only on `CoEdit` and could not claim cross-domain generality.

To meet the scientific rigor required for the research study, the architecture must transition from handcrafted rules to a differentiable, self-discovered latent lifecycle framework evaluated across multi-dataset benchmarks.

## Options considered

1. **Option 1: Retain the 5-state discrete FSM and add more heuristic transition rules.**
   - Rejected: Does not solve designer bias; fails top-tier peer review.
2. **Option 2: Pure continuous black-box representation without explicit lifecycle modeling.**
   - Rejected: Forfeits interpretability, counterfactual intervention, and shortcut resistance.
3. **Option 3: Differentiable Latent Lifecycle Discovery (L3-GNN).**
   - Adopted: Model relation lifecycle as a continuous latent phase distribution $z_{uv}(t) \in \Delta^{K-1}$ with a continuous-time transition kernel and an information-bottleneck parsimony regularizer. Use the discovered lifecycle as a shortcut-resistant survival filter $\Psi(z_{uv}(t))$ to modulate edge existence scoring, and benchmark across Wikipedia, Reddit, MOOC, and LastFM.

## Decision

Adopt Option 3. Transition the core architecture to **Differentiable Latent Lifecycle Discovery (L3-GNN)**:
1. Replace handcrafted discrete states and the static $C$ matrix with a parameter-efficient latent phase distribution $z_{uv}(t)$ and a learned continuous-time transition operator $\frac{dz}{dt} = -\mathbf{D} z + \mathbf{T}_\phi(z, \Delta t)$.
2. Couple the discovered lifecycle state into link prediction via a monotonic survival filter $\Psi(z_{uv}(t))$ that suppresses high-frequency shortcut artifacts on decaying relations.
3. Replace circular hand-initialized interventions with principled time-silence counterfactual interventions $do(\Delta t \to \infty)$ and treatment effect analysis.
4. Expand protocol and evaluation to standard dynamic graph benchmarks (Wikipedia, Reddit, MOOC, LastFM).

## Scientific consequences

- Eliminates designer bias and circular intervention claims.
- Establishes an unforced inductive bias intended to address temporal shortcut overfitting under formal empirical audit.
- Aligns the methodological framework with recent literature on temporal shortcut mitigation (TGB-Seq) and self-interpretable temporal graph learning (ICML/NeurIPS).
- Defines the target architecture to be audited against rigorous protocol gates prior to any formal publication release.

## Evidence and affected IDs

- Protocol: `LCR-P-TKDE-001`.
- Research Questions: `LCR-RQ-DISCOVERY-001`, `LCR-RQ-SHORTCUT-001`, `LCR-RQ-CAUSAL-001`.
- Datasets: `LCR-D-WIKI-001`, `LCR-D-REDDIT-001`, `LCR-D-MOOC-001`, `LCR-D-LASTFM-001`.
- Supersedes the hand-designed FSM approach while preserving legacy records for audit.

## Supersedes / superseded by

Supersedes: Hand-crafted 5-state FSM design in `DEC-0001`. Superseded by: none.

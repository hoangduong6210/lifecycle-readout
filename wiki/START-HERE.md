---
title: Start Here
status: canonical onboarding
last_updated: 2026-09-16
paper_source: false
---

# Start Here

## What this project studies

The project asks whether a five-state hierarchical lifecycle readout can
explain a temporal graph prediction and whether its interpretability path
alters the scored output. Decoder reachability, score invariance, gradient
ancestry, and typed interventions are separate research questions. The
state cardinality belongs to `LCR-P-READOUT-001`; it is a design fact rather
than a measured outcome.

The stage is `split-migration-audit`. Both evidence and paper pointers are
`UNRELEASED`. A local migration check found two CoEdit copies bit-identical
under `LCR-C-COEDIT-IDFIX-001`, `LCR-E-DATA-MIGRATION-001`, and
`LCR-JOB-LOCAL-20260819-001`; upstream identity and licensing remain blocked.
No quantitative lifecycle, invariance, intervention, or generalization claim
has been admitted. Historical results, figures, and working manuscripts are
quarantined outside the source repository.

## Read these five pages first

1. [Project Status](status/Project-Status.md) states the current stage,
   blockers, and next actions.
2. [Dataset Registry](datasets/Dataset-Registry.md) separates local migration
   copies from verified upstream data.
3. [Claim Registry](claims/Current-Claim-Language.md) defines exactly what may
   be said.
4. [Limitations](LIMITATIONS.md) records what has not been demonstrated.
5. [Reproducibility](REPRODUCIBILITY.md) defines the closure required for a
   scientific result.

The [Exhaustive Index](INDEX.md) lists every maintained page and identifier.

## Repository map

| Path | Purpose |
|---|---|
| `src/lifecycle_readout/` | Package-local backbone, readout, datasets, metrics, and training code |
| `experiments/` | Thin study runners, probes, and dataset builders |
| `protocols/` and `configs/` | Versioned study contract, settings, and dependency hashes |
| `resources/` | Corpus identity manifest; local binaries are excluded from Git |
| `evidence/jobs/` | Local structural execution record |
| `results/` | Release pointer and future frozen evidence; audit and historical data stay local |
| `wiki/` | Canonical scientific interpretation and publication source |
| `paper/` | Paper pointer and export guidance; working drafts stay local |
| `publication/` | Blocked public-release gate |

## Safe first checks

From the repository root, run:

```bash
python -m pytest -q
```

The test suite checks structural contracts and any locally available corpus
bytes. It does not run heavy training or create scientific evidence. Training
and intervention batteries require an approved scheduler workflow; see the
[Research Workflow](operations/Research-Workflow.md) and
[Live Execution](status/Live-Execution.md).

## How to change scientific knowledge

Begin with the question, protocol, configuration, and complete execution
record. Preserve failed and negative attempts. Update the evidence ledger and
claim registry only after a frozen release passes its gates; update scientific
status in the wiki before creating a paper snapshot. The immediate priorities
are to close upstream data rights, reconcile legacy coverage, and rerun the
declared protocol. [Project Status](status/Project-Status.md) owns the current
blockers and next actions; [Contributing](CONTRIBUTING.md) records edit rules.

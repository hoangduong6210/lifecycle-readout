---
title: Start Here
status: canonical onboarding
last_updated: 2026-09-16
paper_source: false
---

# Start Here

## Objective and stage

The project studies a five-state, hierarchical, intervene-able temporal
lifecycle readout and tests whether its interpretability path can alter the
scored prediction. The state cardinality is owned by `LCR-P-READOUT-001`,
`LCR-E-RUNTIME-MIGRATION-001`, and job `LCR-JOB-LOCAL-20260819-001`; it is a
design fact, not a measured outcome. The current stage is `split-migration-audit`. Current evidence
release: `UNRELEASED`. Current paper snapshot: `UNRELEASED`.

## Supported now

- The project owns an internal checksum-listed backbone closure and has no
  sibling import.
- The two local CoEdit migration copies are checksum-identical
  (`LCR-C-COEDIT-IDFIX-001`, `LCR-E-DATA-MIGRATION-001`, job
  `LCR-JOB-LOCAL-20260819-001`).
- Legacy claims and blind/named manuscripts are retained locally without
  admission; quarantined artifacts are excluded from the source repository.

## Unsupported or blocked now

- No quantitative lifecycle, invariance, intervention, or generalization claim
  is admitted.
- Upstream dataset identity, licensing, and redistribution remain blocked.
- Legacy task/seed/failure coverage and clean-clone corpus verification are open.

## Running work and ownership

No training or scheduler job is running. The project owner must resolve dataset
rights and provenance; a future evidence reviewer must close execution coverage;
the claim reviewer must approve wording only after a frozen release exists.

## Next three actions

1. Close upstream identity/license records and clean-clone corpus verification.
2. Reconcile legacy attempts against `LCR-P-READOUT-001` and rerun required gates.
3. Freeze the first `LCR-*` evidence release, then review exact claim wording.

## Safe first checks and task routes

Run `python -m pytest -q` for local structural contracts; it does not run heavy
training or create scientific evidence. Contributors start with [Project Status](status/Project-Status.md).
Claim reviewers read [Claim Registry](claims/Current-Claim-Language.md),
[Evidence Ledger](evidence/Evidence-Ledger.md), and [Limitations](LIMITATIONS.md).
Dataset reviewers read [Dataset Registry](datasets/Dataset-Registry.md) and
[Source Map](references/Technical-Source-Map.md). Paper editors read the
[Paper Export Contract](manuscript/Paper-Export-Contract.md).

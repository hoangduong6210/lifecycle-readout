---
title: Decision 0001 - Separate lifecycle readout and link prediction
status: accepted
date: 2026-08-19
paper_source: false
---

# DEC-0001 — Separate lifecycle readout and link prediction

## Context

Two independent paper scopes shared one mutable implementation and evidence tree.
That prevented project-local namespaces, pointers, and claim admission rules.

## Options considered

1. Keep the mixed tree and distinguish studies only by documentation.
2. Share a mutable sibling runtime between two project roots.
3. Create independent roots with temporarily duplicated, checksum-listed runtime
   closures while preserving the mixed source as migration history.

## Decision

Adopt option 3. Create an independent lifecycle-readout root with a copied,
checksum-listable backbone and readout closure. Do not import or symlink a sibling
project. Preserve the mixed source tree until parity and evidence audits complete.

## Scientific consequences

- The split changes artifact ownership, not a scientific conclusion.
- Legacy numerical artifacts and paper prose remain QUARANTINED.
- A new result requires `LCR-P-READOUT-001`, complete execution accounting, a
  frozen evidence release, and explicit claim review.
- Temporary runtime duplication creates drift risk controlled by manifests and tests.

## Evidence and affected IDs

- Runtime migration record: `LCR-E-RUNTIME-MIGRATION-001`.
- Data migration record: `LCR-E-DATA-MIGRATION-001`.
- Quarantined legacy records: `LCR-E-LEGACY-RESULTS-001` and
  `LCR-E-LEGACY-PAPER-001`.
- All `LCR-RQ-*`, `LCR-D-*`, `LCR-P-*`, and `LCR-C-*` objects are project-local.

## Supersedes / superseded by

Supersedes no earlier lifecycle-readout decision record. Superseded by: none.
A later change must create a new decision and link both records; this accepted
record is not edited to rewrite the historical choice.

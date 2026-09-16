---
title: Project Status
status: active migration status
last_updated: 2026-09-16
paper_source: false
---

# Project Status

## Current state

The project root, package namespace, CoEdit registry, legacy result partition,
named/blind paper working tree, and canonical wiki are separated. Evidence and
paper pointers remain `UNRELEASED`. The original mixed tree remains a read-only
migration source because it contains dirty changes and pre-ID-fix provenance.
No scientific conclusion changed during the split.
The standalone source repository excludes local corpora, historical results,
legacy figures, and working manuscripts under the blocked publication gate.

## Verification state

The local repository, import-boundary, checksum-manifest, JSON, wiki-link,
frontmatter, artifact-hygiene, dataset-schema, and anonymity contracts passed on
2026-08-19 (`LCR-E-DATA-MIGRATION-001`, `LCR-E-RUNTIME-MIGRATION-001`,
job `LCR-JOB-LOCAL-20260819-001`). This is a local structural result, not a
frozen evidence release.
Corpus binaries are ignored, so clean-clone CI cannot yet reproduce the local
dataset byte checks.

## Active blockers

- Upstream Wikipedia identity, dataset license, citation, and redistribution
  permission are UNKNOWN/BLOCKED.
- Legacy intervention, invariance, gradient-ancestry, seed, denominator, retry,
  and failure coverage has not been reconciled to `LCR-P-READOUT-001`.
- No source commit pinned to a frozen evidence release, admitted claim, results lock, or
  immutable paper snapshot exists.
- Strong numerical wording in `paper/working/` remains QUARANTINED.

## Next stage

Close dataset provenance and clean-clone verification; reconcile the complete
legacy execution matrix; execute the frozen protocol; then create a checksum-
locked evidence release and review exact claim wording. Paper export remains
blocked until those gates pass.

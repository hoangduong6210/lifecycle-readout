---
title: Numeric Evidence and Publication Hygiene
status: canonical blocked gate
last_updated: 2026-08-19
paper_source: false
---

# Numeric Evidence and Publication Hygiene

Every empirical scalar in the wiki must appear in a claim record that names an
Evidence ID and an execution job. A blocked claim must say `Execution job: NONE`
and may not restate a legacy scalar. Dates, ordered-list labels, identifier
suffixes, hashes, algorithm names, and implementation versions are governance or
structural metadata rather than empirical claims.

The current wiki admits no empirical scalar. Local migration and design assertions
resolve to `LCR-E-DATA-MIGRATION-001`, `LCR-E-RUNTIME-MIGRATION-001`, and
`LCR-JOB-LOCAL-20260819-001`. Legacy paper numbers remain under
`LCR-E-LEGACY-RESULTS-001`, with no normalized job registry, and are prohibited
from release or paper export.

Lexical scanning can detect vendor/persona names and internal-workflow markers;
it cannot determine authorship. No claim of human or AI authorship may be made
from stylometry alone. `paper/working/`, `results/historical/`, audit output, and
the mixed parent tree are outside the publication surface.

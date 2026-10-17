---
title: Hierarchical Lifecycle Readout
status: migration method record
last_updated: 2026-08-19
paper_source: false
---

# Hierarchical Lifecycle Readout

The readout factors state probabilities through birth, alive, and rising gates
instead of relying on a flat ordered interpolation. It consumes detached
per-pair statistics. Its output is interpreted only within the five declared
states and CoEdit scope. This design cardinality is owned by
`LCR-P-READOUT-001`, `LCR-E-RUNTIME-MIGRATION-001`, and job
`LCR-JOB-LOCAL-20260819-001`; it is not an empirical scalar. Current source is under
`src/lifecycle_readout/modeling/`.

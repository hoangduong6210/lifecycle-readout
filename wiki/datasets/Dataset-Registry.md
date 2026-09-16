---
title: Dataset Registry
status: canonical dataset registry
last_updated: 2026-08-19
paper_source: false
---

# Dataset Registry

## LCR-D-COEDIT-001

- **Version and lifecycle:** migration-current local copy; not frozen.
- **Source and provenance:** Derived from a Wikipedia edit stream. Exact upstream
  file identity, acquisition record, and builder-input checksum are UNKNOWN/BLOCKED.
- **License and redistribution:** UNKNOWN/BLOCKED pending owner review.
- **Checksum:** Owned by `resources/manifest.toml` and
  `resources/checksums.sha256`; do not copy the mutable scalar into the wiki.
- **Population and geometry:** Non-bipartite contributor-to-contributor co-edit
  events; exact construction population and exclusions are BLOCKED pending builder audit.
- **Inputs, targets, units, and fidelity:** NPZ schema is owned by
  [Data and Target Contract](Data-and-Target-Contract.md). Timestamps and feature
  units are UNKNOWN; lifecycle states are derived readout quantities, not observed labels.
- **Inclusion, exclusion, and quality gates:** Array lengths, shapes, ID ranges,
  finite-value policy, and time ordering require machine checks; upstream
  inclusion/exclusion rules are UNKNOWN.
- **Split and leakage controls:** Protocol ownership is
  `LCR-P-READOUT-001`; exact frozen split registry and leakage audit are BLOCKED.
- **Known defects:** Missing upstream identity/license closure; local binary is ignored.
- **Compatible claims:** `LCR-C-COEDIT-IDFIX-001` for local byte identity only.

## LCR-D-COEDIT-PREIDFIX-001

- **Version and lifecycle:** HISTORICAL local migration backup; not frozen and not current.
- **Source and provenance:** Same unresolved upstream/build lineage as
  `LCR-D-COEDIT-001`; exact source identity is UNKNOWN/BLOCKED.
- **License and redistribution:** UNKNOWN/BLOCKED pending owner review.
- **Checksum:** Owned by `resources/manifest.toml` and
  `resources/checksums.sha256`.
- **Population and geometry:** Intended historical CoEdit backup; scientific
  population equivalence beyond byte equality is not separately established.
- **Inputs, targets, units, and fidelity:** Historical NPZ under the same schema;
  units and builder provenance remain UNKNOWN.
- **Inclusion, exclusion, and quality gates:** Retained for byte comparison only.
- **Split and leakage controls:** Not eligible for a new split or claim.
- **Known defects:** Ignored local binary with unresolved upstream provenance.
- **Compatible claims:** `LCR-C-COEDIT-IDFIX-001` for local byte identity only.

No dataset is eligible for a frozen evidence release until source identity,
license, checksums, build configuration, split identity, and leakage controls close.

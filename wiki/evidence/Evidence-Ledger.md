---
title: Evidence Ledger
status: canonical evidence ledger
last_updated: 2026-08-19
paper_source: false
---

# Evidence Ledger

Current evidence release: `UNRELEASED`. No entry below is frozen scientific evidence.

## LCR-E-DATA-MIGRATION-001

- **Scientific purpose:** Verify identity of the local CoEdit migration copies.
- **Lifecycle:** VALIDATED for local migration integrity; not ADMITTED.
- **Source commit:** UNKNOWN; the split tree has no frozen release commit.
- **Protocol, configuration, and data hashes:** Protocol/configuration hashes are
  not applicable to byte comparison. Dataset hashes are owned by
  `resources/manifest.toml` and `resources/checksums.sha256`; upstream-source hash
  is BLOCKED.
- **Execution identity:** `LCR-JOB-LOCAL-20260819-001`, recorded at
  `evidence/jobs/LCR-JOB-LOCAL-20260819-001.toml`; local execution, not a
  scheduler or frozen-release job.
- **Artifact path or release URI:** `resources/manifest.toml` and
  `resources/checksums.sha256`; local ignored files under `resources/corpora/`.
- **Artifact checksum:** Dataset bytes are declared in `resources/checksums.sha256`;
  the execution record is pinned by `evidence/jobs/checksums.sha256`. No frozen-
  release checksum manifest exists.
- **Coverage and failures:** Both declared local CoEdit copies were checked with no
  local contract failure. Clean-clone corpus coverage is unavailable because the
  binaries are not redistributed.
- **Acceptance-gate outcome:** Local byte-equality gate PASS; upstream provenance,
  license, and release-closure gates BLOCKED.
- **Supported claim IDs:** `LCR-C-COEDIT-IDFIX-001` at VALIDATED technical status only.
- **Rejected claim IDs:** none.
- **Scientific-use boundary:** Migration identity only; no lifecycle result or
  paper-eligible conclusion.

## LCR-E-RUNTIME-MIGRATION-001

- **Scientific purpose:** Detect drift in the copied runtime closure.
- **Lifecycle:** VALIDATED for local code-copy integrity; not ADMITTED.
- **Source commit:** UNKNOWN; no frozen release commit.
- **Protocol, configuration, and data hashes:** Runtime file hashes are owned by
  `configs/shared-runtime.sha256`; protocol/config/data closure is BLOCKED.
- **Execution identity:** `LCR-JOB-LOCAL-20260819-001`, recorded at
  `evidence/jobs/LCR-JOB-LOCAL-20260819-001.toml`; local execution, not a
  scheduler or frozen-release job.
- **Artifact path or release URI:** `src/` and `configs/shared-runtime.sha256`.
- **Artifact checksum:** Per-file SHA-256 values are in
  `configs/shared-runtime.sha256`; the execution record is pinned by
  `evidence/jobs/checksums.sha256`. No release-level checksum exists.
- **Coverage and failures:** Files listed by the runtime manifest were checked with
  no local contract failure; behavioral parity is not established.
- **Acceptance-gate outcome:** Manifest integrity PASS; scientific parity and
  frozen-release gates BLOCKED.
- **Supported claim IDs:** none.
- **Rejected claim IDs:** none.
- **Scientific-use boundary:** Code-copy integrity only.

## LCR-E-LEGACY-RESULTS-001

- **Scientific purpose:** Preserve pre-split result artifacts for future audit.
- **Lifecycle:** QUARANTINED.
- **Source commit:** UNKNOWN.
- **Protocol, configuration, and data hashes:** UNKNOWN or incomplete; reconciliation
  against `LCR-P-READOUT-001` is BLOCKED.
- **Execution identity:** Historical job identities are not normalized into a
  complete execution registry.
- **Artifact path or release URI:** `results/historical/legacy_import/`.
- **Artifact checksum:** Local preservation hashes are in
  `results/historical/legacy_import/checksums.sha256`; no frozen release checksum.
- **Coverage and failures:** Seed, task, retry, exclusion, and failure coverage are
  UNKNOWN pending audit.
- **Acceptance-gate outcome:** NOT EVALUATED under the current release contract.
- **Supported claim IDs:** none at ADMITTED status.
- **Rejected claim IDs:** none registered; unaudited artifacts do not reject claims.
- **Scientific-use boundary:** Audit input only; not current or paper evidence.

## LCR-E-LEGACY-PAPER-001

- **Scientific purpose:** Preserve named and anonymous pre-split editorial artifacts.
- **Lifecycle:** QUARANTINED.
- **Source commit:** UNKNOWN.
- **Protocol, configuration, and data hashes:** UNKNOWN; no results lock exists.
- **Execution identity:** not applicable.
- **Artifact path or release URI:** `paper/working/`.
- **Artifact checksum:** Local preservation hashes are in
  `paper/working/checksums.sha256`; no immutable paper snapshot hash exists.
- **Coverage and failures:** Named/blind variants are retained, but scientific and
  citation audits are incomplete.
- **Acceptance-gate outcome:** Paper-export gate BLOCKED.
- **Supported claim IDs:** none.
- **Rejected claim IDs:** none.
- **Scientific-use boundary:** Editorial history only; strong wording inside these
  files is not permitted current claim language.

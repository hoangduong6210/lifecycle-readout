---
title: Current Claim Registry
status: canonical claim registry
last_updated: 2026-08-19
paper_source: false
---

# Current Claim Registry

## Admitted claims

None. The split does not promote legacy paper numbers.

## Validated artifacts that are not admitted claims

### LCR-C-COEDIT-IDFIX-001

- **Exact permitted statement:** The locally migrated CoEdit file is bit-identical
  to the locally retained pre-ID-fix backup named by the corpus manifest.
- **Lifecycle status:** VALIDATED (technical migration property only)
- **Scope and population:** The two local CoEdit NPZ identities in
  `resources/manifest.toml`; no upstream or cross-dataset conclusion.
- **Dataset and fidelity:** `LCR-D-COEDIT-001` and
  `LCR-D-COEDIT-PREIDFIX-001`; migration copies, not frozen-release evidence.
- **Metric and uncertainty unit:** SHA-256 byte equality; uncertainty not applicable.
- **Evidence IDs:** `LCR-E-DATA-MIGRATION-001`.
- **Execution job:** `LCR-JOB-LOCAL-20260819-001` (local structural check only).
- **Required qualifiers:** Say "local migration copies" and "bit-identical";
  do not describe the upstream source or scientific equivalence as verified.
- **Known limitations:** Upstream Wikipedia identity, license, and clean-clone
  corpus verification remain blocked.
- **Paper eligibility:** false.
- **Last review date:** 2026-08-19.

## Blocked or proposed claims

### LCR-C-DECODER-001

- **Exact permitted statement:** Historical artifacts report hierarchical decoder
  behavior; current numerical wording is not admitted.
- **Lifecycle status:** BLOCKED.
- **Scope and population:** CoEdit lifecycle-readout migration artifacts only.
- **Dataset and fidelity:** `LCR-D-COEDIT-001`; legacy, non-frozen execution fidelity.
- **Metric and uncertainty unit:** UNKNOWN pending seed, denominator, and
  aggregation audit.
- **Evidence IDs:** `LCR-E-LEGACY-RESULTS-001`; a future frozen evidence ID is required.
- **Execution job:** NONE; BLOCKED pending a normalized frozen job registry.
- **Required qualifiers:** Historical, quarantined, and not current evidence.
- **Known limitations:** Decoder/model identity, failures, seed coverage, and
  aggregation are not closed.
- **Paper eligibility:** false.
- **Last review date:** 2026-08-19.

### LCR-C-INVARIANCE-001

- **Exact permitted statement:** Invariance and gradient-ancestry artifacts are
  retained for audit; no current invariance guarantee is admitted.
- **Lifecycle status:** BLOCKED.
- **Scope and population:** Retained CoEdit model instances only; no architectural
  or cross-instance guarantee.
- **Dataset and fidelity:** `LCR-D-COEDIT-001`; legacy instance-level artifacts.
- **Metric and uncertainty unit:** UNKNOWN pending instance-level rerun and exact
  tolerance contract.
- **Evidence IDs:** `LCR-E-LEGACY-RESULTS-001`; a future frozen evidence ID is required.
- **Execution job:** NONE; BLOCKED pending a normalized frozen job registry.
- **Required qualifiers:** Describe ancestry and invariance as pending
  implementation tests, not a universal guarantee.
- **Known limitations:** The admitted model identity, tolerance, task coverage,
  and failure accounting are not frozen.
- **Paper eligibility:** false.
- **Last review date:** 2026-08-19.

### LCR-C-INTERVENTION-001

- **Exact permitted statement:** Intervention artifacts are retained for audit;
  no current directional percentage is admitted.
- **Lifecycle status:** BLOCKED.
- **Scope and population:** Typed interventions on retained CoEdit instances only.
- **Dataset and fidelity:** `LCR-D-COEDIT-001`; legacy, non-frozen execution fidelity.
- **Metric and uncertainty unit:** UNKNOWN pending denominator, seed, failure, and
  uncertainty audit.
- **Evidence IDs:** `LCR-E-LEGACY-RESULTS-001`; a future frozen evidence ID is required.
- **Execution job:** NONE; BLOCKED pending a normalized frozen job registry.
- **Required qualifiers:** The intervention structure is designer-imposed and is
  not a discovered SCM.
- **Known limitations:** Direction, reversibility, no-op, and score-path gates have
  not been closed under a frozen protocol.
- **Paper eligibility:** false.
- **Last review date:** 2026-08-19.

## Rejected positive claims

None registered. Absence of an admitted claim is not evidence of rejection.

## Prohibited wording

Do not call the readout a discovered SCM, call internal coherence external
validation, call a legacy NPZ frozen evidence, or promote a value from a
quarantined working paper.

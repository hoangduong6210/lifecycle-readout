---
title: Exhaustive Wiki Index
status: canonical index
last_updated: 2026-08-19
paper_source: false
---

# Exhaustive Wiki Index

## Maintained pages and semantic owners

- [Wiki authority](README.md)
- [Start Here](START-HERE.md)
- [Exhaustive Wiki Index](INDEX.md)
- [Glossary](GLOSSARY.md)
- [Contributing](CONTRIBUTING.md)
- [Limitations](LIMITATIONS.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Research System Map](architecture/Research-System-Map.md)
- [Research Questions](questions/Research-Questions.md)
- [Current Claim Language](claims/Current-Claim-Language.md)
- [Historical Claim Ledger](claims/Historical-Claim-Ledger.md)
- [Dataset Registry](datasets/Dataset-Registry.md)
- [Data and Target Contract](datasets/Data-and-Target-Contract.md)
- [Split Decision](decisions/0001-separate-link-prediction-and-lifecycle-readout.md)
- [Evidence Ledger](evidence/Evidence-Ledger.md)
- [License and Assets](governance/License-and-Assets.md)
- [Numeric Evidence and Publication Hygiene](governance/Numeric-Evidence-and-Publication-Hygiene.md)
- [Paper Export Contract](manuscript/Paper-Export-Contract.md)
- [Hierarchical Readout](methods/Hierarchical-Lifecycle-Readout.md)
- [Typed Interventions](methods/Typed-Interventions.md)
- [Research Workflow](operations/Research-Workflow.md)
- [Technical Source Map](references/Technical-Source-Map.md)
- [Faithfulness and Invariance](results/Lifecycle-Faithfulness-and-Invariance.md)
- [Project Status](status/Project-Status.md)
- [Live Execution](status/Live-Execution.md)

## Identifier index

| Kind | Identifier | Semantic owner |
|---|---|---|
| Research question | `LCR-RQ-DECODER-001` | [Research Questions](questions/Research-Questions.md) |
| Research question | `LCR-RQ-INVARIANCE-001` | [Research Questions](questions/Research-Questions.md) |
| Research question | `LCR-RQ-INTERVENTION-001` | [Research Questions](questions/Research-Questions.md) |
| Dataset | `LCR-D-COEDIT-001` | [Dataset Registry](datasets/Dataset-Registry.md) |
| Dataset | `LCR-D-COEDIT-PREIDFIX-001` | [Dataset Registry](datasets/Dataset-Registry.md) |
| Protocol | `LCR-P-READOUT-001` | [Protocol](../protocols/lifecycle_readout_v1.toml) |
| Evidence | `LCR-E-DATA-MIGRATION-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| Evidence | `LCR-E-RUNTIME-MIGRATION-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| Evidence | `LCR-E-LEGACY-RESULTS-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| Evidence | `LCR-E-LEGACY-PAPER-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| Execution job | `LCR-JOB-LOCAL-20260819-001` | [Job record](../evidence/jobs/LCR-JOB-LOCAL-20260819-001.toml) |
| Current claim | `LCR-C-COEDIT-IDFIX-001` | [Current Claim Language](claims/Current-Claim-Language.md) |
| Current claim | `LCR-C-DECODER-001` | [Current Claim Language](claims/Current-Claim-Language.md) |
| Current claim | `LCR-C-INVARIANCE-001` | [Current Claim Language](claims/Current-Claim-Language.md) |
| Current claim | `LCR-C-INTERVENTION-001` | [Current Claim Language](claims/Current-Claim-Language.md) |
| Historical claim | `LCR-H-MIXED-001` | [Historical Claim Ledger](claims/Historical-Claim-Ledger.md) |
| Decision | `DEC-0001` | [Split Decision](decisions/0001-separate-link-prediction-and-lifecycle-readout.md) |

## Paper snapshot index

None. `paper/CURRENT` and `PROJECT.toml` both declare `UNRELEASED`.

Identifier namespaces are project-local: questions `LCR-RQ-*`, datasets
`LCR-D-*`, protocols `LCR-P-*`, evidence `LCR-E-*`, claims `LCR-C-*`, and
historical claims `LCR-H-*`. Decision IDs use `DEC-*` within this project.

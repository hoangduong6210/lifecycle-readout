---
title: Exhaustive Wiki Index
status: canonical index
last_updated: 2026-09-16
paper_source: false
---

# Exhaustive Wiki Index

## Orientation and governance

| Page | Owns |
|---|---|
| [Wiki Home](README.md) | Authority rules, reader routes, and admission path |
| [Start Here](START-HERE.md) | New-contributor orientation and safe first checks |
| [Exhaustive Wiki Index](INDEX.md) | Complete page and identifier inventory |
| [Glossary](GLOSSARY.md) | Lifecycle terminology and notation |
| [Contributing](CONTRIBUTING.md) | Page ownership, review, and update rules |
| [Research System Map](architecture/Research-System-Map.md) | Scored path, readout path, and evidence flow |
| [License and Assets](governance/License-and-Assets.md) | Code, dataset, and asset-rights boundaries |
| [Numeric Evidence and Publication Hygiene](governance/Numeric-Evidence-and-Publication-Hygiene.md) | Quantitative-claim and publication exclusions |
| [Technical Source Map](references/Technical-Source-Map.md) | Source identity and citation review |

## Status, scope, and decisions

| Page | Owns |
|---|---|
| [Project Status](status/Project-Status.md) | Current stage, blockers, and next actions |
| [Live Execution](status/Live-Execution.md) | Current execution state |
| [Research Questions](questions/Research-Questions.md) | Decoder, invariance, and intervention endpoints |
| [Limitations](LIMITATIONS.md) | Boundaries applying across claims and manuscripts |
| [Split Decision](decisions/0001-separate-link-prediction-and-lifecycle-readout.md) | Project-separation rationale and consequences |

## Datasets and graph targets

| Page | Owns |
|---|---|
| [Dataset Registry](datasets/Dataset-Registry.md) | Local CoEdit identities and unresolved upstream provenance |
| [Data and Target Contract](datasets/Data-and-Target-Contract.md) | Dataset schema and target boundary |

## Methods

| Page | Owns |
|---|---|
| [Hierarchical Readout](methods/Hierarchical-Lifecycle-Readout.md) | State factorization and readout design |
| [Typed Interventions](methods/Typed-Interventions.md) | Intervention types and direction/reversibility gates |

## Results and claims

| Page | Owns |
|---|---|
| [Current Claim Language](claims/Current-Claim-Language.md) | Exact wording, scope, evidence, and eligibility |
| [Historical Claim Ledger](claims/Historical-Claim-Ledger.md) | Legacy wording retained for audit |
| [Faithfulness and Invariance](results/Lifecycle-Faithfulness-and-Invariance.md) | Current result status and blocked conclusions |
| [Evidence Ledger](evidence/Evidence-Ledger.md) | Artifact identities, checksums, jobs, and claim links |

## Reproduction and operation

| Page | Owns |
|---|---|
| [Reproducibility](REPRODUCIBILITY.md) | Source, data, attempt, and hash closure for a release |
| [Research Workflow](operations/Research-Workflow.md) | Question-to-evidence-to-paper process |

## Publication source and snapshots

| Page | Owns |
|---|---|
| [Paper Export Contract](manuscript/Paper-Export-Contract.md) | Eligibility gates and future snapshot lock |

No manuscript-source page or immutable paper snapshot is admitted yet. The
authoritative pointer is [`paper/CURRENT`](../paper/CURRENT).

## Identifier index

### Current claims

| IDs | Registry | State class |
|---|---|---|
| `LCR-C-COEDIT-IDFIX-001` | [Current Claim Language](claims/Current-Claim-Language.md) | Validated local migration property; not paper eligible |
| `LCR-C-DECODER-001`, `LCR-C-INVARIANCE-001`, `LCR-C-INTERVENTION-001` | [Current Claim Language](claims/Current-Claim-Language.md) | Blocked pending frozen evidence |

### Historical claims

| IDs | Registry | State class |
|---|---|---|
| `LCR-H-MIXED-001` | [Historical Claim Ledger](claims/Historical-Claim-Ledger.md) | Quarantined legacy wording |

### Evidence, datasets, and decisions

| IDs | Registry |
|---|---|
| `LCR-RQ-DECODER-001`, `LCR-RQ-INVARIANCE-001`, `LCR-RQ-INTERVENTION-001` | [Research Questions](questions/Research-Questions.md) |
| `LCR-D-COEDIT-001`, `LCR-D-COEDIT-PREIDFIX-001` | [Dataset Registry](datasets/Dataset-Registry.md) |
| `LCR-P-READOUT-001` | [Protocol](../protocols/lifecycle_readout_v1.toml) |
| `LCR-E-DATA-MIGRATION-001`, `LCR-E-RUNTIME-MIGRATION-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-E-LEGACY-RESULTS-001`, `LCR-E-LEGACY-PAPER-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-LOCAL-20260819-001` | [Job record](../evidence/jobs/LCR-JOB-LOCAL-20260819-001.toml) |
| `DEC-0001` | [Split Decision](decisions/0001-separate-link-prediction-and-lifecycle-readout.md) |

## Paper snapshot index

None. `paper/CURRENT` and `PROJECT.toml` both declare `UNRELEASED`.

Identifier namespaces are project-local: questions `LCR-RQ-*`, datasets
`LCR-D-*`, protocols `LCR-P-*`, evidence `LCR-E-*`, claims `LCR-C-*`, and
historical claims `LCR-H-*`. Decision IDs use `DEC-*` within this project.

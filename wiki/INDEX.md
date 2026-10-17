---
title: Exhaustive Wiki Index
status: canonical index
last_updated: 2026-10-16
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
| [Contribution Gap Review](questions/Contribution-Gap-Review.md) | Targeted prior-art overlaps and a candidate-specific temporal mechanism question |
| [Journal Research Program](questions/Research-Program.md) | Contribution hypothesis, development and submission gates |
| [Journal Paper Strategy](questions/Research-Paper-Strategy.md) | Article scope, evidence map, venue fit and publication costs |
| [Limitations](LIMITATIONS.md) | Boundaries applying across claims and manuscripts |
| [Split Decision](decisions/0001-separate-link-prediction-and-lifecycle-readout.md) | Project-separation rationale and consequences |
| [Differentiable Lifecycle Decision](decisions/0002-differentiable-latent-lifecycle-discovery.md) | Differentiable latent lifecycle discovery for the research study |

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
| [Differentiable Latent Lifecycle](methods/Differentiable-Latent-Lifecycle.md) | Continuous latent phase discovery and dynamics |
| [Shortcut-Resistant Gating](methods/Shortcut-Resistant-Gating.md) | Latent survival filter against temporal shortcuts |
| [Prediction Target and Sampling](methods/Prediction-Target-and-Sampling.md) | Survival, sampled labels and ranking interpretation boundaries |

## Results and claims

| Page | Owns |
|---|---|
| [Current Claim Language](claims/Current-Claim-Language.md) | Exact wording, scope, evidence, and eligibility |
| [Historical Claim Ledger](claims/Historical-Claim-Ledger.md) | Legacy wording retained for audit |
| [Faithfulness and Invariance](results/Lifecycle-Faithfulness-and-Invariance.md) | Current result status and blocked conclusions |
| [Benchmark and Baseline Results](results/Benchmark-and-Baseline-Results.md) | Audited empirical benchmark performance and baseline comparison |
| [CRAFT Adapter and Feasibility](results/CRAFT-Adapter-and-Feasibility.md) | Author interface, scoped runtime profile, failures and CPU/GPU audit |
| [ML-20M Development Baselines](results/ML20M-Development-Baselines.md) | Fresh first-time-pair validation screen and exact audit scope |
| [Historical Hard Negative Controls](results/Hard-Negative-Controls.md) | Feature-matched controls, support exclusions and exploratory negative outcomes |
| [Evidence Ledger](evidence/Evidence-Ledger.md) | Artifact identities, checksums, jobs, and claim links |
| [research Validation Development](results/Validation-Development.md) | Partial validation study, failed transport reconciliation and development decision |
| [Synthetic Temporal Response Diagnostics](results/Synthetic-Mechanisms.md) | Capacity, null response and misspecified temporal effects |
| [Hard Control Retrieval](evidence/Hard-Control-Retrieval.md) | Exact local results, source archives, audit and manuscript lookup |

## Reproduction and operation

| Page | Owns |
|---|---|
| [Reproducibility](REPRODUCIBILITY.md) | Source, data, attempt, and hash closure for a release |
| [Research Workflow](operations/Research-Workflow.md) | Question-to-evidence-to-paper process |

## Publication source and snapshots

| Page | Owns |
|---|---|
| [Paper Export Contract](manuscript/Paper-Export-Contract.md) | Eligibility gates and future snapshot lock |
| [Article Assembly and Verification](manuscript/Article-Assembly.md) | Authorized publisher-template working export, author facts and release boundary |

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
| `LCR-RQ-DECODER-001`, `LCR-RQ-INVARIANCE-001`, `LCR-RQ-INTERVENTION-001`, `LCR-RQ-DISCOVERY-001`, `LCR-RQ-SHORTCUT-001`, `LCR-RQ-CAUSAL-001` | [Research Questions](questions/Research-Questions.md) |
| `LCR-D-COEDIT-001`, `LCR-D-COEDIT-PREIDFIX-001`, `LCR-D-WIKI-001`, `LCR-D-REDDIT-001`, `LCR-D-MOOC-001`, `LCR-D-LASTFM-001` | [Dataset Registry](datasets/Dataset-Registry.md) |
| `LCR-P-READOUT-001` | [Readout protocol](../protocols/lifecycle_readout_v1.toml) |
| `LCR-P-TKDE-001` | [Earlier benchmark protocol](../protocols/lifecycle_tkde_v1.toml) |
| `LCR-P-TKDE-002` | [Corrected causal protocol](../protocols/lifecycle_tkde_v2.toml) |
| `LCR-P-HARD-001` | [Hard control protocol](../protocols/hard_controls_v1.toml) |
| `LCR-P-Q1-DEV-001` | [Validation development protocol](../protocols/q1_development_v1.toml) |
| `LCR-P-Q1-SYNTH-001` | [Oracle diagnostic protocol](../protocols/q1_synthetic_v1.toml) |
| `LCR-P-TGBSEQ-SCREEN-001` | [Validation baseline screen](../protocols/tgbseq_baseline_screen_v1.toml) |
| `LCR-E-ML20M-SCHEMA-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-ML20M-SCHEMA-001` | [Schema job](../evidence/jobs/LCR-JOB-ML20M-SCHEMA-001.toml) |
| `LCR-E-Q1-SYNTHETIC-001`, `LCR-E-Q1-PREFLIGHT-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-Q1-SYNTHETIC-001` | [Oracle diagnostic job](../evidence/jobs/LCR-JOB-Q1-SYNTHETIC-001.toml) |
| `LCR-JOB-Q1-PREFLIGHT-001` | [Preflight job](../evidence/jobs/LCR-JOB-Q1-PREFLIGHT-001.toml) |
| `LCR-RQ-TARGET-001` | [Research Questions](questions/Research-Questions.md) |
| `LCR-D-TGBSEQ-ML20M-001` | [Dataset Registry](datasets/Dataset-Registry.md) |
| `LCR-E-TGBSEQ-INTAKE-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-TGBSEQ-INTAKE-001` | [Intake job](../evidence/jobs/LCR-JOB-TGBSEQ-INTAKE-001.toml) |
| `LCR-E-Q1-DIAGNOSTIC-001`, `LCR-E-Q1-DEVELOPMENT-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-Q1-DEV-WIKIPEDIA-001` | [Development job](../evidence/jobs/LCR-JOB-Q1-DEV-WIKIPEDIA-001.toml) |
| `LCR-JOB-Q1-DIAGNOSTIC-001` | [Diagnostic job](../evidence/jobs/LCR-JOB-Q1-DIAGNOSTIC-001.toml) |
| `LCR-E-DATA-MIGRATION-001`, `LCR-E-RUNTIME-MIGRATION-001`, `LCR-E-COLAB-WORKFLOW-001`, `LCR-E-TKDE-BENCHMARK-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-E-LEGACY-RESULTS-001`, `LCR-E-LEGACY-PAPER-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-E-HARD-CONTROLS-001` | [Hard control retrieval](evidence/Hard-Control-Retrieval.md) |
| `LCR-JOB-HARD-WIKIPEDIA-001` | [Wikipedia execution](../evidence/jobs/LCR-JOB-HARD-WIKIPEDIA-001.toml) |
| `LCR-JOB-HARD-REDDIT-001` | [Reddit execution](../evidence/jobs/LCR-JOB-HARD-REDDIT-001.toml) |
| `LCR-JOB-COLAB-SMOKE-20261014-001`, `LCR-JOB-COLAB-SMOKE-20261014-002`, `LCR-JOB-LOCAL-20260819-001`, `LCR-JOB-LOCAL-20261013-001`, `LCR-JOB-COLAB-WIKI-001`, `LCR-JOB-COLAB-MOOC-001`, `LCR-JOB-COLAB-REDDIT-001`, `LCR-JOB-COLAB-LASTFM-001`, `LCR-JOB-BASELINE-GRAPHMIXER-WIKI-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `DEC-0001`, `DEC-0002` | [Decisions](decisions/0001-separate-link-prediction-and-lifecycle-readout.md) |

## Paper snapshot index

None. `paper/CURRENT` and `PROJECT.toml` both declare `UNRELEASED`.

Identifier namespaces are project-local: questions `LCR-RQ-*`, datasets
`LCR-D-*`, protocols `LCR-P-*`, evidence `LCR-E-*`, claims `LCR-C-*`, and
historical claims `LCR-H-*`. Decision IDs use `DEC-*` within this project.

| `LCR-E-ML20M-SCREEN-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-ML20M-SCREEN-001` | [Screen job](../evidence/jobs/LCR-JOB-ML20M-SCREEN-001.toml) |

| `LCR-P-CRAFT-PROFILE-001` | [CRAFT feasibility protocol](../protocols/craft_profile_v1.toml) |

| `LCR-P-CRAFT-PROFILE-002` | [CRAFT upload-size amendment](../protocols/craft_profile_v2.toml) |

| `LCR-E-CRAFT-PROFILE-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-CRAFT-PROFILE-001` | [Profile job](../evidence/jobs/LCR-JOB-CRAFT-PROFILE-001.toml) |
| `LCR-JOB-CRAFT-PROFILE-002` | [Profile job](../evidence/jobs/LCR-JOB-CRAFT-PROFILE-002.toml) |
| `LCR-JOB-CRAFT-PROFILE-003` | [Profile job](../evidence/jobs/LCR-JOB-CRAFT-PROFILE-003.toml) |

| `LCR-P-CRAFT-FIT-001` | [Full-support CRAFT validation development](../protocols/craft_fit_v1.toml) |

| `LCR-P-CADENCE-DEV-001` | [Pair-cadence validation development](../protocols/pair_cadence_development_v1.toml) |


| `LCR-E-CRAFT-FIT-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-CRAFT-FIT-001` | [Bounded author-fit job](../evidence/jobs/LCR-JOB-CRAFT-FIT-001.toml) |

| `LCR-P-SESSION-DIAG-001` | [Source-session diagnostic plan](../protocols/source_session_diagnostic_v1.toml) |
| `LCR-E-SESSION-DIAG-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-SESSION-DIAG-001` | [CPU diagnostic job](../evidence/jobs/LCR-JOB-SESSION-DIAG-001.toml) |

- [Pair Cadence Development](results/Pair-Cadence-Development.md)
- [Journal Narrative](manuscript/Article-Narrative.md)

| `LCR-E-SOURCE-PORTABILITY-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-SOURCE-RECOVERY-001` | [Source recovery job](../evidence/jobs/LCR-JOB-SOURCE-RECOVERY-001.toml) |

| `LCR-E-CADENCE-DEV-001` | [Evidence Ledger](evidence/Evidence-Ledger.md) |
| `LCR-JOB-CADENCE-WIKIPEDIA-ORIGINAL-001` | [Cadence job](../evidence/jobs/LCR-JOB-CADENCE-WIKIPEDIA-ORIGINAL-001.toml) |
| `LCR-JOB-CADENCE-WIKIPEDIA-FEATURES-001` | [Cadence job](../evidence/jobs/LCR-JOB-CADENCE-WIKIPEDIA-FEATURES-001.toml) |
| `LCR-JOB-CADENCE-WIKIPEDIA-RELATIVE-001` | [Cadence job](../evidence/jobs/LCR-JOB-CADENCE-WIKIPEDIA-RELATIVE-001.toml) |
| `LCR-JOB-CADENCE-REDDIT-ORIGINAL-001` | [Cadence job](../evidence/jobs/LCR-JOB-CADENCE-REDDIT-ORIGINAL-001.toml) |
| `LCR-JOB-CADENCE-REDDIT-FEATURES-001` | [Cadence job](../evidence/jobs/LCR-JOB-CADENCE-REDDIT-FEATURES-001.toml) |
| `LCR-JOB-CADENCE-REDDIT-RELATIVE-001` | [Cadence job](../evidence/jobs/LCR-JOB-CADENCE-REDDIT-RELATIVE-001.toml) |

## Exploratory simplex development

- [Monotone Simplex Transition Development](methods/Monotone-Simplex-Development.md): selected direction, exact derivative criterion, prototype and limits.
- `LCR-E-SIMPLEX-DEV-001`: scoped local algebra and implementation record.
- `LCR-JOB-SIMPLEX-DEV-001`: local CPU checks; no model fit.

## Competing-destination development

- [Competing Destination Development](methods/Competing-Destination-Development.md): online behavioral exposure, equal-input controls and mechanism scope.
- [Competing Destination Pilot](results/Competing-Destination-Pilot.md): prospective execution and eventual outcome owner.
- `LCR-P-COMPETE-DEV-001`: initial Wikipedia validation-only exploratory pilot.

- `LCR-E-COMPETE-DEV-001`: complete exploratory pilot with negative primary and scoped audits.
- `LCR-JOB-COMPETE-DEV-001`: owned Colab pilot, artifacts retrieved and runtime released.

## Readout-specific preservation and scored-path diagnostics

- [Readout Aware Entropic Preservation](methods/Readout-Aware-Entropic-Preservation.md): exact restricted condition, proof, constructor and strict-separation witness.
- [Scored Path Silence Diagnostic](results/Scored-Path-Silence-Diagnostic.md): learned checkpoint anchor paths and native replay.
- `LCR-P-SCORED-PATH-001`: posthoc fixed validation-anchor diagnostic.
- `LCR-E-SCORED-PATH-001`: scoped learned-model path evidence.
- `LCR-JOB-SCORED-PATH-001`: no-fit CPU diagnostic and native checks.
- `LCR-E-READOUT-TILT-001`: restricted proof, symbolic identities and constructor checks.
- `LCR-JOB-READOUT-TILT-001`: local algebra and prototype verification.

## Full validation scored-path expansion

- `LCR-P-SCORED-FULL-001`: posthoc all-validation expansion of the fixed-anchor path diagnostic; no refitting.
- [Full Validation Scored Paths](results/Full-Validation-Scored-Paths.md): execution, whole-supported-cohort outcomes and limits.

- `LCR-E-SCORED-FULL-001`: complete supported-validation conditional path evidence and scoped verification.
- `LCR-JOB-SCORED-FULL-001`: two sequential no-fit Colab diagnostics, with verified resource release.

## No-refit readout repair

- [Readout Repair Projection](methods/Readout-Repair-Projection.md): constrained matrix-distance repair and controlled reference/flow operations.
- [Readout Repair Validation](results/Readout-Repair-Validation.md): fixed validation ranking/structural endpoints and eventual outcome owner.
- `LCR-P-REPAIR-VAL-001`: prospective posthoc no-refit repair comparison.

- `LCR-E-REPAIR-VAL-001`: restricted repair, ranking-cost controls and complete NumPy lifecycle-score replay.
- `LCR-JOB-REPAIR-VAL-001`: no-fit validation comparison, both owned runtimes released.

## Exact event-driven competition development

- [Event Driven Competitive Lifecycle](methods/Event-Driven-Competitive-Lifecycle.md): finite-jump semantics, exact state operator and exposure-clock equivalence boundary.
- `LCR-E-COMPETE-FLOW-001`: local algebra/operator checks, not admitted.

## Frozen-reference event history

- [Frozen Reference Competition History](methods/Frozen-Reference-Competition-History.md): exact interval dose and timestamp/reset obligations.
- [Competitive Event History Audit](results/Competitive-Event-History-Audit.md): completed scoped real-history audit and descriptive coverage.
- `LCR-P-COMPETE-HISTORY-001`: prospective exploratory input audit.
- `LCR-E-COMPETE-HISTORY-001`: verified development inputs, not admitted.

## Exact-state competition model comparison

- [Exact State Competition Study](methods/Exact-State-Competition-Study.md): prospective reset, flow, primary equal-parameter comparator and audit contract.
- [Exact State Competition Pilot](results/Exact-State-Competition-Pilot.md): fixed endpoints and execution/outcome owner.
- `LCR-P-EVENT-COUPLING-001`: separate exploratory exact-state model comparison.

- `LCR-E-EVENT-COUPLING-001`: completed exact-state validation pilot, full checkpoint inference audit and runtime release.

## Competition input anatomy and dataset proposal

- [Competition Signal Anatomy](methods/Competition-Signal-Anatomy.md): exact source/hash and shuffle-background identities.
- [Competition Signal Anatomy Results](results/Competition-Signal-Anatomy.md): posthoc full-validation decomposition and scope.
- [Competition Dataset Proposal](datasets/Competition-Dataset-Proposal.md): independently supplied content candidates and recurrence preflight.
- `LCR-P-COMPETE-ATTR-001`: fixed posthoc input anatomy.
- `LCR-P-COMPETE-ATTR-FULL-001`: separate full-validation exact-ID expansion.
- `LCR-E-COMPETE-ATTR-001`: initial component and fixed-anchor audit, not admitted.
- `LCR-E-COMPETE-ATTR-FULL-001`: full validation exact-ID decomposition, not admitted.

## Alternative dataset intake

- [KuaiRand Intake Preflight](datasets/KuaiRand-Intake-Preflight.md): fixed training-window intake, source identity, recurrence and resource gates.
- `LCR-P-KUAIRAND-INTAKE-001`: bounded CPU-only candidate-dataset intake before model fitting.

- `LCR-P-KUAIRAND-LOCAL-001`; `LCR-E-KUAIRAND-INTAKE-001` — [KuaiRand intake outcome](datasets/KuaiRand-Intake-Preflight.md).

- [Last.fm Original History Intake](datasets/LastFM-Intake-Preflight.md) — `LCR-P-LASTFM-INTAKE-001`.

- `LCR-E-LASTFM-INTAKE-001` — [Last.fm fixed intake and audit](datasets/LastFM-Intake-Preflight.md).

- [Last.fm Recorded Artist History Audit](results/LastFM-Recorded-Artist-Audit.md) — `LCR-P-LASTFM-ARTIST-001`.

- `LCR-E-LASTFM-ARTIST-001` — [Last.fm recorded-artist inputs](results/LastFM-Recorded-Artist-Audit.md).

- [Last.fm Recorded Artist Model Comparison](methods/LastFM-Recorded-Artist-Model.md) — `LCR-P-LASTFM-MODEL-001`.
- [Last.fm Recorded Artist Pilot](results/LastFM-Recorded-Artist-Pilot.md) — `LCR-E-LASTFM-MODEL-PREP-001`.

- `LCR-E-LASTFM-PRIOR-001` — [Repeat and metadata-inhibition prior review](questions/Contribution-Gap-Review.md#repeat-consumption-and-metadata-inhibition-precedents-for-lastfm).

- [Last.fm complete controlled model outcome](results/LastFM-Recorded-Artist-Pilot.md) — `LCR-E-LASTFM-MODEL-001`.

- [LastFM-History-Strata](results/LastFM-History-Strata.md) — `LCR-P-LASTFM-STRATA-001`, `LCR-E-LASTFM-STRATA-001`.

- [LastFM-Mechanistic-Readout](results/LastFM-Mechanistic-Readout.md) — `LCR-P-LASTFM-READOUT-001`, `LCR-E-LASTFM-READOUT-001`.

- [Fixed Rate Readout Certificate](methods/Fixed-Rate-Readout-Certificate.md) — `LCR-P-FIXED-RATE-TILT-001`, `LCR-E-FIXED-RATE-TILT-001`.

- [Inverse-Tilt-Projection](methods/Inverse-Tilt-Projection.md) — `LCR-P-INVERSE-TILT-001`, `LCR-E-INVERSE-TILT-001`.

- [Fixed-Rate-Repair-Validation](results/Fixed-Rate-Repair-Validation.md) — `LCR-P-RATE-REPAIR-VAL-001`, `LCR-E-RATE-REPAIR-VAL-001`.

- [Last.fm February Holdout](results/LastFM-February-Holdout.md) — `LCR-P-LASTFM-FEB-001`.

- February preparation and heldout evaluation — `LCR-E-LASTFM-FEB-PREP-001`, `LCR-E-LASTFM-FEB-001`, `LCR-JOB-LASTFM-FEB-PREP-001`, `LCR-JOB-LASTFM-FEB-001`; [result owner](results/LastFM-February-Holdout.md).

- [Spectral Ratio Prior Position](questions/Spectral-Ratio-Prior-Position.md) — `LCR-P-SPECTRAL-POSITION-001`, `LCR-E-SPECTRAL-POSITION-001`, `LCR-JOB-SPECTRAL-POSITION-001`.

## Additional scientific and reproduction pages

| Page | Role |
|---|---|
| [Figure-to-File Map](Figures.md) | Canonical scientific or reproduction owner |
| [Distributed Repository Evidence](evidence/Repository-Evidence.md) | Canonical scientific or reproduction owner |

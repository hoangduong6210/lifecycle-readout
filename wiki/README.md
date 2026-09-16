---
title: Lifecycle Readout Research Wiki
status: canonical home
last_updated: 2026-09-16
paper_source: false
---

# Lifecycle Readout Research Wiki

This wiki is the project's scientific source of truth. It owns method
interpretation, claim status, limitations, decisions, and publication-ready
language. A paper is a versioned snapshot of admitted wiki content, not an
independent source of new claims.

Executable code lives under `src/` and `experiments/`; protocol and configuration
live under `protocols/` and `configs/`. The dataset registry identifies local
corpora, and the evidence ledger explains what checked artifacts establish.
Both release pointers are `UNRELEASED`.

## Choose an entry route

| Reader | Begin here | Then read |
|---|---|---|
| New contributor | [Start Here](START-HERE.md) | [Research System Map](architecture/Research-System-Map.md) and [Research Workflow](operations/Research-Workflow.md) |
| Returning owner | [Project Status](status/Project-Status.md) | [Live Execution](status/Live-Execution.md), claims, evidence, and reproducibility |
| Research reader | [Claim Registry](claims/Current-Claim-Language.md) | [Research Questions](questions/Research-Questions.md), [Methods](methods/Hierarchical-Lifecycle-Readout.md), [Results](results/Lifecycle-Faithfulness-and-Invariance.md), and [Limitations](LIMITATIONS.md) |
| Dataset reviewer | [Dataset Registry](datasets/Dataset-Registry.md) | [Data and Target Contract](datasets/Data-and-Target-Contract.md), [Source Map](references/Technical-Source-Map.md), and [License and Assets](governance/License-and-Assets.md) |
| Compute operator | [Research Workflow](operations/Research-Workflow.md) | [Live Execution](status/Live-Execution.md) and the [Reproducibility Contract](REPRODUCIBILITY.md) |
| Paper editor | [Paper Export Contract](manuscript/Paper-Export-Contract.md) | [Claim Registry](claims/Current-Claim-Language.md), [Evidence Ledger](evidence/Evidence-Ledger.md), and [Limitations](LIMITATIONS.md) |

## Canonical index

The [Exhaustive Index](INDEX.md) lists every maintained wiki page by semantic
owner and links the current identifiers to their registries. A page absent from
that index is outside the maintained research record.

## Authority rules

1. `paper_source: true` identifies prose that may be considered for a future
   paper only after its linked claim is admitted.
2. `paper_source: false` identifies operational, historical, or editorial
   material; it does not itself permit a scientific claim.
3. A completed computation does not become a claim automatically. The
   [Claim Registry](claims/Current-Claim-Language.md) owns exact wording,
   population, evidence, qualifiers, and paper eligibility.
4. A frozen release must pin source, data, configuration, attempts, failures,
   and hashes before its evidence can support a claim.
5. `paper/working/` and `results/historical/` remain local quarantine inputs,
   outside the source repository and public-release surface.

## Admission path

```text
research question
    -> frozen protocol and immutable inputs
    -> complete scheduler attempts and numerical gates
    -> checksum-locked evidence release
    -> scoped claim reviewed in the registry
    -> manuscript-source page
    -> versioned paper snapshot
```

The [Project Status](status/Project-Status.md) and [Evidence Ledger](evidence/Evidence-Ledger.md)
currently stop this path before a scientific release or paper snapshot.

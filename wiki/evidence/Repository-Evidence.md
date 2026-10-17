---
title: Distributed Repository Evidence
status: canonical research documentation
last_updated: 2026-10-17
paper_source: false
---

# Distributed evidence and provenance

The [publication manifest](../../evidence/publication-manifest.json) records each
included scientific report, its original SHA-256 and its public SHA-256. Reports
with identical hashes preserve source bytes. A host-path-redacted derivative is
explicitly marked and retains the source hash; numerical scientific fields are
unchanged. Historical retrieval indexes continue to pin original archived files,
including inputs which are not distributed in this repository.

| Question | Direct report | Scientific interpretation |
|---|---|---|
| Does a decreasing latent flow guarantee a decreasing final score? | [Complete path report](../../results/reports/scored-paths.json) | [Complete supported-path diagnostic](../results/Full-Validation-Scored-Paths.md) |
| Does specified-rate certification avoid extra deformation? | [Certificate and matched repair report](../../results/reports/rate-repair.json) | [Retained-model comparison](../results/Fixed-Rate-Repair-Validation.md) |
| Did the selected phase branch beat its flexible control? | [January model comparison](../../results/reports/january-model.json) | [Full January outcome](../results/LastFM-Recorded-Artist-Pilot.md) |
| Was the local January observation confirmed next month? | [January cohorts](../../results/reports/january-cohorts.json), [February follow-up](../../results/reports/february-holdout.json) | [Exploratory strata](../results/LastFM-History-Strata.md), [failed confirmation](../results/LastFM-February-Holdout.md) |

## Evidence levels

Mathematical statements are owned by the exact definitions, assumptions and proofs
in the [method pages](../methods/Fixed-Rate-Readout-Certificate.md). Numerical proof
checks are supporting tests, not a formal proof assistant or independent review.
Empirical summaries describe the declared retained models and conditional candidate
populations, not universal or causal findings. The evidence and paper pointers stay
`UNRELEASED`; publishing auditable working reports does not admit frozen claims.

All aggregate article CSVs are in the [editable article](../../paper/working/article/).
The root README is generated from these distributed reports, with a source link
beside each reported number. Figure data and overview claims can be checked with
`python scripts/verify_public_evidence.py`.

Raw third-party interaction corpora, trained model checkpoints, candidate-level
prediction arrays, downloaded third-party texts, cache files and full transport
archives are not redistributed. Checksums and historical retrieval maps preserve
their identities. Full checkpoint/history replay requires those original inputs;
implementation tests, aggregate reconstruction and figure regeneration do not.

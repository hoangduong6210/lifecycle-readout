---
title: Dataset Registry
status: canonical dataset registry
last_updated: 2026-10-15
paper_source: false
---

# Dataset Registry

## Verified original Last.fm training intake

The [new Last.fm-1K record](LastFM-Intake-Preflight.md) has primary-matching archive and raw TSV MD5 identities plus SHA256 pins, a fixed pre-cutoff sampled history and a passed engineering support gate. It is separate from the earlier processed LastFM benchmark. This is a verified development intake, not a frozen admitted dataset identity or predictive result.


## KuaiRand fixed intake outcome

The [actual intake](KuaiRand-Intake-Preflight.md) supersedes the earlier proposal-only access state. A local archive matches the published MD5 and recorded SHA256; fixed training-window preflight and independent recurrence replay complete. The clicked long-gap support gate fails. This does not admit a corpus identity or predictive claim into a frozen release. Failed network acquisition and CPU cleanup remain separately recorded.


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

## LCR-D-WIKI-001

- **Version and lifecycle:** Public Stanford SNAP Wikipedia temporal network benchmark.
- **Source and provenance:** Stanford SNAP JODIE interaction dataset; download/preparation owned by
  [benchmark.py](../../src/lifecycle_readout/benchmark.py).
- **License and redistribution:** Publicly downloadable; dataset-specific use/redistribution terms remain
  unresolved. Availability alone does not establish an open license.
- **Population and geometry:** Dynamic bipartite user-page interaction events with text features.
- **Protocol ownership:** Earlier corrected benchmark `LCR-P-TKDE-002`; the historically matched study `LCR-P-HARD-001` applies to Wikipedia and Reddit only.

## LCR-D-REDDIT-001

- **Version and lifecycle:** Public Stanford SNAP Reddit temporal interaction graph benchmark.
- **Source and provenance:** Stanford SNAP JODIE interaction dataset; download/preparation owned by
  [benchmark.py](../../src/lifecycle_readout/benchmark.py).
- **License and redistribution:** Publicly downloadable; dataset-specific use/redistribution terms remain
  unresolved. Availability alone does not establish an open license.
- **Population and geometry:** Dynamic user-subreddit posting events with LIWC text-category features.
- **Protocol ownership:** Earlier corrected benchmark `LCR-P-TKDE-002`; the historically matched study `LCR-P-HARD-001` applies to Wikipedia and Reddit only.

## LCR-D-MOOC-001

- **Version and lifecycle:** Public Stanford SNAP MOOC online course interaction graph.
- **Source and provenance:** Stanford SNAP interaction dataset; download/preparation owned by
  [benchmark.py](../../src/lifecycle_readout/benchmark.py).
- **License and redistribution:** Publicly downloadable; dataset-specific use/redistribution terms remain
  unresolved. Availability alone does not establish an open license.
- **Population and geometry:** Dynamic student-activity interaction events.
- **Protocol ownership:** Earlier corrected benchmark `LCR-P-TKDE-002`; the historically matched study `LCR-P-HARD-001` applies to Wikipedia and Reddit only.

## LCR-D-LASTFM-001

- **Version and lifecycle:** Public Stanford SNAP LastFM user-song listening event graph.
- **Source and provenance:** Stanford SNAP interaction dataset; download/preparation owned by
  [benchmark.py](../../src/lifecycle_readout/benchmark.py).
- **License and redistribution:** Publicly downloadable; dataset-specific use/redistribution terms remain
  unresolved. Availability alone does not establish an open license.
- **Population and geometry:** Dynamic user-song interaction sequence. The stored feature channels are all-zero placeholders; the original JODIE paper describes this dataset as having no interaction features.
- **Protocol ownership:** Earlier corrected benchmark `LCR-P-TKDE-002`; the historically matched study `LCR-P-HARD-001` applies to Wikipedia and Reddit only.

No dataset is eligible for a frozen evidence release until source identity,
license, checksums, build configuration, split identity, and leakage controls close.


## Fresh benchmark preflight

TGB-Seq code is pinned to upstream commit
`c1ee801ea4301c2f944f5b3da10cbc5310fa4f68`. Code is MIT licensed; this does not
license the underlying datasets. The official evaluator and local strict-shape
adapter agree on `1009` simulated queries, including all ties. The upstream
loader consumes supplied split masks and official negatives; a future adapter
must preserve those identities rather than reuse our chronological quantiles.
No new real-data metric has been computed. The technical record is
`LCR-E-Q1-PREFLIGHT-001`, with exact source and metadata hashes under
`results/audit/q1_preflight_v1/`.

| Candidate dataset | Metadata revision | Preflight state |
|---|---|---|
| GoogleLocal | `c818133589f185c48c7e0e602f92d32dffe302cb` | Metadata inspected; dataset files not downloaded; underlying terms unresolved |
| Flickr | `64f31b40c054eacecc4f793589b5cef928503951` | Metadata inspected; dataset files not downloaded; underlying terms unresolved |
| ML-20M | `60e27d0ac9a13517d69b9940c34fd29e798d0c63` | Local byte intake now verified; scientific readiness remains open |

Hugging Face metadata supplies no license field for these three snapshots.
The metadata observations preceded ML-20M intake and remain preserved unchanged.
Source-specific terms need separate inspection. The GoogleLocal metadata also
lists large optional feature arrays, so event-stream and feature availability
must be distinguished when budgeting and declaring feature access.

## LCR-D-TGBSEQ-ML20M-001

- **Version and lifecycle:** Exact TGB-Seq dataset repository revision pinned
  above; local staged input only, not a frozen scientific release.
- **Source and provenance:** [TGB-Seq catalogue](https://tgb-seq.github.io/datasets/)
  identifies MovieLens as the original source. Benchmark CSV and official negative
  array match the pinned repository's LFS SHA-256 and size. The transformation
  from original GroupLens bytes to benchmark bytes has not been independently
  reconstructed.
- **License and redistribution:** The
  [original GroupLens README](https://files.grouplens.org/datasets/movielens/ml-20m-README.html)
  allows research use with acknowledgement and no implied endorsement; it restricts
  redistribution and commercial use without separate permission. This intake is
  local research only; no redistribution permission was obtained. The original
  citation is Harper and Konstan, The MovieLens Datasets: History and Context,
  ACM TiiS 2015. A mirror's availability does not establish redistribution rights.
- **Checksum and exact files:**
  [Structured identity](../../resources/tgbseq_ml20m_v1.toml) owns byte hashes,
  sizes, URLs and revision; the source terms and acquisition record are retained
  in [intake evidence](../../results/audit/q1_preflight_v1/ml20m_intake.json).
- **Population and geometry:** User-to-movie rating interactions treated as
  implicit feedback by the benchmark. The intake established byte identities; the later schema/development audit below owns reviewed chronological masks, supplied ID ranges and train/validation preparation. Original-to-benchmark collection and filtering remain open.
- **Split and leakage controls:** The later development adapters preserve supplied split masks and evaluator tie handling. Full official test evaluation remains pending; generated development validation candidates are distinct from supplied test candidates.
  The test negative array was downloaded and hashed without loading its contents.
  No new test scoring, model training or test-driven selection occurred.
- **Execution and evidence:** `LCR-JOB-TGBSEQ-INTAKE-001` and
  `LCR-E-TGBSEQ-INTAKE-001`.
- **Compatible claims:** No scientific claim admitted. Local benchmark byte
  identity is a technical intake property only.


## ML-20M schema and development support audit

`LCR-E-ML20M-SCHEMA-001` now verifies the supplied CSV schema and chronological
split masks. Exact counts are `14000091` train, `364943` validation and `129291`
test structural rows. The source/destination ranges are disjoint as supplied.
No `70/15/15` resplit is applied. The candidate file header is `[129291,100]`
with `int64` values; values were not loaded. Test CSV fields were inspected only
for structure, chronology, ranges and counts; no test recurrence or model score
was calculated.

All `14365034` train/validation pairs are distinct, with zero strictly past
pair-history events. The earlier historically matched recurrent-support protocol
therefore has no eligible positive supervision on these development rows. This
is a limitation of that protocol, not a claim that every lifecycle model fails.

The [derived manifest](../../resources/tgbseq_ml20m_development_v1.toml) owns
current train/validation-only preparation and the
[schema report](../../results/audit/q1_preflight_v1/ml20m_schema.json).
The original intake manifest's earlier schema-review flag records intake state;
this later derived record owns the reviewed development state. No test rows or
test candidate values are included in the prepared NPZ used for Colab.
Original-to-benchmark byte reconstruction remains open; the source-level split interpretation below is now reviewed.


## JODIE source and preparation review

The [official JODIE page](https://snap.stanford.edu/jodie/) and the pinned author download script identify the same corpus locations as the retained identities. The [author README](https://github.com/claws-lab/jodie) distinguishes the state-change label from event features. Our preparation excludes that label column from predictor inputs. The [original paper](https://cs.stanford.edu/people/jure/pubs/jodie-kdd19.pdf) describes LIWC features for Wikipedia and Reddit and no interaction features for LastFM; the locally stored LastFM channels are verified all-zero placeholders.

The versioned `resources/benchmark_causal_v2.toml` now centralizes the declared local raw/derived identities, source locations and preparation rules without redistributing corpus bytes. Source-page and author-code snapshots are retained in `results/audit/benchmark_preparation_v1/source_review/`. The repository MIT license refers to software and associated documentation; it is not recorded here as a separate grant for the externally hosted data. Citation instructions are verified, while dataset-specific redistribution permission and original collection/feature reconstruction remain unresolved.

The full-row local preparation audit is complete for Wikipedia, Reddit, MOOC and LastFM. It reconstructs every prepared source ID, shifted destination ID, timestamp and feature value from retained CSV bytes, with independent Python-parser anchors and all input hashes retained. Row order, timestamps and feature columns are preserved, and the state-label column is excluded. This is a local preparation identity check, distinct from equality to a newly retrieved upstream copy or reconstruction of the original collection pipeline. No dataset or scientific claim is admitted by this review.


## ML-20M split and preprocessing interpretation

The [TGB-Seq paper](https://arxiv.org/html/2502.02975v3) describes chronological splitting followed by training-degree and future-node filtering. The pinned [author preprocessor](https://github.com/TGB-Seq/TGB-Seq/blob/c1ee801ea4301c2f944f5b3da10cbc5310fa4f68/tgb_seq/datasets/preprocess.py) computes time cutoffs before filtering, prunes low-degree training nodes iteratively, then retains future rows only for surviving training nodes. It reuses the original cutoffs when assigning final split masks. Retained split proportions therefore need not equal the nominal pre-filter proportions. The current experiments preserve the supplied masks rather than splitting the retained table again.

This is source inspection, not a reconstruction from original MovieLens ratings. The pinned script's output basename also differs from the supplied benchmark filename; their exact equivalence has not been established. Cached paper/source bytes, source line ranges, hashes and explicit scope are retained in `results/audit/ml20m_preprocessing_review_v1/`. Neither data inclusion nor active baseline settings changed.

## User-authorized alternative competition dataset proposal

[The current proposal](Competition-Dataset-Proposal.md) conditionally prioritizes
KuaiRand-1K for complete sampled-user histories and independently supplied video
tags. MIND-small is secondary and has a different recurrence/clock boundary.
These are documentation-reviewed candidates, not downloaded corpora or newly
admitted dataset identities. The existing registry entries and results remain
unchanged.

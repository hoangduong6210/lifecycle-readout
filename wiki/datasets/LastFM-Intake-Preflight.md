---
title: Last.fm Original History Intake
status: complete and independently audited; fixed support gate passes, no model fit
last_updated: 2026-10-16
paper_source: false
---

# Original user-track history before another fit

`LCR-P-LASTFM-INTAKE-001` fixes this training-only intake after the
[KuaiRand planning gate failed](KuaiRand-Intake-Preflight.md). Both that failure
and the prior negative matched-control comparisons remain part of the evidence.
The [creator-attributed institutional record](https://zenodo.org/records/6090214)
provides archive and raw TSV checksums, schema and non-commercial use terms.
This original Last.fm-1K history is distinct from the earlier processed LastFM
benchmark in this repository.

## Acquisition and input identity

A direct verified-TLS Zenodo request timed out. The institution's HTTPS server
provided the original archive. A sequential transfer was intentionally interrupted
at a known owned process, preserving its prefix, then completed with explicit
nonoverlapping HTTP byte ranges and validated Content-Range headers. Its exit
`130` and unfinished stage status are retained, not reported as successful
independent acquisition. Only the assembled full archive is the verified result.
No third-party processed data or GitHub release binary was used.

Complete archive MD5 matches the primary record; original event and profile TSV
MD5 values independently match their respective published values. SHA256 values,
member names, sizes, source guards, prepared arrays and outputs are retained in
`results/audit/lastfm_intake_execution_v1/`. Only allow-listed regular members
were extracted; this is not an exhaustive archive-member inventory. Full archive
SHA256 was rechecked after analysis with the same value.

## Prespecified sample and scope

Sample 100 profile user IDs using a fixed hash ranking and seed `17`, retaining
inactive users without replacement. All their event rows before
`2009-01-01T00:00:00Z` are retained, including missing-track events in source
activity. Later timestamps are used only to exclude rows, without parsing their
track/artist feedback into analysis. Full raw bytes are read for identity.

Only valid UUID track IDs define target dyads; no title merging, event thinning
or user-artist target substitution is performed. Recorded artist UUID is an
identity proxy, not genre similarity, historically certified catalog metadata,
semantic substitution or causal displacement. Titles/demographics are unused.
No nonempty malformed UUIDs occurred in the selected training rows. Unsupported
missing IDs remain explicitly represented in the prepared event history.

## Generated training-sample outcome

Values below are generated from the exact result artifact, with no fitted score.
The original event file contains 19150868 rows and the profile
file 992 users. Of the fixed 100 sampled IDs, 95
have pre-cutoff events; none of the others was replaced.

| Endpoint | Observed |
|---|---:|
| All retained pre-cutoff event rows | 1762318 |
| Valid-track target event rows | 1581165 (`89.7208%`) |
| Strictly earlier same-track recurrent rows | 1233073 |
| Returns after at least one hour | 1125591 rows / 93 users |
| One-hour returns with intervening source events | 1122295 |
| One-hour returns without intervening source events | 3296 |
| Returns after at least one day | 882269 |
| Recorded valid artist among valid-track rows | `100.0000%` |

The fixed engineering gate requires at least 500 one-hour recurrent valid-track
rows across at least 25 sampled users and at least `80%` valid artist coverage
among valid-track rows. **PASS.** This establishes sampled training support,
not performance, statistical significance, generalization or a new mechanism.
Intervening-event counts include missing-track events, but do not yet measure
same-artist doses or demonstrate a useful competition signal.

## Independent audit and limitations

An independent sequential dictionary implementation replays every prepared row,
queries before whole timestamp-group updates, excludes both boundary groups,
retains multiplicities and reconstructs every recurrence/gap/competitor count.
Coverage is independently recomputed. The separate raw parser verifies all
21025 pre-cutoff records of the first two fixed sampled
users against the prepared arrays. This raw-input reconstruction is an anchor
audit, not a second full raw parser for all sampled users. Source hashes are
unchanged from the prospective preanalysis record.

The source parser, archive pipeline and repository checks passed with `17`
passes and one unrelated optional-corpus skip before this closure; subsequent
wiki/evidence mapping checks are retained separately. No malformed selected
training row caused an execution failure. Known access/interruption outcomes
remain visible in the acquisition records.

`LCR-JOB-LASTFM-INTAKE-001` / `LCR-E-LASTFM-INTAKE-001` map exact artifacts to
`evidence/indexes/lastfm_intake_v1.toml`. No optimizer, predictor/test score,
Colab or GPU allocation occurred. No new scientific claim is admitted; both
publication pointers remain `UNRELEASED`.

## Conditional discriminating comparison after intake

If the fixed support gate passes, the next study can test **recorded same-artist
activity**, using exact artist IDs in intervening listening events rather than
behavioral hashing. A target reference must come from its latest strictly prior
own-track event; a future row or whole-corpus artist map cannot supply that
reference. Missing/changed references need explicit support flags. The intake's
integer UUID maps are storage encodings, not permission to use future metadata
as an online feature.

Compare a phase-specific dose branch with a flexible same-input dose branch,
a proportional clock, activity-only input and a prespecified alignment control.
Keep physical time, total activity, own history and availability channels matched
in the primary comparison. Preserve complete source event histories and tied
query-before-update rules. Candidate eligibility, chronological validation,
budgets, selection and the primary contrast still require their own prospective
protocol before any fit. Intake success alone does not authorize that execution.

An exact same-artist kernel removes the specific behavioral-hash ambiguity; it
does not prove substitution, negative influence, phase semantics or novelty.
Positive evidence would have to survive the same-information flexible comparator
and input controls, with all declared seeds and negative outcomes retained.
The already completed Wikipedia negative comparison remains part of the paper.

The subsequent [recorded-artist history audit](../results/LastFM-Recorded-Artist-Audit.md) now closes strictly prior input availability and within-user total-count nonredundancy on these same training histories. Its direct feature replay is scoped to fixed anchors, and no model utility is established.

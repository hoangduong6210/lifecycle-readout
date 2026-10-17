---
title: Last.fm History Strata
status: completed posthoc descriptive strata; original primary unchanged
last_updated: 2026-10-16
paper_source: false
---

# Is a benefit localized to a particular history?

`LCR-P-LASTFM-STRATA-001` fixes this posthoc question after the complete
[parent aggregate outcome](LastFM-Recorded-Artist-Pilot.md) was known and before
these cohort scores were read. The primary aggregate result remains negative on
average. No threshold, seed, checkpoint or candidate was changed by this analysis.

## Fixed populations and complete primary-contrast table

Cohorts use the positive query's strictly prior gap and same-artist count.
Every selected query retains BOTH its positive and its existing matched negative;
this is a retrospective positive-query population, not a candidate-symmetric
filter or deployable grouping rule. Gap bins are one hour to one day, one day to
seven days, and at least seven days. Dose bins are zero, one through nine, and at
least ten. All marginals and all intersections are retained, including losses.
Counts refer to queries and distinct users, not independent statistical units.

| Positive-query cohort | Queries | Users | Mean phase − MLP AP | Sample SD | Positive seeds |
|---|---:|---:|---|---|---|
| `all` | `40600` | `68` | `-0.000078712` | `0.000531768` | `2/3` |
| `gap_1h_to_1d` | `6166` | `55` | `-0.001181757` | `0.001060590` | `0/3` |
| `gap_1d_to_7d` | `10838` | `60` | `+0.000657894` | `0.001360263` | `2/3` |
| `gap_ge_7d` | `23596` | `68` | `-0.000311534` | `0.000230199` | `0/3` |
| `dose_zero` | `7337` | `66` | `-0.000661984` | `0.000986166` | `1/3` |
| `dose_1_to_9` | `13336` | `67` | `+0.000105008` | `0.000738354` | `2/3` |
| `dose_ge_10` | `19927` | `64` | `-0.000039506` | `0.000361247` | `2/3` |
| `gap_1h_to_1d__dose_zero` | `1645` | `44` | `+0.000469079` | `0.002170255` | `2/3` |
| `gap_1h_to_1d__dose_1_to_9` | `2224` | `51` | `-0.001200222` | `0.002952717` | `1/3` |
| `gap_1h_to_1d__dose_ge_10` | `2297` | `42` | `-0.002506625` | `0.001065243` | `0/3` |
| `gap_1d_to_7d__dose_zero` | `2491` | `55` | `-0.001421358` | `0.002400990` | `1/3` |
| `gap_1d_to_7d__dose_1_to_9` | `4210` | `58` | `+0.001030066` | `0.001278125` | `3/3` |
| `gap_1d_to_7d__dose_ge_10` | `4137` | `45` | `+0.001231336` | `0.001128790` | `2/3` |
| `gap_ge_7d__dose_zero` | `3201` | `65` | `-0.000721387` | `0.001118020` | `1/3` |
| `gap_ge_7d__dose_1_to_9` | `6902` | `67` | `-0.000147345` | `0.000188039` | `0/3` |
| `gap_ge_7d__dose_ge_10` | `13493` | `64` | `-0.000285077` | `0.000306657` | `1/3` |

The one-to-seven-day, one-to-nine-event intersection has a positive phase-versus-
MLP AP difference at each seed. It contains 4210 queries from 58 users. Its
observed mean advantage is `0.001030066 AP`, about `0.103007 pp`; the seed values
are `+0.002501859`, `+0.000388895` and `+0.000199445`. This is an exploratory
localization candidate, not confirmation of a reliable operating region.
The long-gap marginal is negative at every seed, and the short-gap/high-dose
intersection also loses at every seed. Thus neither long silence nor high dose
alone supports a favorable headline. All fixed secondary contrasts, arm metrics
and individual seed values live in the complete result artifact.

## Verification and use boundary

`LCR-E-LASTFM-STRATA-001` / `LCR-JOB-LASTFM-STRATA-001` map
`results/audit/lastfm_strata_v1/`. Prediction identities/checksums and parent
checkpoint audit are verified. A separate implementation reconstructs every
threshold partition and positive-user identity, and independently computes AP
from threshold-group precisions and AUC from tie-aware ranks for all arms and
cohorts. Partition sums, counts and contrasts pass. A setup path/import failure
before analysis was repaired and retained; no outcome was produced by that attempt.

These overlapping cohorts were inspected after the aggregate outcome; validation
also selected checkpoints. No multiplicity-adjusted or population inference,
unseen test, semantic substitution, causal effect, general superiority or
scientific admission follows. A future confirmation would require a prospectively
fixed evaluation. There is no automatic model retry or remote allocation.
Retrieval: `evidence/indexes/lastfm_diagnostics_v1.toml`.


The subsequent [fixed February holdout](LastFM-February-Holdout.md)
does not pass its local confirmation screen. The January localization remains
exploratory; it is not a confirmed operating-region contribution.

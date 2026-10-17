---
title: Last.fm Recorded Artist History Audit
status: complete scoped training-input audit; exact artist dose varies beyond total activity
last_updated: 2026-10-16
paper_source: false
---

# Exact recorded-artist dose before a model claim

`LCR-P-LASTFM-ARTIST-001` fixes a training-input audit after the
[original intake](../datasets/LastFM-Intake-Preflight.md). No user resampling,
new window, fit or predictive endpoint is introduced. The previous Wikipedia
negative outcomes and KuaiRand support failure remain unchanged.

## Strictly prior reference and event partition

For target user-track queries, freeze artist identity from the latest strictly
earlier own-track timestamp group. It is available only when every row at that
group has the same valid artist ID. Missing or conflicting groups remain flagged;
current query artist cannot supply the reference. UUID integer maps are storage
encodings only. The extractor queries every tied row before updating the group.

Between old-own and current timestamps, excluding both full boundary groups,
count total source events, same-reference artist events, known-other artist events
and unknown-artist events. All source events remain, including missing-track rows.
For available references these doses partition the total exactly. Same/other
zeros under an unavailable reference are masked values with a status flag,
not observed absence of competition. Each valid new own group resets its
reference; persistent artist consistency across all own groups is not certified.

Reusable implementation is `src/lifecycle_readout/lastfm_artist_history.py`;
entry point `scripts/audit_lastfm_artist_history.py`. Prefix sums/counters and
artist identity matching are ordinary constructions, without a novelty claim.

## Prospective checks

Synthetic every-row direct-history comparison verifies ties, multiplicities,
unsupported targets, missing/conflicting artists, arbitrary row order, future
append invariance and independence from current query artist. The pipeline
checks parent-byte guards and input rejection before feature creation.
Real extraction reuses the exact original prepared training arrays. All recurrence
and gap denominators must agree with the completed parent intake.

Direct prepared-history anchors are selected by a fixed hash ranking, with up
to 16 rows per available-one-hour, first-own and unavailable-reference stratum.
This is not every-real-row independent replay; raw-TSV lineage inherits the
intake's fixed two-user anchor audit. Source/input hashes precede real doses.

## Descriptive question and decision boundary

Report reference coverage, same/other/unknown versus total doses, zero/mixed
cases and log-count correlation on all recurrent and one-hour rows. Report
within-source fixed-total groups whose same-artist count varies, with a fixed
first lexicographic witness. This checks nonredundancy with total activity in
that scope, not conditional information beyond every other feature or utility.
No favorable correlation, threshold or subgroup becomes a replacement model
primary. Availability and varying exact-ID inputs are insufficient evidence of
substitution, lifecycle phases, negative causal influence or model benefit.

Execution in `results/audit/lastfm_artist_history_execution_v1/` is complete. Source/input guards pass; no optimizer, model/test score, Colab or GPU allocation occurred. Both release pointers stay `UNRELEASED`.

## Complete declared training-input result

The byte-pinned parent history has 1762318 rows. All 1233073
recurrent valid-track rows have a usable strictly prior artist reference. No
missing/conflicting reference occurs in this cohort; the corresponding real
anchor stratum is empty, with no replacement or claim of real-case coverage.
Source counters still include missing-track and unknown-artist source events.
Every declared parent recurrence, gap and intervening-activity denominator agrees.

| One-hour recurring user-track rows | Observed |
|---|---:|
| Total / users | 1125591 / 93 |
| Available prior reference | 1125591 |
| Positive same-artist dose | 938392 |
| Activity present, same-artist dose zero | 183903 |
| Both same and other known artists intervene | 886039 |
| All intervening activity has the reference artist | 50748 |
| Median total / same-artist dose | 418 / 9 |
| Log-count Pearson correlation | `0.299728` |
| Within-user fixed-total groups with varying same dose | 108146 |
| Rows in those varying groups | 943007 |

These are counts of query rows and overlapping history intervals, not counts of
distinct competitor events or independent experimental units. In the fixed
first witness, user index 0 and total dose 1 admit
same-artist doses 0 and 1; exact prepared row
locators are retained in the closure witness artifact. Thus same-artist count
is not a deterministic function of user and total count alone in this cohort.
This does not establish information beyond all other predictors or a performance
effect; physical time and other histories are not conditioned away by that check.

The extractor's full output partition and reported summaries are independently
reconstructed, including within-source fixed-total variability. Direct independent
prepared-history scans pass at all 32 selected real anchors:
16 supported one-hour and 16 first-own rows; unavailable-reference scope is empty.
Every synthetic row, missing/conflicting cases, future/current-artist invariants
and parent-byte rejection pass in six pre-extraction checks. This is not an
independent raw-history dose replay of every real row. Original raw lineage
inherits the previous intake's fixed two-user audit.

## Consequence for the next model comparison

This closes the input-availability and total-count nonredundancy question within
the declared training cohort. It repairs the specific behavioral-hash ambiguity
for this input definition, without repairing the prior Wikipedia results or
proving artist substitution. A separate prospective model comparison must now
match time, activity, own-history and availability inputs against a flexible dose
control, retain proportional-clock and alignment/activity controls, and preserve
all seeds and outcomes. No phase benefit or original mechanism is inferred.

`LCR-E-LASTFM-ARTIST-001` / `LCR-JOB-LASTFM-ARTIST-001` map source, fixed protocol,
parent identities, features, anchors, report and closure snapshots to
`evidence/indexes/lastfm_artist_history_v1.toml`. No scientific claim is admitted.

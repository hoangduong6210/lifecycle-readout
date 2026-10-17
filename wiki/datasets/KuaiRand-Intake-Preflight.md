---
title: KuaiRand Intake Preflight
status: local intake and independent replay complete; fixed recurrence gate failed
last_updated: 2026-10-16
paper_source: false
---

# Fixed bounded intake before a new model

The user's alternative-dataset permission led to a bounded, training-only intake.
`LCR-P-KUAIRAND-INTAKE-001` fixed the sample, endpoints and planning gate before
real recurrence outcomes. `LCR-P-KUAIRAND-LOCAL-001` subsequently amended only
acquisition/execution to use an existing local archive, before those outcomes.
No model fit, predictor score, test score or manuscript export occurred.

## Source identity and preserved access failure

The [official repository](https://github.com/chongminggao/KuaiRand) is pinned to
commit `f8dbf6678b3c9594050e3e813aeff0c942260ec4`. Its README and repository
license were retrieved; repository licensing is distinct from the data record's
rights statement, which was not retrieved. Local verified-TLS Zenodo requests
and the separately allocated CPU-only Colab request timed out. The author-linked
mirror showed a shutdown notice and listing; the bounded download-event wait
expired. Original transfer provenance for the later-discovered Downloads archive
was not conclusively observed and is not attributed to that click.

The existing archive matched the author's published MD5 and the amendment's
recorded SHA256. Its original bytes were read-only and full SHA256 was rechecked
after analysis. Archive, selected regular-file members, source bundle and outputs
have retained identities. Only the earlier standard-log training file, user table
and basic video metadata were parsed; later standard/random logs and monthly
statistics were not queried. This is neither full-corpus nor online feedback
availability validation.

The failed CPU stage remains failed despite CLI/driver exit success: its remote
completion marker and failure report establish a network timeout, with no data
result. The owned CPU runtime was stopped and backend absence verified. Local
processing required no new Colab or GPU. Execution records remain separate:
`LCR-JOB-KUAIRAND-CPU-001` and `LCR-JOB-KUAIRAND-LOCAL-001`.

## Fixed sample and outcomes

The hash-ranked sample fixes 100 user IDs, seed `17`, retaining all their events
in `log_standard_4_08_to_4_21_1k.csv`. Of these, 98 have rows in that window;
no inactive sampled user was replaced. Exact outcomes are generated from
`results/audit/kuairand_local_intake_execution_v1/artifacts/results.json`.

| Training-sample endpoint | Observed |
|---|---:|
| Selected impression rows | 472957 |
| Positive-click rows | 184930 |
| Strictly earlier same-video impression recurrence | 8009 (`1.6934%`) |
| Same-video positive-click recurrence | 811 (`0.4385%`) |
| Positive-click returns after at least one hour | 4 rows / 4 users |
| Impression returns after at least one hour | 4072 rows / 77 users |
| Click-weighted valid numeric tag coverage | `98.0533%` |

The prespecified planning gate required at least 500 one-hour recurrent clicked
rows across at least 25 users, and at least `80%` click-weighted valid-tag
coverage. **FAIL:** metadata coverage passes, clicked recurrence does not.
All selected videos joined basic metadata; no unfamiliar tag syntax was found.
Numeric tag availability is not validated semantic substitutability.

Counts use strictly earlier timestamps. All tied rows query before updating the
group, and multiplicity is preserved. Impression and clicked recurrence use their
respective event streams; intervening competitors in the clicked summary are
other clicked rows, not all impressions. Events at the previous-own and current
timestamp boundaries are excluded. Independent sequential dictionary replay
checks every selected row, both recurrence summaries and gap bins; independent
CSV-reader coverage and full archive SHA256 also agree.

## Decision and scientific boundary

Do not train the proposed long-gap user-video lifecycle comparison on this
sample or lower its threshold after inspection. Do not silently switch edges to
user-author/category or use impression recurrence as clicked-edge evidence.
This fixed-sample failure is not proof of infeasibility for every KuaiRand user,
window or target, nor a predictive outcome. It avoids an inadequately supported
fit. The alternative-dataset proposal records the next conditional candidate.

`LCR-E-KUAIRAND-INTAKE-001` maps this development record to the retrieval index
`evidence/indexes/kuairand_intake_v1.toml`. No scientific claim is admitted;
both publication pointers remain `UNRELEASED`.

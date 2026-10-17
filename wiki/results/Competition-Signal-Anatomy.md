---
title: Competition Signal Anatomy Results
status: completed posthoc full-validation input anatomy; no model fit or admitted claim
last_updated: 2026-10-16
paper_source: false
---

# Full validation exposure anatomy

The [decomposition](../methods/Competition-Signal-Anatomy.md) separates the
unchanged frozen-reference feature into exact source-ID overlap and hash
collisions. The first anchor diagnosis and subsequent full-validation expansion
are explicitly posthoc after model outcomes, not independent confirmation.

| Seed | Validation queries | Own-source overlap | Same-other-source overlap | Cross-source hash collisions |
|---|---|---|---|---|
| `101` | `884` | `44.59%` | `6.95%` | `48.46%` |
| `202` | `884` | `43.71%` | `7.07%` | `49.21%` |
| `303` | `884` | `43.86%` | `7.22%` | `48.92%` |

Fractions are sums of component exposure divided by summed original exposure
across both candidate slots on supported Wikipedia validation queries. They
are exposure-weighted, not the fraction of events, users, AP or semantic content.
The same positive queries recur across seeds, so seed rows are not independent
populations. Original hashed normalizers remain in every term.

The feature's exact same-other-source contribution is small in this scoped
population. Most exposure comes from the querying source's own overlap and
cross-ID hash collisions. Thus the pilot's added-feature improvement cannot be
reported as demonstrated semantic competition. This decomposition does not
identify which component caused any AP change, show that collisions are useless,
or establish the benefit of a hypothetical exact-ID replacement.

## Coverage and verification

The first stage decomposes all supported training/validation inputs into four
own/other hashed terms and a permutation-invariant background. Direct exact-ID
raw-history checks cover the declared evenly spaced anchors. The separate
full-validation expansion computes all three exact-ID components for every
validation slot. It matches original hashed exposure and the earlier own-source
component on every slot, and independently reconstructed exact-ID components at
validation anchors. All synthetic validation slots are checked by direct counts.
This is not every-real-row independent query-wise raw-history replay.

Every declared seed completes without a failed or omitted analysis. Source/input
pins and original candidates remain unchanged. Full arrays retain tiny negative
roundoff before tolerance-bounded clipping. All structural test anatomy arrays
are zero. No optimizer, fitted predictor evaluation, new test score, hash-width
selection or GPU allocation occurs.

## Artifact owners and next action

`LCR-E-COMPETE-ATTR-001` and `LCR-E-COMPETE-ATTR-FULL-001` map the separate
stages and normalized jobs. Execution lives in
`results/audit/competition_attribution_execution_v1/` and
`results/audit/competition_attribution_full_execution_v1/`; closure sources,
complete precision and generated tables are in
`results/audit/competition_attribution_closure_v1/`. The retrieval index is
`evidence/indexes/competition_attribution_v1.toml`.

The result motivates checking independently supplied destination content or
categories on a suitable dataset, while retaining Wikipedia as a negative
mechanism case. The user's authorization to propose another dataset is recorded
in the [dataset proposal](../datasets/Competition-Dataset-Proposal.md).
It does not retroactively validate the current proxy or erase earlier primaries.

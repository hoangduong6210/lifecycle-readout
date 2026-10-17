---
title: Competitive Event History Audit
status: completed exploratory real-history input audit; no trained prediction result
last_updated: 2026-10-16
paper_source: false
---

# Completed event-dose input audit

`LCR-P-COMPETE-HISTORY-001` fixes the [interval-frozen reference](../methods/Frozen-Reference-Competition-History.md)
before its real-data extraction. Every declared seed completes on the same
Wikipedia sampled candidates as the prior residual pilot. New history inputs
and the operator's anchor state calculations pass their scoped checks.
No fitting, predictor evaluation, test exposure or GPU allocation occurs.

## Descriptive validation coverage

| Seed | Validation queries | Reference available | Nonzero exposure | Mean absolute difference from old exposure |
|---|---|---|---|---|
| `101` | `884` | `99.49%` | `78.79%` | `0.973323` |
| `202` | `884` | `99.43%` | `78.73%` | `1.095551` |
| `303` | `884` | `99.55%` | `78.90%` | `1.010306` |

Availability/nonzero fractions count both candidate slots per query.
These are descriptive fractions on the supported conditional task. The same
positive queries recur across seeds; they are not independent populations.
The last column uses native exposure units, not AP or an effect size.
Full training/validation endpoints, including exact floating-point exposure ties,
are retained in the execution result. Exact float ties are not an effective
information or ranking-resolution criterion.

## Audit scope

The source guard, prepared-data byte identity and all declared parent candidate
identity/history arrays match. All supported training and validation feature
rows are extracted; test feature arrays remain zero. A separately declared set
of evenly spaced training/validation anchors is reconstructed directly from raw
history, using event-time profiles and no production prefix/reset logic.
The same anchors check the exact aggregated state against independent sequential
NumPy decay/jump updates, with a fixed initial state and rates. This is not a
trained model, a full-predictor replay or independent raw-history replay of every
real feature row. Floating differences and all denominators are in the immutable
execution report; generated tables come from that report.

The previous residual pilot's negative primary remains unchanged. Its exposure
contract differs from this development. A high availability fraction does not
show additional predictive information beyond activity or matched pair history.
No method superiority, semantics, edge-death label, significance or priority
claim follows. The next discriminating study must define the own-edge state
reset, then compare nonproportional phase rates with a proportional exposure
clock and flexible models receiving the same physical time and event exposure.

## Reproducibility owner

`LCR-E-COMPETE-HISTORY-001` and `LCR-JOB-COMPETE-HISTORY-001` map source,
protocol, outputs and checks. Execution is in
`results/audit/competitive_event_history_execution_v1/`; closure is in
`results/audit/competitive_event_history_closure_v1/`; the retrieval index is
`evidence/indexes/competitive_event_history_v1.toml`.
The operation is bounded local CPU feature extraction and audit. It creates no
owned remote runtime and consumes no allocated Colab GPU session. Existing
publication pointers remain unreleased and the wiki remains the living record.

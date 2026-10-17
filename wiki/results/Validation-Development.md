---
title: research Validation Development Results
status: partial development study with transport failure reconciled
last_updated: 2026-10-15
paper_source: false
---

# research Validation Development Results

`LCR-P-Q1-DEV-001` completed Wikipedia across all declared seeds and models.
Reddit was not allocated after the first development comparison failed to
establish mixture utility beyond the scalar residual. The two-dataset study is
therefore incomplete; this is a disclosed development stop, not a full-protocol
success or confirmatory result. Existing test outcomes were not rescored.

## Validation outcomes

AP mean and sample standard deviation over seeds `101`, `202`, `303`. Each
checkpoint and architecture comparison uses validation; these are optimistic
development estimates. Brier is for the same balanced sampled candidates.

| Model | Validation AP | Validation Brier mean |
|---|---:|---:|
| Base plus history | `0.601335 ± 0.007182` | `0.245217` |
| Multiplicative scalar | `0.536869 ± 0.006135` | `0.391831` |
| Legacy ordered gate | `0.544453 ± 0.006560` | `0.316758` |
| Scalar logit residual | `0.601671 ± 0.008279` | `0.245183` |
| Contextual mixture residual | `0.601669 ± 0.008278` | `0.245183` |
| Unconstrained MLP logit residual | `0.605625 ± 0.008375` | `0.242535` |

The mixture and scalar residual behave almost identically; the neutral-capable
architecture prevents a forced non-neutral starting gate but has not established
a useful temporal component. Selected amplitudes are zero for both residuals at
seed `101`, and very small at the other seeds, as retained in the
[parameter inspection](../../results/audit/q1_development_v1/residual_parameter_inspection.json).
This does not prove all mixture models must fail. Changing architecture and
clock together also prevents causal attribution to either change alone.

## Traceable execution and recovery

- [Normalized execution](../../evidence/jobs/LCR-JOB-Q1-DEV-WIKIPEDIA-001.toml)
- [Original failed transport job](../../evidence/records/lcr-wikipedia-bdec2912ae--job.json)
- [Explicit reconciliation](../../results/audit/lcr-wikipedia-bdec2912ae/transport_reconciliation.json)
- [Retained results](../../results/audit/lcr-wikipedia-bdec2912ae/artifacts/results.json)
- [Independent audit](../../results/audit/q1_development_v1/wikipedia_development_audit.json)
- [Resource inventory](../../results/audit/q1_development_v1/resource_after_wikipedia.json)
- [Diagnostic and audit checksums](../../results/audit/q1_development_v1/checksums.sha256)

Remote training and the closed artifact archive were complete while the local
CLI stdout client remained busy. The client was interrupted, then finally
retrieval verified the archive and released the owned runtime. The original
failed job remains unmodified. Technical training completion does not erase
the transport failure.

The auditor independently checks all artifacts, history/support replay, source
archive, declared seeds/models, validation candidate IDs, train-only clock and
normalization, projected amplitudes, selected epochs and recomputed metrics.
The completion-probe workflow now detects a closed-archive marker independently
of CLI stdout, checks its source identity and then verifies the downloaded archive
hash. Local subprocess tests cover success, remote failure and missing marker;
that new recovery path has not yet been exercised on another live allocation.

## Next research step

Keep the [research gates](../questions/Research-Program.md) open. Before further GPU
sweeps, use controlled simulated cases to test whether the residual family learns
known time effects and remains neutral when time is uninformative; compare the
same scalar and flexible controls. Review count/sequential prior work and define
a contribution that survives those comparisons. Any new real-data confirmation
requires licensed fresh datasets and official full-candidate evaluation.

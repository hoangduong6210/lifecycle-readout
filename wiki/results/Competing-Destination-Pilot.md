---
title: Competing Destination Pilot
status: complete exploratory validation pilot with negative primary contrast; not admitted
last_updated: 2026-10-16
paper_source: false
---

# Completed initial pilot

The user-selected [competition direction](../methods/Competing-Destination-Development.md)
has an implemented causal extractor, equal-capacity scoring controls and a
completed Wikipedia validation-only pilot under `LCR-P-COMPETE-DEV-001`.
All six arms and all three declared seeds completed. The owned Colab runtime
was released and independently observed backend inventory is empty.
Existing hard-control and cadence results remain unchanged.

The primary inhibitory mechanism loses to the flexible profile MLP on every
seed. The profile-information contrast has a small positive mean with mixed
signs, while the correctly aligned profile MLP does not beat the coordinate-shuffled
control on average. These findings do not support a profile-semantic displacement
interpretation or choosing the constrained mechanism as a superior predictor.
No tuning, replacement primary contrast or additional dataset allocation follows
from these outcomes. This pilot does not reject all possible competition models.

## Audited endpoint tables

Cells report seed means and sample standard deviations, not confidence intervals.
They were generated from retained audited predictions/reports by the closure
script; full precision lives in `results/audit/competition_closure_v1/summary.json`.

| Arm | Validation AP | AUC | Sampled Brier | Paired accuracy |
|---|---|---|---|---|
| `masked_mlp` | `0.601824 ± 0.006398` | `0.594866 ± 0.002235` | `0.244440 ± 0.000404` | `0.621041 ± 0.005986` |
| `activity_mlp` | `0.602405 ± 0.007935` | `0.595284 ± 0.003981` | `0.243996 ± 0.000907` | `0.622172 ± 0.016664` |
| `profile_mlp` | `0.603959 ± 0.008366` | `0.594338 ± 0.002651` | `0.246229 ± 0.003573` | `0.615385 ± 0.015219` |
| `shuffled_mlp` | `0.605567 ± 0.006809` | `0.596961 ± 0.004002` | `0.243704 ± 0.000836` | `0.622172 ± 0.014836` |
| `activity_displacement` | `0.603002 ± 0.008171` | `0.595416 ± 0.005181` | `0.243675 ± 0.001062` | `0.616516 ± 0.010791` |
| `profile_displacement` | `0.603182 ± 0.007893` | `0.595540 ± 0.005084` | `0.243663 ± 0.001098` | `0.616139 ± 0.010870` |

## Prespecified contrasts

| Prespecified contrast | Mean AP difference | Sample SD | Positive / tied / negative seeds |
|---|---|---|---|
| `primary_mechanism` | `-0.000776` | `0.000473` | `0 / 0 / 3` |
| `profile_information` | `+0.001553` | `0.001652` | `2 / 0 / 1` |
| `profile_alignment` | `-0.001608` | `0.001559` | `1 / 0 / 2` |
| `activity_information` | `+0.000581` | `0.001610` | `1 / 0 / 2` |
| `profile_vs_activity_hazard` | `+0.000180` | `0.000685` | `1 / 0 / 2` |

The primary contrast is `primary_mechanism`; secondary information/coordinate
contrasts are not replacements for it. All arms have the same `185804` trainable
parameters. Every seed has `2805` supported training queries, `884` validation
queries and `977` structural test rows. No test predictions or test competition
features were produced. The sampled Brier and paired accuracy describe the
conditional generated-candidate task, not natural event prevalence or survival.

## Verification and reproducibility boundary

`LCR-E-COMPETE-DEV-001` and `LCR-JOB-COMPETE-DEV-001` map this development outcome.
The execution plan pins source, configuration, protocol and preallocation tests.
Artifact checksums and normalized job hashes are verified. The parent cadence
candidate identities, exclusions, splits, elapsed times and original six input
channels are exactly reproduced across all seeds. Those parent models are not
additional comparable arms of this new architecture.

All declared raw-history feature anchors pass independent direct reconstruction.
That checks selected queries, not every raw feature value. All retained validation
predictions from all `18` checkpoints pass CPU inference replay against the
exact matching launch source; the largest probability discrepancy is
`2.682209014892578e-7`, within predeclared tolerances. Metrics, training-only
normalization, common candidates and earliest-maximum epoch selection are checked.
This is not optimizer replay or fresh-dependency environment reproduction.

The guarded synthetic/operations/repository suite passed before allocation;
post-closure structural verification is recorded separately. No model fit failed
or was omitted. The separate literature cache's direct PDF-download failures are
retained; primary documents were inspected through browser text extraction.

## Resources and access

The owned attempt is `results/audit/lcr-wikipedia-f83b3eafa4/`; execution, primary
source hashes, inference audit, parent comparison and resource observations are
in `results/audit/competition_execution_v1/`. Generated tables and closure records
are in `results/audit/competition_closure_v1/`. The tracked retrieval manifest is
`evidence/indexes/competition_development_v1.toml`.

The post-run CLI observation reports balance `173.34 CU`, usage `0.00/hr` and
no active backend assignment. The displayed balance is an observation, not a
proof of zero billed compute: billing latency/rounding was not audited.

## Interpretation boundary and disposition

The kernel uses online behavioral co-interaction profiles with a small hash
budget and includes the querying source's own past contributions. It is not
verified semantic substitution. The shuffle perturbs coordinate alignment rather
than all behavioral information; its stronger mean does not prove semantic
irrelevance in general. Data are previously inspected, support is recurrent-only,
selection uses validation, and the epoch budget is fixed. No significance,
equivalence, universal inferiority, causal death labels or fresh confirmation
is inferred from three seeds.

The [direct competition precedents](../questions/Contribution-Gap-Review.md#direct-competition-precedents-found-during-the-pilot)
also preclude claiming generic cross-edge attention displacement as a new principle.
Retain the implementation and complete negative pilot as a development boundary;
this version is not promoted to the article's main new predictive contribution.
A stronger semantic mechanism would first require an independently justified
similarity signal and a separately fixed design, not more fits of this protocol
until a positive table appears. No Word export or scientific admission is made.

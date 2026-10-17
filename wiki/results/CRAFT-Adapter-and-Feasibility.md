---
title: CRAFT Adapter and Feasibility
status: audited bounded validation fit; convergence unestablished
last_updated: 2026-10-16
paper_source: false
---

# CRAFT Adapter and Feasibility

A hash-pinned CRAFT author model can now receive the prepared ML-20M development
stream and the same cached candidates as the count controls. The adapter does
not import the author loader, evaluator or training runner. Source/destination
time indices are immutable; each query truncates strictly before its timestamp.
Past ties retain supplied file order, recent histories are left aligned, zero is
padding, and absolute timestamps remain float64 until interval subtraction.
Original author files remain unchanged; two import statements are relocated into
a private module namespace at load time. MIT license bytes are retained.

## Completed full-support validation fit

`LCR-P-CRAFT-FIT-001`, attempt `lcr-ML-20M-f2af45c59b`, completed the declared
single-seed development fit using the pinned author model and proceedings ML-20M
settings. Each complete epoch uses all `14000091` training events. The same cached
`364943` validation queries and `100` negatives per query are reused from the
count screen; supplied masks and strict-past histories remain unchanged.

The fit completed `27` epochs and reached its declared six-hour in-process
budget during epoch `28`, retaining `4006000` partial-epoch rows and checkpoint
separately. The selected completed checkpoint is epoch `26` with validation MRR
`0.417077`. The first budget stop and earliest strict-maximum selection were
checked. Final validation/artifact closure adds time after the training stop;
this is not an extension of training. Convergence is not demonstrated.

The [independent audit](../../results/audit/lcr-ML-20M-f2af45c59b/independent_audit.json)
passes launch-bound source/configuration/protocol identities, all candidate and
reciprocal-rank endpoints, every epoch's rank vector, selected-checkpoint rank
identity, query/parameter counts and the declared stopping rule. It independently
reconstructs histories and scores at `20` raw-history anchors on CPU. Complete
training-negative replay and optimization reproduction were not performed.

The normalized `LCR-JOB-CRAFT-FIT-001`, `LCR-E-CRAFT-FIT-001` and
[retrieval index](../../evidence/indexes/craft_fit_v1.toml) retain exact source,
checkpoint, partial epoch, predictions, audits and post-run resource identities.
The owned endpoint was released. Separate backend inventory confirms no sessions
or assignments and zero consumption rate. No test was scored and no claim is
admitted. This is a single-seed bounded validation comparison, not a paper-score
reproduction, multi-seed uncertainty estimate or evidence of lifecycle utility.
The [common-candidate result owner](ML20M-Development-Baselines.md) records the
comparison with independently implemented counts. The cadence development study
is the next declared experiment; it was not allocated during this fit.

## Completed scope

`LCR-P-CRAFT-PROFILE-002` profiles author initialization and short BPR/Adam updates,
not a completed predictive baseline. Full train/validation history indices build
in about `3.47` seconds on the allocated L4 host. Two warmup plus eight measured
steps were run separately at each training batch size, from fresh initialization.

| Provisional profile batch | Mean end-to-end step seconds | Projected train epoch seconds | Peak allocated CUDA bytes |
|---|---:|---:|---:|
| `200` | `0.015755` | `1102.85` | `68100608` |
| `1024` | `0.038282` | `523.39` | `190154752` |

Epoch durations are short-sample projections, excluding validation, checkpointing
and warmup. They are not measured complete epochs or estimates for the exact
paper configuration. The timing sample is too short for a deployment cost claim.
The partial fits must not be treated as model convergence or efficacy evidence.

The profile uses two layers, recent `20`, embedding `128`, dropout `0.1`. The
[official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/036912a83bdbb1fd792baf6532f102d8-Paper-Conference.pdf),
Table 8, gives ML-20M one layer, recent `60`, batch `400`, embedding `128` and
all three dropout rates `0.2`. The author README's Table 6 pointer does not match
the proceedings table numbering. The cached source and extracted parameter record
are under `results/audit/q1_preflight_v1/baseline_review/craft/`. A predictive
baseline must use a prospectively declared fit configuration and stopping rule;
no paper-score reproduction is asserted here.

## Audit and retained failures

Local checks cover tied/future-event exclusion, missing histories, UNIX timestamp
precision, source checksums, candidate permutations and finite gradients through
the author architecture. A launcher-entrypoint test guards omission of the
optional seed flag. The GPU profile's 20 validation queries use exact copied
rows from the previous candidate cache, without resampling. No test values enter.

The [independent auditor](../../scripts/audit_craft_profile.py) verifies all saved
source histories against raw events, all global candidate update times against
strictly past prefixes, parent candidate identity, source/archive hashes and
timing calculations. Reconstructing the saved model on CPU matches GPU logits:
maximum absolute difference below `4e-7`. No MRR was computed.

`LCR-JOB-CRAFT-PROFILE-001` preserves the rejected source upload, and
`LCR-JOB-CRAFT-PROFILE-002` preserves the default-seed entrypoint failure. Both
occurred before model execution. `LCR-JOB-CRAFT-PROFILE-003` completed the scoped
profile and audit. All owned runtimes were released. Separate backend inventory
reports no sessions or assignments and `0.00/hr`; displayed balance `183.40` CU
is unchanged at CLI precision, not evidence of exactly zero billed consumption.

`LCR-E-CRAFT-PROFILE-001` and
[the retrieval index](../../evidence/indexes/craft_profile_v1.toml) locate exact
local evidence. Clean-clone reproduction remains open. The bounded validation fit above now
provides an initial trained comparator; convergence and lifecycle benefit remain
unestablished.


## Historical full-fit audit preparation

Before the fit completed, the following audit checks were prepared; the completed scoped audit is now recorded above. The local fit auditor now binds reported settings to the launch-pinned configuration and protocol, checks earliest strict-maximum selection and the first patience stop, rejects undeclared stopping reasons, and compares selected per-query ranks with the final prediction vector. Query and model parameter counts are also checked. Mutation tests exercise changed settings, altered launch hashes, impossible ranks, tied means with different rank vectors, and continuation after a patience trigger. The review and verification output are retained in `results/audit/craft_fit_auditor_review_v1/`.

The complete launch source ZIP remains unchanged. These checks strengthen artifact consistency; they do not independently replay training negatives or reproduce the complete optimization trajectory. Independent checkpoint score reconstruction remains scoped to the declared raw-history anchors. The [dataset registry](../datasets/Dataset-Registry.md) now explains pre-filter time cutoffs and supplied-mask preservation. The eventual result remains a bounded validation development comparison, with original-to-benchmark byte reconstruction still open.

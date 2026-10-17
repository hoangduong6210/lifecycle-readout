---
title: Reproducibility Contract
status: canonical contract with journal analysis scope
last_updated: 2026-10-16
paper_source: false
---

# Reproducibility Contract

Scientific release requires a retrievable source state, protocol and configuration
identities, data identities and preparation, environment records, complete
outcomes and failures, independently checked endpoints, and claim mappings.
Mutable verification belongs in `results/audit/`; immutable admitted evidence
belongs in `results/frozen/<release-id>/`. `results/CURRENT`, `PROJECT.toml`, the
evidence ledger and paper `results.lock.yaml` must agree before a submission
snapshot. All current pointers remain unreleased.

## Journal analysis scope

The [journal strategy](questions/Research-Paper-Strategy.md) defines the current
article. Each retained table, figure and mathematical statement must have its own
evidence chain. A working manuscript is not itself evidence admission.

| Evidence component | Required reproducibility record |
|---|---|
| Exact silence result and candidate-transform counterexample | Derivation assumptions, native equations, executable checks, figure arrays and deterministic figure construction; distinguish constructed cases from trained-checkpoint behavior. |
| Historical hard-control comparison | Original launch source and manifest, sealed protocol/configuration, supplied/prepared corpus identities, all declared seeds and variants, all support exclusions, full candidate identities, training/validation histories, selected checkpoints, predictions and independent endpoint replay. |
| Earlier exploratory benchmark diagnostics | Preserve their own sources, raw-second clock, candidate generation, information access, exclusions, failed gate verdicts and provenance. Do not merge their populations with the prospective hard controls. |
| ML-20M development comparisons | Supplied masks, train/validation-only prepared input, shared generated candidates and tie rule; count/control source and full-fit stopping/selection records. Preserve validation-only and budget-limited scope. |
| Cost and execution | Observed runtime versions and dependency exports, hardware, elapsed timing, resource observations, transfer failures/recoveries, artifact hashes and verified owned-runtime release. |

The current source checkout is dirty. Archived launch bundles identify the bytes
actually executed; they do not imply that a clean commit contains every required
file. A release needs a reviewed, reproducible source snapshot and retrieval
instructions for any artifacts omitted from the source repository. Dataset
redistribution is a separate rights decision; identity checks do not grant rights.

## Verification levels must remain distinct

- Metric reconstruction checks retained predictions against reported values.
- Candidate and history replay checks timing, support and information access.
- Checkpoint inference replay checks whether archived models and reconstructed
  inputs reproduce retained scores within declared numerical tolerances.
- Training reproduction reruns optimization with pinned dependencies, seeds,
  candidates and selection. Inference replay does not establish this level.
- A fresh environment or clean-clone check must actually use that environment or
  clone; extraction of archived source with existing host dependencies is weaker.
- Scientific admission additionally requires the publication gate and an explicit
  frozen evidence/claim decision. Passing any technical check is insufficient.

Record observed package versions separately from an installation lock. Existing
runtime dependency exports do not alone pin wheel hashes, the complete host
runtime or GPU kernels. Cross-device comparisons must state tolerances and scope;
a numerical match does not prove identical training trajectories.

## Legacy CoEdit scope

Earlier lifecycle-readout release requirements additionally include CoEdit
identity, faithfulness-dump schema and intervention outputs for claims using that
study. Those requirements remain attached to those claims. They are not evidence
for the current temporal-gating analysis, whose datasets and scored paths are
owned by its study-specific records. CoEdit redistribution remains blocked.

## Current closure boundary

The [hard-control retrieval guide](evidence/Hard-Control-Retrieval.md) locates
completed execution and metric/history audits. Local archived-checkpoint inference replay now passes the complete retained
hard-control test cohorts, with independent history inputs and recorded numerical
discrepancies in `results/audit/hard_checkpoint_replay_v1/`. This uses existing
host dependencies rather than a fresh installation. The [baseline owner](results/CRAFT-Adapter-and-Feasibility.md)
records the completed bounded CRAFT fit, scoped independent audit and runtime
release. The declared pair-cadence study is also complete: arm/cross-arm audits
and full retained validation checkpoint inference pass, with independent history,
cadence and normalization reconstruction. Its result owner and retrieval index
retain exact scope. No fit remains active and final owned-runtime cleanup was
independently verified.
The source-session diagnostic independently replays its complete action-gap
clock and retained rank-vector aggregation; it does not add a trained model.
Full optimization replay,
fresh-environment reproduction, code ownership, artifact distribution and
scientific release admission remain open. The
[publication gate](../publication/PUBLICATION_GATE.toml) remains blocked.

## Restoring pinned CRAFT author assets

`LCR-E-SOURCE-PORTABILITY-001` and `LCR-JOB-SOURCE-RECOVERY-001` map the
recovery and isolated-copy checks below. They are technical records only.

The [asset restorer](../scripts/fetch_craft_assets.py) and the small provenance
files under `resources/` recover the author inputs referenced by the sealed
CRAFT fit configuration. From a source tree containing the current reviewed
files, using Python `3.11` or later:

```bash
python3 scripts/fetch_craft_assets.py \
  --output results/audit/q1_preflight_v1/baseline_review/craft \
  --report results/audit/craft_asset_recovery.json
```

Use a fresh report path. An optional `--ca-file` supplies a trusted CA bundle;
TLS verification remains enabled. The restorer validates metadata against the
existing configuration hashes, requires commit-pinned upstream URLs, preserves
the original license and refuses to overwrite mismatched local assets. It
downloads source bytes without executing them. The original metadata flags
describe the earlier pre-fit review; the completed fit is owned by the CRAFT
result page, not those historical flags.

A recovery into an empty directory and a second no-download recheck pass in
`results/audit/craft_asset_recovery_v1/`. A separate source-copy smoke outside
the main checkout imports the copied adapter and recovered author code and
checks a small synthetic forward pass and candidate permutation. Its report
is in `results/audit/source_candidate_review_v1/`. This checks source-path
portability with current host dependencies; it is not a fresh dependency
installation, a clone of the old committed HEAD, optimization reproduction or
a new real-data benchmark.

The new files are not yet part of a reviewed source commit. Launch bundles also
contain data inputs, so they must not be substituted for a source-only public
release. Source ownership, distribution, admission and final author declarations
remain separate from these successful byte-recovery and portability checks.

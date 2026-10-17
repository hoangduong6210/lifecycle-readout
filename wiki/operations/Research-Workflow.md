---
title: Research Workflow
status: canonical operations
last_updated: 2026-10-15
paper_source: false
---

# Research Workflow

Research state is owned by [Project Status](../status/Project-Status.md). The
completed prospective control study is [LCR-P-HARD-001](../../protocols/hard_controls_v1.toml).
The [retrieval guide](../evidence/Hard-Control-Retrieval.md) links its exact evidence;
[LCR-P-TKDE-002](../../protocols/lifecycle_tkde_v2.toml) remains an earlier audit study.
Earlier audit artifacts are preserved and are not promoted by a successful run.

## Colab execution

The official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
is pinned in [compute settings](../../configs/tkde_causal_v2.toml). Heavy training
and intervention work belongs on Colab. Local work is limited to structural,
unit, and small synthetic pipeline checks.

Authenticate once by running `colab sessions` and completing the Google browser
flow. Never put credentials or authorization codes into repository files.

From the repository root, use:

```bash
python3 scripts/run_colab_benchmark.py --dataset wikipedia --seeds 42 --smoke
python3 scripts/run_colab_benchmark.py --dataset wikipedia
```

[The workflow](../../src/lifecycle_readout/colab_workflow.py) inventories backend
assignments, creates a uniquely named owned session, uploads a source snapshot,
executes the thin benchmark runner, retrieves artifacts, and stops that session
in a finally path. It verifies that the owned endpoint is absent from backend
inventory. Allocation, upload, execution, retrieval, and cleanup failures remain
in the attempt's `job.json` and `execution.txt`.

Do not leave an allocated GPU idle during code changes. Each dataset attempt is
bounded by configured execution and wall timeouts. Never stop unrelated sessions.
If release verification fails, the attempt is marked `cleanup_required`; resolve
that owned runtime before allocating another. Do not interpret a local missing
session record as proof of server-side resource release.

## Evidence and admission

Outputs are stored under a fresh `results/audit/lcr-<dataset>-<attempt>/` directory.
They include source/config/protocol hashes, raw data identity, environment,
checkpoints, per-event predictions, split/negative identities, exclusions,
validation history, phase profiles, and model intervention trajectories.
No failed or unfavorable result is removed or overwritten.

A numerical gate pass is not publication admission. Complete dataset/seed/model
coverage, matched comparator evaluation, licenses, scientific interpretation,
and reproducibility must be reviewed before freezing a release. Update the
claim and evidence registries before exporting an immutable paper snapshot.

## Feature-matched hard controls

The authorized prospective protocol `LCR-P-HARD-001` has its own
[configuration](../../configs/hard_controls_v1.toml),
[training entry point](../../scripts/train_hard_controls.py) and
[Colab launcher](../../scripts/run_hard_controls_colab.py).
Run each dataset in a fresh bounded attempt:

```bash
python3 scripts/run_hard_controls_colab.py --dataset wikipedia
python3 scripts/run_hard_controls_colab.py --dataset reddit
```

The launcher bundles the selected entry script, configuration and protocol,
prepares data before allocation, and uses the same owned-session finally cleanup.
After retrieval, run
[scripts/audit_hard_controls.py](../../scripts/audit_hard_controls.py) with the
attempt directory and an output JSON. It independently replays every event's
matching support, checks no eligible event was omitted, and recomputes metrics
from retained predictions. Partial observations never justify allocating a
second copy of a live run. Confirm the existing process and backend first.


## Validation development and client completion recovery

The separate `LCR-P-Q1-DEV-001` configuration is selected explicitly:

```bash
python3 scripts/run_hard_controls_colab.py --config configs/q1_development_v1.toml --dataset wikipedia
```

It emits validation predictions only. Inspect the
[development disposition](../results/Validation-Development.md) before another
allocation; the incomplete planned coverage is not a full-protocol success.
The bootstrap now writes a source-identified completion marker only after closing
the artifact archive. The launcher observes it independently of stdout, terminates
a stalled client after verified remote completion, then checks the archive hash,
retrieves and finally releases its owned runtime. Marker absence never implies
completion. Historical recovered failures retain their original job plus a
separate hash-pinned reconciliation; do not silently change a failed exit code.


## Small synthetic response diagnostic

The bounded one-thread oracle check is local; it does not train a graph model:

```bash
python3 scripts/run_synthetic_response.py --output <fresh-attempt>
python3 scripts/audit_synthetic_response.py <attempt> --output <fresh-audit.json>
```

The launcher archives exact source/configuration/protocol bytes. Dataset counts
and step budgets beyond the declared small CPU guard require Colab. Review
[its completed outcomes](../results/Synthetic-Mechanisms.md) before considering
another configuration; preserve the original outcomes when amending a design.
Large official datasets remain outside Git, under the conditions recorded by the
[dataset registry](../datasets/Dataset-Registry.md).

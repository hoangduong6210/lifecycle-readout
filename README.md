# Faithful Lifecycle Readout for Temporal Graphs

> **SCIENTIFIC SCOPE BOUNDARY.** This repository is in a split-migration audit.
> Historical experiments and manuscripts are retained locally for review, but
> they are not a frozen evidence release and do not support current numerical
> lifecycle, invariance, intervention, or generalization claims.

This project studies a five-state BIRTH/REINFORCE/DECAY/DEATH/IDLE readout for
temporal graphs. It pairs a hierarchical lifecycle branch with a scored link
prediction path, then specifies gradient-ancestry, score-invariance, and typed
intervention checks. The package owns a checksum-listed copy of its backbone
runtime and does not import from the sibling link-prediction project.

**Research status:** `split-migration-audit`. Both the evidence release and the
paper snapshot are `UNRELEASED`; see [Project Status](wiki/status/Project-Status.md)
and [Current Claim Language](wiki/claims/Current-Claim-Language.md). The
[version-controlled wiki](wiki/README.md) is the source of truth for scientific
interpretation. Begin with [Start Here](wiki/START-HERE.md) or browse the
[Exhaustive Index](wiki/INDEX.md).

## Manuscript packages

The paper lifecycle is separate from experiment evidence:

| Location | State | Purpose |
|---|---|---|
| [`paper/CURRENT`](paper/CURRENT) | `UNRELEASED` | Authoritative paper-snapshot pointer |
| `paper/working/` | Local and excluded from Git | Quarantined named and blind legacy drafts; not permitted claim language |
| `paper/snapshots/` | No snapshot yet | Reserved for an immutable export after the [Paper Export Contract](wiki/manuscript/Paper-Export-Contract.md) passes |

The [paper lifecycle notes](paper/README.md) explain the quarantine. No current
manuscript package can be built or cited from this source repository.

## Current evidence status

| Topic | Current state | Record |
|---|---|---|
| Copied backbone runtime | Local checksum integrity validated; behavioral parity and release closure remain open | [Runtime evidence](wiki/evidence/Evidence-Ledger.md), [`shared-runtime.sha256`](configs/shared-runtime.sha256) |
| CoEdit migration copies | Locally bit-identical; upstream identity and redistribution rights remain blocked | [Dataset Registry](wiki/datasets/Dataset-Registry.md), [Evidence Ledger](wiki/evidence/Evidence-Ledger.md) |
| Decoder, invariance, and intervention outcomes | No quantitative claim admitted | [Claim Registry](wiki/claims/Current-Claim-Language.md), [Research Questions](wiki/questions/Research-Questions.md) |
| Frozen results and paper | `UNRELEASED` | [`results/CURRENT`](results/CURRENT), [`paper/CURRENT`](paper/CURRENT) |

The hand-specified lifecycle structure is not a discovered causal model.
Legacy result files and figures are excluded from Git because their claims have
not passed the current protocol. See [Limitations](wiki/LIMITATIONS.md) and the
[publication gate](publication/PUBLICATION_GATE.toml).

## Layout

```text
src/lifecycle_readout/  reusable datasets, models, metrics, and training code
experiments/            thin runners, probes, and dataset builders
protocols/              readout and intervention contract
configs/                versioned parameters, dependency and runtime hashes
resources/              dataset manifest; local corpora are excluded from Git
evidence/jobs/           local structural execution record
results/                 release pointer; audit and historical data excluded
figures/                 figure guidance; legacy generated figures excluded
paper/                   paper pointer and export guidance; drafts excluded
wiki/                    canonical status, methods, claims, and evidence
publication/             blocked public-release gate
tests/                   repository and provenance contracts
```

Implementation is package-relative under `src/`. The experiment entry points
are described in [experiments/README.md](experiments/README.md); heavy work must
run through an approved scheduler workflow.

## Install

Python 3.10 or newer is required. In an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

The test suite checks repository structure, imports, links, manifests, and
available local dataset bytes. It does not run model training or establish a
scientific result.

## Data

The repository tracks [dataset identities](resources/manifest.toml) and
[checksums](resources/checksums.sha256), not the CoEdit corpus binaries. Local
copies belong under `resources/corpora/`, or set `LIFECYCLE_DATA_DIR` to a
complete checksum-verified directory. Missing data never triggers an implicit
download or build.

The upstream Wikipedia identity, license, citation, and redistribution
permission are unresolved. Review the [Dataset Registry](wiki/datasets/Dataset-Registry.md),
[Data and Target Contract](wiki/datasets/Data-and-Target-Contract.md), and
[License and Assets](wiki/governance/License-and-Assets.md) before using or
sharing a corpus.

## Reproducing a result

There is no frozen lifecycle result to reproduce yet. The safe local check is:

```bash
python -m pytest -q
```

When a corpus is available locally, the tests also verify its declared hashes
and schema. A future result must follow the [readout protocol](protocols/lifecycle_readout_v1.toml)
and [reproducibility contract](wiki/REPRODUCIBILITY.md), retain every attempt
and failure, and pass evidence and claim review before appearing here.

### Provenance

| Record | What it establishes | Boundary |
|---|---|---|
| [`LCR-JOB-LOCAL-20260819-001`](evidence/jobs/LCR-JOB-LOCAL-20260819-001.toml) | Local structural migration checks | No training or frozen scientific evidence |
| [Evidence Ledger](wiki/evidence/Evidence-Ledger.md) | Artifact identities and current acceptance status | Historical results and working papers remain quarantined |
| [`PROJECT.toml`](PROJECT.toml) | Current project stage and release pointers | Both pointers remain `UNRELEASED` |

## Known limitations

- No quantitative lifecycle, invariance, intervention, or cross-domain claim is
  admitted.
- Dataset provenance and redistribution rights are unresolved; clean clones do
  not contain corpus bytes.
- Historical runs lack reconciled seed, denominator, retry, and failure coverage.
- The available tests verify structural contracts, not behavioral parity or
  generalization. See the full [limitations register](wiki/LIMITATIONS.md).

## License

No project code license has been selected. Do not infer a grant from this
repository. Dataset and third-party terms require separate review; see
[License and Assets](wiki/governance/License-and-Assets.md).

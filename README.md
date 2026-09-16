# Faithful Lifecycle Readout for Temporal Graphs

This project studies a five-state BIRTH/REINFORCE/DECAY/DEATH/IDLE readout,
hierarchical decoding, typed interventions, gradient ancestry, score
invariance, and causal-coherence diagnostics on CoEdit.

The repository is independent from the link-prediction project. It owns a
pinned internal backbone snapshot because the readout's faithfulness tests
must execute against the exact scored path, but it has no sibling imports or
filesystem links. Claims and evidence use the `LCR-*` namespace.

## Start here

1. Read [wiki/START-HERE.md](wiki/START-HERE.md).
2. Inspect [PROJECT.toml](PROJECT.toml) for release pointers.
3. Install with `python -m pip install -e .` in an isolated environment.
4. Run `python -m pytest -q` for repository contracts.

Local migration workspaces may retain named and anonymous legacy manuscripts in
`paper/working/` and historical results in `results/historical/`. These
quarantined assets, legacy figures, and the local corpora are excluded from this Git repository;
see [the publication gate](publication/PUBLICATION_GATE.toml). Heavy training
must run through an approved scheduler workflow, not on a login node.

License selection is pending owner review; see
[wiki/governance/License-and-Assets.md](wiki/governance/License-and-Assets.md).

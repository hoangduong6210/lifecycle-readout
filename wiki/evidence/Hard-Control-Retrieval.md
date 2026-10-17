---
title: Hard Control Evidence Retrieval
status: canonical local retrieval guide
last_updated: 2026-10-15
paper_source: false
---

# Hard Control Evidence Retrieval

Use `LCR-E-HARD-CONTROLS-001` to retrieve the completed `LCR-P-HARD-001`
comparison. The [results owner](../results/Hard-Negative-Controls.md) owns numerical
interpretation; the [ledger](Evidence-Ledger.md) owns evidence status. The
[tracked checksum index](../../evidence/indexes/hard_controls_v1.toml) pins the
exact bytes of the local evidence and corrected Word revision.

## Locate the completed attempts

| Dataset | Execution record | Retained result | Independent audit |
|---|---|---|---|
| Wikipedia | [Job](../../evidence/jobs/LCR-JOB-HARD-WIKIPEDIA-001.toml) | [Results](../../results/audit/lcr-wikipedia-256094e9e7/artifacts/results.json) | [Replay and metrics](../../results/audit/hard_controls_v1/wikipedia_audit.json) |
| Reddit | [Job](../../evidence/jobs/LCR-JOB-HARD-REDDIT-001.toml) | [Results](../../results/audit/lcr-reddit-bf7053e7a2/artifacts/results.json) | [Replay and metrics](../../results/audit/hard_controls_v1/reddit_audit.json) |

Each attempt directory contains `job.json`, `execution.txt`, `source.zip` and
`artifacts/`. The source archive records the actual uploaded worktree; it is
more specific than the repository's current uncommitted source. The artifact
manifest pins predictions, checkpoints, candidates, exclusions, environment and
aggregate results. Both normalized jobs identify protocol/configuration/source
hashes, seed coverage and cleanup state.

Study design is owned by the [protocol](../../protocols/hard_controls_v1.toml)
and [configuration](../../configs/hard_controls_v1.toml). The independent
[auditor](../../scripts/audit_hard_controls.py) replays history and recomputes
metrics. Recheck a retained attempt without allocating compute:

```bash
python3 scripts/audit_hard_controls.py results/audit/lcr-wikipedia-256094e9e7 --output /tmp/lcr-wikipedia-recheck.json
python3 scripts/audit_hard_controls.py results/audit/lcr-reddit-bf7053e7a2 --output /tmp/lcr-reddit-recheck.json
```

## Retrieve the manuscript and resource record

- [Corrected mutable Word](../../paper/working/legacy/Lifecycle_Manuscript.docx)
- [Preserved original Word](../../paper/working/revisions/20261015-pre-hard-controls.docx)
- [Revision hashes and recorded render inspection](../../results/audit/hard_controls_v1/manuscript_revision.json)
- [Final recorded Colab inventory and usage](../../results/audit/hard_controls_v1/resource_final.json)
- [Audit and revision checksum manifest](../../results/audit/hard_controls_v1/checksums.sha256)

The resource record is a timestamped observation, not a live account balance.
The revision record identifies the wiki bytes used for the Word correction;
subsequent navigation edits do not regenerate that historical editorial record.
The [export contract](../manuscript/Paper-Export-Contract.md) owns manuscript status.

## Availability and integrity

Wiki, protocols, configuration, normalized jobs and the retrieval index are
source-controlled material. `results/audit/` and `paper/working/` are deliberately
ignored by Git. The linked large artifacts exist in this local project directory;
a clean clone alone cannot recover them. No immutable release or external artifact
archive has been published. Preserve the original local directories together with
their manifests; a checksum proves byte identity but cannot recover missing bytes.

The index records repository-relative paths, SHA-256 hashes and whether each file
is a local artifact. Verify it from the repository root using Python:

```python
from pathlib import Path
import hashlib
import tomllib

root = Path.cwd().resolve()
index = tomllib.loads((root / "evidence/indexes/hard_controls_v1.toml").read_text())
for entry in index["artifacts"]:
    path = (root / entry["path"]).resolve()
    assert path.is_relative_to(root) and path.is_file(), entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
print("All indexed evidence bytes verified")
```

This snippet uses Python `3.11` or later for `tomllib`. Full scientific release,
clean-clone reproduction and publication admission remain open. Mixed results
and support exclusions must remain visible when using the retrieved artifacts.


## Supplementary checkpoint replay

The original retrieval index and its pinned files remain unchanged. `results/audit/hard_checkpoint_replay_v1/summary.json` locates a later complete test-input and checkpoint inference replay for both datasets, all declared seeds and variants. It records the original index hash, reverified artifact/archive identities, independently reconstructed pre-event inputs, archived source imports and CPU/GPU discrepancies. The accompanying manifest pins the auditor, fixture, extracted model source and reports. This is local inference reproducibility with existing dependencies, not fresh-environment training reproduction or evidence admission.

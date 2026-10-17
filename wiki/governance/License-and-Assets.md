---
title: License and Assets
status: blocked governance
last_updated: 2026-10-16
paper_source: false
---

# License and Assets

No project code license was present in the migration source. The public source
repository grants no reuse license; rights remain with their respective
holders. Choosing an open-source license requires a review of ownership and
authorization for the copied backbone and other contributions. Public source
visibility does not admit a scientific result or paper snapshot.

Corpus binaries, historical results, working manuscripts, and legacy figures
are excluded from Git. Dataset identity, citation, terms, and redistribution
permission remain blocked pending a separate registry review. `IEEEtran.cls`
retains its own LPPL notice in the local working-paper tree and is not
relicensed by this project. Named and blind manuscripts must remain explicitly
distinguished.

## Current source and upstream asset review

The source inventory in `results/audit/source_candidate_review_v1/` records
current package/script hashes and their Git state. The repository root still
has no license file. Git history, file headers and an author name do not by
themselves establish ownership of all copied or modified contributions. The
requested ownership information remains unanswered; no project reuse
license has been selected or granted.

The CRAFT author files have a separately retained upstream `LICENSE` and pinned
commit. The [asset restorer](../../scripts/fetch_craft_assets.py) recovers those
exact files and license notice using tracked provenance metadata, validates
them against the existing fit configuration, and preserves unfamiliar local
bytes. It does not execute downloaded code or grant a license to this project.
The [reproduction instructions](../REPRODUCIBILITY.md#restoring-pinned-craft-author-assets)
state the recovery scope and historical metadata flags.

The reviewed Colab launch bundles named `source.zip` also contain prepared data
inputs. They are execution evidence, not source-only distribution packages.
Any eventual public package must use the existing allowlist publication gate
and its rights review. No archive was published or redistribution permission
inferred from source-code license notices.

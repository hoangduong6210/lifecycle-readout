# Dataset builders

These are migrated provenance utilities, not an implicit runtime path. They stage
outputs under `resources/staging/`; they never overwrite the checksum-pinned corpus.
Promote a staged file only after recording its raw-source digest, license, builder
revision, processed checksum, and dataset-registry review. `apply_idfix.py` reads the
preserved inputs under `resources/corpora/pre_idfix/`.

The upstream raw-file digests and licenses are currently `UNKNOWN/BLOCKED`; do not
use a fresh download as evidence for an existing claim or release.
`download.py` therefore requires a reviewed `<DATASET>_RAW_SHA256` environment
variable and rejects an unverified raw CSV.

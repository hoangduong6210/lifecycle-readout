# Corpus contract

`corpora/` contains a local, ignored CoEdit copy for migration verification.
Tracked identity is owned by `manifest.toml` and `checksums.sha256`. Set
`LIFECYCLE_DATA_DIR` to another complete checksum-verified directory when
running elsewhere. Missing data never triggers an implicit build.

CoEdit is bit-identical to its pre-ID-fix backup. The Wikipedia source identity
used by the CoEdit builder must still be recorded before a frozen release.

# Evidence records

This directory owns structured execution identity. Wiki claims may cite an
Evidence ID, but every assertion produced by a command must also resolve to a
record under `jobs/` and to `jobs/checksums.sha256`.

These records cover local migration checks, Colab execution, and independent
exploratory-result audits. They are not a frozen scientific evidence release and
do not admit performance claims.

For the completed hard-control study, start with the
[wiki retrieval guide](../wiki/evidence/Hard-Control-Retrieval.md). The
[structured index](indexes/hard_controls_v1.toml) pins exact local result, source,
audit and manuscript bytes. Large evidence files remain local and ignored by Git;
the index alone cannot recover missing files.

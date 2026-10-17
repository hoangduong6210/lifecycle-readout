# Study and reproduction scripts

The training, scoring, intake and diagnostic entry points use the matching files
in `protocols/` and `configs/`. Numerical code lives in `src/lifecycle_readout/`.
Raw data and retained checkpoints must be supplied explicitly when required.

- `build_article.py --output <directory>` explicitly assembles an article from
  canonical wiki text and distributed reports; it does not run a compiler.
- `build_readme.py` generates the research overview from distributed reports.
- `verify_public_evidence.py` verifies source/public hashes, aggregate counts,
  article CSV agreement, linked overview values and wiki navigation.
- Figure generators are in `figures/source/` and need no remote compute.

Scientific evidence and immutable article pointers remain `UNRELEASED`; these
source/reproduction tools do not admit scientific claims.

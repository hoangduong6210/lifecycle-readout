---
title: Reproducibility Contract
status: canonical contract
last_updated: 2026-08-19
paper_source: false
---

# Reproducibility Contract

A release must pin source commit, runtime snapshot hash, protocol/config hashes,
dependency lock, CoEdit identity, complete seed coverage, failures, model state,
faithfulness dump schema, intervention outputs, and claim mappings. Mutable work
goes to `results/audit/`; immutable releases go to
`results/frozen/<release-id>/`. `results/CURRENT`, `PROJECT.toml`, the evidence
ledger, and paper `results.lock.yaml` must agree before export.

# Migration notes

This project was copied from the mixed SR-GNN working tree on 2026-08-19.
The lifecycle readout and its exact v3.3 backbone closure were converted to
package-relative imports under `src/lifecycle_readout/`. CoEdit, lifecycle
experiments/results/figures, and named/blind Paper 2 artifacts were imported to
the local migration workspace. Corpus bytes, historical results, legacy figures,
and working manuscripts are excluded from the source repository. No imported
result was promoted to frozen evidence.

# Lifecycle Readout for Temporal Graphs

Research implementation for **Monotonicity of Temporal Graph Readouts: Exact Certificates and Predictive Limits**.

The project studies when normalized reweighting preserves a decreasing readout of an absorbing simplex flow. It includes an exact certificate for specified decay rates, finite rate-cone checks, inverse-tilt projection, temporal prediction controls and diagnostics of the complete scoring path.

The certificate fixes the readout, tilt and candidate context along each elapsed-time path. It does not certify an arbitrary changing transition or the whole learned predictor. Structural preservation does not imply better ranking: matched repair comparisons show no mean AP advantage, and the exploratory January Last.fm subgroup benefit was not confirmed in February. Latent coordinates are not verified causal or semantic lifecycle stages.

- [Scientific article and editable source](paper/README.md)
- [Wiki overview](wiki/README.md)
- [All eight figures, vector artwork and numeric inputs](wiki/Figures.md)
- [Aggregate-data definitions](paper/working/sncs/Supplementary_Data_Guide.txt)

## Installation and checks

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests use synthetic inputs and do not launch remote compute. To rebuild all eight figures from the included aggregate CSVs:

```bash
python figures/source/sncs_all_figures.py --output figures/generated/sncs
```

## Code map

| Location | Contents |
|---|---|
| `src/lifecycle_readout/modeling/` | Temporal encoders, phase readouts, certificate and projection operators |
| `src/lifecycle_readout/` | Historical features, controlled comparisons, dataset interfaces and diagnostics |
| `scripts/` | Training, scoring, auditing and data-preparation entry points |
| `experiments/` | Earlier model runners and diagnostic probes |
| `configs/`, `protocols/` | Study settings, endpoints, scope and input identities |
| `tests/` | Implementation tests |
| `figures/source/` | Matplotlib figure and pipeline generators |
| `paper/working/sncs/` | Article source, PDF, vector figures and aggregate CSVs |

## Data and reproducibility limits

Source interaction datasets, trained checkpoints and local run archives are not distributed. Obtain data from the original providers under their terms. Configurations referring to `results/audit/` identify retained inputs; those files are not bundled, so the full retained-model evaluations cannot be replayed from this checkout alone. Figure regeneration and implementation tests work without them. Aggregate results retain negative, tied and unconfirmed outcomes.

This is research source and a working article, not an accepted publication or a frozen evidence release. No reuse license is granted for project code; third-party assets retain their own terms.

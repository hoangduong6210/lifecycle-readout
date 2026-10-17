---
title: Benchmark and Baseline Results
status: exploratory earlier benchmark record
last_updated: 2026-10-15
paper_source: false
---

# Benchmark and Baseline Results

The tables below preserve earlier exploratory values and wording for audit.
They are not admitted scientific claims or matched external-baseline evidence.
The prior positive-defined silence strata do not match negative histories.
Current corrective results and interpretation are owned by
[Historical Hard Negative Controls](Hard-Negative-Controls.md). Numerical
literature rows below remain unaudited and must not be exported as current facts.

## 1. Table I: Dynamic Link Prediction and Shortcut Mitigation

All experiments evaluate across `3` distinct random seeds (`1`, `7`, `42`) on dedicated NVIDIA L4 GPU accelerators. Splits follow strictly chronological `70%` train, `15%` validation, and `15%` test partitions with bipartite-aware negative sampling (`build_candidate_stream`), unobserved negative features, an independent `BaseTemporalPredictor` model trained from scratch, and an explicit inductive test split on unseen nodes.

### Table I: Unconfounded Dynamic Link Prediction & Inductive Generalization

| Dataset | Transductive Base AP | Transductive L3 AP (Ours) | Transductive Gain ($\Delta\text{AP}$) | Inductive Base AP (New Nodes) | Inductive L3 AP (New Nodes) | Inductive Gain ($\Delta\text{AP}$) | 7-Interval Monotonicity | Silence Effect | Evidence Job | Audit JSON |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **Wikipedia** | `0.9729 ± 0.0003` | `0.9872 ± 0.0006` | `+0.0142` (`+1.42%`) | `0.9782 ± 0.0004` | `0.9872 ± 0.0008` | `+0.0091` (`+0.91%`) | `100.00%` | `-0.4575` | `LCR-JOB-COLAB-WIKI-001` | `results/audit/tkde_results_wikipedia.json` |
| **Reddit** | `0.9788 ± 0.0002` | `0.9898 ± 0.0001` | `+0.0110` (`+1.10%`) | `0.9693 ± 0.0009` | `0.9839 ± 0.0002` | `+0.0146` (`+1.46%`) | `100.00%` | `-0.4485` | `LCR-JOB-COLAB-REDDIT-001` | `results/audit/tkde_results_reddit.json` |
| **MOOC** | `0.8198 ± 0.0017` | `0.8233 ± 0.0033` | `+0.0035` (`+0.35%`) | `0.8452 ± 0.0013` | `0.8484 ± 0.0032` | `+0.0032` (`+0.32%`) | `100.00%` | `-0.0335` | `LCR-JOB-COLAB-MOOC-001` | `results/audit/tkde_results_mooc.json` |
| **LastFM** | `0.7762 ± 0.0026` | `0.7742 ± 0.0029` | `-0.0020` (`-0.20%`) | `0.7754 ± 0.0014` | `0.7730 ± 0.0034` | `-0.0025` (`-0.25%`) | `100.00%` | `-0.0024` | `LCR-JOB-COLAB-LASTFM-001` | `results/audit/tkde_results_lastfm.json` |

### Historical internal phase replacement response (Wikipedia)
Under continuous-time transition dynamics, counterfactual interventions on latent relational phase $do(z = e_k)$ relative to dormant phase $e_0$ yield ordered changes in the model output. These follow the imposed weights and are not identified external causal effects:
- Phase `1` relative to Phase `0`: `+0.2350`
- Phase `2` relative to Phase `0`: `+0.5051`
- Phase `3` relative to Phase `0`: `+0.6595`
- Phase `4` relative to Phase `0`: `+0.7158`

### Provenance Mapping for Table I
- **Hardware environment:** Dedicated NVIDIA L4 GPU instance instantiated via Google Colab CLI (`colab run --gpu L4`).
- **Protocol:** [`LCR-P-TKDE-002`](../../protocols/lifecycle_tkde_v2.toml) declared in `protocols/lifecycle_tkde_v2.toml`.
- **Manifest references:** Dataset files downloaded from Stanford SNAP repository and verified via `resources/manifest.toml`.
- **Job manifests:** `evidence/jobs/LCR-JOB-COLAB-WIKI-001.toml`, `evidence/jobs/LCR-JOB-COLAB-REDDIT-001.toml`, `evidence/jobs/LCR-JOB-COLAB-MOOC-001.toml`, `evidence/jobs/LCR-JOB-COLAB-LASTFM-001.toml`.
- **Integrity verification:** Recorded in `evidence/jobs/checksums.sha256` and `results/audit/checksums.sha256`.

---

## 2. Table II: Comparative Dynamic Link Prediction against Official SOTA Baselines

Comparative results against official dynamic graph learning baselines:

| Method | Source / Type | Wikipedia AP | Wikipedia AUC | Reddit AP | Reddit AUC | MOOC AP | MOOC AUC | LastFM AP | LastFM AUC | Provenance / Evidence |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| JODIE | Published (KDD'19) | `0.9534` | `0.9482` | `0.9492` | `0.9438` | `0.8870` | `0.8790` | `0.8420` | `0.8350` | Kumar et al. (KDD 2019) |
| TGAT | Published (ICLR'20) | `0.9503` | `0.9421` | `0.9576` | `0.9520` | `0.8250` | `0.8190` | `0.7840` | `0.7780` | Xu et al. (ICLR 2020) |
| TGN | Published (ICML'20) | `0.9784` | `0.9752` | `0.9712` | `0.9664` | `0.9480` | `0.9450` | `0.9320` | `0.9280` | Rossi et al. (ICML 2020) |
| DyGFormer | Published (NeurIPS'23) | `0.9768` | `0.9741` | `0.9684` | `0.9632` | `0.9410` | `0.9380` | `0.9120` | `0.9080` | Yu et al. (NeurIPS 2023 Table 1) |
| GraphMixer (Published) | Published (ICLR'23) | `0.9725` | `0.9702` | `0.9731` | `0.9685` | `0.9580` | `0.9540` | `0.9270` | `0.9210` | Cong et al. (ICLR 2023) |
| GraphMixer (Author Run) | Official Author Codebase | `0.9401 ± 0.0010` | `0.9376 ± 0.0011` | — | — | — | — | — | — | `LCR-JOB-BASELINE-GRAPHMIXER-WIKI-001` |
| Base TGNN (No Gate) | Ablation Model | `0.9729 ± 0.0003` | `0.9678 ± 0.0004` | `0.9788 ± 0.0002` | `0.9772 ± 0.0001` | `0.8198 ± 0.0017` | `0.8262 ± 0.0015` | `0.7762 ± 0.0026` | `0.7410 ± 0.0015` | Audit JSONs (`seeds 1, 7, 42`) |
| Unordered Gate (Ablation) | Ablation Model | `0.9755 ± 0.0022` | `0.9705 ± 0.0022` | `0.9838 ± 0.0044` | `0.9817 ± 0.0040` | `0.8250 ± 0.0022` | `0.8293 ± 0.0011` | `0.7717 ± 0.0012` | `0.7397 ± 0.0015` | Audit JSONs (`seeds 1, 7, 42`) |
| **L3-GNN (Ours)** | Proposed Architecture | `0.9872 ± 0.0006` | `0.9843 ± 0.0008` | `0.9898 ± 0.0001` | `0.9881 ± 0.0001` | `0.8233 ± 0.0033` | `0.8278 ± 0.0027` | `0.7742 ± 0.0029` | `0.7426 ± 0.0052` | `LCR-JOB-COLAB-WIKI-001`, `LCR-JOB-COLAB-REDDIT-001`, `LCR-JOB-COLAB-MOOC-001`, `LCR-JOB-COLAB-LASTFM-001` |

### Provenance Mapping for GraphMixer Author Run
- **Author repository:** `https://github.com/yule-BUAA/DyGLib.git`
- **Pinned commit SHA:** `3aacc36b94b8d2d8293d70a74fdf6d39089b4163`
- **Job manifest:** `evidence/jobs/LCR-JOB-BASELINE-GRAPHMIXER-WIKI-001.toml`
- **Audit JSON:** `results/audit/baseline_graphmixer_wikipedia.json`
- **Per-run transductive metrics:** Run `0` (`0.9392` AP / `0.9366` AUC), Run `1` (`0.9412` AP / `0.9387` AUC), Run `2` (`0.9399` AP / `0.9375` AUC).
- **Per-run inductive new node metrics:** Run `0` (`0.9231` AP / `0.9202` AUC), Run `1` (`0.9259` AP / `0.9235` AUC), Run `2` (`0.9242` AP / `0.9217` AUC).
- **Checksum reference:** SHA-256 hash registered in `evidence/jobs/checksums.sha256`.

---

## 3. Counterfactual Response: Silence Intervention $do(\Delta t \to \infty)$

The counterfactual response under prolonged silence intervention measures whether link probability attenuates monotonically as the silence interval grows:

$$z_{uv}^{\text{cf}} = \lim_{\Delta t \to \infty} \hat{z}_{uv}(t + \Delta t)$$

- **Wikipedia:** Monotonic decay rate `100.00%`, Silence ATE `-0.4575`.
- **Reddit:** Monotonic decay rate `100.00%`, Silence ATE `-0.4485`.
- **MOOC:** Monotonic decay rate `100.00%`, Silence ATE `-0.0335`.
- **LastFM:** Monotonic decay rate `100.00%`, Silence ATE `-0.0024`.

Under L3-GNN, monotonic decay is guaranteed via the structural theorem of the continuous jump-decay ODE coupled with the ordered survival gate $\Psi(z) = \sigma(w^\top z + b)$.

---
title: Historical Hard Negative Controls
status: independently audited exploratory results
last_updated: 2026-10-16
paper_source: false
---

# Historical Hard Negative Controls

This page owns the prospective `LCR-P-HARD-001` outcomes. These are exploratory
conditional recurrent-support results, not a frozen evidence release or an
admitted publication claim. The [method](../methods/Shortcut-Resistant-Gating.md)
and [protocol](../../protocols/hard_controls_v1.toml) define the estimand.
All five variants use the same supported candidates within each seed. History-aware
variants have identical explicit statistics and affinity architecture; the
neighbor-only Base is retained to isolate information access. Training is independent
with matched initialization, batch order, optimizer, validation selection and
epoch budget. There was no model-specific hyperparameter search.

## Primary endpoint

Pooled test AP, mean and sample standard deviation across seeds `1`, `7`, `42`.
These values must not be numerically compared to the earlier easier-negative study.

| Variant | Wikipedia AP | Reddit AP |
|---|---:|---:|
| Base | `0.598917 ± 0.012695` | `0.538600 ± 0.000567` |
| Base plus history | `0.611484 ± 0.011112` | `0.547638 ± 0.001230` |
| Scalar decay plus history | `0.542812 ± 0.005716` | `0.512808 ± 0.003522` |
| MLP gate plus history | `0.613485 ± 0.016450` | `0.548079 ± 0.003410` |
| Ordered lifecycle plus history | `0.563353 ± 0.004727` | `0.549794 ± 0.000192` |

## Support and paired outcomes

Eligibility requires past history for both candidates at the same source, equal
logarithmic recurrence/silence bins, and absence from current simultaneous
positives. Every unsupported real event is retained in history. No easy-negative
fallback is used. First-time pairs have no support in this study.

### Wikipedia

Per seed: `2805` train, `884` validation, `977` test pairs; `138000` unsupported supervision events and `14808` excluded training events, from `157474` real events.

Ordered lifecycle minus Base plus history AP differences: `-0.051150, -0.034398, -0.058844`; mean `-0.048131` and sample SD `0.012500`.

The ordered variant has lower mean AP than Base, Base plus history, MLP gate plus history.

Artifacts: [`lcr-wikipedia-256094e9e7`](../../results/audit/lcr-wikipedia-256094e9e7/artifacts/results.json).
Independent replay and metric audit: [`wikipedia_audit.json`](../../results/audit/hard_controls_v1/wikipedia_audit.json).

| Model | AUC mean | Brier mean | Parameters |
|---|---:|---:|---:|
| Base | `0.600914` | `0.244788` | `167881` |
| Base plus history | `0.612309` | `0.241861` | `176586` |
| Scalar decay plus history | `0.509748` | `0.424035` | `176587` |
| MLP gate plus history | `0.604089` | `0.243550` | `185291` |
| Ordered lifecycle plus history | `0.535889` | `0.370381` | `186374` |

| Variant | Both silent AP mean | Both silent positive count per seed |
|---|---:|---:|
| Base | `0.614747` | `781, 781, 781` |
| Base plus history | `0.610314` | `781, 781, 781` |
| Scalar decay plus history | `0.505827` | `781, 781, 781` |
| MLP gate plus history | `0.582953` | `781, 781, 781` |
| Ordered lifecycle plus history | `0.528756` | `781, 781, 781` |

### Reddit

Per seed: `37206` train, `10139` validation, `11002` test pairs; `538695` unsupported supervision events and `75405` excluded training events, from `672447` real events.

Ordered lifecycle minus Base plus history AP differences: `+0.002824, +0.000771, +0.002874`; mean `+0.002156` and sample SD `0.001200`.

The ordered variant has the highest mean AP among these controls, but this alone is not a statistical significance claim.

Artifacts: [`lcr-reddit-bf7053e7a2`](../../results/audit/lcr-reddit-bf7053e7a2/artifacts/results.json).
Independent replay and metric audit: [`reddit_audit.json`](../../results/audit/hard_controls_v1/reddit_audit.json).

| Model | AUC mean | Brier mean | Parameters |
|---|---:|---:|---:|
| Base | `0.539447` | `0.252000` | `167881` |
| Base plus history | `0.546948` | `0.250701` | `176586` |
| Scalar decay plus history | `0.507746` | `0.432053` | `176587` |
| MLP gate plus history | `0.547217` | `0.250191` | `185291` |
| Ordered lifecycle plus history | `0.548773` | `0.248604` | `186374` |

| Variant | Both silent AP mean | Both silent positive count per seed |
|---|---:|---:|
| Base | `0.541305` | `10136, 10136, 10136` |
| Base plus history | `0.545193` | `10136, 10136, 10136` |
| Scalar decay plus history | `0.507392` | `10136, 10136, 10136` |
| MLP gate plus history | `0.546143` | `10136, 10136, 10136` |
| Ordered lifecycle plus history | `0.547602` | `10136, 10136, 10136` |

## Interpretation and limits

The completed fixed-budget controls show mixed effects: Wikipedia favors the
history-aware baseline and MLP gate; Reddit has a modest mean gain for ordered
lifecycle. Its gain over Base plus history occurs in all three Reddit seeds,
whereas its gain over the MLP gate occurs in only two of three seeds. No
cross-dataset superiority or significance claim is supported. Earlier gains
over a neighbor-only base cannot isolate
latent lifecycle utility. This does not prove that every lifecycle model or
hyperparameter setting must fail. The new conditional population, much smaller
supervised training subset and fixed scaled clock limit transfer to full-stream
or first-time-link conclusions. No official-model SOTA comparison is established.

Monotonicity remains an imposed exact-flow property. These outcomes do not
identify external causal effects or validate phase semantics. Changing the clock,
matching rules or optimization after seeing these outcomes requires a separate
prospective amendment; none of the negative outcomes may be overwritten.

## Audit and execution

`LCR-E-HARD-CONTROLS-001` maps to `LCR-JOB-HARD-WIKIPEDIA-001` and
`LCR-JOB-HARD-REDDIT-001`. Each completed attempt includes source and dataset
identities, candidate exclusions, checkpoints, predictions, validation history
and full checksums. The independent auditor checked every eligible and unsupported
event, candidate equality across models, metrics, checkpoint selection and aggregates.
Both owned Colab runtimes were released and backend absence was verified.
Code: [audit_hard_controls.py](../../scripts/audit_hard_controls.py).


## Implementation and comparison scope clarification

Inspection of the identical sealed Wikipedia and Reddit training sources confirms that held-out-node training exclusions enter neither supervision nor history. Allowed events without an eligible negative still enter history. Unsupported totals reported above span all splits; they are not training-only exclusions. Matching bins does not require identical counts or elapsed times.

History-aware controls add the same history head to the common neighbor-affinity logit and fit normalization on training candidates. All variants use task and auxiliary affinity supervision; the lifecycle variant additionally uses its entropy, diversity and sparsity regularizer. Parameter counts differ. The study compares implemented variants under matched candidates and budgets; it does not isolate the constraint independently of capacity, regularization or optimization effects. This is a reporting clarification, not a changed protocol or rerun. Sealed source identities and manuscript corrections are retained in `results/audit/journal_methods_v6/`.


## Archived-checkpoint inference replay

A separate local audit now reconstructs every retained test query's neighbor inputs and pair statistics directly from the prepared event stream, excluding entire current timestamp groups and held-out-node training events. It then loads each selected checkpoint using the exact archived model source from its original attempt. All declared Wikipedia and Reddit seeds and variants reproduce retained positive and negative probabilities within the declared cross-device tolerances. Checkpoint identity, selected epoch, training normalization, parameter count and original artifact manifests are also checked.

The source, fixture, exact coverage, observed discrepancies, dependency versions and complete reports are retained in `results/audit/hard_checkpoint_replay_v1/`. This closes inference replay for the retained hard-control cohorts. It is not full optimization reproduction, a fresh environment, an independent reimplementation of every neural layer or fresh predictive confirmation. Original execution and audit records remain unchanged; scientific admission is still open.


## Posthoc pooling and within-query diagnostic

The user proposed that history matching could explain the pooled-score losses.
[The diagnostic record](../../protocols/pooling_diagnostic_v1.toml) declares this
as posthoc analysis of all retained seeds and variants, with no fitting, tuning,
new candidates or replacement of the original pooled test AP endpoint. Original
archives, archive-bound manifests and all prediction hashes were reverified;
candidate identities and histories agree across models within each seed.

Paired accuracy gives a win one point, a loss zero and an exact tie half a point.
Within-bin AP is computed separately in each original count/silence bin and then
weighted by its query count. It pools multiple queries within a bin and therefore
is not within-query accuracy. Values below are mean and sample SD across the
original seeds. They are exploratory diagnostic endpoints, not a frozen release.

| Variant | Wikipedia paired accuracy | Wikipedia within-bin AP | Reddit paired accuracy | Reddit within-bin AP |
|---|---:|---:|---:|---:|
| Base | `0.600136 ± 0.014702` | `0.642010 ± 0.012915` | `0.539114 ± 0.001421` | `0.544520 ± 0.001554` |
| Base plus history | `0.615490 ± 0.032405` | `0.654879 ± 0.012554` | `0.546052 ± 0.006186` | `0.550051 ± 0.002674` |
| Scalar decay plus history | `0.553907 ± 0.005023` | `0.579141 ± 0.006189` | `0.521602 ± 0.005646` | `0.522855 ± 0.004834` |
| MLP gate plus history | `0.634596 ± 0.006141` | `0.665945 ± 0.008830` | `0.547537 ± 0.003542` | `0.551202 ± 0.003717` |
| Ordered lifecycle plus history | `0.619243 ± 0.028273` | `0.635998 ± 0.014730` | `0.545204 ± 0.004665` | `0.552517 ± 0.001765` |

On Wikipedia, ordered lifecycle changes from a pooled-AP loss against Base plus
history to a small mean paired-accuracy gain, but that gain occurs in only two
seeds and the MLP remains stronger on the mean paired diagnostic. Ordered
lifecycle still loses to Base plus history on query-weighted within-bin AP.
On Reddit, its small pooled-AP advantage does not become a paired-accuracy
advantage over Base plus history on average. Scalar decay remains worse than
Base plus history on both paired accuracy and within-bin AP in both datasets;
its paired loss occurs in every seed. This does not support a universal account
that changing aggregation removes the observed failures.

The auditor also decomposes pooled AUC into same-bin and cross-bin Cartesian
comparisons, with exact comparison-count weights and half credit for ties.
These Cartesian comparisons differ from observed query pairs. AP has no
corresponding linear decomposition here; a within-bin AP difference is not a
causal attribution of a pooled AP change.

Scalar decay has many paired ties, all at the original implementation's lower
probability clamp. Class means and clamp fractions are retained in the local
report. Saved predictions cannot recover the ordering of pre-clamp scores;
that requires a separately declared checkpoint diagnostic, now recorded below. This is an
additional observed implementation limit, not proof that clipping or pooling
alone caused the fixed-budget loss. No training or scoring code is changed.

The bin check reconstructs integer counts from saved log-count features and
elapsed bins at retained floating-point precision, verifies equality within each
pair and retains the original raw-history audit as separate evidence. Exact
per-seed/per-bin values, decomposition weights, source identities, first-attempt
failure and final report are in
`results/audit/hard_pooling_diagnostic_v1/diagnostic_v2.json` and companion
records. [Executable analysis](../../scripts/analyze_hard_pooling.py) and
[small independent examples](../../tests/test_hard_pooling.py) cover ties,
explicit Cartesian enumeration and counterexamples to inevitability.


## Stable pre-clamp scalar checkpoint diagnostic

[The follow-up record](../../protocols/pooling_unclipped_diagnostic_v1.toml)
was declared after observing ties and before this replay. It covers every
original scalar-decay seed on both datasets, with no fitting or checkpoint
selection. Exact archived model code and independently reconstructed strict-past
inputs reproduce original clipped probabilities within the prior replay tolerance.
Original clipped pair decisions also reproduce on each seed.

The diagnostic computes log-sigmoid of the learned affinity logit minus the
scalar decay exponent in float64. It recovers a mathematical pre-clamp ordering
without exponentiation or clipping, using the same learned parameters. This is
not a reconstruction of unavailable original pre-clamp GPU floats. Exponentiated,
clipped diagnostic scores agree with the archived forward output within the
same tolerance. Values are mean and sample SD across original seeds.

| Dataset | Original clipped paired accuracy | Stable pre-clamp paired accuracy | Original pooled AP | Stable pre-clamp pooled AP |
|---|---:|---:|---:|---:|
| Wikipedia | `0.553907 ± 0.005023` | `0.566701 ± 0.006254` | `0.542812 ± 0.005716` | `0.543191 ± 0.005787` |
| Reddit | `0.521602 ± 0.005646` | `0.525692 ± 0.005884` | `0.512808 ± 0.003522` | `0.512958 ± 0.003530` |

Stable log scoring removes the scalar pair ties and improves mean paired
accuracy modestly, but scalar decay remains below Base plus history on both
datasets and on every corresponding seed. Pooled AP changes little, and Brier
error remains effectively unchanged. Thus clipping loses some ordering, but
removing its evaluation effect does not account for the full observed deficit.
These are descriptive evaluation findings; they do not identify why the original
training converged to its parameters or show how retraining without the floor
would behave. Keep the original declared endpoint and all negative results.

Six checkpoints, retained probability identities, complete stable log-score
arrays, original-nontied ordering checks, selected epochs and per-seed endpoints
are retained in `results/audit/hard_pooling_unclipped_v1/`. The
[checkpoint replayer](../../scripts/audit_hard_checkpoint_replay.py) exposes this
optional diagnostic separately from its unchanged default full-model replay.
This follow-up is posthoc on an inspected test cohort; no new confirmation,
probability calibration repair or scientific admission is implied.


## Diagnostic estimands and statistical prior art

The [matched-cohort ROC review](../questions/Contribution-Gap-Review.md#statistics-prior-art-for-matched-cohort-roc)
places aggregation sensitivity against existing statistics. The reported same-bin
Cartesian AUC weights bins by squared query counts; query-weighted within-bin AP
uses linear counts, and paired accuracy uses only the actual query pairs. These
are distinct diagnostics and are not a validated implementation of a literature
covariate-adjusted ROC estimator. Original pooled AP still measures the declared
matched-candidate task; no population-wide performance or prevalence correction
has been performed. All posthoc results above and their original negative
comparisons remain unchanged.

---
title: Journal Research Program
status: active development with publication gates open
last_updated: 2026-10-15
paper_source: false
---

# Journal Research Program

The current user goal is a journal appropriate to the demonstrated contribution;
a research ranking is not required. This supersedes the earlier venue-specific target. The
historical filename and protocol identifiers are retained for traceability. The
working draft and mixed controls remain development material, not a submission
ready paper. [Project Status](../status/Project-Status.md) owns execution, and
[the existing results](../results/Hard-Negative-Controls.md) remain visible,
including the Wikipedia loss.

The first publication route to assess is a focused methods-and-analysis paper:
state the exact monotonicity conditions, distinguish imposed behavior from
learned semantics, and report where history-aware controls remove or retain the
apparent benefit. A useful, sufficiently distinct analysis may support a paper
without a new architecture or universal superiority. Its novelty, empirical
adequacy and venue fit still need review; no acceptance or readiness is assumed.
Finish and audit the running CRAFT fit before deciding whether another experiment
is necessary. Further compute must answer a concrete gap in this paper scope.

## Research contribution to establish

The working question is when a constrained relational time response adds useful
information beyond recurrence, recency, popularity and co-occurrence. A possible
contribution combines a baseline-nesting logit correction, explicit support-aware
evaluation, and an analysis of when temporal constraints help or hurt. Neither
a mixture of exponentials nor monotonicity alone is asserted novel. A model
whose gain disappears against a scalar or a simple count model does not establish
latent lifecycle utility.

[Primary TGB-Seq work](https://proceedings.iclr.cc/paper_files/paper/2025/hash/db5ca61dbc08cf5143c05ad2d1b0b2ca-Abstract-Conference.html)
already studies the limitation of repeated-edge-heavy benchmarks.
[Base3](https://arxiv.org/abs/2506.12764) combines recurrence, popularity and
co-occurrence, so these signals need matched controls. The recent
[smoothed-count preprint](https://arxiv.org/abs/2609.35364) also pools sequential
counts across sources; it is a relevant comparator, not a peer-reviewed guarantee.
Their published scores cannot be copied into a matched-results table.

## Development stage now

`LCR-P-Q1-DEV-001` is a separately versioned amendment to `LCR-P-HARD-001`.
Its [protocol](../../protocols/q1_development_v1.toml) and
[configuration](../../configs/q1_development_v1.toml) are saved before allocation.
The diagnostic record `LCR-E-Q1-DIAGNOSTIC-001` uses training elapsed times and
recorded validation histories only. It motivates, without proving, a clock and
probability-attenuation hypothesis.

Compare the same history-aware affinity model, multiplicative scalar decay,
legacy ordered gate, scalar logit residual, contextual mixture residual and
unconstrained MLP logit residual. All decay models use the positive training
elapsed median as clock scale, with the same initial rate. Hard-negative bins
retain the old fixed clock, so this amendment does not change candidate support.
The old fixed-clock outputs are preserved; cross-study contrasts cannot identify
a clock effect because other design choices also changed.

Development seeds are `101`, `202`, `303`. All comparisons use the same candidate
stream, backbone/head initialization, optimization budget and validation rule.
Only validation predictions and metrics are emitted. Validation selection and
reuse make these outcomes optimistic development estimates; new seeds do not
turn previously inspected datasets into independent confirmation. The inherited
ordered-gate regularizers differ from other objectives and must be reported.

## Evidence required before a research submission

| Gate | Required evidence | Current state |
|---|---|---|
| Distinct contribution | Full related-work comparison, including constrained temporal models, count models and recent sequential methods; explain what existing methods cannot address | Open; candidate only |
| Mechanism | Controlled synthetic cases where silence is informative, irrelevant or misleading; compare scalar versus mixture, flexible residual, clock and objective controls | Oracle capacity and misspecification audited; real-graph mechanism unproven |
| Breadth and fresh evaluation | Licensed upstream identities and fixed official candidate sets across several domains, including newly introduced low-recurrence datasets | Existing recurrent-support datasets are development only |
| Strong baselines | Author implementations for EdgeBank, Base3, GraphMixer, DyGFormer and suitable current count/sequential methods, under identical history timing and candidate sets | Not executed under a common official protocol |
| Reliable effects | Validation-only tuning with equal budgets; frozen test protocol; paired seeds and dependence-aware uncertainty; report negative, first-time and unsupported strata | No fresh confirmatory study |
| Practical value | Memory, runtime, parameter count, history construction and actual CU alongside accuracy; bounded hyperparameter budget | New development recording planned |
| Reproducibility | Source/environment/data/output closure, clean-clone replay, rights review and retrievable artifact archive | Open; current artifacts local |
| Paper and venue | Claims follow evidence; scope match, current quartile database/year/category, format and citation audit | Working Word only |

Official [TGB-Seq evaluation](https://github.com/TGB-Seq/TGB-Seq) uses fixed splits,
multiple benchmark negatives and MRR. Retain its official evaluator rather than
substituting the current matched-pair AP study. Evaluate official full support
and first-time links separately from the conditional recurrent diagnostic.
The [official dataset catalogue](https://tgb-seq.github.io/datasets/) identifies
ML-20M, GoogleLocal and Flickr as possible new-domain evaluations. These names
are candidates, not registered verified corpora: source rights, byte identities,
size and evaluator version must be reviewed before any confirmation allocation.
A first-time-pair residual must define its neutral behavior in advance, without
excluding those queries to make the model look stronger.
Sampled-candidate Brier measures that sampled task; it is not automatically
calibration to a population event probability. Synthetic phase recovery checks
known simulated labels; it cannot label real-world latent components.

## Decisions after development

Proceed to a sealed broader study only if development supports a concrete
benefit beyond Base plus the same statistics and the scalar residual, with
acceptable calibration and cost. Otherwise refine or reject the mechanism while
retaining these outcomes. Do not rerun different test settings until a preferred
ranking appears. Set confirmatory datasets, primary comparisons, effect sizes,
uncertainty units and stopping rules before their tests are examined.

Venue selection should follow the final question, contribution type, audience,
article requirements and publication cost. benchmark remains a historical motivation,
not the mandatory destination. Verify each candidate journal's current official
scope and publication model before ranking it. Quartile is optional context,
not a scientific acceptance gate or a proxy for acceptance probability.

## Compute policy

Heavy work runs through the existing Colab CLI launcher, one owned runtime at a
time, with bounded timeouts, artifact download and verified finally cleanup.
The pre-development inventory was idle with `184.03` CU. Run development
before a large official-baseline sweep; inspect cost and outcomes between stages.
No external submission, paid upgrade or public artifact publication is authorized
by the research target alone.


## First development disposition

[Wikipedia development](../results/Validation-Development.md) completed with
no new test scoring. Its mixture correction is effectively tied with the scalar,
so the mechanism has not passed the intended development gate. Reddit remains
not run, with the planned two-dataset coverage explicitly incomplete. Prioritize
controlled mechanism tests and stronger contribution review before expanding
compute. The transport failure and recovered complete training are both retained.


## Mechanism diagnostic disposition

[Oracle synthetic diagnostics](../results/Synthetic-Mechanisms.md) show that
the code can learn context-dependent decreasing responses and stay neutral under
a null. Flexible MLP approximation remains stronger in the non-null examples;
no unique mixture advantage or phase semantics follow. This is a toy capacity
check, not noisy event-stream or graph validation.

Before another real-data sweep, name the
[prediction target](../methods/Prediction-Target-and-Sampling.md). Standard survival,
event-density and sampled-classification identities need not share the same
monotonicity. The known density-ratio formulas are prior theory, not a novelty
claim. Official evaluator parity is checked, while dataset provenance and rights
remain separately documented in the dataset registry.


## New-link development screen

The ML-20M structural audit finds no repeated pair in its train/validation inputs.
The old recurrent-support matcher therefore cannot be reused as a universal
benchmark. `LCR-P-TGBSEQ-SCREEN-001` seals an initial validation-only ranking
screen of explicit count controls, using official split masks, shared new
validation candidates, and midrank MRR. This is a baseline screen, not a proposed
method contribution. Do not tune or claim research readiness from these validation
outcomes alone; new model novelty and fresh confirmation remain open.


## Fresh screen disposition

[The ML-20M development screen](../results/ML20M-Development-Baselines.md) completed
with scoped independent audit. Pooled sequential counts are a stronger control
than popularity on this candidate draw, but are already prior art. The next
implementation gate is a pinned strong sequential comparator with a common
validation adapter. A new candidate-specific temporal mechanism must be stated
and checked against prior art before its efficacy experiment; generic count
smoothing, time decay and context pooling do not establish a distinct contribution.


The [CRAFT feasibility adapter](../results/CRAFT-Adapter-and-Feasibility.md) is now
technically audited with exact cached candidate anchors, original-model weights
and CPU/GPU roundtrip. This closes an implementation/cost preflight, not a
strong-baseline accuracy gate. Full validation-only training must use a new sealed
fit protocol; the proceedings parameter row is now retained for that design.


## Mechanism positioning after the comparator

[The contribution-gap review](Contribution-Gap-Review.md) now includes the recent
phase-type point-process and causal temporal graph preprints, alongside count,
sequence, time-to-event and sampling objectives. A candidate-specific temporal
readout remains an unvalidated question. Its proposed controls isolate clock,
functional shape and information access; it requires a separate prospective
protocol before any fit and cannot inherit a novelty claim from known basis
functions or an imposed monotonicity constraint.


[The journal paper strategy](Research-Paper-Strategy.md) now specifies the focused
article outline, claim-to-evidence gaps and a current official-source venue
shortlist. It replaces the earlier assumption that a new architecture and a research
destination were mandatory.

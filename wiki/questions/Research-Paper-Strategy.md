---
title: Research Paper Strategy
status: canonical research documentation
last_updated: 2026-10-17
paper_source: false
---

# Research paper strategy

The article develops an exact, restricted-domain monotonicity characterization,
its repair geometry and controlled evidence about predictive limits. The failed
next-month confirmation and negative primary comparisons remain part of the
scientific argument. Publication venue and access-model decisions are external
to the research contribution and are not prescribed by this repository.

## Current recommendation after the February outcome

Lead with the fixed-rate certificate, inverse-tilt geometry and complete scoring
path diagnostics. Explain the assumptions and matched comparisons before any
performance interpretation. Independent proof review, scoped prior-work
comparison and evidence locking remain requirements for a final submission.

## Paper to develop

The [working article narrative](../manuscript/Article-Narrative.md) now supplies
the abstract, introduction, connected interpretation, discussion and assembly
map. Method definitions and numerical tables remain owned by their canonical
pages; the narrative does not duplicate or override those records. The main
article follows the scored-path, shared-history and explicit-clock questions.
Synthetic, pooling/floor and first-time-boundary material supplies scoped
supplementary evidence. It is a review draft, not a frozen submission.

Working title: **Monotone Temporal Gating for Link Prediction: Guarantees and
Limits under History-Matched Evaluation**. The central question is when an
imposed silence response contributes beyond information already present in
history statistics. The article should connect an exact, conditional mathematical
property to practical evaluation boundaries. It is a candidate analysis paper,
not a claim of discovered lifecycles or universal predictive superiority.

The [journal program](Research-Program.md) owns overall research gates.
Historical protocol and configuration identifiers are retained for reproducibility.
Changing the venue target does not admit numerical claims or make a draft ready
for submission. Both release pointers remain unreleased.


## Contribution and evidence map

The [closest benchmark comparison](Contribution-Gap-Review.md#closest-benchmark-comparison-after-cadence-completion)
now explicitly reserves established source-time cancellation, query-common
offset invariance and exponential-mixture constructions to prior work. The
article's proposed distinctness must rest on its specified scored-path analysis
and controlled evidence, not those ingredients. That scope distinction still
does not settle whether the contribution is sufficient for the journal.

| Article contribution | Current evidence owner | Required closure |
|---|---|---|
| Exact conditions for monotonicity under a fixed affinity score and pure silence | [Gating method](../methods/Shortcut-Resistant-Gating.md) | Present assumptions and proof; distinguish sufficient active-weight ordering from the weaker necessary-and-sufficient dormant-gap condition. Review distinction from existing monotone and temporal models. |
| Delimit preservation through a candidate transformation | [Transition analysis](../methods/Shortcut-Resistant-Gating.md#a-sufficient-magnitude-and-drift-bound-for-the-ideal-many-phase-map) | Separate the exact restricted two-phase condition, constructive reversals and conservative many-phase magnitude-and-drift bound. The latter excludes clamps and changing affinity/initial state. Independent proof and novelty review remain open; no trained-checkpoint guarantee follows. |
| Diagnose the information and candidate-support dependence of apparent gains | [Hard controls](../results/Hard-Negative-Controls.md) and [benchmark audit](../results/Benchmark-and-Baseline-Results.md) | Keep shared-feature controls, all seeds and unsupported-query denominators. Do not compare absolute AP across different candidate tasks as an improvement. |
| Separate aggregation and numerical-floor effects from ranking utility | [Pooling diagnostic](../results/Hard-Negative-Controls.md#posthoc-pooling-and-within-query-diagnostic) and [stable-log replay](../results/Hard-Negative-Controls.md#stable-pre-clamp-scalar-checkpoint-diagnostic) | Label both posthoc; retain the original pooled AP endpoint, every seed, paired ties and pre-clamp scope. Descriptive differences do not establish why training failed. |
| Show the boundary between imposed silence behavior and predictive usefulness | [Validation development](../results/Validation-Development.md) and [synthetic diagnostics](../results/Synthetic-Mechanisms.md) | Preserve mixed and negative results; oracle approximation is a capacity check, not real-world causality or phase recovery. |
| Contextualize the method against a strong sequence model and simple counts | [ML-20M controls](../results/ML20M-Development-Baselines.md) and [CRAFT interface](../results/CRAFT-Adapter-and-Feasibility.md) | The bounded fit and scoped audit are complete; retain its cap, seed, selection and convergence limits. ML-20M is a validation-only boundary case, not a new confirmed test result or an official-paper reproduction. |

The theorem alone may be too elementary for an original article. The potential
contribution depends on a coherent, reproducible empirical analysis that adds
insight beyond known benchmark shortcut observations. The
[contribution-gap review](Contribution-Gap-Review.md) is targeted prior-art work;
it does not establish this novelty. CRAFT accuracy alone cannot resolve the
novelty of this analysis.


## How the new diagnostics change the article argument

The empirical analysis should distinguish three questions: whether the output
obeys the fixed-context silence requirement, whether it ranks the two candidates
within a query, and whether scores remain comparable across queries. A guarantee
for the first does not settle the other two. The hard-control diagnostic shows
that aggregation can change a model comparison, but changing aggregation does
not remove the scalar losses. Stable pre-clamp checkpoint scores recover some
ordering hidden by probability clipping while leaving the larger deficit.
These findings belong together as an operating-limit analysis with the original
negative results intact.

Place the paired/within-bin results after the original hard-control table, then
present stable-log replay as a numerical sensitivity check. Keep the original
primary metric visually and verbally primary. Explain ties and the different
comparison populations; do not frame the supplementary metric as a corrected
benchmark winner. The simple matched-bin identity and the transition-norm
counterexample delimit proposed explanations. Neither is currently a cleared
novelty claim, and neither supplies a general impossibility theorem. The
[statistics prior-art review](Contribution-Gap-Review.md#statistics-prior-art-for-matched-cohort-roc)
also establishes that matching/aggregation sensitivity predates this study;
the diagnostic estimands must remain explicit.

The [three-direction review](Contribution-Gap-Review.md#review-of-the-three-proposed-breakthrough-directions)
records the relevant prior art and rejects the strongest unsupported claims in
the brainstorm. Source-session attenuation remains an optional hypothesis, not
an allocated study or a new contribution by definition. Complete the existing
declared cadence development comparison, after the completed CRAFT fit, before deciding
whether another fit addresses a concrete unresolved question. Future paper
assembly draws this argument from the wiki; this update does not trigger a
Word revision or relax the submission gates.

The [source-gap diagnostic](../results/ML20M-Development-Baselines.md#posthoc-source-session-inactivity-diagnostic)
provides a limited empirical motivation for that optional question: transition
advantage is smaller in longer-inactivity intervals. It uses existing ranks and
train-derived boundaries, not a new trained model. Keep it secondary and
observational; it does not establish an optimal decay, a new session clock or
a monotone-gate gain. It need not become another architecture branch for the
focused journal article.


## Prospective cadence interpretation

The [sealed cadence protocol](../../protocols/pair_cadence_development_v1.toml)
owns the endpoint, selection, features and compute budget. The interpretation
below adds no threshold, significance test, acceptance criterion or new study.
It distinguishes what the already-declared contrasts can establish before any
real-data cadence fit is observed.

| Observed development pattern | Interpretation to retain in the wiki | Conclusion that does not follow |
|---|---|---|
| Exposed cadence features improve over their masked control, while the relative scalar clock does not improve over the absolute scalar clock with the same exposed features | Additional cadence information may help the declared ranking task; the chosen relative attenuation has not shown added value | A new clock or a benefit attributable to changing the explicit clock |
| The relative scalar clock improves over the absolute scalar clock with identical exposed features | Evidence of an incremental clock choice under this protocol; report each dataset, paired seed, selection and uncertainty | Fresh-test generalization, causal identification, phase semantics or originality of relative freshness |
| Only the mixture contrast improves | A secondary exploratory shape result; the primary scalar endpoint must still be reported | Success of the primary endpoint, or a reason to replace it after seeing results |
| A flexible MLP matches or exceeds the constrained residual | Available information and flexible response remain viable explanations; retain the constraint as a conditional design property | Unique lifecycle utility from a positive constrained-model contrast alone |
| Effects differ by dataset or seed, or no contrast improves | Preserve heterogeneity or the negative result and assess its operating-limit value | Universal superiority, a pooled win over unlike populations, or automatic justification for more tuning |
| Missing-history coverage is poor | State the estimable population and fallback behavior alongside the outcome | A pair-specific measured rhythm for unavailable histories, or performance on first-time pairs |

Any favorable pattern remains validation development on previously inspected
datasets. Report the declared contrasts together, including ties and losses.
Neither an implementation check nor a favorable contrast establishes novelty;
the [direct relative-freshness predecessor](Contribution-Gap-Review.md#direct-relative-freshness-prior-art)
already covers the core normalization. Further confirmation or redesigned tuning
would require a separate prospective record, rather than reinterpreting this
study after its outcomes.


## Article structure

1. **Problem and task.** Define sampled destination ranking and distinguish it
   from survival estimation. State the operational silence requirement and the
   information available to each model.
2. **Temporal gate and guarantee.** Give the dynamics, score and monotonicity
   assumptions. Explain why the unconstrained gate can invert but need not do so.
   Describe connected gradients without claiming guaranteed nonzero gradients.
3. **Evaluation design.** Separate the older easy-negative benchmark from the
   prospective history-matched study. Define positive and negative histories,
   strict timestamp rules, candidate support and the conditional population.
4. **Results and operating limits.** Include every declared comparison and seed,
   the Wikipedia loss and limited Reddit gain, with support denominators. Keep
   development-only studies separate from test evidence. Do not pool incomparable
   metrics or treat selection on validation as independent confirmation.
5. **Implications and limitations.** Explain that an architectural guarantee can
   hold without improving ranking. Discuss first-time pairs, sampled labels,
   missing lifecycle labels, compute budgets and generalization boundaries.
6. **Reproducibility and disclosures.** Freeze source, data identities, protocol,
   environment, artifacts and independently checked metrics. Cite source datasets
   and licenses; provide retrieval instructions where redistribution is restricted.

The [journal working Word draft](../../paper/working/Monotone_Temporal_Gating_Article_Manuscript_v8.docx)
now includes the focused framing and targeted related-work corrections, while preserving the earlier drafts.
The current v8 has a 233-word abstract and 21 inspected rendered pages. It adds the restricted fixed-transition condition, a constructive counterexample and an analytic figure, and identifies LastFM feature channels as zero placeholders. All earlier native mathematical objects and all seven tables are preserved. The data tables remain
unchanged and exploratory; CRAFT results are not inserted while training is active.
The v3 pass distinguishes EvolveGCN from DySAT, separates bottleneck objectives, narrows generalization wording and adds Base3, CRAFT and smoothed-count positioning. The v3 pass retained all seven tables and 421 mathematical objects byte-identically to v2; v4 adds two native equations without changing those objects. This is a targeted statement review; full bibliography and novelty closure remain open. Reconcile the audited baseline and complete those reviews before
forming any frozen, paper-eligible tables. Do not add a new
architecture merely to make the article look novel.


## Next executable steps

The v5 pass also corrects GNNExplainer pagination, formats DeepSurv as an article number and directly cites neural cumulative-intensity prior work. The current-numbered locator map is retained in `results/audit/journal_bibliography_v5/`; this does not certify every cited assertion.

The owned CRAFT fit, artifact recovery, scoped independent audit and runtime
release are complete. Reconcile its bounded validation interpretation from
[the result owner](../results/CRAFT-Adapter-and-Feasibility.md#completed-full-support-validation-fit)
alongside the [completed cadence comparison](../results/Pair-Cadence-Development.md). Review
whether the analysis is sufficiently distinct from prior benchmark work. Follow
the [wiki-first hard rule](../../AGENTS.md#9-wiki-and-paper-snapshots): do not
issue an incremental manuscript revision for this update. Assemble the paper
from the reconciled wiki at the final manuscript stage, or export a working
draft only on explicit user request. Only allocate
additional experiments for an explicit unresolved claim under a prospective
protocol. Close data provenance, reproducibility and evidence-release gates
before producing a submission snapshot. No journal submission is authorized by
this planning record.


The v6 reporting pass clarifies allowed-event history updates, all-split unsupported totals, matched-bin versus exact-value equality, and the differing capacity and lifecycle regularization of the hard-control variants. It names the candidate transformation as a pre-label scoring operation and distinguishes empty history from held-out node identity. Sealed sources from both completed hard-control attempts agree on these implementation details. Equations, tables, references and figure bytes are unchanged; six changed rendered pages were inspected, with other pages identical to reviewed v5. Exact records are in `results/audit/journal_methods_v6/`. This revision does not add a baseline endpoint or admit claims.


The v7 working draft adds an explicitly unmeasured pair-relative clock extension in the discussion, with arithmetic-mean cadence, availability/fallback rules, a relative-time formula and the capacity-matched development design. It cites personalized-interval and adaptive-sampling prior work and does not claim novelty or predictive efficacy from implementation checks. All earlier math, tables and figure bytes are retained, with two native equations added. New sources, input-audit scope and render identities are in `results/audit/journal_cadence_v7/`. Finish the baseline and prospective cadence comparison before presenting efficacy endpoints or promoting the extension as a scientific contribution.



## Completed cadence and the final article argument

The [complete cadence comparison](../results/Pair-Cadence-Development.md)
records a negative primary clock-replacement difference on every declared seed
of both corpora. Added contextual features help some families, especially on
Reddit, but that is a different contrast. Both sides of the primary comparison
already receive relative freshness as a feature. The study therefore does not
show that relative-time information is useless; it shows no observed AP gain
from imposing this alternative explicit attenuation clock under the declared
settings. Small negative effects do not establish equivalence or universal
inferiority.

The paper should retain the analysis framing: fixed-context guarantees,
information/candidate-support dependence, aggregation and numerical effects,
and the distinction between a contextual clock and an imposed response. Do not
recast the known clock as a new method, substitute secondary mixture seeds for
the primary result, or launch another architecture solely to obtain a positive
table. The observational source-session diagnostic remains a motivated future
question with known prior art, not a required new branch for this article.

All declared cadence fits and checkpoint inference checks are complete, and
Colab is idle. Next reconcile the final contribution and author-review questions
against these wiki owners and prepare the reviewed evidence/source snapshot.
No fresh test scoring or routine Word export is implied. Final submission
readiness remains unproved while novelty/contribution, source availability,
rights, declarations and scientific admission are unresolved.


## Contribution strengthening after the competition pilot

The primary competition pilot is negative and is not promoted as a successful
new predictor. Two stronger candidate additions now support the analysis route:

| Candidate contribution | Exact current support | Remaining boundary |
|---|---|---|
| [Readout-specific entropic preservation](../methods/Readout-Aware-Entropic-Preservation.md) | Necessary-and-sufficient restricted all-rate condition, constructive parameterization and exact separation witness | Constant tilts and fixed readout/context; human proof and exact-result priority review remain open |
| [Learned scored-path diagnostic](../results/Scored-Path-Silence-Diagnostic.md) | Archived-source validation anchors, direct derivative check and native full-model replay; Reddit reversals and Wikipedia threshold non-crossings both retained | Finite fixed anchor/grid intervention, not whole-cohort prevalence or physical causal silence |

The revised [working narrative](../manuscript/Article-Narrative.md) integrates
these additions while preserving shared-history controls, negative cadence and
competition outcomes, and established-method attribution. This strengthens the
specific analytical/diagnostic argument; it does not settle journal readiness,
scientific admission or novelty of the entire paper. Complete supported-validation coverage has since been verified in the
[full-cohort owner](../results/Full-Validation-Scored-Paths.md). The closest
preservation-related preview comparison narrows the candidate distinction but
does not clear priority. A fixed repair/ranking-distortion study is the next
useful method test; failed pilot fitting is not repeated until a positive
result appears.

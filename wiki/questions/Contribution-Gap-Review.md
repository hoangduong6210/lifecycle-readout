---
title: Contribution Gap Review
status: targeted prior-art review and unvalidated mechanism question
last_updated: 2026-10-16
paper_source: false
---

# Contribution Gap Review

The journal target requires a clear, sufficiently distinct contribution. A
focused analysis paper is a possible route; a new architecture is optional.
Method-superiority claims require support beyond strong count and sequence controls. The current author-baseline fit addresses comparator strength;
it does not itself supply a new method. This page records a targeted overlap
review and an explicit question for subsequent development. It is not exhaustive
novelty clearance or an admitted claim.

## Closest benchmark comparison after cadence completion

Zhang and colleagues' [v1 preprint](https://arxiv.org/html/2610.04869v1),
*Your Temporal Link Predictor Is Blind to Who Is Active*, already explains
sampler density-ratio cancellation and within-query invariance of a source-time
offset. The closer reading retains these overlaps explicitly.

| Inspected locus | Existing construction | Current study's narrower object |
|---|---|---|
| Section `4.4`, equation `10`; Appendix `A.1` | Additive log source intensity; unchanged destination ranking within a fixed query | Candidate-dependent gates and the path from a phase flow to a scored response |
| Appendix `B.3` | Positive exponential mixtures for slow decay | A controlled nested residual; mixture form is not a new ingredient |
| Appendix `E.3` | Batch-level AP/AUC; historical negatives may change source | Same-source matched-history candidates; pooled primary and posthoc diagnostics |
| Section `4.4`; Appendix `A.3` | Separate event-intensity likelihood | Sampled classification; no inherited intensity or survival interpretation |

These are differences in objects and protocols, not proof of originality.
The proposed contribution remains the specific scored-path analysis and
capacity/information-controlled clock comparison, with their negative results.
No published score is imported or sampler reproduction claimed. Inspected
locators and source identity are retained in
`results/audit/journal_closest_prior_art_v1/`.

## Constructions already covered by related work

| Proposed ingredient | Primary prior work inspected | Remaining requirement in this project |
|---|---|---|
| Recurrence, popularity and shared sequential evidence | [Base3](https://arxiv.org/abs/2506.12764) and the [smoothed-count preprint](https://arxiv.org/html/2609.35364v1) | Isolate any additional signal with identical candidates and information access. Renaming these statistics as phases does not establish a contribution. |
| Candidate-aware recent-history matching | [CRAFT, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html) | The bounded shared-candidate author fit is complete; retain its seed, budget and validation-selection limits. Any added-readout efficacy claim remains a separate question. |
| Phase-type states, delayed Erlang responses and exact event likelihoods | [EPT-X v1, October 2026 preprint](https://arxiv.org/html/2610.14948v1), construction and neural prediction sections | A generic phase-type or renewal replacement is insufficient novelty. A link-ranking formulation would need a demonstrated distinction from marked event prediction and a concrete benefit beyond simpler controls. |
| Synthetic causal graph validation | [CTIG v1, February 2026 preprint](https://arxiv.org/abs/2602.02161v1) | A known synthetic generator can test a specified causal mechanism; internal phase replacement and the current oracle response fits do not substitute for that experiment. |
| Time-to-link appearance as a probability distribution | [NPGLM v1](https://arxiv.org/abs/1706.06783v1) | Separate time-to-event estimation from the present sampled destination-ranking task; survival terminology alone is not a new formulation. |
| Positive-dependent noise and spectral density-ratio tools | [Conditional NCE](https://proceedings.mlr.press/v80/ceylan18a.html) and [GradientDICE](https://proceedings.mlr.press/v119/zhang20r.html) | Compare the actual objectives and assumptions. The working posterior equation is not a validated new estimator or a novelty claim. |

EPT-X uses a phase-type state space with closed-form intensity/integral updates
and a renewal channel. CTIG proposes generated interaction graphs for controlled
causal shifts. These are relevant overlaps, not reproductions performed here;
their reported empirical superiority is not imported into our result tables.
NPGLM's inspected abstract concerns arrival-time distributions. Cached primary
identities and inspected source snapshots are retained in
`results/audit/q1_preflight_v1/mechanism_review/`.

## Candidate question after the strong comparator

Does the shape of a candidate-specific temporal response add reliable ranking
information after recent-sequence encoding and simple count signals are held
fixed? This is a narrower, testable mechanism question within the journal program;
this optional mechanism route is not a prerequisite for an analysis paper.

The low-recurrence ML-20M inputs make a per-pair clock unavailable for every
validation positive. A candidate-global clock and a source-context clock remain
observable. A proposed readout must state which clock it uses, handle missing
history explicitly and explain which candidate scores it changes. A common
positive multiplier cannot alter within-query ordering in exact arithmetic.

A useful controlled probe would freeze a specified affinity model and compare a
neutral readout, scalar time response, monotone exponential basis, delayed-response
basis and a flexible MLP with the same input statistics and an explicit parameter
budget. Zero residual must recover the parent scorer. Phase-type basis functions
would be a prior-art modeling choice; observed gains would need to distinguish
functional shape from added features, parameters, objective and tuning budget.
No latent phase receives a real-world lifecycle label without external evidence.

The probe is a research question, not a sealed executable study. It has not been
allocated or scored. A separate prospective protocol must fix feature/clock
construction from training inputs, candidate identities, training/selection
budgets, model controls and stopping rules before fitting. If scalar or flexible
controls explain the effect, the lifecycle-specific interpretation remains
unsupported. Current CRAFT settings and stopping rules remain unchanged.

## Decision boundary

Complete and audit the strong comparator first. Then assess whether a concrete
temporal residual hypothesis merits an experiment; do not infer a need for one
from the mere existence of baseline errors. Retain negative outcomes, avoid
test-guided redesign and require fresh confirmation across domains before paper
superiority wording. The [journal program](Research-Program.md) continues to own
submission gates; [Project Status](../status/Project-Status.md) owns execution.

## Training history availability for a pair cadence

A local read-only audit checks whether the proposed past mean inter-event clock is observable on the existing hard-control training candidates. It does not fit a model, inspect validation/test feature histories or estimate predictive utility. Both candidates are queried from the same strictly earlier history; simultaneous events enter only after all queries. Held-out-node training events stay excluded, while events without a matched negative still enter history.

The candidate clock is `(last-first)/(prior_count-1)` for at least two observations, with positive observed span. This is an arithmetic mean, not an exponential moving average. Counts below two and zero-span histories are kept separate; no fallback or regularization constant is fitted.

The local audit retains exploratory coverage counts, cadence ratios and support denominators outside the publication surface. The raw allowed-event denominator includes events without an eligible negative; earlier easy-negative training denominators and hard-supported supervision are distinct populations. An unsupported-event total over all splits is not a training-only exclusion count. Support scarcity alone does not establish the cause of model loss.

The Reddit replay includes repeated candidate histories with zero observed span. Replacing a missing or zero cadence with an arbitrary tiny denominator would conflate sparse evidence with extremely rapid repetition. A prospective experiment must declare smoothing and an availability indicator, expose identical clocks to flexible/history controls, and hold architecture and training support fixed while testing the feature change.

Matched bins do not require identical elapsed time or identical counts. The observed cadence spread establishes variation inside the matched cohort, not a relationship between cadence and the correct destination. Pair clocks are unavailable for all first-time ML-20M train/validation positives; that remains a separate operating boundary.

Source corpus/candidate hashes, all six training-history arrays, exact denominators and independently scanned candidate-history anchors are retained in `results/audit/pair_cadence_support_v1/`. A deterministic fixture covers tied events, missing/zero-span cadence and held-out exclusions. Existing protocols, results, training settings and manuscript tables are unchanged.


## Gate and transition compatibility

The [gating method](../methods/Shortcut-Resistant-Gating.md) now gives an explicit constant-transition counterexample to extending the pure-silence guarantee to the scored state, and a necessary-and-sufficient condition for the restricted two-phase interior map. Full-module checks include the configured five-phase architecture. This makes the operating boundary constructive rather than relying only on a verbal caveat. It is a candidate component of the focused analysis paper, not a claim of new general monotonicity theory or observed checkpoint failure frequency. The v4 working manuscript now includes the restricted derivation, constructive illustration and LastFM placeholder-feature clarification. The general novelty and evidence-admission boundaries remain open.


## Pair-cadence implementation and prior-art boundary

The user requested implementation of the pair clock. The opt-in [method extension](../methods/Shortcut-Resistant-Gating.md) and [prospective cadence design](../../protocols/pair_cadence_development_v1.toml) now separate added information from clock choice using capacity-matched controls. This implementation is not a real-data efficacy result, and no scientific or novelty claim is admitted.

The broad idea of personalized temporal modeling predates this extension. [Liu and colleagues](https://link.springer.com/chapter/10.1007/978-3-030-30952-7_40) describe personalized time intervals with node embeddings and Hawkes modeling; only the publisher abstract was accessible, so exact formula equivalence is unverified. [DPS](https://arxiv.org/html/2111.11886v1) learns node-specific decay rates for historical-neighbor sampling. That differs from a past-mean pair clock used in a candidate-scoring residual, but prevents claiming personalized decay itself as new. A [network-memory study](https://link.springer.com/article/10.1007/s41109-023-00597-w) scales descriptive inter-event distributions by group means; its descriptive construction is not the same as the online dyadic feature here. Cached primary-source bytes and this scope are in `results/audit/pair_cadence_implementation_v1/`.

A potentially publishable question is whether pair-relative time contributes incremental ranking information under matched historical support after identical feature access, capacity and flexible controls. It remains a question: the mean-gap identity, positive rescaling and exponential-basis residual alone do not establish novelty. Exact overlap review and empirical effect are both open. Do not market the arithmetic mean as a learned biological clock, a relationship-death label or a guaranteed improvement.


## Direct relative-freshness prior art

The full primary article by [Navarro, Miritello, Canales and Moro](https://link.springer.com/article/10.1140/epjds/s13688-017-0127-3) directly defines relative freshness as time since the last call divided by the same tie's mean inter-event interval, and uses it in tie-persistence prediction. Its feature-preparation section also log-transforms relative freshness. This is substantially closer than generic personalized-time or node-specific decay work. The core pair-normalized recency feature is established prior art; the implemented smoothing and log-one-plus transform do not establish a new clock.

The article's target is future-window tie persistence, whereas the current development target is sampled destination ranking under strict-past histories and matched support. That distinction motivates an evaluation question, not a claim that the normalization itself is original. The implemented feature, masked/exposed controls and relative-clock experiment remain useful for testing incremental predictive value; their novelty and utility cannot be inferred from new terminology. The source, exact inspected sections and scope comparison are retained in `results/audit/pair_cadence_prior_art_v2/`. No protocol, configuration, candidate set or running fit is changed.


## Review of the three proposed breakthrough directions

The supplied brainstorm is a set of hypotheses, not evidence of novelty. The
first direction has now been tried on every retained hard-control prediction;
the second now has counterexamples and a scoped many-phase sufficient bound.
The third remains a conditional experiment idea. The bounded CRAFT fit and the [pair-cadence comparison](../results/Pair-Cadence-Development.md) have completed.

### Matched bins and pooled metrics

For one positive and one negative per supported query with the same bin B,
empirical bin frequencies match exactly between classes. Thus the sampled
posterior P(Y=1 given B=b) is one half on occupied bins. On common support,
Bayes' rule reduces the conditional posterior log-odds to the log ratio of
p(h given Y=1,b) and p(h given Y=0,b). Equal bin marginals do not remove
conditional information in h, enforce equality of continuous elapsed times, or
establish strictly decreasing history log-odds in the unmatched population.
That monotonicity is an empirical assumption in the brainstorm.

A strictly silence-decreasing score can still perfectly rank both classes:
s(h,t)=sigmoid(M(2h−1)−εt) with a sufficiently predictive binary covariate h
and bounded evaluated times separates the classes for large enough M. Its time
derivative at fixed h is negative. An executable example gives perfect pooled
ordering and small Brier error; increasing the separation margin makes the
error arbitrarily small. A universal incompatibility theorem, inevitable AP
collapse and inevitable Brier explosion therefore do not follow from matching.

A narrower statement is exact. Suppose scores strictly separate entire balanced
bins in a fixed order and let π_b=n_b/N. Cross-bin AUC is then one half, and

    AUC_pooled = 1/2 + sum_b π_b² (AUC_b − 1/2).

This explains how perfect within-bin ordering can coexist with degraded pooled
ordering in a constructed example. Monotonicity at fixed context alone does not
imply the dominance of entire bins required by this identity. More generally,
pooled AUC splits exactly into within-bin and cross-bin Cartesian components;
the actual diagnostic reports both without imposing bin dominance. A score
depending only on a balanced bin has AP and AUC one half, with
Brier = 1/4 + E[(s(B)−1/2)²]. This bin-only identity does not apply to a scorer
that also uses predictive covariates.

The [posthoc results](../results/Hard-Negative-Controls.md#posthoc-pooling-and-within-query-diagnostic)
show aggregation-sensitive model ordering, but do not remove all scalar/ordered
losses or identify their causes. Keep the original primary endpoint. These
checks support an operating-limit analysis, not a newly confirmed method.

The [Zhang and colleagues v1 preprint](https://arxiv.org/html/2610.04869v1),
Appendix A.1, already derives the negative-sampler density ratio and source-time
cancellation when negatives share source/time. Its historical/inductive protocol
can change the source and is not this repository's same-source history matching.
Appendix E.3 reports batch-averaged AP/AUC rather than whole-split pooled metrics.
The connection is useful prior art; calling our bin cancellation a direct new
opposing theorem or importing its reported gains is unjustified. This targeted
review does not establish priority for the elementary decomposition above.

### Candidate-transform theory and transition norms

The restricted exact two-phase condition and general Jacobian already belong
to the [gating method](../methods/Shortcut-Resistant-Gating.md). A full necessary
and sufficient theorem for arbitrary phase count, a varying transition network
and clamps has not been proved. A large Frobenius norm does not establish
reversal: the admissible two-phase matrix with a=−M, b=M has norm √2|M| but
a+b=0, so it satisfies the exact constant-map condition for all interior states.
A full-module autograd check with a large norm retains decreasing scores. No
retained checkpoint's violation frequency is inferred from norm summaries.

The monotone residual is a valid design route under fixed context, but is not
the unique possible repair. Restricting the constant two-phase map is another
route in that restricted setting. Neither the counterexample nor implementation
establishes new general theory or predictive utility.

A subsequent analytic attempt supplies a
[sufficient magnitude-and-drift bound](../methods/Shortcut-Resistant-Gating.md#a-sufficient-magnitude-and-drift-bound-for-the-ideal-many-phase-map)
for any phase count in the ideal interior map. It controls both transition
entries and their total time derivative along the path. A high-frequency,
small-magnitude counterexample shows why drift cannot be ignored for a varying
matrix. The derivation and executable derivative checks are retained in
`results/audit/simplex_silence_bound_v1/`. This is a conservative sufficient
condition, not an exact general threshold, a guarantee with numerical clamps,
or evidence that trained checkpoints satisfy it. No training constraint is
implemented by these checks. Prior-art and independent proof review remain
open before treating it as an original theoretical contribution.

The [entropic-update positioning](../methods/Shortcut-Resistant-Gating.md#established-exponential-update-form-and-the-narrower-question)
also records classical precedent for the normalized exponential map. The
candidate contribution must concern the scoped temporal-preservation analysis
and empirical limits, not invention of softmax reweighting. This targeted source
review neither proves nor disproves priority of the specific sufficient bound.

### Source-session transition attenuation

Time since the source's last action can be observed for a new dyad; that makes
the proposed clock relevant to ML-20M. However,
[Time-LSTM](https://www.ijcai.org/proceedings/2017/504) explicitly models intervals
between user actions. Its second variant constrains a time-gate weight to be
nonpositive to reduce the last consumed item's influence with elapsed time
while separately modeling longer-term interest. The claim that all earlier
decay methods use only a dyadic clock, and that source-session attenuation is
entirely new, is false. This is a targeted overlap, not exact model identity or
an exhaustive novelty search.

An additive stationary branch plus a candidate-varying attenuated transition
branch can change within-query ranking. A common additive source offset, or a
positive multiplier of the entire score, cannot. The count-screen contrast
compares different information; it does not show that a learned time attenuation
improves that transition control. Shrinking a signed transition logit also need
not decrease every candidate score: the derivative depends on its sign. Memory
attenuation and decline of each link score are different requirements.

No source-session attenuation fit is allocated. Any later comparison
must prospectively hold candidates, transition/popularity information,
normalization, budget and validation selection fixed, compare no decay, scalar
attenuation and a flexible interval control, and identify the actual clock.
That comparison is not claimed as executed here. Primary-source snapshots and
scoped review are retained with the pooling diagnostic. No manuscript export or
scientific admission follows from the brainstorming review.

A subsequent [CPU source-gap diagnostic](../results/ML20M-Development-Baselines.md#posthoc-source-session-inactivity-diagnostic)
now stratifies the original count-control reciprocal ranks using intervals fixed
from training gaps. The transition advantage decreases across the intervals,
while staying positive. Full independent clock and metric replay passes. This
supports considering a separate attenuation comparison after the cadence study,
but observational user/history differences prevent interpreting the trend as
a causal time effect. No attenuation model has been fit and the prior-art
boundary above remains unchanged.


## Follow-up on numerical ties in the pooling hypothesis

The [stable pre-clamp replay](../results/Hard-Negative-Controls.md#stable-pre-clamp-scalar-checkpoint-diagnostic)
now checks the missing ranking information at the scalar probability floor on
all original scalar checkpoints. Stable log scoring removes ties and recovers
some paired ordering, but does not recover the history-aware baseline on either
dataset. Pooled AP changes little. The result narrows the hypothesis further:
aggregation and clipping are relevant operating limits, but neither currently
explains away all scalar losses. This analysis does not estimate a retrained
model's performance or justify replacing the declared primary endpoint.


## Statistics prior art for matched-cohort ROC

[Janes and Pepe (2008)](https://doi.org/10.1111/j.1541-0420.2007.00823.x)
already study matching in classification-accuracy analysis and explain that
unadjusted summaries can be biased. Their publisher abstract and registered
metadata were inspected; the indexed first-page illustration also concerns
pooled versus covariate-specific ROC. [Janes and Pepe (2009)](https://doi.org/10.1093/biomet/asp002)
propose covariate-adjusted ROC estimation. The general matching/aggregation
issue is established statistics, not a new principle supplied by this project.
This targeted review does not establish the earliest such work or certify the
full journal proofs; exact access limits are retained in the audit record.

The repository diagnostics are different objects. Pair accuracy compares the
actual two candidates for each query. Same-bin Cartesian AUC compares every
positive with every negative inside a bin and weights its bin AUC by squared
bin counts. Query-weighted within-bin AP uses linear query counts. Neither
implemented diagnostic has been validated as the covariate-adjusted ROC estimator
in the cited papers. Do not call the paired metric adjusted AUC or transfer
literature guarantees to it.

The biomedical studies also distinguish a target population from a matched
sample. Here the negative candidates are generated and history-conditioned;
original pooled AP remains a valid declared endpoint for that conditional task.
A different conditional estimand does not by itself invalidate it or recover
natural event prevalence, whole-stream ranking or a population ROC. Any attempt
to estimate a broader target needs a separately specified sampling/weighting
model, rather than relabeling the posthoc diagnostics as a correction.

Keep the experimentally demonstrated model-order differences and numerical-floor
limits, with their posthoc status. Novelty must rest on a demonstrated distinction
in the gate analysis and temporal-ranking operating boundaries, not the generic
fact that matching and pooling interact. Metadata, the registered abstract and
explicit failed full-text routes are retained in
`results/audit/pooling_statistics_prior_art_v1/`. No model, protocol or primary
endpoint changes follow from this positioning review.


## Follow-up: impossibility, competing edges and invariant readouts

The supplied follow-up brainstorm is a hypothesis source, not evidence of
priority or an instruction to assert its conclusions. The existing
[simplex development](../methods/Monotone-Simplex-Development.md) remains a
fixed-context constructive result with explicit restrictions.

### Limits of the proposed impossibility theorem

A partial time derivative certificate does not imply a full chronological
certificate when context changes. The correct chain rule includes
$\partial_\tau f\,\dot\tau+\nabla_cf\cdot\dot c$; the first factor is not
necessarily unity for a normalized clock. This is a useful scope distinction,
not proof of universal incompatibility or of an empirically observed BCE
mechanism. Fixed-context certificates still have their stated mathematical
meaning. Real trajectories also have discrete history jumps.

Monotonicity plus a global Lipschitz constant gives an interval difference
bound proportional to interval width, not an exponential-in-age bound.
An exponential kernel supplies a localized exponential derivative bound, but
only for that component. For example,
$s_i=b_i-\alpha\tau_i$ with $\alpha>0$ is strictly decreasing in its explicit
clock. If both candidates have the same clock value, $s_+-s_-=b_+-b_-$:
any perfectly discriminating baseline remains perfectly ordered. This is an
admissible special case of same-bin matching, so universal ranking collapse
under that sampling condition is false. An exponentially small temporal
contribution is likewise not proof that the complete score cannot discriminate.
No theorem that all monotone repairs reduce to the linear redundant variant has
been established; the implemented nonlinear map is not the same linear case.
Finite negative benchmark outcomes do not prove a function-class impossibility.

### Competing-destination hypothesis

Distinguishing source inactivity from activity redirected to other destinations
is the strongest candidate for a new empirical question in this brainstorm.
Its generic ingredients have precedents:
[Neural Hawkes](https://www.cs.jhu.edu/~jason/papers/mei%2Beisner.nips17.pdf)
explicitly models cross-type inhibition and excitation, with consumer
substitution/complementarity examples in its introduction;
[DyRep](https://openreview.net/pdf?id=HyePrhR5KX) models graph-structural
interaction dynamics; [Butts's relational event framework](https://doi.org/10.1111/j.1467-9531.2008.00203.x)
already describes recency, persistence, structural dependencies and participation
shifts in its abstract. These inspected passages reject blanket claims that
previous models are independent one-dimensional dyads. They do not establish
that any source has exactly the proposed semantic displacement operator or
resolve priority of a narrower method.

The proposed linear event jump can make a nonnegative state negative if its
jump coefficient exceeds unity. A nonnegative competitive kernel with update
$z^+=z^-\exp(-\gamma\kappa)$ avoids that problem for pure inhibition.
Signed complementary effects require a separate bounded state/intensity design.
Neither update identifies real edge death from positive-only event data.
Semantic similarity is not necessarily substitution; attention conservation is
a modeling assumption, not an observed law in these corpora. Cross-edge coupling
alone also does not define a simplicial model.

A useful future screen must compare candidate-specific semantic competition
against source event count, pair count/recency, and a flexible history-aware
control receiving the identical new inputs. Query-common competition offsets
can cancel in pairwise ranking. All features and semantic embeddings must be
constructed from strict-past or training-only information. Compare learned,
constant and shuffled semantic kernels before attributing an effect to semantic
displacement. Candidate bins are approximate matching, not equality of histories.
Existing support restrictions and archived negative results remain unchanged.
No diagnostic, new fit, runtime estimate or GPU allocation is asserted here.

### Readout invariance is distinct from faithful explanation

An auxiliary readout unused by a frozen predictor leaves its score invariant
under interventions confined to that readout. Stop-gradient alone does not
prevent forward dependence, shared mutable state or training changes through
other routes, and does not imply orthogonality, causal semantics or explanation
faithfulness. A five-dimensional intermediate representation need not lose
accuracy if it is sufficient for the target; dimensionality alone cannot prove
unavoidable interference. [Concept Embedding Models](https://papers.nips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html)
already studies accuracy/explanation tradeoffs and reports competitive accuracy;
the inspected abstract is a positioning boundary, not a full architecture review.
The actual [split decision](../decisions/0001-separate-link-prediction-and-lifecycle-readout.md)
concerns repository/scientific ownership separation, not a proof of detached
readout superiority. Existing phase labels and anomaly semantics remain unvalidated.

The cited [EPT-X preprint](https://arxiv.org/abs/2610.14948) was checked at the
abstract/metadata level: it combines a history-dependent Erlang renewal channel
with a Hawkes backbone. That inspection is insufficient to assert absence of a
specific competing-destination mechanism from the full paper.

### Recommendation

Retain the current analysis article as scoped guarantees, counterexamples and
matched-history operating limits; do not rename it a universal impossibility
result. If pursuing an additional mechanism, investigate whether strictly-past
candidate-specific semantic competition contains conditional ranking information
beyond shared-history and source-activity controls. This is a research question,
not an established novelty or promised improvement. The user requested an
assessment; this review does not seal a new data protocol, start a fit, change
primary endpoints, export a manuscript or admit a claim.


## Direct competition precedents found during the pilot

[Gelardi, Le Bail, Barrat and Claidiere (2021)](https://doi.org/10.1098/rspb.2021.1164),
“From temporal network data to the dynamics of social relationships,” directly
models interaction-driven weakening of other ties under limited attention.
The author-hosted full text's introduction and framework equations
`(2.1)`–`(2.2)` were inspected: inactivity alone leaves weights unchanged, while
interactions reinforce the participating tie and weaken other ties. This is a
closer conceptual precedent than a generic Markov or neural point-process model.
Cross-edge displacement and the offline-versus-redirected-attention distinction
therefore cannot be claimed as new principles of this project.

[Valera and Gomez-Rodriguez](https://people.mpi-sws.org/~manuelgr/pubs/competing-icdm15.pdf),
“Modeling Adoption and Usage of Competing Products,” already models continuous-time
recurrent product use with recency, competition and positive/negative influence.
The abstract and introduction were inspected; the complete derivation has not
been audited here. Local source copies/access outcomes and scoped comparison
are retained in `results/audit/competition_literature_v1/`.

The initial [competition pilot](../methods/Competing-Destination-Development.md)
was already fixed before this closer literature pass. Its design is unchanged.
The narrower open question is the utility of strict-past behavioral-profile
weighted exposure under matched-history ranking, relative to equal-information
flexible controls and an activity-only explicit inhibitory pathway. This
combination has not received priority clearance and no positive outcome is
presumed. Behavioral co-interaction similarity, especially with a small hash
budget, is not a semantic substitution label; the source's own past interactions
contribute to those profiles and remain a possible ordinary-history signal.


## Closest inspected preservation-related constructions

The targeted review now finds additional overlap beyond generic monotone networks.
The [MoST publisher preview](https://www.sciencedirect.com/science/article/pii/S0893608026006143)
discusses competitive softmax normalization, a monotonicity/expressivity tradeoff,
and a positive-kernel architecture that avoids competitive normalization. Its
certificate is for set-inclusion monotonicity and rank-dependent set aggregation.
The current restricted temporal result instead retains normalized reweighting and
certifies a specified readout along absorbing phase decay. This is a candidate
distinction, not a proof of originality. The indexed publisher preview was
inspected; direct full-page access was forbidden and complete proofs were not
reviewed. The publisher lists a future issue month relative to this task date;
first online priority is unverified and must not be inferred from that issue date.

An older [institution-hosted weighting draft](https://ca3-uninova.org/docs/2003-BJ-FSS.pdf)
already studies restrictions that preserve monotonicity under normalized
input-dependent weighting, including a linear generating family. Indexed primary
passages and its stated linear-family theorem were inspected; direct PDF access
timed out. Its satisfaction values and their weights vary together, whereas the
current restricted proposition fixes readout gaps and varies simplex mass under
absorbing decay. The institutional bibliography uses a different published title
for the corresponding journal/page identity; final citation identity still needs
reconciliation. Neither the normalized ratio nor the generic need to constrain
its derivative is an invention here. Exact access scope and metadata ambiguity
are retained in `results/audit/restricted_preservation_prior_art_v1/review.json`.

The [readout-aware condition](../methods/Readout-Aware-Entropic-Preservation.md)
and its constructive active-mass redistribution example remain a restricted
problem-specific candidate contribution. The [full validation diagnostic](../results/Full-Validation-Scored-Paths.md)
adds learned-checkpoint coverage distinct from abstract normalization examples.
Do not claim the first normalization failure, first monotonic architecture,
first attention competition model or an exhaustive literature exclusion.

## Repeat-consumption and metadata-inhibition precedents for Last.fm

This targeted primary-source pass was performed while the already sealed
[Last.fm comparison](../results/LastFM-Recorded-Artist-Pilot.md) was running.
It changes no model, candidate, seed, budget or primary metric.

[Anderson and colleagues, WWW 2014](https://cs.stanford.edu/~ashton/pubs/repeat-consumption-www2014.pdf)
already formulate choice among previously consumed items, with event-index
recency and quality-weighted competition. Their inspected introduction and
recency/quality construction account for intervening consumptions and competing
alternatives. Thus neither a consumption-count clock nor conditional repeat
ranking is a new problem principle here. The current artist-dose variation is
not identical to total event count, but that fact is not priority clearance.

[RepeatNet, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4408)
already learns repeat/explore behavior and recommends from consumed history.
Its repeat/explore decoder and mode construction are direct prior art against
claiming learned repeat recommendation as new. Its session recommendation task
and catalog metrics differ from the current matched recurrent-candidate task;
published numbers are not imported as comparable baselines.

[ReCODE, SIGIR 2024](https://arxiv.org/html/2405.16550v1) explicitly combines
base preference with a dynamic repeat component and uses neural ODEs for pair
repeat dynamics. The additive prediction equation `(2)` is a direct predecessor
to a generic base-plus-repeat branch narrative. A closed-form absorbing basis
and restricted readout certificate require their own precise comparison; the
presence of a dynamic branch, pair state or ODE alone is not a new contribution.

[Huang and colleagues' SDIL/TPNE preprint](https://arxiv.org/html/2410.22013v1)
models temporal positive/negative item interaction signals, including relation
indicators for substitute items and lag-dependent kernels. The inspected
sections `4.1`–`4.2.2`, equations `(8)`–`(12)`, directly preclude a first
metadata-related inhibition claim. Its stated lag-dependent construction differs
from the current frozen-reference summed dose, which cannot distinguish competitor
event positions within an interval at fixed total dose. This is a restricted
architecture distinction, not an originality proof. Its sign notation, intensity
positivity and implementation have not been independently audited here.

All four primary source files were retrieved and hashed. This is a targeted
method/positioning review, not exhaustive priority clearance or implementation
reproduction. The narrower open contribution is the controlled evidence for the
restricted construction and the existing readout/operating-limit analysis.
Even a positive Last.fm primary would not establish state of the art over these
author models, causal substitution or novel generic temporal inhibition.
`LCR-E-LASTFM-PRIOR-001` maps exact sources and reviewed scope in
`results/audit/lastfm_prior_art_review_v1/` and its retrieval index. No working
manuscript export or scientific admission follows from this source review.

## Decision after the recorded-artist controlled outcome

The complete [Last.fm comparison](../results/LastFM-Recorded-Artist-Pilot.md),
`LCR-E-LASTFM-MODEL-001`, does not establish a predictive advantage for the
specified constant-reset phase branch over its flexible same-information control.
Replacing hashed histories with recorded artists did not rescue this version
in the declared setting; differences between datasets do not isolate a cause.
Feature availability, initialization verification and execution correctness must
not be advertised as predictive or novelty contributions.

The strongest remaining candidate account combines the
[restricted readout preservation characterization](../methods/Readout-Aware-Entropic-Preservation.md),
[complete learned-score-path diagnostics](../results/Full-Validation-Scored-Paths.md)
and [repair cost/control evidence](../results/Readout-Repair-Validation.md).
This is a concrete analysis route, but its exact priority/proof review and final
scientific admission remain open. It does not fulfill a stronger-new-predictor
objective by relabeling a negative pilot as an architectural success.

The next strengthening decision must resolve the closest normalized-weighting
precedents against the precise assumptions and necessity direction of the
restricted result. A future architecture direction also needs a demonstrable
functional distinction from the current aggregate time/dose family and the
repeat-dynamics predecessors, followed by a new prospective comparison. The
current diagonal operator's order invariance is already documented in its
[method owner](../methods/Event-Driven-Competitive-Lifecycle.md), not a new
finding from these scores. No new runtime, tuned retry or unseen test evaluation
is justified merely by the negative primary.

## Expanded access to weighted-aggregation priors

The institutional draft is now retrieved in full. Its actual header is
“Aggregation with generalized mixture operators using weighting functions,”
matching [the publisher record](https://www.sciencedirect.com/science/article/pii/S0165011402004311),
with DOI `10.1016/S0165-0114(02)00431-1`. The earlier tentative draft-title label
was wrong; a secondary `(00)` DOI variant is also not the primary registration.
Sections three through five and the linear/quadratic constraint proofs were
inspected. These characterize chosen sufficient restrictions on generating
functions; they are not a general exact characterization of every monotone
normalized operator. Fixed satisfaction-dependent weighting differs from the
present fixed-readout absorbing-mass path; this distinction does not clear priority.

[Mesiar and Spirkova](https://www.kybernetika.cz/content/2006/2/151/paper.pdf)
define mixture/generalized mixture operators and give derivative restrictions
and sufficient monotonicity conditions in section three. The inspected equations
and propositions already establish both normalization failures and refinements
of sufficient restrictions. Generic monotonicity repair remains prior work.

The [fixed-rate specialization](../methods/Fixed-Rate-Readout-Certificate.md)
uses standard finite LP support and binary copositivity ideas, with its complete
model-specific proof written explicitly. It is not a new general copositivity
algorithm or moment-support theorem. Its narrower candidate distinction is a
sharp all-state fixed-rate/conic certificate and a constructive quantifier tradeoff.
No exhaustive search or empirical predictor benefit follows. Exact source bytes,
access scope, primary DOI registration and theorem checks are retained under
`LCR-E-FIXED-RATE-TILT-001`.


## Spectral-ratio hypothesis comparison

The [primary-source ratio comparison](Spectral-Ratio-Prior-Position.md) now gives the exact exponential-ratio representation and rational safe examples outside the inspected positive-coefficient or unimodal-quotient predicates. Laplace/exponential-ratio monotonicity is established prior work. This identifies a specific state-uniform absorbing-family candidate; it does not rule out an equivalent specialization of earlier theory or establish priority. No predictive endpoint changes.

---
title: Journal Narrative
status: working narrative with strengthened restricted theory and scoped learned-path evidence
last_updated: 2026-10-16
paper_source: false
---

# Monotone Temporal Gating for Link Prediction: Guarantees and Limits under History-Matched Evaluation

This page owns the connective article prose and assembly decisions. Definitions,
proofs and numerical tables remain with the linked method and result owners.
The text is a review draft, not an admitted manuscript or submission snapshot.
Both release pointers remain `UNRELEASED`.

## Contribution decision after the frozen next-month check

The [February holdout](../results/LastFM-February-Holdout.md) does not confirm
the January-discovered operating-region gain. The paper cannot use that
localization to rescue the negative predictive comparisons. Its strongest
current working structure is a focused analysis of guarantees and their limits:

- An exact [specified-rate preservation condition](../methods/Fixed-Rate-Readout-Certificate.md)
  distinguishes a guarantee for given rates or a finite rate cone from the
  stronger requirement over every positive rate. Its proof and strict-separation
  examples are explicit; generic copositivity and support arguments are prior
  mathematics, and priority remains unresolved.
- The [inverse-tilt geometry](../methods/Inverse-Tilt-Projection.md) supplies a
  same-metric convex repair. Its [retained-model application](../results/Fixed-Rate-Repair-Validation.md)
  shows that sharper certification can avoid extra deformation of an already
  frozen-reference comparator. This does not preserve the original dynamic
  predictor or establish better ranking.
- Complete scored-path diagnostics, matched-information controls and the
  frozen next-month check distinguish structural guarantees, added information,
  and predictive utility. Negative results are part of this account, rather
  than separate favorable performance claims.

The [exact readable Last.fm branch](../results/LastFM-Mechanistic-Readout.md)
supports a limited mechanism inspection: its response components and direct
influence can be reconstructed. It does not explain the dominant learned base,
identify semantic lifecycle phases or establish improved whole-model
interpretability. It should support the scope discussion, rather than serve as
an independent explainability headline. These are working article candidates,
not admitted claims; focused prior-art assessment and external proof review
remain necessary before a publication contribution is settled.

## Abstract

Monotone temporal gating promises a decreasing response to silence, but this
property depends on the complete scoring path. We analyze temporal link
predictors that combine a decaying simplex state, a candidate transformation,
and a learned readout. We derive the pure-flow condition, exhibit score reversal
after transformation, and give a necessary-and-sufficient preservation condition
for a restricted constant-reweighting family over all positive decay rates.
A constructive parameterization permits active-phase redistribution while
preserving the chosen readout, separating readout monotonicity from coordinatewise
decline. A complete supported-validation diagnostic of archived models shows frequent
scored-path reversal in Reddit checkpoints despite a passing pure-flow check;
Wikipedia does not cross the declared violation threshold. These are fixed-context model paths, not chronological or
causal relationship dynamics. A no-refit repair removes detected reversals but lowers Reddit AP;
controls distinguish reference freezing from the smaller incremental readout
constraint. Matched-history predictive comparisons show
dataset-dependent effects of ordered gating. Separate validation development
distinguishes added cadence information from explicit clock replacement, whose
primary contrast is negative at every recorded seed on both corpora.
An inhibitory competing-destination pilot and an exact-state follow-up supply
no mean primary advantage over their respective flexible controls. Metric reconstruction and native checkpoint
replay support the recorded observations. Together, the analysis separates the
response guaranteed by an architecture from its predictive value under controlled
information access. Neither the restricted repair nor the diagnostics establish
phase semantics, general predictive superiority, or priority over established monotonic and
normalized-reweighting methods.

Keywords: temporal link prediction; monotonicity; negative sampling; historical
features; conditional ranking; reproducibility.

## Introduction

An event stream records interactions, while a temporal link predictor ranks
candidate destinations at a query time. A long interval without a recorded
interaction can motivate a decreasing model response. In the present task,
however, a sampled negative is a contrastive candidate, not an observed label
that a relationship has ended. The prediction target and the sampling rule must
therefore accompany any interpretation of a temporal gate. A sampled classifier
does not acquire a time-to-event likelihood merely by multiplying its score by
a decaying factor.

There is also a distinction between imposing a response on an intermediate
state and imposing it on the final prediction. An ordered linear readout of a
decaying phase distribution can be non-increasing at fixed affinity and initial
state. A subsequent transformation can reverse that response. Changes to the
encoder or contextual features can create further dependencies on time. These
paths must be examined before an internal trajectory diagnostic is interpreted
as a guarantee for the deployed predictor.

Predictive value raises a separate question. A history-aware comparison can
already use recurrence, elapsed time and neighborhood information. Adding a
constrained gate changes the model family, but a gain over a weaker model with
less information does not isolate the constraint's value. Likewise, matching
positive and negative candidates in coarse historical bins changes the
comparison population without making their continuous histories identical.
Performance must be interpreted on that declared population.

This article examines three questions. What precisely is guaranteed by the
silence flow, and which scoring operations preserve it? Does an ordered
lifecycle readout contribute beyond common explicit history on matched
candidates? Does changing an explicit attenuation clock help when pair-relative
recency is already available to the contextual predictor? The empirical answer
is mixed for the ordered readout and negative for the primary clock-replacement
comparison under the recorded development settings.

The strengthened candidate contribution has two concrete additions. An exact
restricted characterization certifies the readout without requiring every active
phase to decline, and a constructive witness demonstrates the distinction.
An archived-model diagnostic shows that a passing pure-flow check can coexist
with scored-path reversal in learned Reddit checkpoints under the fixed-context
intervention on the supported validation cohort, while Wikipedia does not cross
the declared threshold.
Together with shared-history predictive controls, these results separate
structural guarantees, diagnostic coverage and predictive utility. They do not
invent a pair-normalized clock, normalized exponential update, linear-fractional
function or generic cross-edge attention competition. Priority and proof review
of the specific characterization remain open.

A supplementary [competition pilot](../results/Competing-Destination-Pilot.md)
now separates candidate-specific profile information from an imposed inhibitory
residual. Its negative primary contrast and lack of advantage over shuffled
profile alignment are retained. It is not the main new predictor contribution.

## Relation to established constructions

The [closest benchmark comparison](../questions/Contribution-Gap-Review.md#closest-benchmark-comparison-after-cadence-completion)
narrows the proposed contribution. Its existing cancellation and query-ranking
explanation must be acknowledged, not claimed as a discovery here.

Pair-relative recency has a direct predecessor in
[Navarro and colleagues](https://link.springer.com/article/10.1140/epjds/s13688-017-0127-3).
The present online construction adds declared handling of sparse histories and
uses a sampled-ranking target. Those choices motivate a controlled evaluation;
they do not establish a newly invented clock.

The normalized exponential candidate map also has an established entropic
optimization form, as discussed by
[Beck and Teboulle](https://www.tau.ac.il/~becka/3.pdf).
Our temporal-preservation question concerns a path through changing states and
transformations. A frozen-step algebraic correspondence does not transfer a
convex optimization convergence guarantee to an arbitrary learned transition.

[Omi, Ueda and Aihara](https://proceedings.neurips.cc/paper/2019/hash/39e4973ba3321b80f37d9b55f63ed8b8-Abstract.html)
provide direct prior work connecting monotone neural functions to point-process
modeling. The gate analyzed here is evaluated as a sampled predictor; it does
not inherit a cumulative-intensity likelihood. Likewise, the statistics of
[matched classification evaluation](https://doi.org/10.1111/j.1541-0420.2007.00823.x)
predate this study. The implemented paired and within-bin diagnostics are not
presented as a new adjusted-ROC estimator.

The [technical source map](../references/Technical-Source-Map.md) owns source
identity, access limitations and the remaining literature. In particular, recent
preprints must retain their actual version and publication status in final
reference preparation. No exhaustive priority claim follows from this targeted
review.

## Mathematical account to assemble

The [gating method](../methods/Shortcut-Resistant-Gating.md) owns the definitions
and derivations. The main article should present the exact pure-flow condition,
its assumptions and a constructive failure after a candidate transformation.
Ordering every active weight above the dormant weight suffices; ordering active
weights among themselves is unnecessary. A large transition norm alone does
not diagnose reversal in a trained checkpoint.

The nested residual supplies exact baseline recovery at zero amplitude and a
nonpositive time partial derivative at fixed context. It is a bounded logit
correction, not a proper general survival distribution. Context contains temporal
statistics, so its fixed-context result must not be restated as monotonicity of
the complete predictor along actual changes in query time.

The exact restricted two-phase threshold and the conservative many-phase
magnitude-and-drift bound belong in the technical appendix with their full
assumptions. The latter excludes numerical clamps and changing affinity or
initial encoder state. Executable checks support the derivation but do not
replace proof review or show that a trained transition satisfies the bound.
The many-phase bound is a working analytic result, not the headline claim of
an established new general theorem.

The [readout-aware theorem](../methods/Readout-Aware-Entropic-Preservation.md)
now adds a necessary-and-sufficient result for constant normalized positive
reweighting and all positive absorbing-rate vectors. Its proof reduces partial
readout derivatives to affine numerators on the active simplex. The feasible
constructor and strict-separation witness belong with that exact statement;
they do not certify arbitrary dynamic transformations or changing contexts.
The standard pointwise softmax derivative supports diagnostics but is not itself
an invented theorem.

The [initial learned-path audit](../results/Scored-Path-Silence-Diagnostic.md)
and its [complete supported-validation expansion](../results/Full-Validation-Scored-Paths.md)
belong beside the constructive counterexample. Archived source, native observed
input reconstruction, dual-mode derivative checks and independent flag
reconstruction support the fixed-context finite-grid observation. The expansion
was fixed after anchor outcomes were seen, so it is not independent confirmation.
It does not represent chronological physical silence or external causal effects.
The initial failed whole-backbone precision conversion is preserved and corrected
by retaining native anchors.

## Recorded artist follow-up and contribution boundary

The [Last.fm recorded-artist follow-up](../results/LastFM-Recorded-Artist-Pilot.md)
replaces behavioral hashes with strictly prior observed artist identities and
matches candidates by prior recurrence and elapsed-time bins. Its prospective
same-input, equal-parameter primary supplies no mean phase advantage; the phase
arm is below its common history-only model at every seed. Recorded artist identity does not rescue this version in the declared Last.fm
setting. The cross-dataset comparison does not isolate the reason for the
failure or establish universal inferiority.
No positive secondary can replace the fixed primary. Development selection,
conditional sampled ranking, environment drift and scoped replay remain explicit.

The [closer source review](../questions/Contribution-Gap-Review.md#repeat-consumption-and-metadata-inhibition-precedents-for-lastfm)
also disallows broad first-event-clock, first dynamic-repeat-residual and first
metadata-inhibition wording. The present article must earn its contribution
through its restricted preservation characterization and model-path diagnostics,
with controlled predictive outcomes as operating evidence. A new superior
competition predictor is not an established contribution of this version.

## Local benefit and explanation coverage

The [complete posthoc history strata](../results/LastFM-History-Strata.md)
identify a small same-sign phase/MLP difference in one fixed silence/dose
intersection. All other cohorts, including consistent losses, remain reported.
Validation selection and posthoc localization prevent a confirmed operating-region
claim and leave the original negative aggregate primary unchanged.

The [retained branch decomposition](../results/LastFM-Mechanistic-Readout.md)
exposes an exact additive finite-exponential account and a global fixed-base
probability-influence bound. Its direct effect is small in these checkpoints;
this says nothing about its full training influence. The four component labels
are permutation-nonidentifiable and have no verified lifecycle meaning. The
common encoder remains opaque. Neither algebraic decomposability nor a readable
small branch establishes improved human or whole-model interpretability.

## Sharper fixed-rate characterization

The [fixed-rate analysis](../methods/Fixed-Rate-Readout-Certificate.md) separates
uniformity over every decay vector from the rates actually specified. A
structured derivative admits a two-support reduction and an exact pairwise
condition, with a finite rate-cone extension. A co-ordered-rate corollary permits
arbitrary constant tilts and has an untrained differentiable constructor, including
a strict-separation witness beyond the earlier all-rate feasible set. This
strengthens the restricted theoretical account rather than the empirical accuracy
claim. Its standard mathematical tools, priority/proof limits and corrected
boundary-gradient implementation issue must remain explicit in final assembly.

## Exact repair geometry and unnecessary uniform-rate deformation

The [inverse-tilt specialization](../methods/Inverse-Tilt-Projection.md) converts
the sharp feasible set into nonnegative linear lower constraints and gives a
unique same-coordinate Euclidean repair with no arbitrary positivity floor.
The metric-specific distance comparison is exact; no AP bound follows from it.
A [new complete no-refit comparison](../results/Fixed-Rate-Repair-Validation.md)
finds that the retained frozen-reference gradient comparators already satisfy the
specified-rate certificate at every candidate. Stronger all-rate bounds change
many Reddit configurations unnecessarily for those rates. The prespecified AP
comparison does not show a mean advantage, and the original dynamic-to-frozen
loss persists. This supplies a conditional certification observation beyond the
parent finite-grid check, not a superior predictor or general priority claim.

## Empirical argument

The hard-control study is the main predictive comparison. Its primary endpoint
is pooled test average precision (AP) under the declared matched historical candidate protocol.
All history-aware variants receive the same explicit statistics and affinity
architecture. Their parameter counts differ, and the lifecycle variant has an
additional regularizer; the comparison is not a constraint-only causal ablation.
The neighbor-only baseline serves as an information-access control.

The scalar control here applies a multiplicative exponential factor. The later
cadence study uses baseline-nesting additive logit residuals, including scalar
and mixture versions. These are different model families; their absolute scores
and behavior must not be conflated across studies.

The [hard-control outcomes](../results/Hard-Negative-Controls.md) show that the
ordered lifecycle variant loses to the history-aware controls on Wikipedia.
On Reddit its mean AP is higher than the history baseline, but its paired gain
over the flexible gate is not uniform across seeds. These results support a
limited, dataset-dependent operating account. They do not establish that
temporal constraints are always harmful, or that a lifecycle gate is generally
superior.

The cadence study asks a different question using validation selection and
evaluation. The masked control and exposed-feature arm have the same input
dimension and within-family capacity. The absolute and relative clock arms
receive exactly the same exposed features. Relative freshness is included in
both, so the primary contrast changes where the explicit attenuation uses time,
not whether the predictor can access relative-time information.

The [complete cadence outcomes](../results/Pair-Cadence-Development.md) retain
negative primary scalar differences on both datasets at every declared seed.
The added contextual bundle improves the recorded scalar and mixture variants,
with broader feature gains on Reddit. A bundle comparison does not identify
which channel helps. Neither the positive secondary effects nor selected
mixture seeds replace the primary clock comparison. The small negative clock
effects also do not prove equivalence or universal inferiority.

Exact tables, denominators, seed values, selection rules and metric definitions
are imported from the result owners during final assembly. Test results from
the hard study and validation-selected development results from cadence must
remain separately labeled. They are not pooled into a single generalization
estimate.

## Supplementary explanations and boundaries

The [pooling diagnostic](../results/Hard-Negative-Controls.md#posthoc-pooling-and-within-query-diagnostic)
changes the comparison population to inspect within-query ordering and within-bin
performance. It shows aggregation-sensitive model ordering but leaves the original scalar multiplier's
deficits. The [stable-log replay](../results/Hard-Negative-Controls.md#stable-pre-clamp-scalar-checkpoint-diagnostic)
recovers some ordering hidden by probability-floor ties, without recovering the
history baseline. These checks limit two proposed explanations; they do not
identify the cause of the remaining training or ranking deficit. The original
primary pooled endpoint remains unchanged.

Matched bins alone do not imply inevitable pooled ranking failure. A predictive
context can separate labels while a score still decreases with silence at fixed
context. The narrower bin-separation identity requires its stated extra
assumptions. Conditional information, actual query pairs and same-bin Cartesian
comparisons must remain distinct in the article.

The [ML-20M boundary case](../results/ML20M-Development-Baselines.md) examines
supplied positive pairs without observed dyadic recurrence. The count controls
and bounded author-model fit share generated validation candidates. Their
different optimization and selection budgets, single candidate draw and limited
training-seed coverage preclude an equal-search-budget or official-test claim.
The CRAFT run's cap and retained partial epoch remain visible.

A source-action clock remains observable when a dyadic cadence is unavailable.
The posthoc source-gap diagnostic finds smaller transition advantages in longer
inactivity intervals, but user and history composition can differ between those
intervals. No source-session gate is fitted. Action-interval gating has direct
precedent in [Time-LSTM](https://www.ijcai.org/proceedings/2017/504), so this
observation is a future evaluation question with known prior art.

## Discussion and conclusion

A designed response, information access and ranking utility are distinct
properties. A valid intermediate-state guarantee can be lost on a scored path.
A guarantee that holds at fixed context can also coexist with a changing full
prediction when that context depends on time. These distinctions explain why a
trajectory consistency result cannot by itself validate semantic phases or
predictive usefulness.

The experiments show the value of separating feature information from an
explicit temporal response. The cadence primary comparison already exposes the
relative-time feature to both models. It therefore supplies evidence about the
specified clock replacement under those settings, while leaving other uses of
that information open. Adding history information and constraining its use
must be reported as separate comparisons.

The empirical scope remains narrow. Hard matching restricts the recurrent
cohort and leaves unsupported events outside supervision. Fixed budgets do not
establish convergence. Descriptive seed dispersion is not a significance test.
Cadence uses previously inspected validation populations, while supplementary
diagnostics are posthoc. Candidate sampling also defines the meaning of the
reported Brier score. No population relationship-survival probability, latent
phase semantics or external causal effect is identified.

The reproducibility records verify source identity, histories, candidate
information, metric aggregation and retained checkpoint inference within declared
cross-device tolerances. They do not reproduce optimization in a fresh dependency
environment. Local retrievability is distinct from public artifact access and
from permission to redistribute data or code.

The warranted conclusion is that a monotonicity guarantee must be stated on its
actual scored path and evaluated separately from predictive value. In the
recorded studies, ordered gating has mixed predictive effects, and replacing an
explicit absolute clock with pair-relative time gives no observed primary AP
gain. The analysis retains positive feature effects and negative clock outcomes
under the same declared evidence boundaries.

## Assembly map and author review

| Article component | Canonical source | Placement |
|---|---|---|
| Connective prose and abstract | This page | Main text |
| Prediction target and positive-dependent sampling | [Target owner](../methods/Prediction-Target-and-Sampling.md) | Main setup; detailed derivation in appendix |
| Exact flow condition, scored-path counterexample and nested residual | [Method owner](../methods/Shortcut-Resistant-Gating.md) | Main theory and method |
| Main matched-history results and support | [Hard controls](../results/Hard-Negative-Controls.md) | Main empirical table; auxiliary diagnostics in supplement |
| Primary cadence contrast, secondary feature effects and support | [Cadence owner](../results/Pair-Cadence-Development.md) | Main development analysis, clearly separated from test results |
| Many-phase sufficient bound | [Method derivation](../methods/Shortcut-Resistant-Gating.md#a-sufficient-magnitude-and-drift-bound-for-the-ideal-many-phase-map) | Technical appendix, with working-result scope |
| Synthetic capacity checks and first-time boundary | [Synthetic owner](../results/Synthetic-Mechanisms.md), [ML-20M owner](../results/ML20M-Development-Baselines.md) | Supplementary context |
| Sources and disclosure/retrieval limitations | [Source map](../references/Technical-Source-Map.md), [reproducibility contract](../REPRODUCIBILITY.md) | Related work, availability and limitations |

Do not import the earlier unmatched GraphMixer number as a comparable main
baseline, or retain confounded easy-negative gains as evidence that the gate
alone improves performance. Exact supplementary selection should support the
three stated questions, without converting every audit into a separate claimed
contribution.

Author review must decide whether this focused analysis adds sufficient insight
beyond the inspected benchmark and statistics literature. The earlier Word draft
already supplies an author order, affiliations and correspondence for draft use.
Funding, competing interests and source ownership remain unrecorded; they must
not be filled with an invented negative declaration. Actual artifact access and
the final AI-use disclosure also require factual author review. Public availability of the new source and
evidence has not been verified. The [export contract](Paper-Export-Contract.md) and publication gate
remain in force; writing this narrative does not admit claims or authorize a
Word export, fee, journal contact or submission.

## Restricted repair and ranking distortion

The [nearest-matrix repair](../methods/Readout-Repair-Projection.md) applies the
restricted theorem without fitting. Its [complete comparison](../results/Readout-Repair-Validation.md)
retains both shape and ranking endpoints. The repair removes threshold
violations but lowers Reddit AP relative to the original dynamic scorer. Most
of that observed decrease appears in the constant-reference control before
applying the readout bound. The final bound's incremental change relative to
the unconstrained gradient family is much smaller. This is a controlled
operational contrast, not a universal causal price of monotonicity.

Readout-specific and coordinatewise repairs share the same matrix-distance
objective. Their comparison supports retaining only the constraint required
by the specified readout, with mixed seed-level ranking effects. The original
negative primary remains central; it is not replaced by the favorable restricted-
family secondary comparison. No new overall superiority claim is made.

## Exact-state competition follow-up

After the earlier residual pilot, a [separate exact-state comparison](../results/Exact-State-Competition-Pilot.md)
fixes an own-edge reset and tests independent clock/event rates against a
same-information, equal-parameter small MLP. The primary contrast is negative on average. The specified exact-state phase residual does not beat its equal-parameter dose MLP on the declared primary endpoint. Negative and positive seed outcomes remain in the result rather than being replaced by a secondary contrast.
Its role is a controlled follow-up, not a replacement for earlier negative
experiments or a claim that a phase basis supplies new historical information.
The physical-time and event-exposure sufficient statistics must be available
to the comparator. Generic competing-tie weakening and exponential phase
responses remain prior ingredients; priority and general utility are not
established by this bounded comparison.

The exact-state follow-up additionally distinguishes predictive feature effects
from a proposed mechanism: exposure helps the flexible model relative to an
activity mask, while reference-coordinate alignment fails its directional
control. Extra phase-specific rates supply no observed benefit over the
proportional clock in this fixed budget. These are separate outcomes, rather
than interchangeable evidence for a new interaction mechanism.

## Competition proxy anatomy and construct validity

A [posthoc full-validation input decomposition](../results/Competition-Signal-Anatomy.md)
shows that the hashed exposure mainly contains the querying source's own
overlap and cross-ID collisions in the supported Wikipedia population. This
clarifies what the prior added-feature observation measures: a behavior-derived
proxy, not demonstrated semantic substitution. A coordinate shuffle also retains
a nonnegative common background term, so the control does not erase all input
information. These exact input observations do not causally explain AP or
validate a replacement. Independently supplied content and adequate recurrence
are requirements for a next mechanism study, not facts supplied by the current
experiments.

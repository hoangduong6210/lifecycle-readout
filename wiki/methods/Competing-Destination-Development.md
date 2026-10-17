---
title: Competing Destination Development
status: implemented with complete negative-primary pilot; no admitted claim
last_updated: 2026-10-16
paper_source: false
---

# Question and scope

The user selected cross-destination competitive displacement after the
[brainstorm review](../questions/Contribution-Gap-Review.md#follow-up-impossibility-competing-edges-and-invariant-readouts).
The initial question is whether a candidate-specific summary of competing
source events adds ranking information beyond pair history, source activity and
availability controls, and whether a constrained inhibitory residual improves
on a flexible residual receiving exactly the same features.

`LCR-P-COMPETE-DEV-001` is a new [exploratory protocol](../../protocols/competition_development_v1.toml),
with [fixed settings](../../configs/competition_development_v1.toml). It does not
amend the completed hard-control or cadence outcomes. This initial pilot is
Wikipedia only, with the seeds, epoch budget, primary contrast and audit anchors
fixed before allocation. Later dataset expansion requires a separate recorded
design; no expansion is implicitly triggered by a positive result.

## Causal behavioral profiles

A destination profile is a nonnegative histogram of hashed source identities
from allowed strictly earlier events, normalized to unit Euclidean norm when
nonempty. The fixed hash and coordinate budget are versioned settings. These
are behavioral affinity proxies with collisions, not ground-truth topic labels
or learned textual embeddings. Empty profiles remain zero and availability is
exposed separately.

For a recurrent candidate, let $s_{uv}$ be its last interaction timestamp. Define

$$E_{uv}(t)=\sum_{s_{uv}<s<t:\,(u,w,s)\in\mathcal H}
\langle e_v(t^-),e_w(s^-)\rangle.$$

All source events in this open interval are necessarily to other destinations.
The kernel is a nonnegative cosine between behavioral profiles. Query-time
candidate and event-time competitor profiles are intentionally different clock
snapshots. This is not a kernel fixed over the entire interval or a guarantee
that the integrated exposure itself is nondecreasing when profiles drift.

The extractor maintains source prefix vectors and takes differences from a
snapshot after the candidate's last timestamp group. All queries at a timestamp
precede all history updates, and all events in the last timestamp group are
excluded from the open interval. Excluded held-out training events enter no
history; unsupported allowed events still enter history. Test exposure is not
queried; only pre-existing split/support structure is retained. The parent
sampler still uses its original inductive split construction and approximate
count/time bins, not exact equality of candidate histories.

## Shared inputs and controls

Each arm reserves the same added channels: log competing-source event count,
log profile-weighted exposure, candidate-profile availability and log count of
competitor events with available profiles. Logs use the nonnegative log-one-plus
transform. Normalization is fitted on supported training candidate slots only.

| Arm | Available added information or scoring constraint |
|---|---|
| `masked_mlp` | All added channels masked; equal input dimension and parameter count |
| `activity_mlp` | Count and the same availability/usable-count channels; profile exposure masked |
| `profile_mlp` | Full profile channels and flexible residual |
| `shuffled_mlp` | Full channels, but permute candidate profile coordinates only before exposure |
| `activity_displacement` | Full profile information; explicit inhibition uses competing event count |
| `profile_displacement` | Identical full information; explicit inhibition uses profile exposure |

The coordinate shuffle preserves profile norms and disrupts alignment. It is
not a destination-label shuffle and may leave common activity components.
All arms share predictor/head architecture, initial parameter tensors, optimizer,
BCE and affinity auxiliary objective, candidates, seeds and epoch budget. Flexible
and inhibitory residual heads have equal parameter counts. This controls
capacity, not optimization equivalence between function classes.

## Mechanism and restricted guarantee

The flexible residual is $a\,g(c)$, where $a$ starts at zero. The inhibitory
residual is

$$r(c,E)=-\max(a,0)\operatorname{softplus}(g(c))\log(1+E).$$

The inhibition amplitude is projected nonnegative after optimizer updates. Both
branches exactly recover the shared history-aware score at initialization.
At fixed context, amplitude and base score, the partial derivative in explicit
exposure is nonpositive. Context itself receives exposure; the guarantee therefore
is not the total derivative through context, not a chronological certificate,
and not a proof of monotonicity under arbitrary new graph events.

A positive latent state update $z^+=z^-\exp(-\gamma\kappa)$ motivates inhibition,
but this pilot implements an additive logit residual, not a new inferred
multiphase lifecycle, attention conservation law, identified causal hazard or
observed edge-death process. The fixed behavioral kernel is not learned semantic
substitution. Prior inhibition/interaction precedents remain acknowledged in the
[contribution review](../questions/Contribution-Gap-Review.md).

## Endpoints and verification

The primary endpoint is paired selected validation AP of `profile_displacement`
minus `profile_mlp`. Secondary contrasts separate added activity, profile
information, coordinate alignment and profile-versus-count inhibition. Other
metrics include AUC, sampled-candidate Brier, paired ranking accuracy, coverage,
parameter count, selected amplitude and elapsed time. Earliest maximum validation
AP chooses the checkpoint. No test prediction is authorized by this pilot.

Synthetic tests compare every queried feature with direct raw-history
reconstruction, including ties and held-out exclusions, and check future/current
feature independence, feature masks, equal initialization/capacity and a guarded
validation-only end-to-end pipeline. Real-data feature auditing uses the
predeclared direct-reconstruction anchors and must not be described as full
independent feature replay. The post-retrieval auditor verifies source identity,
metrics, normalization, selection, common candidate reproduction and full
checkpoint validation inference with current host dependencies. It does not
replay optimization or establish clean-environment reproducibility.

Heavy work uses one bounded owned Colab runtime, artifact retrieval and finally
teardown with backend absence verification. Results and limitations are owned by
the [pilot result page](../results/Competing-Destination-Pilot.md). No Word export,
novelty clearance or scientific admission follows from this development.

## Closer precedent discovered after design lock

The [direct competition review](../questions/Contribution-Gap-Review.md#direct-competition-precedents-found-during-the-pilot)
now records interaction-driven other-tie weakening and finite attention in an
earlier published model, as well as competing-product point processes. Generic
cross-edge competition is established prior work. The locked pilot still tests
its narrower profile-information and conditional-ranking question; no design or
primary endpoint changes follow from this literature discovery.

## Completed pilot disposition

The [complete pilot owner](../results/Competing-Destination-Pilot.md) records
negative primary mechanism contrasts on all seeds and no average advantage of
correct profile alignment over the coordinate-shuffled control. The code is
implemented, but the new scoring constraint is not supported as a superior
method. Those outcomes are retained without changing the sealed protocol.

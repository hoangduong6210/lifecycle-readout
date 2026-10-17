---
title: Start Here
status: canonical onboarding
last_updated: 2026-10-17
paper_source: false
---

# Start Here

## What this project studies

The project studies exact monotonicity certificates for temporal graph readouts,
convex inverse-tilt repair, complete scoring paths and history-matched predictive
controls. The current goal is a focused analysis article supported by the
demonstrated structural and empirical results.
Submission remains conditional on the
[research program gates](questions/Research-Program.md);
`DEC-0002` preserves the earlier benchmark motivation. Dormant phase,
phase count, and monotonic silence direction are design choices; causal or
shortcut-resistance claims require independent evidence.

Both evidence and paper pointers remain `UNRELEASED`. The bounded development
study `LCR-P-CRAFT-FIT-001` has completed and passed its scoped audit, with the
owned runtime released. It supplies a validation-only trained comparator against
audited counts; convergence is unestablished. The sealed pair-cadence validation
comparison is now complete with full validation checkpoint inference replay;
[its result owner](results/Pair-Cadence-Development.md) separates the negative
primary clock contrast from added-feature effects. All owned runtimes are released.
The completed `LCR-P-HARD-001` study tests lifecycle utility beyond explicit
history statistics using common historical hard negatives.
The earlier corrected study `LCR-P-TKDE-002` remains preserved for audit. Earlier benchmark outputs remain exploratory and are
quarantined in [the audit disposition](results/Benchmark-and-Baseline-Results.md).
No quantitative lifecycle or generalization claim has been admitted.

A separately authorized [simplex preservation prototype](methods/Monotone-Simplex-Development.md)
now explores a structural repair for candidate-transform reversal. It is not
part of the completed studies and has no empirical superiority or novelty claim.

The user-selected [competing-destination pilot](results/Competing-Destination-Pilot.md)
is now completed and audited within its declared scope, with the owned runtime
released. Its negative primary and coordinate-shuffle results are retained;
this version supplies no demonstrated new predictive advantage.

Current strengthening work adds a [readout-specific exact restricted condition](methods/Readout-Aware-Entropic-Preservation.md)
and [complete supported-validation scored-path evidence](results/Full-Validation-Scored-Paths.md),
with scope and priority limitations explicit.

The [no-refit repair comparison](results/Readout-Repair-Validation.md) now
quantifies structural success and ranking cost, with controls separating the
reference change from the final readout constraint. No superior predictor is
claimed from this result.

## Read these pages first

1. [Latest research summary](../README.md) and [distributed evidence](evidence/Repository-Evidence.md) give the current contributions, results and source links.
2. [Project Status](status/Project-Status.md) states the current stage,
   blockers, and next actions.
2. [Dataset Registry](datasets/Dataset-Registry.md) separates local migration
   copies from verified upstream data.
3. [Claim Registry](claims/Current-Claim-Language.md) defines exactly what may
   be said.
4. [Limitations](LIMITATIONS.md) records what has not been demonstrated.
5. [Reproducibility](REPRODUCIBILITY.md) defines the closure required for a
   scientific result.

The [Exhaustive Index](INDEX.md) lists every maintained page and identifier.

## Repository map

| Path | Purpose |
|---|---|
| `src/lifecycle_readout/` | Package-local backbone, readout, datasets, metrics, and training code |
| `experiments/` | Thin study runners, probes, and dataset builders |
| `protocols/` and `configs/` | Versioned study contract, settings, and dependency hashes |
| `resources/` | Corpus identity manifest; local binaries are excluded from Git |
| `evidence/jobs/` | Local structural execution record |
| `results/` | Release pointer, distributed aggregate reports and retained-input identities |
| `wiki/` | Canonical scientific interpretation and publication source |
| `paper/` | Current working article, editable source and marked historical exports |
| `publication/` | Scientific-admission gate; distinct from public source availability |

## Safe first checks

From the repository root, run:

```bash
python -m pytest -q
```

The test suite checks structural contracts and any locally available corpus
bytes. It does not run heavy training or create scientific evidence. Training
and intervention batteries require an approved scheduler workflow; see the
[Research Workflow](operations/Research-Workflow.md) and
[Live Execution](status/Live-Execution.md).

## How to change scientific knowledge

Follow the [wiki-first hard rule](../AGENTS.md#9-wiki-and-paper-snapshots):
routine research updates belong in the wiki. Assemble the final paper from the
wiki and its verified evidence; do not issue a new manuscript after every update.
Working manuscript exports require an explicit user request or the final
manuscript stage.

Begin with the question, protocol, configuration, and complete execution
record. Preserve failed and negative attempts. Update the evidence ledger and
claim registry only after a frozen release passes its gates; update scientific
status in the wiki before creating a paper snapshot. The hard-negative controls, independent replay, Word corrections and Colab
teardown are complete. Use the [evidence retrieval guide](evidence/Hard-Control-Retrieval.md)
to locate the exact results, source snapshots, audits and corrected manuscript.
The [journal research program](questions/Research-Program.md) now includes completed oracle
capacity diagnostics. The prediction-target note separates survival from sampled labels. Count-control
and CRAFT interface audits are complete; the bounded CRAFT validation fit
has completed with scoped audit and verified runtime release. A new mechanism is an optional research route; the immediate task is
to assess a focused analysis paper against matched controls and prior work. Any
new confirmatory study still requires a sealed protocol. Scientific and dataset provenance audits remain open. [Project Status](status/Project-Status.md) owns the current
blockers and next actions; [Contributing](CONTRIBUTING.md) records edit rules.

The renewed competition route now has an [exact event-driven state operator](methods/Event-Driven-Competitive-Lifecycle.md),
with explicit jump semantics and an exposure-clock equivalence boundary. Its fitted comparison is owned by the exact-state pilot below; no new
predictive advantage is established, and the earlier negative residual pilot
stays part of the evidence.

The [frozen-reference history audit](results/Competitive-Event-History-Audit.md)
has now checked the exact event-dose inputs on the retained Wikipedia candidates.
It supports historical input validity within the audit scope; prediction gains
and a trained architecture remain unestablished.

The [exact-state competition pilot](results/Exact-State-Competition-Pilot.md)
now has a complete, audited model comparison, with all prespecified primary and
secondary outcomes retained. Its bounded validation scope and prior-art limits
remain explicit; the owned runtime is released.

The [competition input anatomy](results/Competition-Signal-Anatomy.md) now
quantifies source-overlap/hash contributions and motivates a
[source-checked alternative dataset proposal](datasets/Competition-Dataset-Proposal.md).
That anatomy review did not acquire data. A subsequent [bounded KuaiRand intake](datasets/KuaiRand-Intake-Preflight.md) now completes with a failed long-gap clicked-recurrence gate; no new model was fitted.

The [original Last.fm intake](datasets/LastFM-Intake-Preflight.md) now passes its fixed training-support gate, with verified source bytes and independent scoped replay. It provides a suitable next development input, without a new fitted model or scientific claim.

The [exact recorded-artist input audit](results/LastFM-Recorded-Artist-Audit.md) now verifies scoped historical references and count nonredundancy on Last.fm, preparing the next controlled model comparison without a new fit or admitted superiority claim.

The [complete Last.fm recorded-artist model comparison](results/LastFM-Recorded-Artist-Pilot.md)
now retains a negative mean primary and every prespecified secondary, with full
checkpoint replay and independent same-source initialization reconstruction.
The owned GPU and initialization-only CPU runtimes are released. This narrows
the predictive contribution rather than establishing a superior phase model.

The [fixed-rate readout certificate](methods/Fixed-Rate-Readout-Certificate.md)
now sharpens the earlier universal-rate condition for specified rates and finite
rate cones, with explicit proof and an untrained ordered-rate constructor.
It broadens the configurations certified for fixed rates without claiming better
AP, general dynamic-transition coverage or exhaustive priority.

The [inverse-tilt repair geometry](methods/Inverse-Tilt-Projection.md) and its
[complete retained-model application](results/Fixed-Rate-Repair-Validation.md)
now show that specified-rate certification can avoid extra deformation of the
frozen-reference gradient comparator. The matched AP primary has no mean
advantage, and losses from converting the original dynamic model remain.


The subsequent [fixed February holdout](results/LastFM-February-Holdout.md)
does not pass its local confirmation screen. The January localization remains
exploratory; it is not a confirmed operating-region contribution.

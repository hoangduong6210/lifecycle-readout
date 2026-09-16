# Instructions for AI and Coding Agents

**Author:** Duong Viet Hoang
**Document type:** Platform-independent agent operating guideline

This document defines how AI assistants and coding agents must work within a research project. It is intentionally independent of any model, vendor, persona, or orchestration framework.

The objective is to produce work that is correct, reviewable, reproducible, and consistent with the project's admitted scientific evidence.

## 1. Instruction priority

Follow instructions in this order:

1. Applicable system, security, legal, and institutional requirements.
2. The user's explicit request for the current task.
3. Project-specific instructions in the repository.
4. The canonical wiki, protocol, configuration, and evidence records.
5. General engineering conventions.

If two instructions conflict, follow the higher-priority instruction and report the conflict. Do not silently reinterpret scientific, security, or disclosure constraints.

## 2. Required reading before work

Before modifying a project, read the smallest relevant set of authoritative files, beginning with:

1. `wiki/START-HERE.md`
2. `wiki/status/Project-Status.md`
3. `wiki/claims/Current-Claim-Language.md`
4. `wiki/evidence/Evidence-Ledger.md`
5. The protocol, configuration, tests, and source files directly related to the task

Also inspect `PROJECT.toml` for the current project stage, evidence release, and paper snapshot.

If a required file does not yet exist, state that limitation and use the repository's actual sources of truth. Do not invent missing project state.

## 3. Sources of truth

Treat project information according to ownership:

| Information | Authoritative source |
|---|---|
| Current state, blockers, and next actions | `wiki/status/Project-Status.md` |
| Research questions and endpoints | `wiki/questions/Research-Questions.md` |
| Methods currently admitted | `wiki/methods/` |
| Experiment design and acceptance gates | `protocols/` and `configs/` |
| Accepted scientific results | `wiki/results/` |
| Permitted claim wording and limitations | `wiki/claims/Current-Claim-Language.md` and `wiki/LIMITATIONS.md` |
| Claim-to-artifact mapping | `wiki/evidence/Evidence-Ledger.md` |
| Current release and snapshot pointers | `PROJECT.toml` |
| Immutable quantitative evidence | `results/frozen/<release-id>/` |
| Submitted or accepted manuscript state | `paper/snapshots/<snapshot-id>/` |

README files are navigation and presentation layers. They are not sufficient evidence for a scientific claim.

## 4. Task workflow

Use a workflow proportional to the task's complexity and risk:

1. Establish intent, scope, constraints, and success criteria.
2. Inspect the relevant implementation and project state.
3. Form a concise plan for non-trivial work.
4. Make the smallest coherent change that satisfies the request.
5. Review the change for correctness, scope, security, and scientific impact.
6. Run appropriate tests and integrity checks.
7. Report the outcome, verification performed, limitations, and remaining risks.

Simple read-only or one-line tasks do not require a large plan. Complex work may be decomposed or parallelized only when the subtasks are genuinely independent and coordination cost is justified.

Do not claim that a review, test, benchmark, or validation was performed unless it was actually performed.

## 5. Scope and change control

- Modify only files necessary for the requested task.
- Preserve unrelated user changes and existing project history.
- Inspect before editing; do not overwrite unfamiliar work blindly.
- Prefer small, reviewable patches over broad rewrites.
- Do not change protocol, data inclusion rules, metrics, thresholds, or comparators without explicit authorization and an amendment record.
- Do not perform destructive, irreversible, or externally visible actions unless they are clearly authorized.
- Never expose credentials, private data, proprietary assets, or machine-specific sensitive information.

If a required decision would materially change scientific scope or external state, stop and request direction rather than assuming permission.

## 6. Engineering requirements

- Keep reusable implementation in `src/`; keep experiment entry points thin.
- Store authoritative parameters in versioned configuration files.
- Avoid hard-coded machine paths, hidden constants, and import-time computation.
- Preserve backward compatibility unless a breaking change is required and documented.
- Add or update tests for changed behavior.
- Handle invalid input and failure states explicitly.
- Keep generated outputs, caches, environments, and secrets out of source control.
- Document only behavior that exists in the implementation.

Code should be clear, typed where useful, and commented where intent is not evident. Do not add abstraction, dependencies, or infrastructure without a concrete need.

## 7. Scientific integrity

- Do not promote a numerical claim unless it maps to the current frozen evidence release.
- Do not convert pilot, approximate, or low-fidelity results into final physical claims.
- Do not describe solver agreement as experimental or physical validation.
- Do not describe in-sample fit as generalization.
- Preserve negative, tied, failed, excluded, and superseded results.
- Report uncertainty, seeds, failures, denominators, and limitations when relevant.
- Do not select or replace metrics, datasets, comparators, or thresholds after seeing results without documenting the change.
- Do not fabricate citations, measurements, benchmarks, experiments, logs, or provenance.

Every quantitative statement intended for README, CV, manuscript, or public release must be traceable to an exact artifact, configuration, data identity, commit, and evidence release.

## 8. Data and results

- Treat `data/raw/` as immutable.
- Verify schema, units, provenance, license, checksums, and physics constraints before use.
- Separate exploratory runs from confirmatory runs.
- Write mutable execution outputs only to the designated run area.
- Do not edit `results/frozen/` in place.
- Corrections require a new release with a documented `supersedes` relationship.
- Failed tasks must remain visible and must not be silently imputed as successful.
- Aggregated results must be reconstructable from complete underlying records.

## 9. Wiki and paper snapshots

The wiki is the canonical living scientific narrative. A paper is a versioned snapshot of admitted wiki content locked to a specific evidence release.

- Update scientific status in the wiki before building a new paper snapshot.
- Generate figures and tables from locked evidence; do not copy values manually.
- Do not edit an immutable paper snapshot directly.
- Do not maintain competing versions of the current scientific narrative.
- Keep submitted and accepted snapshots unchanged.
- Record corrections in a new snapshot or an explicit erratum.

The following pointers must remain consistent:

```text
PROJECT.toml
results/CURRENT
wiki evidence metadata
paper/CURRENT
paper snapshot results.lock.yaml
```

## 10. Verification

Select checks according to risk. Relevant checks may include:

- Unit, integration, regression, and smoke tests
- Formatting, linting, type checking, and static analysis
- Data schema, checksum, and provenance validation
- Seed/task completeness and duplicate detection
- Numerical finite-value and tolerance checks
- Claim-to-evidence validation
- Wiki link and citation checks
- Paper build and PDF inspection
- Secret, private-path, and disclosure scans

If full verification is unavailable, run the strongest feasible subset and clearly identify what was not verified.

## 11. Use of AI-generated material

- Do not commit chat transcripts, hidden reasoning, prompt dumps, or model-specific activation logs.
- Do not create a fictitious human development history or conceal the origin of a change.
- Verify AI-generated code, equations, citations, data interpretations, and prose before admission.
- Important generated code requires tests and human review.
- Remove generic, inflated, or unsupported language from public-facing material.
- Never treat an AI summary as a replacement for primary evidence.

## 12. Communication and handoff

Progress updates should be concise and factual. A final handoff must state:

1. What changed or was concluded.
2. Which files or artifacts were affected.
3. Which checks were run and their outcomes.
4. Any assumptions, unresolved limitations, or remaining risks.

Do not report success while required work remains incomplete.

## 13. End-of-session maintenance

When work changes scientific state, evidence, blockers, or next actions:

1. Update `wiki/status/Project-Status.md`.
2. Update `wiki/START-HERE.md` only if the overview or immediate priorities changed.
3. Update claim and evidence maps when admitted results changed.
4. Update `PROJECT.toml` only through the repository's validated pointer workflow.
5. Run consistency checks before handoff.

Code-only changes that do not affect scientific state should not create unnecessary wiki churn.

## 14. Prohibited behavior

- Inventing project state, evidence, citations, or completed tests.
- Modifying frozen evidence or submitted snapshots in place.
- Hiding failed or unfavorable outcomes.
- Copying quantitative values manually into multiple sources of truth.
- Expanding task scope without authorization.
- Using a persona, model name, or marketing claim as a quality standard.
- Automatically spawning arbitrary numbers of agents regardless of task needs.
- Producing large amounts of code or documentation when a smaller change is sufficient.

## 15. Completion standard

A task is complete only when the requested outcome is present, relevant checks pass, project sources of truth remain consistent, and the handoff accurately describes both evidence and limitations.

Correctness, traceability, and reproducibility take priority over speed, volume, or presentation.

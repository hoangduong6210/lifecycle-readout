---
title: Glossary
status: canonical terminology
last_updated: 2026-08-19
paper_source: false
---

# Glossary

| Term | Controlled meaning |
|---|---|
| Lifecycle state | One of IDLE, BIRTH, REINFORCE, DECAY, or DEATH; cardinality is owned by `LCR-P-READOUT-001`, `LCR-E-RUNTIME-MIGRATION-001`, job `LCR-JOB-LOCAL-20260819-001`. |
| Interpretable path | Detached branch that emits lifecycle state and interventions. |
| Scored path | Branch that emits the link-existence score used for AP. |
| Score invariance | Fixed-model scored outputs are unchanged by a readout-only toggle. |
| Gradient ancestry | Autograd reachability test from scored output to parameter groups. |
| Legacy import | Preserved artifact that is neither frozen nor claim-admitted. |

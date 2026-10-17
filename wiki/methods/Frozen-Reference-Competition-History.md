---
title: Frozen Reference Competition History
status: implemented strict-past input extraction; separate from earlier residual pilot
last_updated: 2026-10-16
paper_source: false
---

# Exact interval exposure from a frozen own-event reference

The [exact absorbing event operator](Event-Driven-Competitive-Lifecycle.md)
needs a predictable nonnegative event dose. The implementation in
`src/lifecycle_readout/competitive_event_stream.py` supplies one specified
contract, independently versioned from the completed
[query-time-profile pilot](Competing-Destination-Development.md).
The new input protocol is `LCR-P-COMPETE-HISTORY-001`.
No earlier protocol or outcome changes.

At a candidate's latest own-edge timestamp $s$, freeze the destination profile
$\rho_{uv}=e_v(s^-)$ before the whole timestamp group. Let
$B_u(t^-)=\sum_{r<t:(u,w,r)\in\mathcal H}e_w(r^-)$.
After all allowed events at $s$, save the source prefix $B_u(s^+)$.
Then query with

$$ E_{uv}(t)=\rho_{uv}^{\top}[B_u(t^-)-B_u(s^+)]. $$

Equivalently, it is the sum of kernels
$\kappa_{uvw}(r)=\langle\rho_{uv},e_w(r^-)\rangle$ over source events
$s<r<t$. No intervening event is to the target, by definition of $s$.
All kernels are nonnegative cosine similarities between normalized hashed
behavioral profiles, with zero vectors for missing profiles. The target
reference is interval-frozen, not recomputed at each competitor event or query.
The competitor vector is evaluated strictly before its timestamp group.

## Timing, exclusions and state resets

All queries precede all updates at tied timestamps. All allowed events,
including events without supported sampled negatives, update histories.
Excluded held-out training events update nothing. Destination profile updates
occur only after event vectors and source-prefix/reset snapshots are formed.
The whole last-own timestamp group is excluded from the open exposure interval,
including other destinations in that group. First-time pairs are outside this
recurrent-support contract; a recurrent pair may still have a zero reference.
Reference availability is supplied as an explicit feature, not imputed semantics.
Only training/validation feature rows are queried. Reconstructing structural test
identities for sampler consistency is not querying test exposure or states.

This is a history extractor. It does not choose a learned own-edge reset,
encode a last-event neural context, fit rates, or train a prediction model.
For the state audit only, an explicitly fixed initial simplex and fixed rates
are used identically in the aggregated and independent sequential calculations.
Those constants are numerical witnesses, not estimated lifecycle parameters.

## Why the prefix is exact

Holding $\rho_{uv}$ fixed makes the inner product distributive over the prefix
difference. This recovers exactly the event-frozen kernel sum in real arithmetic.
Between own-edge reset groups, increments of $E$ are sums of nonnegative kernels,
so $E$ cannot decrease. Resetting the reference starts a new interval and can
change the dose to zero. Physical time and event dose feed the existing exact
absorbing operator; the proportional-rate exposure-clock equivalence and
commuting-event limits still apply. No competitor-order sensitivity is gained.

This contrasts with the old pilot's $e_v(t^-)^\top[B_u(t^-)-B_u(s^+)]$:
query-time profile drift changes the weights applied to earlier events.
The old feature is not erroneous for its declared contract. It simply is not
the same predictable cumulative dose, and its negative result is preserved.

With profile width $B$, total processed history events $H$, reserved candidate
slots $Q$ (including unqueried structural test rows), and requested pairs $P$, the prefix operations take
$\mathcal O((H+Q)B)$ work apart from input validation and timestamp indexing.
Working profile/prefix/reset storage is
$\mathcal O((|U|+|V|+P)B)$, with separate output storage
$\mathcal O(QB)$. Only requested pairs retain reset/count state, while every
allowed event still updates source prefixes and destination profiles. The code
uses timestamp searches and finite-precision bounds checks; this complexity
statement is not a measured speedup or a novel prefix-sum algorithm claim.

## Verification and scientific boundary

The [real-history audit](../results/Competitive-Event-History-Audit.md) owns
execution and coverage. Synthetic tests compare every queried slot with direct
raw-history summation, match states to independent event-by-event NumPy replay,
permute event ordering within tied timestamp groups, alter excluded/future events,
and check within-interval nondecreasing exposure.

Profiles include permitted prior contributions from the source itself and have
hash collisions. They remain behavioral proxies, not independently verified
semantic substitution. Finite attention and interaction-driven other-tie
weakening have prior work recorded in the
[competition review](../questions/Contribution-Gap-Review.md#direct-competition-precedents-found-during-the-pilot).
Neither exact implementation nor historical availability establishes useful
ranking information, causal displacement or a new architecture contribution.
Any model experiment needs a new prospective contract with identical-information
controls, a specified own-edge reset and no replacement of the earlier negative
primary. No manuscript export follows from this input development.

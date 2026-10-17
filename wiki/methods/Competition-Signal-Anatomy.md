---
title: Competition Signal Anatomy
status: exact input decomposition with posthoc real-data audit; no semantic or causal identification
last_updated: 2026-10-16
paper_source: false
---

# What a hashed behavioral cosine contains

The [exact-state comparison](Exact-State-Competition-Study.md) failed its
primary advantage and directional alignment control. This separate diagnosis
examines the unchanged frozen-reference feature, not the trained score.
`LCR-P-COMPETE-ATTR-001` fixes an initial posthoc decomposition;
`LCR-P-COMPETE-ATTR-FULL-001` separately expands exact-ID coverage after seeing
the first stage. Neither is a confirmatory experiment or changes earlier inputs.

For source $u$, let its hash bucket be $b$. Write each normalized profile as
$a=\alpha e_b+\bar a$, $c=\beta e_b+\bar c$, where $\alpha,\beta$ include
only counts from the exact source $u$ and the barred profiles contain all other
sources under the original hashed normalizers. Then

$$a^\top c=\alpha\beta+\alpha\bar c_b+\bar a_b\beta+\bar a^\top\bar c.$$

These terms are nonnegative. The two cross terms can only arise from other
source IDs sharing $u$'s bucket. The last term mixes exact same-other-source
overlap with further collisions. An own-source-only pair has cosine one even
with no other contributing user; disjoint source sets can also have cosine one
if their IDs collide. Neither is proof of semantic substitutability.
These are elementary identities and witnesses, not a claimed new similarity law.

## Exact-ID decomposition under the original normalizers

Let $n_v(a)$ and $n_w(a)$ be strictly prior counts for exact source ID $a$,
and $A,B$ the original hashed profile norms. The full kernel numerator splits
as follows:

$$\kappa=\frac{n_v(u)n_w(u)}{AB}
+\frac{\sum_{a\ne u}n_v(a)n_w(a)}{AB}
+\frac{\sum_{a\ne c:\,h(a)=h(c)}n_v(a)n_w(c)}{AB}.$$

The terms are own-source overlap, exact same-other-source overlap and cross-ID
hash collisions. All retain $A,B$; replacing them with exact-ID norms would
change the feature and is not done here. Empty profiles give zero terms.
Summing over the same strictly prior competitor events decomposes exposure.

The full-validation implementation uses sparse source prefixes of exact-ID
counts divided by each competitor's hashed norm, plus a hashed prefix. At each
last-own timestamp, a fixed target reference and its scalar prefix products
are saved after the whole timestamp group. This preserves the earlier open
interval and exclusion contract. Only supported validation slots are queried;
training history still updates prefixes. Source-prefix memory is bounded and
aborts rather than silently truncating history. Raw negative roundoff is saved
before clipping within the declared numeric tolerance.

## What a coordinate shuffle retains

For any profiles $a,c$ with width $B$,

$$a^\top c=\frac{(\mathbf1^\top a)(\mathbf1^\top c)}{B}
+(a-\bar a\mathbf1)^\top(c-\bar c\mathbf1),$$

where scalar bars here denote coordinate means. Every coordinate permutation
preserves the first term. Averaging over all uniformly chosen permutations
eliminates the centered term, giving the first term as expectation. This ideal
uniform reference is not the actual fixed-shuffle experiment: that uses one
seeded permutation, with a rule to avoid identity, and supplies no exchangeable
randomization test or p-value. A uniform dense profile is unchanged by every
permutation, so shuffle does not erase all behavioral information.

The background/centered split overlaps the exact-ID decomposition rather than
adding more disjoint source components. Its background-to-total ratio need not
be at most one because centered alignment can be negative.

## Scientific limits

These are legitimate strictly past features, not a diagnosis of future-data
leakage. Source overlap and hash aliasing may carry predictive information.
Their exposure fractions are not fractions of AP, semantic content or a causal
explanation for the shuffled model's performance. Even exact other-user overlap
is behavioral evidence rather than an observed semantic relationship.
No feature replacement, hash-width tuning, retraining or new predictor score
is performed. The [result owner](../results/Competition-Signal-Anatomy.md)
records scope, coverage, guards and all numerical outcomes.

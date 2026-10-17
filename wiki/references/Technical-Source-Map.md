---
title: Technical Source Map
status: canonical reference map
last_updated: 2026-10-16
paper_source: false
---

# Technical Source Map

## Entropic simplex update precedent

[Beck and Teboulle, 2003](https://www.tau.ac.il/~becka/3.pdf),
*Mirror descent and nonlinear projected subgradient methods for convex
optimization*, Operations Research Letters 31, 167–175,
DOI `10.1016/S0167-6377(02)00231-6`, Section 5, supplies a normalized exponential
simplex update. The author-hosted full paper was retrieved and the explicit
update inspected. The [method owner](../methods/Shortcut-Resistant-Gating.md#established-exponential-update-form-and-the-narrower-question)
states the algebraic connection and limits on transferring optimization results.
Retained primary bytes and scope review are in
`results/audit/simplex_map_prior_art_v1/`. The publisher page opened unsuccessfully;
the author PDF supplies the inspected full source. This is targeted attribution,
not exhaustive priority clearance for the temporal-preservation condition.

The source map records literature for positioning and audit. Its historical
rows are not blanket verification or scientific admission. Individually checked
identities and remaining review boundaries are recorded below.

| Topic / Source Owner | Intended Use | Primary Source Identity | Verification State | Admission Boundary |
|---|---|---|---|---|
| Temporal GNN Architectures | Method positioning and baselines | Rossi et al. (ICML 2020, TGN); Kumar et al. (KDD 2019, JODIE); Cong et al. (ICLR 2023, GraphMixer); Yu et al. (NeurIPS 2023, DyGFormer) | SOURCE SPECIFIC REVIEW REQUIRED | Comparative baselines for dynamic link prediction |
| Temporal Benchmark & Shortcuts | Problem formulation and benchmark datasets | Huang et al. (NeurIPS 2023, TGB); TGB-Seq (ICLR 2025) on frequency memorization shortcuts | SOURCE SPECIFIC REVIEW REQUIRED | Motivation for shortcut-resistant survival gating |
| Concept Bottlenecks & Interpretability | Relational phase representation | Koh et al. (ICML 2020, CBM); Chen et al. (arXiv 2026, ST-TGExplainer); Tishby et al. (2000, Information Bottleneck) | SOURCE SPECIFIC REVIEW REQUIRED | Conceptual context only; no information-bottleneck bound established |
| Point Processes & Continuous Dynamics | Jump-decay transition dynamics | Hawkes (Biometrika 1971); Continuous-Time Markov / Jump-Diffusion Processes | SOURCE SPECIFIC REVIEW REQUIRED | Parameterization of continuous decay and event jumps |
| Causal Representation & Interventions | Counterfactual evaluation formulation | Pearl (Cambridge Univ Press 2009); Song et al. (NeurIPS 2024, Nonstationary Sparse Transitions) | SOURCE SPECIFIC REVIEW REQUIRED | Internal model-response context only; external causal identification unestablished |
| Benchmark Interaction Datasets | Empirical evaluation data | Stanford SNAP Continuous-Time Networks (Wikipedia, Reddit, MOOC, LastFM; Kumar et al. 2019) | SOURCE SPECIFIC REVIEW REQUIRED | Multi-dataset evaluation under `LCR-P-TKDE-001` |
| CoEdit Legacy Migration Stream | Historical audit only | Pre-split non-bipartite CoEdit NPZ stream | QUARANTINED | Retained for migration audit; not admitted for primary claims |

## Corrective source review

The blanket earlier verification statement is withdrawn. Reference identities
and numerical tables require individual checks; citing a source does not
validate the proposed mathematical relationship or an empirical comparison.
The following primary identities were rechecked for the manuscript revision:

- [TGB-Seq](https://arxiv.org/abs/2502.02975): Lu Yi, Jie Peng, Yanping Zheng,
  Fengran Mo, Zhewei Wei, Yuhang Ye, Zixuan Yue and Zengfeng Huang; ICLR 2025.
- [ST-TGExplainer](https://arxiv.org/abs/2605.19822): Hongjiang Chen, Xin Zheng,
  Pengfei Jiao, Huan Liu, Zhidong Zhao, Huaming Wu, Feng Xia and Shirui Pan;
  the verified arXiv identity is from 2026. The earlier author list and 2024
  venue assignment were incorrect.
- [Nonstationary sparse transitions](https://arxiv.org/abs/2409.03142):
  Xiangchen Song, Zijian Li, Guangyi Chen, Yujia Zheng, Yewen Fan, Xinshuai
  Dong and Kun Zhang. Their identifiability result is not automatically
  inherited by the lifecycle gate.

Earlier unverified literature numbers are withheld in the revised Word draft.
Official baseline execution still requires matched candidates and evaluation
subsets before a superiority claim.


## research positioning sources checked on 2026-10-15

- [TGB official implementation](https://github.com/shenyangHuang/TGB) owns dataset
  loaders, candidate evaluation and baseline examples; package/data versions need
  pinning for the new study.
- [TGB-Seq author implementation](https://github.com/TGB-Seq/TGB-Seq) owns fixed
  benchmark splits, multi-negative ranking and MRR evaluation.
- [Base3](https://arxiv.org/abs/2506.12764) and
  [author code](https://github.com/ekmpa/Base3) provide recurrence, popularity and
  temporal co-occurrence controls. Abstract identity and code route checked;
  local matched reproduction is pending.
- [Smoothed-count temporal prediction](https://arxiv.org/abs/2609.35364), Lisi
  Qarkaxhija and Ingo Scholtes, submitted September 2026, is a recent preprint
  covering pooled transitions, smoothing and recency. Its abstract and method
  overview were inspected; code reproduction and peer-review status are not
  established here.

The proposed residual is not certified novel by this targeted source check.
The [journal program](../questions/Research-Program.md) requires a broader related-work
comparison before writing a novelty claim.


## Prediction target sources

[Gutmann and Hyvärinen, AISTATS 2010](https://proceedings.mlr.press/v9/gutmann10a.html)
provides the noise/data classification and density-ratio relationship. The
conditional formulas in the target note are standard derivations, not a new
estimator or an inherited identifiability guarantee.
[Mei and Eisner, Neural Hawkes 2017](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
is relevant event-process prior work; event likelihood and sampled edge labels
must not be conflated. A complete phase-type, renewal and neural survival prior-art
review remains necessary before proposing another lifecycle architecture.


## Full-method overlap check for the fresh screen

The [smoothed-count v1 method](https://arxiv.org/html/2609.35364v1), sections `4.2–4.5`,
already contains pooled transitions, exponential count decay, Dirichlet backoff,
multiple recency scales and a learned shared score. Its appendix also examines
strict timestamp updates. The new simple count screen therefore cannot supply a
novel-method claim. [CRAFT's NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html)
verify the original architecture identity and candidate-aware recent-history
matching. Cached primary pages and hashes are retained under
`results/audit/q1_preflight_v1/novelty_review/`. This is a targeted overlap review,
not exhaustive novelty clearance or execution of either author model.


CRAFT author source is now pinned to commit
`81590a78dddf1f0f589bc710fec2effd83f6cd5a`; selected model, loader,
evaluator, neighbor utility, README and MIT LICENSE files are cached with exact
hashes in `results/audit/q1_preflight_v1/baseline_review/craft/source_identity.json`.
No author code was executed. The inspected interface accepts recent destination
IDs/times, query times, positive plus negative candidates and each candidate's
last strictly prior update time. Its `predict` translates destination IDs using
`dst_min_idx`; a common adapter must preserve zero padding and verify that map.
The original loader reads test candidates and the evaluator may generate its own
validation negatives, so use of its default runner would not implement our
validation-only data/candidate boundary. The adapter must explicitly consume the
existing cached candidates, supplied masks and strict timestamp histories;
correct interface construction is still pending.


CRAFT's official proceedings Table 8 (PDF page index 26) supplies the ML-20M
parameter row; it differs from our provisional feasibility profile. The exact
PDF and extracted settings record are locally cached with source hash. The
common adapter now executes the pinned author architecture in technical tests
and a bounded GPU profile; this supersedes the earlier source-intake observation
that no author code had been executed. Full predictive baseline training remains
pending. See [the profile owner](../results/CRAFT-Adapter-and-Feasibility.md).


## Positive-dependent sampling objective review

[Conditional NCE, ICML 2018](https://proceedings.mlr.press/v80/ceylan18a.html)
uses a paired-order discrimination objective with forward/reverse proposal
terms; it is directly relevant prior art before proposing corrections to
positive-dependent negatives. [GradientDICE, ICML 2020](https://proceedings.mlr.press/v119/zhang20r.html)
uses Perron–Frobenius for stationary density-ratio estimation in a different
setting. The [prediction-target note](../methods/Prediction-Target-and-Sampling.md)
now derives a finite-catalogue posterior equation and a closed-component scale
ambiguity, explicitly as working analysis and established mathematical tools.
This is not novelty clearance, a new fitted method or a changed CRAFT objective.
Primary source snapshots and an exact rational counterexample are retained in
`results/audit/q1_preflight_v1/sampling_source_review/`.


## Temporal mechanism overlap review

[EPT-X v1](https://arxiv.org/html/2610.14948v1) is an October 2026 preprint with
phase-type state constructions and a renewal channel; construction and prediction
sections were inspected for overlap. [CTIG v1](https://arxiv.org/abs/2602.02161v1)
is a February 2026 preprint on synthetic counterfactual validation of temporal
interaction graphs. [NPGLM v1](https://arxiv.org/abs/1706.06783v1) already models
link appearance times probabilistically. The
[contribution-gap review](../questions/Contribution-Gap-Review.md) owns the
positioning implications and limits of this targeted review. No result from these
papers has been reproduced or admitted as a matched baseline here.


## First journal bibliography identity pass

The first pass covers primary identities for 20 of the current 46 references;
it does not close the entire bibliography or every cited statement. Exact
source URLs, downloaded bytes, metadata fields and the per-reference remaining
checks are retained in
[the review record](../../results/audit/journal_reference_v1/review.json).
Twenty-one cached primary files include the benchmark team page in addition to
the 20 paper identities. General metadata search candidates were not treated
as verified matches.

[The official FreeDyG proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html)
identify Yuxing Tian, Yiyan Qi and Fan Guo. The working reference incorrectly
used G. Qi and F. Li; the corrected reference uses Y. Qi and F. Guo.
For TGB-Seq, the printed paper and
[benchmark team](https://tgb-seq.github.io/team/) identify Zixuan Yue. The
reference now uses Z. Yue. The proceedings/arXiv landing metadata use the
reversed form Yue Zixuan; that source discrepancy remains recorded.

Monotonic Networks, Deep Lattice Networks and UMNN establish that imposed
monotonicity is prior art. Poursafaei and colleagues already examine easy
negatives and historical candidate evaluation. These sources position the
current analysis; they do not certify a new constraint or evaluation principle.
The potential contribution must be the specific gate analysis and empirical
operating boundaries, with its distinction from earlier evaluations made clear.
ArXiv identities for the three ICLR-2017 relaxation/IB sources were checked,
but their conference venue metadata remain open in this pass. The other 26
reference identities and statement-level coverage still require review.


## Statement level journal review

The second pass locates primary papers or publisher records for the remaining
26 original references. Identity/location checks do not close every page-range,
conference record or article-level statement. The retained record is
`results/audit/journal_reference_v2/`; the first-pass files remain historical.

The working narrative must describe EvolveGCN as recurrent evolution of GCN
parameters, and DySAT as structural and temporal self-attention. Treating both
as RNNs over embeddings is inaccurate. Sources:
[EvolveGCN](https://ojs.aaai.org/index.php/AAAI/article/view/5984),
[DySAT author paper](https://aravindsankar28.github.io/files/DySAT-WSDM2020.pdf).
IRM studies invariance across environments; it does not formalize every form of
shortcut learning. GSAT's assumed spurious-pattern guarantee and ST-TGExplainer's
stability/transition bottleneck are distinct objectives. The manuscript should
describe them separately rather than transfer a shared invariance guarantee.

The [publisher Cox article](https://academic.oup.com/jrsssb/article/34/2/187/7027194)
lists pages 187–202. The older 187–220 span includes discussion in common
citations; v3 follows the publisher's article record.
The information-bottleneck preprint identity is verified, while the original
Allerton proceedings pagination has not been independently retrieved; cite
[the actual arXiv version](https://arxiv.org/abs/physics/0004057) explicitly.
The first pages of the Deep VIB, Gumbel-Softmax and Concrete author PDFs confirm
ICLR 2017, closing the earlier venue-only uncertainty for those three entries.

The current related work also needs explicit treatment of
[Base3](https://arxiv.org/abs/2506.12764),
[CRAFT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html)
and the [smoothed-count preprint](https://arxiv.org/abs/2609.35364).
Base3 combines recurrence, popularity and temporal co-occurrence; its printed
author name is Emma Kondrup despite reversed landing metadata. CRAFT uses
learnable identifiers and candidate-to-recent-history cross-attention. The
count preprint pools transition/co-occurrence estimates with smoothing and
recency. Their reported performance is not imported into our tables. Count and
phase encodings are not automatically distinct scientific contributions.
Unused Pearl, causal-representation survey and CtrlNS bibliography entries were
removed from v3 while remaining available as reviewed positioning sources here.
Original monotonic-network entries 44–46 are now 41–43; Base3, CRAFT and the
smoothed-count preprint occupy 44–46. The mapping is retained in the revision audit. A zero-shot representation for an unseen node is an architectural
capability, not a guarantee of predictive generalization.

The v3 targeted edits are complete and visually reviewed. The original 20-reference pass remains historical; subsequent source location and statement corrections are retained separately. This does not certify every bibliographic field or every article claim.


## Publisher metadata and cumulative intensity closure

The final metadata pass retains official NeurIPS metadata for Neural Hawkes, GNNExplainer, Neural ODEs, Latent ODEs and the monotonic-network references. It confirms the existing monotonic-network pagination and corrects the GNNExplainer page range to the [official conference record](https://proceedings.neurips.cc/paper_files/paper/2019/file/d80b7040b773199015de6d3b4293c8ff-Metadata.json). This is bibliographic metadata, not a new scientific endpoint. Publisher-registered metadata also supports the Hawkes, Cox, EvolveGCN and DeepHit identities. The DeepHit publisher PDF independently provides its page range. DyRep is identified in the indexed primary conference PDF, but direct local OpenReview retrieval remains unavailable; a failed retrieval is retained rather than replaced with the differently titled earlier preprint.

[Omi, Ueda and Aihara, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/39e4973ba3321b80f37d9b55f63ed8b8-Abstract.html) directly connect monotone neural networks to point-process modeling. They parameterize a cumulative intensity/hazard, differentiate it for the intensity and train an event-sequence likelihood. Their positive-weight construction is prior art for a neural monotone temporal function. The current sampled-candidate gate and bounded logit residual do not inherit that likelihood interpretation. Include this direct overlap in the manuscript rather than imply that neural cumulative attenuation alone establishes novelty. Source bytes, exact fields and remaining review scope are retained in `results/audit/journal_bibliography_v5/`.


## DyRep conference source retrieval closure

The [Google Research publication page](https://research.google/pubs/dyrep-learning-representations-over-dynamic-graphs/)
links a [conference PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/6186.pdf)
that is now retained locally. Its title page confirms the existing DyRep title,
author list and ICLR venue/year. This closes local full-paper availability through
an author-affiliated primary route; OpenReview remains challenged and its earlier
failed retrievals remain preserved. The earlier locator maps are historical.

The inspected event-type definitions distinguish observed association and
communication events. Its learning section fits event intensities with an
integrated total-intensity term. These support the existing related-work
positioning; they do not establish the current gate's latent phase semantics,
survival interpretation or monotonicity. No reported performance is imported.

Exact source hashes, the author-page-to-PDF link, inspected sections and the
scoped locator update are in `results/audit/journal_dyrep_source_v1/`. No working
Word draft was modified. This closes one concrete retrieval gap, rather than
certifying the entire bibliography or every cited assertion.


## Matching and ROC positioning sources

[Janes and Pepe, Biometrics 2008](https://doi.org/10.1111/j.1541-0420.2007.00823.x)
and [Biometrika 2009](https://doi.org/10.1093/biomet/asp002) are direct
classification-accuracy and covariate-adjusted ROC prior art. The
[gap review](../questions/Contribution-Gap-Review.md#statistics-prior-art-for-matched-cohort-roc)
owns the interpretation, distinction from the implemented graph diagnostics and
precise abstract/metadata-only access limits. Source identities are retained in
`results/audit/pooling_statistics_prior_art_v1/`. This is no claim of a new
adjusted estimator or an exhaustive literature clearance.

## Last.fm repeat and inhibitory-prior source closure

The [targeted gap review](../questions/Contribution-Gap-Review.md#repeat-consumption-and-metadata-inhibition-precedents-for-lastfm) now records four retained primary sources: the author-hosted WWW repeat-consumption paper, official AAAI RepeatNet paper, ReCODE author preprint and SDIL/TPNE author preprint. Specific inspected sections, hashes and non-equivalence limits are retained in `results/audit/lastfm_prior_art_review_v1/review.json`. Generic repeat dynamics, dynamic add-on scoring and relation-based inhibition cannot be claimed new. This pass neither changes the running experiment nor independently verifies those authors' implementations.

## Expanded primary access for the fixed-rate analysis

The [expanded weighting-source review](../questions/Contribution-Gap-Review.md#expanded-access-to-weighted-aggregation-priors)
resolves the institutional draft's actual title and primary `(02)` DOI, with its
normalized-weight restrictions and the mixture-monotonicity follow-up inspected.
The [fixed-rate result owner](../methods/Fixed-Rate-Readout-Certificate.md)
separates these precedents and standard binary copositivity/moment-support tools
from the narrower proposed specialization. `LCR-E-FIXED-RATE-TILT-001` owns source
bytes and exact access scope. No exhaustive priority statement follows.


## Exponential and Laplace ratio precedents

The [spectral-ratio positioning owner](../questions/Spectral-Ratio-Prior-Position.md) records Yang–Tian, Mao–Tian and Karp–Vishnyakova–Zhang primary preprints, the inspected theorem/corollary hypotheses, exact representation and predicate-separation witnesses. All primary HTML and access failures are retained under `results/audit/spectral_prior_position_v1/`. The Karp publisher identity was retrieved directly. Existing endpoint refinements are acknowledged; these fixtures are not counterexamples to the cited theorems or exhaustive priority clearance.

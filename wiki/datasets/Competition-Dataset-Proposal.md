---
title: Competition Dataset Proposal
status: KuaiRand fixed intake failed; Last.fm-1K conditional next candidate
last_updated: 2026-10-16
paper_source: false
---

# Candidate for independently observed destination content

The user explicitly permits proposing a different suitable dataset. The
[Wikipedia anatomy](../results/Competition-Signal-Anatomy.md) motivates this:
a behavioral hash is not independent evidence of topic substitution. Preserve
that negative case and the existing estimand; select a new dataset for observable
inputs and adequate recurrent support, not for favorable AP.

## First candidate: KuaiRand-1K

The [official documentation](https://kuairand.com/) supplies timestamped feeds,
feedback, random-exposure flags and video category tags. It recommends the
full-history variants for sequential research; Pure omits out-of-pool events.
Therefore prefer `KuaiRand-1K`, using basic tags and retained complete user
histories. Published uncompressed logs are about `829 MB`; basic video features
about `368 MB`. Monthly aggregate statistics are not valid past-only inputs.
New caption/category supplements require ID-join and timestamp checks first.
This is a conditional candidate, not evidence of recurrent support or utility.

## Secondary candidate: MIND-small

[Microsoft's documentation](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news)
provides article categories/text, timestamped impressions and click/non-click
labels. Historical click lists lack per-click times. An article-level recurring
edge may be poorly supported and needs measurement. A user-topic graph would
change the target and requires a separate explicit study. Keep MIND secondary
for content-aligned recommendation rather than assume it tests long-lived dyads.

## Candidates not selected now

The original Last.fm-1K owner page could not be read in this review. The
[official LFM-1b page](https://www.cp.jku.at/datasets/LFM-1b/) explicitly says
downloads are no longer available due to license issues. It is not the same
LastFM file already used in this repo. Do not substitute a mirror as verified
current official access or recommend it as an immediately available fallback.

## Preflight before any new fit

First inspect acquisition terms, archive/schema identity and required files.
Sample users deterministically and retain their full event histories; thinning
individual events would corrupt competition exposure. Count recurring pairs,
long-gap returns, observed competitor events and metadata coverage on training
inputs before defining a model comparison. Treat metadata availability, item
creation times, exposure/feedback timing and candidate eligibility explicitly.
Do not pool random-exposure and standard-policy outcomes without defining the
target. Random exposure is not a measured causal label of relationship death.

If recurrent user-video support is inadequate, report that failure rather than
silently change to user-author or user-category. That alternative target needs
its own definition. No fit, metric search, download or runtime follows merely
from this proposal; a bounded intake/preflight is the next action.

## Subsequent measured decision and new source access

The [fixed KuaiRand intake](KuaiRand-Intake-Preflight.md) now fails its
prespecified long-gap clicked-recurrence gate. Preserve that outcome; no model
fit or posthoc target change follows. The source-review paragraphs above record
the earlier proposal stage, not the current acquisition status.

A newly located [creator-attributed institutional dataset record](https://zenodo.org/records/6090214)
provides the Last.fm-1K archive, published checksum, schema and non-commercial
use statement. It describes timestamped user/artist/song listening events and
near-complete recorded listening histories through the collection cutoff.
This resolves a documentation-access gap; neither binary reachability nor local
archive identity, recurrence, missing-ID handling or semantic tags is yet checked.
It is distinct from the repository's earlier processed LastFM benchmark.

Last.fm-1K is the next conditional intake candidate for recurring **user-track**
edges, with recorded artist identity as an explicit limited content proxy.
Artist identity is not genre similarity or causal competition. A prospective
training-window/user-sample protocol must fix track-ID handling, tied events,
long-gap recurrence and metadata coverage before outcomes. Keep intervening
listening events even where track IDs are missing, with explicit supported-edge
rules. Do not silently merge by track title or substitute user-artist edges.
No download, fit or scientific superiority is claimed by this source review.

## Last.fm original intake now measured

The [fixed Last.fm preflight](LastFM-Intake-Preflight.md) now passes its declared
training-sample long-gap recurrence and recorded-artist coverage gate. Original
archive/raw TSV checksums and independent scoped replay are verified. Earlier
statements that access/support were unmeasured are historical proposal stages.
The next action is exact strict-past input validation and a separate controlled
model protocol; no predictive or novelty claim follows from this feasibility.

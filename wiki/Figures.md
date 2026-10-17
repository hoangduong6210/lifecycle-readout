---
title: Figure-to-File Map
status: canonical research documentation
last_updated: 2026-10-17
paper_source: false
---

# Figure-to-file map

| Figure | Content | Preview | Vector artwork | Numeric input |
|---|---|---|---|---|
| 1 | Tensor-level neural scoring pipeline | [PNG](../figures/generated/article/Fig1.png) | [PDF](../paper/working/article/Fig1.pdf), [EPS](../paper/working/article/Fig1.eps) | Schematic; definitions in article |
| 2 | Fixed-candidate certificate workflow | [PNG](../figures/generated/article/Fig2.png) | [PDF](../paper/working/article/Fig2.pdf), [EPS](../paper/working/article/Fig2.eps) | Schematic; definitions in article |
| 3 | Safe/unsafe trajectories and redistribution | [PNG](../figures/generated/article/Fig3.png) | [PDF](../paper/working/article/Fig3.pdf), [EPS](../paper/working/article/Fig3.eps) | [CSV](../paper/working/article/Figure3_data.csv) |
| 4 | Halfspace and projection geometry | [PNG](../figures/generated/article/Fig4.png) | [PDF](../paper/working/article/Fig4.pdf), [EPS](../paper/working/article/Fig4.eps) | [CSV](../paper/working/article/Figure4_data.csv) |
| 5 | Reference conversion and no-refit repair | [PNG](../figures/generated/article/Fig5.png) | [PDF](../paper/working/article/Fig5.pdf), [EPS](../paper/working/article/Fig5.eps) | Schematic; definitions in article |
| 6 | Complete learned-path diagnostics | [PNG](../figures/generated/article/Fig6.png) | [PDF](../paper/working/article/Fig6.pdf), [EPS](../paper/working/article/Fig6.eps) | [CSV](../paper/working/article/Figure6_data.csv) |
| 7 | Additional tilt deformation | [PNG](../figures/generated/article/Fig7.png) | [PDF](../paper/working/article/Fig7.pdf), [EPS](../paper/working/article/Fig7.eps) | [CSV](../paper/working/article/Figure7_data.csv) |
| 8 | January discovery and February follow-up | [PNG](../figures/generated/article/Fig8.png) | [PDF](../paper/working/article/Fig8.pdf), [EPS](../paper/working/article/Fig8.eps) | [CSV](../paper/working/article/Figure8_data.csv); [interval](../paper/working/article/Figure8_interval.csv) |

[Rebuild all eight figures](../figures/source/article_figures.py) from the included CSVs and analytic examples. The [pipeline generator](../figures/source/article_pipeline_diagrams.py) defines Figures 1, 2 and 5. Captions and equations are in the [article source](../paper/working/article/main.tex).

In Figure 7, Wikipedia has zero changed tilts under both repairs at all displayed seeds. Specified-rate Reddit values are also zero. The explicit baseline markers denote observed zeros, not missing data. The sharper certificate avoids additional deformation of the reference comparator; it does not establish a mean AP improvement.

## Repository overview figures

| Visual | Preview | Source and evidence |
|---|---|---|
| Complete scoring path and certificate domain | [PNG](../figures/generated/overview/scoring-path.png), [SVG](../figures/generated/overview/scoring-path.svg) | [Generator](../figures/source/readme_overview.py), [certificate assumptions](methods/Fixed-Rate-Readout-Certificate.md) |
| Structural behavior and predictive follow-up | [PNG](../figures/generated/overview/research-outcomes.png), [SVG](../figures/generated/overview/research-outcomes.svg) | [Generator](../figures/source/readme_overview.py), [numeric inputs and source hashes](../figures/generated/overview/plot-data.json) |

# Figure-to-file map

| Figure | Content | Preview | Vector artwork | Numeric input |
|---|---|---|---|---|
| 1 | Tensor-level neural scoring pipeline | [PNG](../figures/generated/sncs/Fig1.png) | [PDF](../paper/working/sncs/Fig1.pdf), [EPS](../paper/working/sncs/Fig1.eps) | Schematic; definitions in article |
| 2 | Fixed-candidate certificate workflow | [PNG](../figures/generated/sncs/Fig2.png) | [PDF](../paper/working/sncs/Fig2.pdf), [EPS](../paper/working/sncs/Fig2.eps) | Schematic; definitions in article |
| 3 | Safe/unsafe trajectories and redistribution | [PNG](../figures/generated/sncs/Fig3.png) | [PDF](../paper/working/sncs/Fig3.pdf), [EPS](../paper/working/sncs/Fig3.eps) | [CSV](../paper/working/sncs/Figure3_data.csv) |
| 4 | Halfspace and projection geometry | [PNG](../figures/generated/sncs/Fig4.png) | [PDF](../paper/working/sncs/Fig4.pdf), [EPS](../paper/working/sncs/Fig4.eps) | [CSV](../paper/working/sncs/Figure4_data.csv) |
| 5 | Reference conversion and no-refit repair | [PNG](../figures/generated/sncs/Fig5.png) | [PDF](../paper/working/sncs/Fig5.pdf), [EPS](../paper/working/sncs/Fig5.eps) | Schematic; definitions in article |
| 6 | Complete learned-path diagnostics | [PNG](../figures/generated/sncs/Fig6.png) | [PDF](../paper/working/sncs/Fig6.pdf), [EPS](../paper/working/sncs/Fig6.eps) | [CSV](../paper/working/sncs/Figure6_data.csv) |
| 7 | Additional tilt deformation | [PNG](../figures/generated/sncs/Fig7.png) | [PDF](../paper/working/sncs/Fig7.pdf), [EPS](../paper/working/sncs/Fig7.eps) | [CSV](../paper/working/sncs/Figure7_data.csv) |
| 8 | January discovery and February follow-up | [PNG](../figures/generated/sncs/Fig8.png) | [PDF](../paper/working/sncs/Fig8.pdf), [EPS](../paper/working/sncs/Fig8.eps) | [CSV](../paper/working/sncs/Figure8_data.csv); [interval](../paper/working/sncs/Figure8_interval.csv) |

[Rebuild all eight figures](../figures/source/sncs_all_figures.py) from the included CSVs and analytic examples. The [pipeline generator](../figures/source/sncs_pipeline_diagrams.py) defines Figures 1, 2 and 5. Captions and equations are in the [article source](../paper/working/sncs/main.tex).

In Figure 7, Wikipedia has zero changed tilts under both repairs at all displayed seeds. Specified-rate Reddit values are also zero. The explicit baseline markers denote observed zeros, not missing data. The sharper certificate avoids additional deformation of the reference comparator; it does not establish a mean AP improvement.

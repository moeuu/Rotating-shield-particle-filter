# RA-L Figure Quality Policy

This note defines mandatory checks before any generated figure is described as
ready for the RA-L manuscript.

## Mandatory Visual QA

After generating or updating a manuscript figure, always inspect the rendered
image, not only the source code or the LaTeX include command. For PDF figures,
also generate a raster review copy and inspect that copy at the approximate
paper size.

Reject and revise the figure if any of the following are visible:

- text, legends, panel labels, markers, axes, or arrows overlap in a way that
  makes either item hard to read;
- a panel label covers data, annotations, axes, or title text;
- axis aspect ratios or tick spacing distort the metric geometry;
- floor/background fills visually cover obstacles, sources, paths, or
  particles; geometry-bearing layers must be drawn above the background with a
  clear z-order;
- a 2-D map has redundant room frames or double borders from both a drawn room
  rectangle and axis spines;
- a plotted path implies motion through obstacles when only station locations
  are known;
- a panel does not make a concrete scientific point that is explained by the
  caption or main text;
- schematic elements imply physics, geometry, measurement counts, or algorithm
  behavior that is not actually used by the method or experiment.
- labels in the standalone figure or compiled PDF are below the readable
  RA-L/IEEE two-column scale. Unless there is a documented reason, generated
  figure labels, ticks, and legends should be at least 7 pt at final inclusion
  size, with panel titles around 8 pt and panel labels around 9 pt.

## Logical QA

Every panel must answer a specific manuscript question. If a panel only shows a
decorative component, a generic workflow, or an unlabeled implementation detail,
replace it with a figure that directly supports a claim in the paper.

Do not use a figure as a text container. A figure is justified only when it
clarifies something that is hard to understand from prose alone: geometry,
motion, occlusion, response signatures, uncertainty, model-order behavior, or
quantitative comparisons. If the content can be expressed equally well as a
short paragraph or an itemized list, keep it in the text instead of making a
text-only diagram.

Before redesigning a manuscript figure, inspect the closest related figures or
tables from prior work and record the design decision in the working notes or
commit message. For this RA-L manuscript, the default comparison set is:

- recursive Bayesian/PF radiation surveying figures showing algorithm stages,
  particle behavior, measurement paths, and convergence metrics;
- scene-aware attenuation or sparse reconstruction figures showing LiDAR/voxel
  geometry, source truth/estimates, detector trajectories, and count traces;
- source-separation result tables reporting cardinality, localization error,
  strength error, and runtime.

The new figure should explain what is visually missing from these prior figures
for this paper, namely the controlled Fe/Pb shield-time code and its coupling
to surface-constrained PF updates and active station selection.

For RA-L page budgeting, "eight pages" means the paper should use the eighth
page effectively, not merely stay below the limit. Do not create artificial
white space by over-compressing explanations or figures. Before reporting the
manuscript ready, inspect page 8 of the compiled PDF and revise if it is mostly
blank; use the space for necessary discussion, limitations, experimental
interpretation, or references while staying within eight pages.

For the current RA-L figures:

- Fig. 1 should show the problem setting and why rotating Fe/Pb postures create
  a temporal response code for separating surface sources.
- Fig. 2 should show how one station window turns known surfaces, obstacle
  paths, shield postures, spectra, and residual PF ambiguity into the next
  station/program decision. It must use rendered or explicitly 3-D views when
  explaining the Fe/Pb shield posture or 3-D obstacle/source geometry, and it
  must contain geometry, response signatures, or posterior/diagnostic graphics,
  not only text boxes.
- Fig. 1 and Fig. 2 must not spend their main panel on the same rendered view.
  Fig. 1 should establish the robotic problem setting; Fig. 2 should explain
  the shield hardware/program and inference mechanism.
- Experiment figures should show metric source-estimation results, obstacle
  geometry, final PF particle support, the saved obstacle-aware robot route,
  online model-order behavior, and compact ablation metrics.
  Result-map panels must state whether the plotted estimates are from the
  proposed method or a baseline and must include a marker legend for stations,
  saved route, PF particles, truth, estimates, and isotope colors.
  When the reported metric is 3-D localization, the main result panels should
  include a 3-D view or an equivalent paired projection that makes height
  errors visible.

Main-paper result tables should not be made unreadable to save space. Use a
consistent body font size across adjacent result tables, preferably `\small`;
if a table only fits with `\scriptsize`, simplify column labels or split
content before accepting it.

## Review Artifacts

`scripts/build_ral_figures.py` writes raster review copies by default to:

```bash
results/ral_figure_review/
```

Inspect these PNG files before reporting that the figure update is finished.
If the LaTeX PDF is rebuilt, inspect the compiled page as well, because a figure
that is readable as a standalone asset can still be too small or crowded in the
paper layout.

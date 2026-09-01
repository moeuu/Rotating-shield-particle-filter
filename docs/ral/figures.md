# RA-L Figure Policy

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

Every panel must answer a manuscript question and directly support a claim. Use
figures for geometry, motion, occlusion, response signatures, uncertainty,
model-order behavior, or quantitative comparison rather than as text
containers. Before redesign, inspect the closest PF-surveying,
scene-attenuation, or source-separation figures in prior work and record what
the new design adds: the Fe/Pb attenuation code and its coupling to
surface-constrained PF updates and active station selection.

Use the eighth page effectively. Inspect page 8 before reporting the manuscript
ready and restore necessary discussion, limitations, interpretation, or
references if it is mostly blank while remaining within the page limit.

## Figure Roles

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

The attenuation-code response matrix belongs in the method figure rather than
the result figure. This gives it a distinct explanatory role and leaves result
space for truth-estimate accuracy.

## Main Result Figure

The main result is the split-aware proposed-method PF result for the current
Cs4/Co3 task, not a generic dashboard. Use this five-panel grammar:

1. A metric floor projection with 2 m tick spacing, equal x-y aspect, known
   obstacles, saved obstacle-aware route, final PF particle support, truth,
   reported estimates, and truth-estimate match segments.
2. A metric height projection with 2 m tick spacing and equal scaling so wall,
   floor, obstacle, and high-surface errors remain visible.
3. A compact numerical source panel containing truth ID, merged raw-component
   count, 3-D centroid and RMS position errors, and signed aggregate strength
   error. Do not add a redundant pass/fail column.
4. An online diagnostic panel showing isotope-wise cardinality evolution and
   the hard-cap threshold. Raw cardinality is diagnostic only.
5. A per-true-source panel showing RMS position and relative strength error
   against the predeclared 0.5 m and 25% targets.

Identify sources by isotope and stable truth index. Include a clear marker
legend for stations, route, PF particles, truth, estimates, and isotope colors.
Keep the particle cloud visually secondary to truth and final estimates.

Do not connect stations with straight line segments. Draw a route only from
persisted obstacle-aware path waypoints; otherwise show station markers without
implying an unrecorded collision-free trajectory.

## Obstacle Rendering

Render obstacles from the authenticated environment artifact used by the run.
For grid environments, draw occupied-cell footprints. For component-based
environments, draw available component footprints and traversal-blocking
occupancy. Geometry that affects attenuation or reachability must remain
visible above background fills.

Main-paper result tables should not be made unreadable to save space. Use a
consistent body font size across adjacent result tables, preferably `\small`;
if a table only fits with `\scriptsize`, simplify column labels or split
content before accepting it.

## Figure-Source Data Preservation

Every result figure must remain reproducible after the run without rerunning
Geant4 or PF inference. Keep the numerical source data separate from rendered
PNG/PDF/SVG assets, and retain enough information to change axes, aggregation,
color, panel layout, normalization, or residual presentation later.

For full-simulation results, preserve the authenticated MeasurementLog
artifacts (including full spectra, exact energy-bin edges, detector poses,
shield indices, live times, and environment geometry), the final posterior and
weighted particle snapshot, the station/planner trace, and the evaluation
inputs. Diagnostics that introduce derived values must additionally save their
model predictions and the raw values from which residuals or summaries were
calculated; a rendered curve or aggregate statistic alone is insufficient.

Every derived figure-data payload must state units, bin coordinates, residual
or normalization formulas, filtering/exclusion rules, missing-value semantics,
and stochastic provenance. Preserve unrounded numerical values and apply
rounding only in the presentation layer. Publication artifact inventories must
hash the machine-readable source data together with the other run artifacts.

## Review Artifacts

Build current figures from an authenticated completed-run bundle and its exact
split-aware evaluation. Run-specific bundle construction and private paths
belong with that run, not in this policy. The reusable entry point is:

```bash
uv run python scripts/build_ral_figures.py \
  --completed-run-dir COMPLETED_BUNDLE \
  --split-aware-evaluation EVALUATION_JSON
```

Use `--skip-concepts` when only the result figure should change. The script
writes raster review copies by default to:

```bash
results/ral_figure_review/
```

Inspect these PNG files before reporting that the figure update is finished.
If the LaTeX PDF is rebuilt, inspect the compiled page as well, because a figure
that is readable as a standalone asset can still be too small or crowded in the
paper layout.

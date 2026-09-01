# RA-L Figure Policy

This document defines the permanent evidence, design, and review contract for
RA-L figures. Run-specific paths and conclusions belong in the run bundle, not
in this policy.

## Evidence-only rendering

Scientific figures must be rendered deterministically from authenticated
numeric artifacts. AI-generated or AI-edited imagery is prohibited. Do not
invent a route, obstacle height, posterior sample, uncertainty region,
measurement, or response curve for presentation. Simulation camera images may
be used only when they are reproducible outputs of the declared simulation
scene and their role is contextual rather than quantitative.

Keep source arrays, transformations, and presentation separate. Rounding,
normalization, resampling for display, alpha, color, and camera selection are
presentation operations and must not overwrite source data.

## Design basis from prior 3-D radiation mapping work

Use the following precedent as a design basis, not as evidence for this
project's performance:

- Vavrek et al.'s
  [3-D scene-data-fusion source reconstruction](https://arxiv.org/abs/2009.07303)
  combines contextual plan views with a separate 3-D source reconstruction
  and orthogonal quantitative projections.
- Bandstra et al.'s
  [free-moving quantitative gamma-ray imaging](https://arxiv.org/abs/2104.11318)
  shows detector motion and 3-D material occupancy for context, then uses
  top/side views and spatial confidence information for quantitative reading.
- Lee et al.'s
  [mobile robot radiation mapping](https://arxiv.org/abs/1802.06072) uses the
  reconstructed 3-D scene, robot trajectory, and different shapes for truth
  and estimates in the same spatial frame.
- Pavlovsky et al.'s
  [3-D radiation mapping in real time](https://arxiv.org/abs/1908.06114)
  demonstrates the value of fusing radiation results with contextual 3-D scene
  geometry rather than showing radiation coordinates in an empty box.

The resulting project rule is: a 3-D panel establishes physical context and
occlusion, but it is never the sole localization-accuracy display. Pair it with
equal-scale orthogonal projections and a compact numerical panel. Perspective
occlusion and camera angle must not determine whether an error is visible.

## Geometry semantics

Navigation occupancy and physical obstacle geometry are different data:

- navigation occupancy states where the robot may travel;
- collision geometry states the physical volumes that block traversal; and
- transport geometry states the material volumes used for attenuation.

In a 3-D overview, render the exact authenticated transport components when
available, otherwise exact collision components. Use translucent solid faces
and visible edges so nested or hollow structures remain legible. Preserve the
physical x-y-z aspect ratio and use an orthographic camera by default. Do not
flatten components into floor patches.

In the floor projection, navigation cells may appear as a faint background,
while physical component footprints use a separate darker encoding. In the
height projection, show component z extents. If only grid occupancy exists, an
extruded grid fallback is allowed only when the figure or caption explicitly
labels it as an occupancy-derived approximation; it must not be described as
the true obstacle shape.

The room floor or axes must not obscure obstacles, sources, paths, or posterior
support. Metric axes require equal scale in each displayed coordinate pair.

## Live CUI and final run views

The truth-free CUI and the saved final CUI images must use the same scene
semantics:

- exact physical obstacle components in the 3-D PF view;
- a distinct navigation-occupancy layer in the plan view;
- current detector pose, measurement stations, and only the persisted runtime
  travel waypoints as the route;
- isotope color plus marker shape for posterior components and point estimates;
- metric axes, physical box aspect, and an orthographic 3-D camera; and
- truth only in the separately authorized evaluation overlay, never in the PF
  control view.

Do not connect station locations with straight segments. When no persisted
travel waypoints exist, show station markers and state that the route is
unavailable. A line must never imply motion through an obstacle merely because
the endpoints are measurement stations.

## Completed-run case audit

The deterministic completed-run audit uses six panels:

1. an authenticated 3-D scene containing physical obstacles, stations, any
   saved route, posterior support, truth, raw PF components, merged centroids,
   and truth-to-centroid links;
2. an equal-scale floor projection that distinguishes navigation occupancy
   from physical component footprints;
3. an equal-scale depth-height projection that exposes vertical error and
   obstacle height;
4. a compact numerical source table with truth ID, assigned raw-component
   count, merged-centroid error, split-width-sensitive RMS position error, and
   signed aggregate strength error, without a redundant pass/fail column;
5. online isotope-wise cardinality and hard-cap diagnostics; and
6. per-source RMS position and strength errors normalized by the declared
   0.5 m and 25% targets.

Raw components remain visible because they explain splitting; the merged
centroid is the one-source summary. Particle support must remain visually
secondary. Use isotope color and marker shape redundantly, and provide one
shared legend for obstacles, stations, route when present, PF support, truth,
raw components, merged centroids, and error links.

This case audit is not automatically a headline manuscript result. Until all
four prespecified variants in one valid comparison batch are complete, keep it
as a review or supplementary artifact. Do not let a proposed-only run imply a
completed ablation comparison.

## Final manuscript comparison

The final comparison must be comparison-first and use the same environment,
scales, evaluation rule, and visual encodings for all four variants. Show
source-level distributions or paired values together with aggregate summaries;
do not reduce the evidence to one favorable scene image. A compact shared-scene
3-D context panel may accompany the comparison, but orthogonal projections or
explicit 3-D error metrics remain mandatory for a 3-D localization claim.

The current manuscript has two live figure roles. Adding a result figure later
requires a deliberate manuscript-budget decision: replace or restructure an
existing figure, or explicitly revise the budget after the complete comparison
exists. Do not silently publish the current case audit as a third figure.

The attenuation-code response matrix belongs in the method figure, not the
result figure. This preserves result space for localization, uncertainty,
model-order behavior, and the four-variant comparison.

## Figure-source data preservation

Every result must remain redrawable without rerunning Geant4 or PF inference.
Retain and hash:

- the authenticated MeasurementLog, including full spectra, exact energy-bin
  edges, detector poses, shield indices, live times, station identities, route
  waypoint metadata, and environment geometry;
- the final posterior and weighted particle snapshot;
- station, planner, residual, cardinality-transition, and performance traces;
- the exact evaluation input and evaluation artifact; and
- `pf_figure_data.json`, which binds truth-free route and station display data
  to the run ID and MeasurementLog digest.

For legacy runs without `pf_figure_data.json`, the renderer may read exact
`travel_waypoints_xyz` values from authenticated `observation_metadata.jsonl`.
When both forms exist, they must agree exactly or figure generation fails.

Derived diagnostic payloads must preserve predictions and raw observations,
not only residual plots or aggregate values. State units, bin coordinates,
formulas, filtering rules, missing-value semantics, and stochastic provenance.
Save unrounded values and round only in the renderer.

## Mandatory visual QA

After every figure change, inspect the rendered image itself at approximate
paper size. For a PDF, also inspect a raster review copy and the compiled paper
page when the figure is live. Reject and revise if:

- text, legends, titles, markers, axes, or arrows overlap;
- the metric aspect ratio or tick spacing is distorted;
- obstacle faces hide the sources or make component shape unreadable;
- translucent layers combine into an opaque mass that conceals evidence;
- a route crosses obstacles because unsaved segments were inferred;
- the 3-D camera hides a source/error that the companion projections do not
  recover;
- color is the only distinction between scientific categories;
- a panel does not support a stated manuscript question; or
- any element implies physics, geometry, counts, uncertainty, or algorithm
  behavior not present in the authenticated data.

At final inclusion size, labels, ticks, and legends should normally be at least
7 pt, panel titles about 8 pt, and panel labels about 9 pt. Simplify content
before reducing below those sizes.

## Reusable build path

Build the case audit from an authenticated completed-run bundle and its exact
split-aware evaluation:

```bash
uv run python scripts/build_ral_figures.py \
  --skip-concepts \
  --completed-run-dir COMPLETED_BUNDLE \
  --split-aware-evaluation EVALUATION_JSON
```

The default output and raster review copies stay in
`results/ral_figure_review/`. Promotion into the external manuscript workspace
is a separate, explicit step after the comparison and page budget are ready.

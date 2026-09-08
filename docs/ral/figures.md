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

An Isaac Sim environment capture must be rebuilt from the persisted room,
obstacle, source, robot, detector, shield, route, and station records, and its
camera, renderer, source paths, hashes, and shield-pair identities must be
recorded. A line labelled as a Geant4 particle track must consist only of
recorded points from an actual Geant4 event history. Do not draw, interpolate,
or bend illustrative rays. If sparse event tracks are selected for legibility,
retain the unmodified raw track artifact and record the selection rule, source
index, primary-history identifier, track identifier, transport mode, random
seed, and detector-entry status for every displayed track. A straight
source--detector segment may be shown only as a geometric relation and must be
labelled as such, never as a particle history.

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

The CUI and manuscript result projections must call the same metric scene
renderer for physical obstacles, persisted route segments, station markers,
station order, coordinate projection, and aspect ratio. Manuscript-only truth
and evaluation overlays may be added after inference, but must not replace or
silently restyle that common base. Show one coordinate frame only: do not draw
a second full-room rectangle inside normal plot axes.

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

The compact manuscript scene view uses two large, matched metric projections of
the same authenticated bundle. Treat the x--y floor map and x--z elevation as
equal evidence: keep both readable at the same visual level and repeat the same
station acquisition numbers in both panels so their positions can be matched.
Use the saved CUI source-marker grammar in both panels: a star denotes truth, a
large diagonal cross denotes the estimate, and red/blue plus the adjacent label
denote Cs-137/Co-60. Do not substitute filled crosses or isotope-specific truth
shapes in the manuscript version.
The floor map shows the recorded route and obstacle footprints; the elevation
shows obstacle and station heights. This layout inherits the readable spatial
grammar of the saved CUI result while replacing its navigation blocks and raw
modes with exact physical components and split-aware centroids. Do not add a
truth--estimate connector to this compact view: the marker displacement already
shows the error, while exact 3-D errors belong in the adjacent table. Fig. 1
provides the companion 3-D shape context. The result view may use one
representative completed run only when the caption states that truth is a
post-acquisition evaluation overlay and all comparative claims remain grounded
in the matched all-variant tables.

## Final manuscript comparison

The final comparison must be comparison-first and use the same environment,
scales, evaluation rule, and visual encodings for all four variants. Show
source-level distributions or paired values together with aggregate summaries;
do not reduce the evidence to one favorable scene image. A compact shared-scene
3-D context panel may accompany the comparison, but orthogonal projections or
explicit 3-D error metrics remain mandatory for a 3-D localization claim.

The current manuscript has three live figure roles:

1. a wide Isaac Sim environment rendering reconstructed from the evaluated
   room, showing the mobile robot, actual obstacle shapes, surface sources,
   recorded route/stations, and the detector head in one spatial context;
2. four readable close-up renders of one recorded detector/shield sequence,
   showing a fixed detector pose and independent Fe/Pb octant rotation; and
3. equally weighted floor and elevation result projections that expose the
   robot route, matching station order, physical obstacle dimensions, source
   height, and split-aware estimate.

Context renders must label the major visual entities inside the image as well
as in the legend: robot/sensing head, physical obstacles, recorded route and
stations, and each isotope class. Use sentence case for prose labels and panel
titles. Lowercase mathematical axis symbols such as $x$, $y$, and $z$ retain
their conventional form.

The environment overview uses one green encoding for actual isotropically
emitted Geant4 histories. Display three directionally separated histories per
source so emission from every source is visible without turning the scene into
a dense track cloud. Do not add a second detector-entry track class when the
figure's role is only to establish emitted-radiation context.

When selecting shield-mechanism panels, keep the detector pose and camera fixed,
render every acquired Fe/Pb pair candidate, and choose the displayed subset by
a deterministic spatial-legibility rule. Preserve all candidate renders and
their recorded pair identities. The four displayed panels must make the two
octants' independent spatial motion easier to read than a temporal label alone.
Keep the legend to physical components (detector, Fe, and Pb); describe fixed
pose and independent rotation in the caption instead of assigning those
statements line-symbol entries that do not identify visible objects.

The proposed inference/planning loop belongs in equations and a compact
algorithm whose steps match the method subsections. Do not spend a figure on a
text-box flowchart when it conveys no spatial, physical, or quantitative
evidence. The six-panel case audit remains a review artifact; the third live
figure is the compact scene view and does not replace the all-variant tables.

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

When event trajectories appear, also retain the raw Geant4 step records,
trajectory-export configuration, executable and source digests, physics list,
geometry identity, source and transport seeds, source/history/track identities,
and the deterministic display selection. A raster or selected polyline set is
not a substitute for the raw event records.

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

First regenerate the contextual raster captures with Isaac Sim using the
project capture script. Then build all three manuscript figures and the case
audit from the authenticated completed run and its exact split-aware
evaluation:

```bash
uv run python scripts/build_ral_figures.py \
  --completed-run-dir COMPLETED_BUNDLE \
  --split-aware-evaluation EVALUATION_JSON
```

The build command writes the six-panel audit and raster review copies under
`results/ral_figure_review/`, writes the three live figures to the external
manuscript workspace, and binds every output to the Isaac capture provenance,
run, MeasurementLog, truth manifest, and split-aware evaluation inputs.

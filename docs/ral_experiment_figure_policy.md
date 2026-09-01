# RA-L Experiment Figure Policy

This note defines the figures that should be produced for RA-L result updates.
It is intended to keep future simulations comparable and to avoid ad-hoc
screenshots that do not directly support the paper claims.

Before reporting a figure as ready, follow
`docs/ral_figure_quality_policy.md`: inspect the rendered image, reject any
text/marker overlap, and confirm that every panel has a clear scientific role.

## Main Result Figure

The main paper result figure is a PF result figure for the
`cs4_co3_multi_source_cardinality` task, not a generic ablation dashboard. It
should show where the robot measured, where the sources actually were, where
the proposed PF/reporting pipeline placed the final estimates, and whether the
final PF particle support is consistent with the reported result.

Use the following five-panel grammar:

1. A metric floor projection with 2 m tick spacing, equal x-y aspect, known
   obstacles, saved obstacle-aware robot route, final PF particle support,
   ground-truth sources, final reported estimates, and truth-estimate match
   segments.
2. A metric height projection (`y-z` or `x-z`, whichever better separates the
   sources) with 2 m tick spacing and equal axis scaling so wall, floor,
   obstacle, and high-surface errors are visible.
3. A compact numerical source-result panel showing the truth ID, number of raw
   components in its merged cluster, 3-D centroid and RMS position errors, and
   signed aggregate strength error. Do not add a redundant pass/fail column.
4. An online diagnostic panel showing isotope-wise cardinality posterior or MAP
   evolution and the hard-cap threshold. Raw cardinality is diagnostic only.
5. A per-true-source error panel showing 3-D position error and relative
   strength error against the prespecified 0.5 m and 25% performance targets.
   Identify sources by isotope and stable truth index.

The floor and height projections are the primary result panels. Together they
make the 3-D localization error auditable at RA-L print scale without relying
on a perspective view. A 3-D view can be added in supplementary material, but it
should not displace the metric projections in the main paper.

The attenuation-code response matrix belongs in the method figure, not the
result figure. This gives it a distinct explanatory role and leaves enough
result space for truth-estimate accuracy.

Do not connect measurement stations with straight path lines. Draw route lines
only from saved obstacle-aware path waypoints. A straight line between stations
can falsely imply that the robot drove through obstacles.

## Obstacle Rendering

Obstacles should be rendered from the same known environment manifest used by
the run. For grid-based layouts, draw occupied cell footprints. For arbitrary
or Manchester-derived objects, draw the component footprints when available and
the traversal-blocking occupancy underneath. The goal is not a decorative 3-D
render; the figure should show the geometry that affects source-detector
occlusion, planning, and PF attenuation.

## Rebuild Command

For the completed proposed run, bind the durable PF output, truth-free
MeasurementLog, and private truth manifest into a temporary read-only bundle,
then render it with the schema-v3 split-aware evaluation:

```bash
bundle_dir="$(mktemp -d /tmp/ral-paper-result.XXXXXX)"
ln -s "$PWD/results/ral_ablation/runs/ral_a3fde7067c4ac222_proposed" \
  "$bundle_dir/pf_output"
ln -s "$PWD/results/ral_ablation/measurement_logs/ral_a3fde7067c4ac222_proposed" \
  "$bundle_dir/measurement_log"
ln -s "$PWD/../Rotating-shield-simulation-runtime/private_runs/ral_ablation/truth_manifests/ral_a3fde7067c4ac222_proposed.json" \
  "$bundle_dir/truth_manifest.json"
uv run python scripts/build_ral_figures.py \
  --skip-concepts \
  --completed-run-dir "$bundle_dir" \
  --split-aware-evaluation ../Rotating-shield-simulation-runtime/private_runs/ral_ablation/evaluations/ral_a3fde7067c4ac222_proposed_split_aware_v3.json
```

After the fresh Cs4/Co3 batch is complete, generate the main paper result from the
four evaluator outputs for `proposed`, `no_shield_native_path`,
`round_robin_shield`, and `eig_only_path`. Never mix result files from different
opaque batch IDs.

The output is written to:

- `sections/05_experiments/figures/ral_result_overview.pdf`

Run the same script without `--skip-concepts` whenever Fig. 1 or Fig. 2 needs
to be refreshed:

```bash
uv run python scripts/build_ral_figures.py
```

The script also writes review PNGs to `results/ral_figure_review/` by default.
Use these images for the mandatory visual QA pass.

## Supplementary Figures

The final main paper figure should focus on the Cs4/Co3 multi-source task because it contains
spectral isotope separation, variable isotope-wise source cardinality, and
same-isotope spatial ambiguity in one scene.

The main result figure should include final PF particle support when the run
summary contains it. The particle cloud must be visually secondary to truth and
final report markers, but it is important evidence that the reported point
estimate is supported by the posterior.

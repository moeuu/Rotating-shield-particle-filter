# RA-L Minimal Ablation Plan

This file is the shared project note for the RA-L paper ablation scope. New
Codex sessions should use this plan unless the user explicitly changes it.

## Decision

Generate one fresh scene seed for each new experiment batch. Record that seed
only in the sibling runtime's private config, scenario, truth manifest, and
private experiment manifest. PF configs, PF output tags, MeasurementLog, and
estimator-visible adaptive events must not contain it.

- Omit `--seeds` for a new independent batch.
- Use an explicit `--seeds <recorded-seed>` only to repeat a recorded live batch.
- PF seeds are generated independently and reused across variants in the same
  comparison batch. For an exact replay, pass a separately recorded
  `--pf-seeds <recorded-pf-seed>`; it must not equal the scene seed.
- Geant4 transport seeds are also generated independently. The transport seed
  may remain in physical provenance, but it must not equal the private scene
  seed or PF seed and therefore cannot reconstruct source placement.
- All four methods in one comparison batch use the same generated environment
  and truth layout. A later batch must use a new seed.

Use one main RA-L task. Past Case01/Case02/Case03 paper cases are not part of
the standard RA-L ablation implementation:

- `mix9_multi_isotope_cardinality`
- ground truth source cardinality: `4 Cs-137 + 3 Co-60 + 2 Eu-154`
- source support: continuous walls, floor, ceiling, and every exposed
  transport-component face from the shared physical surface geometry
- random placement is uniform with respect to physical surface area,
  conditioned on a predeclared 3.0 m minimum 3-D Euclidean distance between
  sources of the same isotope
- no height preference, visibility filter, ceiling-count cap, or
  response-observability screening is applied

The same-isotope distance is a geometry-only hard-core condition applied to a
complete proposed layout. It is not computed from detector poses, simulated
counts, PF responses, or holdout results. The complete-layout rejection sampler
therefore remains symmetric in source order and samples the physical-area
measure conditioned on the declared separation event. Three metres is the
fixed RA-L experiment-design value: it removes configurations that are not a
meaningful test of separate-source recovery without screening locations for
favourable visibility or response conditioning.

The runtime is the single production source of the RA-L physical environment.
`runtime.scenarios.RAL_ENVIRONMENT_CONFIG` in the sibling simulation-runtime
repository defines the 10 x 15 x 5 m room, and both the runtime scene and the
truth-free environment payload are derived from that object. Estimator
repositories must not redeclare those dimensions; they consume the environment
from the adaptive-session context and finalized MeasurementLog.

Use the following fixed PF acquisition contract for every paper variant:

- at most 16 stations;
- 8 shield views per complete station;
- 20.0 s live time per view;
- at most 128 measurements;
- 2560 s (42 min 40 s) maximum detector live time;
- `min_station_separation_m = 3.0` and `coverage_radius_m = 3.0`.

The station-separation term remains a planner penalty rather than a hard
geometric exclusion. Physical reachability and collision clearance continue to
come only from the runtime candidate contract.

Run only four closed-loop full-simulation variants for the main paper table:

- `proposed`
- `baseline_passive_equal_time_no_shield`
- `round_robin_shield`
- `eig_only_path`

Use the same source-count support across all variants:

- `max_sources = 5` per isotope

This is a method-level search support, not a value inferred from the known
ground-truth source count in the task.

## Rationale

The paper claim is multi-isotope source-term estimation with isotope-wise source
cardinality, 3-D localization, strength estimation, and same-isotope ambiguity
inside multiple radionuclide channels. The single MIX-9 task exercises all of
these mechanisms in one expensive run.

- `proposed` is the reference method.
- `baseline_passive_equal_time_no_shield` disables shield coding while
  preserving the same per-station physical live-time budget. It tests whether
  longer nondirectional dwell alone is sufficient.
- `round_robin_shield` keeps the Fe/Pb shield and the same posture budget but
  removes posterior-adaptive shield-program selection. It tests whether the
  hardware alone is sufficient.
- `eig_only_path` keeps active planning and shield programs but removes the
  explicit same-isotope signature, response-correlation, obstacle-shadow, and
  elevation terms from DSS-PP. It tests whether ordinary information-driven
  planning is sufficient.

Every variant uses `pure_pf_schema_version: 1`, the `pf_strict` profile, and the
same exact reversible-jump particle filter. Ablations vary only the declared
shield and planning policies above.

## Generated Files

Regenerate the exhaustive manifest and then the compact paper subset:

```bash
PYTHONPATH=src uv run python -m baselines.ral_ablation.cli
uv run python scripts/build_ral_paper_subset.py
```

The truth-bearing paper subset files are owner-readable artifacts below the
sibling runtime repository:

- `../Rotating-shield-simulation-runtime/private_runs/ral_ablation/ral_paper_subset_manifest.csv`
- `../Rotating-shield-simulation-runtime/private_runs/ral_ablation/run_paper_subset.sh`

Each row first invokes the sibling runtime's
`rotating-shield-sim generate-ral-scenario`, which writes both the private
scenario and a separate private truth manifest keyed by opaque `run_id`. It then
starts the RA-L-only session adapter. The adapter gives the generic PF controller
only an owner-only Unix socket, a truth-free PF config, and a separate RA-L control
policy. The PF process receives no scenario path, source profile, scene seed, or
source RNG provenance.

Truth-bearing scenarios, truth manifests, physical runtime overrides, manifests,
and run scripts are kept below the sibling repository's ignored
`private_runs/ral_ablation/`; they are never written under this repository's
`results/`. Each session publishes a unique truth-free MeasurementLog below
`results/ral_ablation/measurement_logs/<output-tag>` and PF outputs below
`results/ral_ablation/runs/<output-tag>`. These targets must not already exist
when the run starts; archive them before repeating a recorded live batch.
Post-run evaluation must call
`evaluation.private_truth.load_private_truth_for_completed_result` with
`closed_loop_result.json` and the corresponding private truth manifest. The
loader rejects incomplete results and mismatched `run_id` values.

Run the selected full simulations with:

```bash
bash ../Rotating-shield-simulation-runtime/private_runs/ral_ablation/run_paper_subset.sh
```

Regenerate the RA-L manuscript figures after paper-scope results are available:

```bash
uv run python scripts/build_ral_figures.py
```

The experiment figure policy is recorded in
`docs/ral_experiment_figure_policy.md`. Before marking any figure ready, apply
the visual and logical QA checklist in `docs/ral_figure_quality_policy.md` and
inspect the generated review PNGs.

## Current Result Notes

The current PDF may contain placeholder or old representative values while new
MIX-9 experiments are pending. Treat those values as replaceable table entries;
do not change the agreed experiment scope just to match an old result table.

Every variant uses the same immutable joint full-spectrum observation model.
There is no shield-pair-specific variance inflation, isotope-count rescue, or
baseline-specific observation correction.

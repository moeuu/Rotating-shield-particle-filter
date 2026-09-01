# RA-L Minimal Ablation Plan

This file is the shared project note for the RA-L paper ablation scope. New
Codex sessions should use this plan unless the user explicitly changes it.

## Decision

Generate one fresh scene seed for each new experiment batch. Record that seed
only in the sibling runtime's private config, scenario, truth manifest, and
private experiment manifest. PF configs, PF output tags, MeasurementLog, and
estimator-visible adaptive events must not contain it.

- Omit `--seeds` for a new independent batch.
- Use explicit `--seeds`, `--pf-seeds`, `--transport-seeds`, and `--batch-ids`
  together only to repeat a recorded live batch. Partial replay provenance is an
  error; the generator never invents a missing replay seed or batch identity.
- PF seeds are generated independently and reused across variants in the same
  comparison batch. For an exact replay, pass a separately recorded
  `--pf-seeds <recorded-pf-seed>`; it must not equal the scene seed.
- Geant4 transport seeds are also generated independently. The transport seed
  may remain in physical provenance, but it must not equal the private scene
  seed or PF seed and therefore cannot reconstruct source placement.
- All four methods in one comparison batch use the same generated environment
  and truth layout. The private manifest binds them to one opaque batch ID,
  experiment profile, `cs4-co3` scene variant, scene seed, PF seed, and transport
  seed. A later batch must use a new seed and batch ID.

Use one main RA-L task. Past Case01/Case02/Case03 paper cases are not part of
the standard RA-L ablation implementation:

- `cs4_co3_multi_source_cardinality`
- ground truth source cardinality: `4 Cs-137 + 3 Co-60`
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
`runtime.experiment_profiles.CS_CO_SURFACE_SEARCH_PROFILE` in the sibling runtime
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
- `no_shield_native_path`
- `round_robin_shield`
- `eig_only_path`

Use the same source-count support across all variants:

- `max_sources = 5` per isotope

This is a method-level search support, not a value inferred from the known
ground-truth source count in the task.

## Rationale

The paper claim is joint multi-isotope source-term estimation with unknown
isotope-wise cardinality, 3-D localization, strength estimation, and
posterior-adaptive attenuation coding. The single Cs4/Co3 task exercises these
mechanisms in one expensive run.

- `proposed` jointly uses the Fe/Pb pose-pair code in the full-spectrum SMC/RJ
  likelihood and chooses each eight-view code and detector station from the
  current posterior.
- `no_shield_native_path` physically removes Fe/Pb attenuation while preserving
  the exact native DSS-PP implementation, candidate contract, station budget,
  eight views, and 20 s live time per view. It tests whether the directional
  attenuation code adds information beyond movement and dwell. "Native path"
  means the same planner algorithm, not forced equality of the resulting poses;
  removing the shield changes the predictive distribution and can therefore
  change the selected pose.
- `round_robin_shield` keeps the Fe/Pb shield and the same posture budget but
  injects an independently implemented deterministic round-robin eight-pair
  code. Native DSS-PP still scores and selects detector poses using that forced
  code. It tests whether posterior-adaptive code design adds value beyond the
  hardware itself.
- `eig_only_path` keeps active planning and shield programs but selects poses
  only by full-spectrum EIG minus the runtime-authored horizontal, mast, and
  settling-time costs. Its coverage score, coverage floor, coverage-reserved
  shortlist slot, bearing/frontier/revisit/turn terms, local-orbit branch, and
  elevation branch are explicitly disabled. Disabled local-orbit and elevation
  parameters use schema-v2 null/empty sentinels, so no numeric setting is
  silently retained behind a zero weight.

Every variant uses `pure_pf_schema_version: 2`, the `pf_strict` profile, and the
same shield-conditioned full-spectrum reversible-jump particle filter. The
ablation interventions live in the RA-L-only adapter/config factory and do not
add fallback branches to the proposed estimator.

## Generated Files

Regenerate the exhaustive manifest and then the compact paper subset:

```bash
PYTHONPATH=src uv run python -m baselines.ral_ablation.cli
uv run python scripts/build_ral_paper_subset.py
```

If the exhaustive manifest contains more than one recorded batch, select one
explicitly with `--batch-id <recorded-batch-id>`. Scene-seed-only subset
selection is intentionally unsupported because it permits cross-batch row
binding.

The truth-bearing paper subset files are owner-readable artifacts below the
sibling runtime repository:

- `../Rotating-shield-simulation-runtime/private_runs/ral_ablation/ral_paper_subset_manifest.csv`
- `../Rotating-shield-simulation-runtime/private_runs/ral_ablation/run_paper_subset.sh`

Each row first invokes the sibling runtime's
`rotating-shield-sim generate-scenario`, which writes both the private
scenario and a separate private truth manifest keyed by opaque `run_id`. The
command binds the experiment profile and `cs4-co3` scene variant explicitly;
runtime defaults are not part of the paper contract. It then
starts the RA-L-only session adapter. The adapter gives the generic PF controller
an owner-only adaptive Unix socket, an opaque renderer-overlay endpoint, a
truth-free PF config, and a separate RA-L control policy. The PF process receives no
scenario path, truth payload, source profile, scene seed, or source RNG provenance.
Only the asynchronous renderer child reads the overlay endpoint; it labels truth in
the CUI without placing truth in PF frames or controller results.

The generated run script authors all four scenarios before starting any adaptive
session. It then runs `baselines.ral_ablation.batch_contract`, which requires the
four authored scenarios to have byte-canonical equality of the comparison-bearing
environment, obstacle geometry, source positions and strengths, isotope set,
scene RNG provenance, and acquisition contract. Run IDs and output/config paths
remain variant-specific and are excluded from the shared comparison digest. The
owner-only contract is written below `private_runs/ral_ablation/batch_contracts/`;
any mismatch aborts the whole batch before the proposed run begins.

Each control-policy file is an exact schema-version-2 document with exactly
`schema_version`, `variant`, and `shield_policy`; it has no aliases, defaults,
or unknown members. The private manifest records the SHA-256 digest of
its exact source bytes, and the session command must supply that digest back as
`--expected-control-policy-sha256`. The controller validates the digest and the
complete discriminated policy before opening the runtime socket. The same sealed
source/canonical identity and policy content are included in the live resolved
configuration hash, final posterior provenance, serialized PF state, and checkpoint
manifest. Replacing a policy file with another valid variant policy is therefore a
hard preflight error.

All four variants require a complete native `DSSPPConfig` and
`planning_eig_samples >= 2`; the retired fixed-path/null-planner lifecycle is
not representable in the current policy schema. Round-robin retains native pose
planning but explicitly sets both
`shield_view_count_shadow_enabled = false` and
`conditional_greedy_one_swap = false` because its shield program is externally
forced. These are required inactive-mode sentinels rather than inherited defaults.

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
The private session runner performs this join automatically after every
successful acquisition, using the fixed cluster-accuracy policy in
`docs/post_run_cluster_accuracy_policy.md`, and writes the detailed report to
the sibling runtime's ignored `private_runs/ral_ablation/evaluations/`
directory. The live truth payload is delivered only to the renderer child through
the dedicated single-response socket and is never serialized into controller state.

Every generated private runtime configuration is a complete strict production
document. Runtime-config inheritance and the retired weighted/capped-history
controls are not emitted. The no-shield ablation declares only
`shield_transmission_target = 1.0`; the shared shield geometry resolves that
physical target to zero Fe/Pb thickness.

Run the selected full simulations with:

```bash
bash ../Rotating-shield-simulation-runtime/private_runs/ral_ablation/run_paper_subset.sh
```

Each private session prints an initially absent stop-sentinel path below
`private_runs/ral_ablation/stop_requests/`. To retain a long but valid prefix,
create that exact empty file with `touch`. A request made earlier remains pending;
after ten completed stations, the controller finishes the current complete station,
seals its causal PF state, asks the runtime to finalize the exact MeasurementLog
prefix, publishes the normal result bundle, and runs the standard private-truth
evaluation with `stop_reason=station_boundary_stop_requested`. A symlink, stale
pre-run sentinel, or nonempty sentinel is an error. Killing a process or encountering
an inference invariant failure still aborts and never becomes a successful result;
the sentinel is the only supported partial-budget finalization path.

Every run also publishes `pf_station_performance.jsonl`. It separates acquisition,
PF update and its internal report stage, live-health diagnostics, CUI enqueue work,
and next-station planning. CUI still publishes every acquired view, but immutable
particle-display arrays are reused within a station and the former duplicate
station-end posterior redraw is removed. Live health uses the compact diagnostic
path and no longer copies unused rejuvenation/cache detail. These changes affect
only reporting and execution overhead; observations, particles, likelihood, RJ
moves, DSS-PP, EIG sample counts, and Geant4 histories are unchanged.

Regenerate the RA-L manuscript figures after paper-scope results are available:

```bash
uv run python scripts/build_ral_figures.py
```

The experiment figure policy is recorded in
`docs/ral_experiment_figure_policy.md`. Before marking any figure ready, apply
the visual and logical QA checklist in `docs/ral_figure_quality_policy.md` and
inspect the generated review PNGs.

## Result Reporting Contract

Do not place placeholder comparison numbers in the paper. Until the paired
four-run Cs4/Co3 batch is complete, an older completed run may appear only as a
clearly labelled predecessor-code diagnostic. It may demonstrate plotting and
failure analysis, but it cannot establish the proposed method's current
accuracy or comparative advantage.

Report the paired four-run batch with the fixed policy in
`docs/post_run_cluster_accuracy_policy.md`. The main outcomes are true-source
association recall; per-source merged-centroid error, strength-weighted RMS
3-D position error, split dispersion, and relative strength error; the joint
0.5 m RMS/25% pass fraction; response-distinct remote-component count; hard-cap
mass; and station/time to a stable pass. Raw RJ component count is diagnostic,
not a success target, because nearby components may represent one physical
source cluster. Also report mission motion time separately from the fixed
2560 s detector live-time budget.

The four variants share one scene within one fresh batch, so report paired
descriptive differences. Do not treat seven sources as independent experiments
or attach unsupported p-values to a single scene.

Every variant uses the same immutable joint full-spectrum observation model.
There is no shield-pair-specific variance inflation, isotope-count rescue, or
baseline-specific observation correction.

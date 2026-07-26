# Pure PF architecture

## Estimator boundary

The scientific runtime has one estimator profile, `pf_strict`, and one
sequential data flow:

```text
raw spectrum + detector pose + shield state
        -> response_poisson count extraction and covariance
        -> append one MeasurementLog record
        -> sequential particle-filter update
        -> PF posterior snapshot
             |-> PF-only DSS-PP belief
             `-> PF posterior point report
```

The log append precedes the update. Ground truth is stored separately and is
used only for evaluation. Runtime and replay do not contain a second estimator,
an accumulated-history fit, a report-time position or strength optimizer, or a
posterior-external source rescue path.

`pf.profiles.resolve_estimator_profile` accepts only `pf_strict`. Source
cardinality, three-dimensional position, strength, and background are particle
state throughout the run. Compatibility configuration keys for the deleted
estimator and rescue paths are invalid rather than dormant switches.

## Physical source support

The standard Python, Geant4, and Isaac Sim entry points set
`source_surface_prior=true`. Source candidates are generated on the physical
environment surfaces:

- floor and ceiling;
- room walls;
- exposed obstacle tops and sides.

The same surface support is used for initial particles, structural proposals,
roughening, and the final posterior projection. The standard RAL factory does
not provide a full-volume source-prior ablation. Detector measurement poses
remain collision-free points in reachable free space; detector support and
source support are intentionally different physical domains.

## Sequential likelihood and structural moves

The PF likelihood consumes the logged `response_poisson` isotope counts and
their covariance. Transport-model, spectrum-model, counting, and correlated
station-view uncertainty are composed once by the production likelihood.
Residual birth and death evidence is evaluated with that same effective
likelihood and covariance semantics.

Birth, death, split, and merge remain PF-internal structural proposals. They do
not invoke an independent strength fit. A birth uses a residual-conditioned
strength proposal bounded by the predeclared physical support, then assesses
that fixed proposed state with the PF likelihood.
Birth is admitted by positive likelihood evidence; death is admitted when
removing a source improves the same likelihood after the configured structural
penalty. There is no unconditional forced-birth path and no rule that suppresses
death merely because a raw residual gate fired.

The structural matching-pursuit proposal is still an approximate
data-conditioned proposal unless its forward/reverse density, prior ratio, and
dimension-matching Jacobian are included. Result provenance must therefore
distinguish ordinary pure-PF execution from exact reversible-jump inference.

## Posterior reporting

For each isotope, particle weights are accumulated by source cardinality. The
MAP cardinality stratum uses deterministic lowest-cardinality tie breaking.
Source slots are aligned within particles and summarized into posterior
position, covariance, credible radius, strength interval, and background
interval.

The public reporting API is:

- `posterior_cardinality_distribution()`;
- `posterior_modes()`;
- `posterior_point_estimate()`;
- `posterior_snapshot()`.

Compatibility `estimates()` is only a projection of the current PF posterior.
It cannot refit strengths, choose source count with a second model, refine
positions against accumulated history, or substitute a best-so-far snapshot.

## Planner and mission control

DSS-PP receives only current PF posterior and causal tentative-source modes.
It cannot read report-rescue or global-surface-rescue modes. Cardinality
pressure is derived from the normalized PF posterior cardinality distribution.
Expected future response discrimination remains a planning heuristic evaluated
from PF modes and hypothetical future observations.

Standard experiment configs use a fixed measurement/action budget and disable
adaptive batch-evidence stopping. Continuous XYZ detector candidates,
collision-aware workspace filtering, height changes, collision-free motion,
and shared Geant4/PF obstacle attenuation remain unchanged.

## Execution model

When CUDA is available, expected-count kernels, observation likelihoods,
spectrum processing, and shield-pair information-gain grids use batched torch
operations. CPU execution uses the same equations in batched NumPy form.

Geant4 transport uses native worker threads. Candidate-pose evaluation, DSS-PP
program evaluation, and PF structural trials use explicit worker or batched
paths. Per-isotope update ordering remains deterministic; parallel work occurs
inside the particle, candidate, and structural-trial kernels.

## MeasurementLog replay

A schema-v1 log bundle contains `run_manifest.json`,
`runtime_config.resolved.json`, `environment.json`,
`forward_model_manifest.json`, `observations.npz`,
`observation_metadata.jsonl`, and `repository_commit.txt`. Truth files are
rejected below the log root.

Records contain ordered step, action, and station identifiers; detector XYZ and
quaternion; Fe/Pb shield indices; raw spectrum and optional spectrum variance;
energy-bin edges; isotope counts and covariance; and live, travel, and shield
actuation time. The forward-model manifest binds the production line response,
shield attenuation table, units, source-rate semantics, and hashes.

Replay consumes rows exactly once and in order. A station is finalized only by
the writer-owned `station_complete=true` marker. Prefix replay does not inspect
future rows, the total record count, or truth.

```text
PYTHONPATH=src uv run python -m pf.replay \
  --measurement-log LOG_DIR \
  --config PF_CONFIG \
  --profile pf_strict \
  --output-dir OUTPUT_DIR \
  --seed 0
```

Replay writes `pf_posterior.json`, `pf_trace.jsonl`, and
`pf_diagnostics.json`. Configuration and forward-model hashes bind the result
to the exact physical and statistical runtime settings.

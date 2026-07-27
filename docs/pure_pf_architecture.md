# Pure PF architecture

## Estimator boundary

The scientific runtime requires `pure_pf_schema_version: 1`, accepts the single
estimator profile `pf_strict`, and follows one sequential data flow:

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
used only for evaluation. Source cardinality, three-dimensional position,
strength, and background remain particle state throughout runtime, replay,
planning, and reporting. The versioned schema is the complete compatibility
boundary; runtime configurations do not depend on a catalog of historical
options. Unknown or retired estimator keys, likelihood aliases, and incomplete
replay records fail closed instead of selecting historical behavior.

## Physical source support

Python, Geant4, and Isaac Sim entry points construct the source state space
directly from the physical environment. The exact structural kernel discretizes
it into a finite dictionary of surface patches:

- floor and ceiling;
- room walls;
- exposed obstacle tops and sides.

Each patch carries its physical area. Initial positions and structural
proposals use the same area-weighted dictionary, prohibit duplicate patch
indices within one isotope state, and store positions in canonical patch-index
order. The finite patch spacing is part of the declared model and sets a
discretization error floor; exactness below refers to this finite-state target,
not to an undiscretized continuous surface.

All position transitions operate on patch indices and preserve the declared
area-weighted surface measure. Detector measurement poses remain collision-free
points in reachable free space; detector support and source support are
intentionally different physical domains.

## Sequential likelihood and structural moves

The PF likelihood consumes the logged `response_poisson` isotope counts and
their covariance. Transport-model, spectrum-model, counting, and correlated
station-view uncertainty are composed once by the production likelihood.
Sequential weight updates and structural trials call the same likelihood
implementation.

The PF applies one exact, internal reversible-jump resample-move kernel whose
invariant target is the current finite-surface posterior:

- the cardinality prior is declared before inference;
- a distinct set of source patches has probability proportional to the product
  of its patch areas;
- each source strength is sampled from the same normalized physical prior used
  at initialization;
- birth and death include the forward and reverse move probabilities, proposal
  densities, cardinality and source priors, and the unit dimension-matching
  Jacobian in the Metropolis-Hastings ratio;
- within-cardinality patch moves compose an area-prior independence proposal
  for global reachability with a local proposal over physically adjacent,
  unoccupied patches; the local acceptance ratio includes the forward/reverse
  available-neighbor degree correction;
- strength moves use a reversible prior-independence
  Metropolis-Hastings transition; and
- accepted rejuvenation moves leave the outer particle weights unchanged.

Move probabilities are normalized at the `K=0` and `K=max_sources`
boundaries. Thus both fixed-cardinality operation and variable-cardinality
operation have explicit target-preserving semantics within the same PF.

There is no second full-spectrum PF likelihood. Spectrum processing produces
the isotope count vector and propagated covariance once; runtime updates,
structural moves, planning, and replay all consume that same statistical
contract.

## Posterior reporting

For each isotope, particle weights are accumulated by source cardinality. The
MAP cardinality stratum uses deterministic lowest-cardinality tie breaking.
Source slots are aligned within particles and summarized into posterior
position, covariance, credible radius, strength interval, and background
interval.

The public reporting API is:

- `posterior_cardinality_distribution()`;
- `posterior_point_estimate()`;
- `posterior_snapshot()`.

Compatibility `estimates()` is a deterministic projection of the current PF
posterior.

## Planner and mission control

DSS-PP receives the current PF posterior. Cardinality pressure is derived from
the normalized PF posterior cardinality distribution. Expected future response
discrimination is evaluated from PF modes and hypothetical future observations.
Every candidate station, including a same-XY height change, is scored with its
own optimized shield program. The previous station's shield sequence is never
reused as a height-transition special case.

Standard experiment configs use a fixed measurement/action budget. Continuous
XYZ detector candidates, collision-aware workspace filtering, height changes,
collision-free motion, and shared Geant4/PF obstacle attenuation remain
unchanged.

## Execution model

When CUDA is available, expected-count kernels, observation likelihoods,
spectrum processing, and shield-pair information-gain grids use batched torch
operations. CPU execution uses the same equations in batched NumPy form.

Geant4 transport uses native worker threads. Candidate-pose and DSS-PP program
evaluation use worker or batched paths; exact RJ response and likelihood
evaluation is batched across proposed particle states. Per-isotope update
ordering remains deterministic.

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

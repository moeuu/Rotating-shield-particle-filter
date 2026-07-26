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
`source_surface_prior=true`. The exact structural kernel discretizes the
physical environment into a finite dictionary of surface patches:

- floor and ceiling;
- room walls;
- exposed obstacle tops and sides.

Each patch carries its physical area. Initial positions and structural
proposals use the same area-weighted dictionary, prohibit duplicate patch
indices within one isotope state, and store positions in canonical patch-index
order. The finite patch spacing is part of the declared model and sets a
discretization error floor; exactness below refers to this finite-state target,
not to an undiscretized continuous surface.

The standard exact kernel does not project or jitter positions after
resampling, exclude source patches near measured detector poses, or impose an
initial-only pairwise separation rule. Those operations would change the
declared position prior without a reversible transition. The standard RAL
factory does not provide a full-volume source-prior ablation. Detector
measurement poses remain collision-free points in reachable free space;
detector support and source support are intentionally different physical
domains.

## Sequential likelihood and structural moves

The PF likelihood consumes the logged `response_poisson` isotope counts and
their covariance. Transport-model, spectrum-model, counting, and correlated
station-view uncertainty are composed once by the production likelihood.
Sequential weight updates and structural trials call the same likelihood
implementation.

The standard RA-L Geant4 configuration selects
`structural_kernel_mode=rj_mh`. It applies a PF-internal resample-move kernel
whose invariant target is the current finite-surface PF posterior:

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
boundaries. Split/merge, BIC thresholds, residual matching pursuit,
pseudo-source pruning, mode-preserving injection, and post-resample
position/strength roughening are disabled in this standard mode. No MLE, batch
fit, surface-map rescue, or strength refit is invoked.

The legacy `structural_kernel_mode=heuristic` remains available only for
explicit diagnostics and historical replay. It uses data-conditioned residual
proposals and likelihood/BIC gates without a reverse proposal, complete prior
ratio, or Jacobian. Its provenance is therefore
`structural_kernel_target_preserving=false`, and its cardinality mass must not
be interpreted as an exact Bayesian posterior.

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

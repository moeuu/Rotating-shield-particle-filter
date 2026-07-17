# Pure PF architecture

## Estimator boundary

The active data flow is:

```text
raw spectrum + pose + shield state
        -> response_poisson count front-end
        -> append MeasurementLog v1 record
        -> sequential PF update
        -> PF posterior snapshot
             |-> PF-only DSS-PP belief
             `-> PF posterior report
```

The log append precedes the PF update. Truth is a separate evaluation asset
and is never opened by replay. The PF core has no dependency on sparse
evidence, surface maps, report MLE, batch model order, or off-grid refinement.

## Variants and capabilities

`pf.profiles.resolve_estimator_profile` is the single authority. Legacy
configuration booleans cannot grant a capability that the profile denies.

- `pf_strict` (default): source cardinalities, positions, and strengths remain
  particle state; the configured background rate is carried in the state
  schema. Conditional strength profiling and every all-history/batch capability
  are disabled.
- `pf_profiled`: identical except that causal conditional strength profiling
  may be applied during the sequential PF update. It is not described as a
  strict PF.

Both variants report `final_estimate_source=pf_posterior`, an empty
`batch_methods_invoked`, and false values for all-history fit, surface-map,
batch-model-order, and batch-feedback provenance.

The standard Geant4 and Python runtime configurations declare `pf_strict`
directly and keep forbidden batch flags false in the files themselves.
`enforce_pure_runtime_settings` and `apply_profile_to_config` remain fail-closed
guards for inherited configs and hostile API overrides. Inherited legacy
report-MLE methods are not part of the `PurePFEstimator` boundary: direct calls
to report position refinement, report-strength fitting, dictionary rescue, or
surface reconstruction raise `PurePFBoundaryError` and are recorded in purity
diagnostics.

PF-internal trans-dimensional proposals are a separate boundary. Residual
birth/split/merge proposals may evaluate a bounded causal measurement block and
optimize trial source strengths at fixed candidate positions before accepting
or rejecting a proposed particle state. Those trial values are never emitted as
a second estimate and never replace the weighted PF posterior. Consequently,
`pf_strict` means PF-only localization without all-history batch refinement; it
does not mean that every data-informed proposal must be sampled from the prior.

## Execution model

The standard runtime automatically enables the torch/CUDA path when CUDA is
available. Expected-count kernels, observation likelihoods, spectrum processing,
and full shield-pair information-gain grids use batched GPU operations. Without
CUDA, the same equations use batched NumPy kernels.

CPU parallelism is explicit for PF structural trials, candidate-pose evaluation,
DSS-PP program evaluation, motion planning, and Geant4 transport. The standard
configs request 32 workers/threads for these paths. Per-isotope PF updates remain
serial because the filters currently share NumPy's global random stream; running
them concurrently would make random draws scheduling-dependent. Parallel work is
therefore applied inside deterministic particle, candidate, and structural-trial
kernels rather than around the stochastic isotope-update order.

## PF posterior reporting

For each isotope, particle weights are accumulated by source cardinality. The
MAP cardinality stratum is selected with deterministic lowest-cardinality tie
breaking. Source slots are deterministically aligned within every particle,
then position, covariance, credible radius, strength interval, and background
interval are aggregated with conditional particle weights. No response fit,
BIC, local refinement, or final strength refit is performed.

The public API is:

- `posterior_cardinality_distribution()`;
- `posterior_modes()`;
- `posterior_point_estimate()`;
- `posterior_snapshot()`.

Compatibility `estimates()` is a projection of `posterior_point_estimate()`;
it no longer invokes the former mixed report path.
The primary live-run `estimated_sources` projection also bypasses legacy
strength/display thresholds and count/SNR absent-isotope filters. Those
post-posterior filters cannot alter the scientific pure-PF result.

## Planner and mission control

DSS-PP may use `pf_posterior` and `pf_tentative` modes only. Cardinality
pressure is normalized PF posterior cardinality entropy, not a sparse-evidence
or BIC gap. Expected future response discrimination remains a design
heuristic because it is evaluated from PF modes and hypothetical future
observations rather than by refitting past data.

Strict experiment configurations use a fixed measurement/action budget,
fixed or count/SNR dwell, and disabled adaptive mission stop. Continuous XYZ
candidate generation, collision-aware `MeasurementWorkspace` filtering,
height changes, and collision-free motion paths are unchanged.
PF-internal station updates are also unchanged: joint or block-sequential
updates and delayed station-boundary resampling remain available because they
do not fit accumulated history with a second estimator.

## MeasurementLog v1

A log bundle contains `run_manifest.json`, `runtime_config.resolved.json`,
`environment.json`, `forward_model_manifest.json`, `observations.npz`,
`observation_metadata.jsonl`, and `repository_commit.txt`. Truth is stored in
a separate evaluation directory. Both `truth.json` and the legacy
`truth_sources.json` are rejected if they occur anywhere below the log root;
the replay process never opens either file. Schema-v1 readers require the
canonical commit filename and fields and do not silently accept legacy
aliases.

Records are ordered rows and contain step/action/station identifiers, exact
detector XYZ, `wxyz` quaternion, Fe/Pb indices, raw spectrum, optional spectrum
variance, energy-bin edges, optional isotope counts/covariance, live time,
travel time, and shield-actuation time. Manifests record resolved physical
configuration, model identifiers/hashes, source-rate semantics, repository
commit, and resolved-config hash. The forward manifest contains the exact
production `ContinuousKernel.line_mu_by_isotope` projection, including line
energy, normalized weight, Fe/Pb attenuation in `cm^-1`, and the fixed line
order. Its shield hash binds the full table and its spectrum hash binds the
energy/weight projection. Unknown or mismatched model identifiers, hashes,
units, response semantics, or line tables fail closed.

The scientific PF likelihood consumes only the logged `response_poisson`
isotope counts (plus their variance/covariance). Raw bins and energy edges are
retained for independent spectral MLE/ablation work but are never passed into
either pure PF variant as a second likelihood.

Replay consumes rows once in order. Prefix execution never inspects the total
record count, a future row, or truth. New live logs persist explicit
`joint_observation_update` and `delayed_resample_update` modes in the effective
replay block. Joint/deferred stations are finalized only by the writer-owned
`station_complete=true` marker on the last row; EOF is never inferred as a
station boundary. Logs created before this contract that have no effective
block and omit (or set null) both top-level modes are replayed one row at a
time for compatibility. Conflicting top-level/effective modes fail closed.
The CLI is:

```text
PYTHONPATH=src uv run python -m pf.replay --measurement-log LOG_DIR --config PF_CONFIG \
  --profile pf_strict --output-dir OUTPUT_DIR --seed 0
```

It writes `pf_posterior.json`, `pf_trace.jsonl`, and `pf_diagnostics.json`.
Provenance distinguishes the raw config-file SHA-256 (`config_sha256`) from
the canonical complete resolved replay-config SHA-256
(`resolved_config_sha256`), including candidate-grid inputs outside the PF
dataclass.

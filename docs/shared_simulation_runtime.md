# Shared simulation runtime and estimator boundary

This repository is the authoritative acquisition runtime for the rotating-shield
projects. Geant4 transport, detector response, environment generation, robot and
shield state, observation finalization, and MeasurementLog publication are maintained
here only. Estimator repositories consume the published, truth-free log; they do not
carry a second production simulator.

## Boundary

```text
PF repository (one acquisition)
  environment + sources + Geant4 + detector + shield + timing
                         |
                         v
       raw integer full-spectrum MeasurementLog v2
                         |
          +--------------+----------------+
          |              |                |
       pure PF       surface MLE       PF+MLE controller
       replay        spectral replay   versioned prefix adapter
```

MeasurementLog v2 is the estimator-neutral connection. It contains finalized raw
integer spectra, the exact energy axis, pose/quaternion, Fe/Pb orientations, live and
motion times, environment geometry, and immutable forward-model identity. It contains
neither realized source truth nor projected isotope counts. An estimator must fail
closed if the schema, model identity, units, source-rate semantics, or artifact hashes
do not match its declared contract.

## Two scientifically different execution modes

For a same-observation estimator comparison, acquire once and replay every estimator
against the identical log. A fixed or estimator-independent acquisition policy gives
the cleanest comparison. Replaying MLE on a PF-planned trajectory is valid, but its
interpretation is "MLE conditioned on PF-selected measurements," not an MLE-controlled
closed-loop mission.

For estimator-controlled planning, each estimator can choose a different next action.
Those runs cannot share the same future observations. They must share the runtime
implementation and scenario seed but execute separate causal missions. Treating one
estimator's realized adaptive trajectory as if every estimator had selected it would
be a contract error.

## Canonical commands

Acquire once with the current high-fidelity runtime:

```bash
uv run python main.py --full-simulation \
  --measurement-log-output results/measurement_logs/EXPERIMENT_ID
```

Replay the local PF without rerunning Geant4:

```bash
uv run python -m pf.replay \
  --measurement-log results/measurement_logs/EXPERIMENT_ID \
  --config configs/geant4/variance_reduction_external_no_isaac_32threads.json \
  --profile pf_strict \
  --output-dir results/estimates/EXPERIMENT_ID/pf
```

The standalone MLE and the PF+MLE orchestrator are invoked from their own repositories
through subprocess adapters. They receive only the log path and their estimator
configuration. No sibling Python imports, copied simulator code, or runtime sync
scripts are part of this boundary. The archived PF+MLE hybrid-v1 algorithm still
requires MeasurementLog v1; it must not manufacture projected isotope counts from a
v2 log. A raw-spectrum causal hybrid therefore needs its own versioned spectral-prefix
contract, while still using this same simulation owner and log boundary.

## Ownership

- This repository owns all production simulation and observation-generation code.
- `3D_estimation` owns only MLE algorithms and MLE result formatting for shared runs.
- `Rotating-shield-estimation-orchestrator` owns estimator selection, subprocess
  isolation, causal hybrid control, revision pins, and cross-estimator manifests.
- Evaluation truth remains outside MeasurementLog and is opened only after estimation
  artifacts are finalized.

# Rotating-shield particle filter

This repository owns only the rotating-shield PF estimator: continuous-surface
SMC/exact-RJ inference, PF-specific planning, diagnostics, evaluation, and posterior
reporting. It does not own Geant4, environments, detector/shield physics, spectrum
generation, or MeasurementLog serialization.

Physical acquisition is implemented once in the sibling
`Rotating-shield-simulation-runtime` repository. That runtime publishes a truth-free
raw full-spectrum MeasurementLog v2. This repository consumes the immutable log and
an estimator-only PF configuration:

```text
Rotating-shield-simulation-runtime
  Geant4 + environment + detector/shield + raw spectra
                         |
                         v
                MeasurementLog v2
                         |
                         v
Rotating-shield-particle-filter
  continuous-surface joint full-spectrum PF / exact-RJ
```

## Install and test

```bash
uv sync
uv run pytest
```

The local development checkout uses an editable sibling dependency on the shared
runtime. A release should pin the corresponding runtime revision in the deployment
lock file.

## Replay

```bash
uv run rotating-shield-pf \
  --measurement-log /path/to/measurement_log \
  --config configs/pf/pf_strict_3d.json \
  --profile pf_strict \
  --output-dir results/pf-replay
```

The replay fails closed if the log schema, source-rate semantics, model identity,
energy axis, environment geometry, or full-spectrum contract is incompatible. Source
truth is not accepted as estimator input.

To generate a new log, run `rotating-shield-sim` from the shared runtime repository.
See [the repository boundary](docs/shared_simulation_runtime.md).

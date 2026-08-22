# Rotating-shield particle filter

This repository owns only the rotating-shield PF estimator: continuous-surface
SMC/exact-RJ inference, PF-specific planning, diagnostics, evaluation, and posterior
reporting. It does not own Geant4, environments, detector/shield physics, spectrum
generation, or MeasurementLog serialization.

Physical acquisition is implemented once in the sibling
`Rotating-shield-simulation-runtime` repository. The online PF controller sends one
causal action at a time to the common adaptive-session API. The runtime durably stages
the resulting truth-free raw full-spectrum observation in MeasurementLog before
returning it, and only then does PF update. The same estimator can also consume a
completed immutable log with an estimator-only PF configuration:

```text
Rotating-shield-simulation-runtime
  Geant4 + environment + detector/shield + raw spectra
                         |
                         v
                MeasurementLog
                         |
                         v
Rotating-shield-particle-filter
  online controller + continuous-surface joint full-spectrum PF / exact-RJ
```

## Install and test

```bash
uv sync
uv run pytest
```

The local development checkout uses an editable sibling dependency on the shared
runtime. A release should pin the corresponding runtime revision in the deployment
lock file.

## PF-controlled acquisition

```bash
uv run rotating-shield-pf-live \
  --scenario /private/runtime/run-001.json \
  --runtime-root ../Rotating-shield-simulation-runtime \
  --config configs/pf/pf_strict_3d.json \
  --profile pf_strict \
  --seed 1 \
  --output-dir results/pf-live-run-001
```

Create the private, action-free scenario with the shared runtime's
`generate-ral-scenario` command. PF never opens its realized source truth. The PF
configuration owns its particle count, planner objective, station/view/measurement
budgets, and stopping rule. An MLE session may use entirely different estimator
settings while connecting to the same runtime protocol.

The runtime owns reachable candidate poses and their physical motion costs. PF owns
candidate ranking and shield-program selection. Every selected station writes
`planner_audit.jsonl`, including the full action count, proxy rank, exact-EIG count,
selected and best exact EIG, score/EIG leaders, top-ranked actions, shortlist
certificate, and MC seed. Independent-seed rank stability remains an explicit
offline diagnostic so it cannot silently double closed-loop planning time.

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

## Independent estimator service

The dedicated PF service exposes exactly the estimator-service contract's two fixed
verbs:

```bash
uv run rotating-shield-pf-service capabilities \
  --response /absolute/path/capabilities.json

uv run rotating-shield-pf-service execute \
  --request /absolute/path/request.json \
  --response /absolute/path/response.json
```

`execute` accepts an authenticated truth-free `MeasurementLogRef`, a self-contained
`radiation.pf-config` v1 file, a request seed, and a new result-directory target. It
calls the same public `replay_measurement_log()` path as the replay CLI and returns a
content-addressed `radiation.pf-result` v1 directory plus named posterior, trace, and
diagnostics file references. Config inheritance and extra input-artifact roles are not
accepted at this independent boundary, so no undeclared files or realized source truth
can enter the PF process.

The service writes an `ExecuteResponse`; it does not manufacture an
`ExecutionReceipt`. The orchestrator that launches the subprocess owns that receipt
because only it can attest the executable, request/response evidence, captured stdout
and stderr, timing, exit status, timeout, and peak memory.

To generate a fixed-plan log without an online estimator, run
`rotating-shield-sim` from the shared runtime repository. See
[the repository boundary](docs/shared_simulation_runtime.md).

## Citation and license

If this software contributes to research, use the metadata in
[`CITATION.cff`](CITATION.cff) to cite the exact software repository. Citation is
a scholarly request, not an additional license condition. Repository-authored
software and documentation are released under the [MIT License](LICENSE);
third-party dependencies and externally sourced data retain their own terms.

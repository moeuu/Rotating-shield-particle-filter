# Shared runtime boundary

`Rotating-shield-simulation-runtime` is the sole owner of Geant4 transport,
environment and source realization, detector and Fe/Pb shield physics, action
execution, raw-spectrum observation generation, and MeasurementLog writing.

This PF repository imports the shared physical response interfaces required to
evaluate the logged likelihood, but does not copy their implementation. It owns only
PF state transitions, SMC/exact-RJ logic, PF-specific planning, diagnostics,
evaluation, and output formatting.

Each estimator-controlled experiment runs its own causal session through the same
shared runtime API. Future observations cannot be shared after planners choose
different actions. The runtime contract, physical configuration, and MeasurementLog
schema are common; PF particles and posterior presentation remain PF-owned.

```bash
# In Rotating-shield-simulation-runtime: author physics, not actions
uv run rotating-shield-sim generate-scenario /private/run-001.json \
  --truth-manifest-output /private/truth/run-001.json \
  --measurement-log-output /private/logs/run-001 \
  --run-id run-001 \
  --runtime-config configs/geant4/variance_reduction_external_no_isaac_32threads.json

# Runtime serves the private scenario over an opaque local socket
uv run rotating-shield-sim serve-adaptive-session-socket /private/run-001.json \
  --socket-path /run/user/1000/rotating-shield/run-001.sock

# In this repository: PF sees only the socket and owns inference/planning policy
uv run rotating-shield-pf-live \
  --session-socket /run/user/1000/rotating-shield/run-001.sock \
  --runtime-root ../Rotating-shield-simulation-runtime \
  --config configs/pf/pf_strict_3d.json \
  --output-dir results/pf-live-run-001
```

The online command does not copy or replace acquisition code. Its controller sends
one action to the sibling runtime, waits until that raw spectrum is durably staged,
and then calls the PF station update. After the final decision, PF asks the runtime
to publish the immutable log and binds the posterior provenance to its digest. The
published log is not accepted as a new batch inference input. The runtime scenario
contains no action list, shield program, or estimator stop rule. It does publish the
truth-free acquisition contract: the standard profile owns the 16-station, 8-view,
20-second-per-view, 128-observation limits and the shared 3 m separation/coverage
radii. PF owns only statistical stopping and action choice.

The private runtime scenario and the separately joined private truth manifest may
contain realized source truth. MeasurementLog and every estimator-visible adaptive
event reject private scene variant, scene seed/RNG provenance, and realized source
fields.

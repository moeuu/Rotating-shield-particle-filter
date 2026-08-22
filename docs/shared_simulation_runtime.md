# Shared runtime boundary

`Rotating-shield-simulation-runtime` is the sole owner of Geant4 transport,
environment and source realization, detector and Fe/Pb shield physics, action
execution, raw-spectrum observation generation, and MeasurementLog writing.

This PF repository imports the shared physical response interfaces required to
evaluate the logged likelihood, but does not copy their implementation. It owns only
PF state transitions, SMC/exact-RJ logic, PF-specific planning, diagnostics,
evaluation, and output formatting.

For a same-observation comparison, acquire once and replay PF and MLE against the
same immutable log. For estimator-controlled planning, run separate causal sessions
through the same shared runtime implementation; future observations cannot be shared
after planners choose different actions.

```bash
# In Rotating-shield-simulation-runtime: author physics, not actions
uv run rotating-shield-sim generate-ral-scenario /private/run-001.json \
  --measurement-log-output /private/logs/run-001 \
  --run-id run-001 \
  --runtime-config configs/geant4/variance_reduction_external_no_isaac_32threads.json

# In this repository: PF owns its planner and mission budget
uv run rotating-shield-pf-live \
  --scenario /private/run-001.json \
  --runtime-root ../Rotating-shield-simulation-runtime \
  --config configs/pf/pf_strict_3d.json \
  --output-dir results/pf-live-run-001

# Or replay one already completed shared log
uv run rotating-shield-pf \
  --measurement-log /path/to/measurement_log \
  --config configs/pf/pf_strict_3d.json \
  --profile pf_strict \
  --output-dir results/pf-replay
```

The online command does not copy or replace acquisition code. Its controller sends
one action to the sibling runtime, waits until that raw spectrum is durably staged,
and then calls the PF station update. The runtime scenario contains no action list,
station count, view count, shield program, or estimator stop rule. The 20-station,
8-view, 160-observation RA-L limits are PF/experiment-harness settings rather than
physical-runtime settings.

The private runtime scenario may contain realized source truth. MeasurementLog
and every estimator-visible adaptive event must not.

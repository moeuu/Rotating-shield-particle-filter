# Shared runtime boundary

`Rotating-shield-simulation-runtime` is the sole owner of Geant4 transport,
environment and source realization, detector and Fe/Pb shield physics, action
execution, raw-spectrum observation generation, and MeasurementLog v2 writing.

This PF repository imports the shared physical response interfaces required to
evaluate the logged likelihood, but does not copy their implementation. It owns only
PF state transitions, SMC/exact-RJ logic, PF-specific planning, diagnostics,
evaluation, and output formatting.

For a same-observation comparison, acquire once and replay PF and MLE against the
same immutable log. For estimator-controlled planning, run separate causal sessions
through the same shared runtime implementation; future observations cannot be shared
after planners choose different actions.

```bash
# In Rotating-shield-simulation-runtime
uv run rotating-shield-sim run-plan /path/to/private-plan.json

# In this repository
uv run python main.py --full-simulation

# Or replay one already completed shared log
uv run rotating-shield-pf \
  --measurement-log /path/to/measurement_log \
  --config configs/pf/pf_strict_3d.json \
  --profile pf_strict \
  --output-dir results/pf-replay
```

The online command does not copy or replace acquisition code. Its controller sends
actions to the sibling runtime, stages each returned raw spectrum through the shared
MeasurementLog writer, and then calls the PF station update. The simulator receives
only the physical configuration; PF and planner settings remain local to this
repository.

The private simulation plan may contain realized source truth. MeasurementLog v2 and
all estimator inputs must not.

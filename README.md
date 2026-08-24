# Rotating-shield particle filter

This repository owns only the rotating-shield PF estimator: continuous-surface
SMC/exact-RJ inference, PF-specific planning, diagnostics, evaluation, and posterior
reporting. It does not own Geant4, environments, detector/shield physics, spectrum
generation, or MeasurementLog serialization.

Physical acquisition is implemented once in the sibling
`Rotating-shield-simulation-runtime` repository. The online PF controller sends one
causal action at a time to the common adaptive-session API. The runtime durably stages
the resulting truth-free raw full-spectrum observation in MeasurementLog before
returning it, and only then does PF update. The finalized MeasurementLog is retained
as immutable provenance for that live session; it is not a batch inference input:

```text
Rotating-shield-simulation-runtime
  Geant4 + environment + detector/shield + raw spectra
                         |
                         v
        typed adaptive event + durable record
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
  --session-socket /run/user/1000/rotating-shield/run-001.sock \
  --runtime-root ../Rotating-shield-simulation-runtime \
  --config configs/pf/pf_strict_3d.json \
  --profile pf_strict \
  --seed 1 \
  --output-dir results/pf-live-run-001
```

Create and serve the private, action-free scenario from the shared runtime. The
generic PF command accepts only its opaque Unix socket and never receives a private
scenario path, scene seed, or source profile. The PF configuration owns its particle
count, planner objective, and statistical stopping rule; the shared runtime contract
owns station, view, and measurement budgets. The production particle count is
declared once as `num_particles` in `configs/pf/pf_strict_3d.json`. Diagnostic PF
profiles inherit that value, so changing this one field updates live and RA-L PF
runs without duplicating the setting. An MLE session may use entirely different
estimator settings while connecting to the same runtime protocol.

The runtime owns reachable candidate poses and their physical motion costs. PF owns
candidate ranking and shield-program selection. Every selected station writes
`planner_audit.jsonl`, including the full action count, proxy rank, exact-EIG count,
selected and best exact EIG, score/EIG leaders, top-ranked actions, shortlist
certificate, and MC seed. Independent-seed rank stability remains an explicit
offline diagnostic so it cannot silently double closed-loop planning time.

The live command fails closed if the runtime context, source-rate semantics, model
identity, energy axis, environment geometry, or full-spectrum contract is
incompatible. Source truth is never accepted as estimator input. See
[the repository boundary](docs/shared_simulation_runtime.md).

For in-process integrations, `pf.PFLiveSession` is the package-owned lifecycle
boundary. Construct it from the runtime `RunContext`, PF configuration/profile,
seed, and runtime asset root; deliver only durably persisted records (or one complete
persisted station); request its copied read-only particle snapshot for planning and
PF-specific display; then call `complete_live_state()` before the runtime publishes
the log and `bind_published_log()` afterward. Binding authenticates the exact run,
context, and full ordered-record digest and never assimilates measurements. The
resulting `PFBoundLiveState` contains canonical posterior bytes and the already
completed state bytes so an outer application can publish them without reaching
back into the estimator.

The session also exposes an optional, typed `PFExternalSurfaceGuidance` proposal
boundary for the sibling orchestrator's MLE-guided particle estimator. A surface grid
must bind the exact incoming run and record prefix. PF performs the batched mapping to
its own continuous-surface charts, mixes the result only into its full-support
structural proposal, and returns a `PFExternalSurfaceGuidanceReceipt` after every
configured isotope evaluated it. The existing forward/reverse MH/RJ terms remain
authoritative: external density never becomes another likelihood, directly changes
outer particle weights, or introduces an MLE dependency into the standalone PF.

The planning snapshot also carries a canonical truth-free posterior summary for the
PF particle display. It does not invent an estimator-neutral grid or a predictive
spectrum: `PurePFEstimator` currently has no public deterministic latest-spectrum
snapshot API. The existing PF CUI therefore continues to render its native particle
cloud and the latest raw spectrum from the persisted runtime record.

## Citation and license

If this software contributes to research, use the metadata in
[`CITATION.cff`](CITATION.cff) to cite the exact software repository. Citation is
a scholarly request, not an additional license condition. Repository-authored
software and documentation are released under the [MIT License](LICENSE);
third-party dependencies and externally sourced data retain their own terms.

# PF Inference Fidelity Policy

This policy governs PF-owned live ingestion, inference, planning, diagnostics,
and posterior publication. Geant4, detector/shield physics, spectrum generation,
environment authoring, and MeasurementLog writing belong exclusively to the
sibling `Rotating-shield-simulation-runtime` repository and must satisfy its
simulation-fidelity policy.

## Production Acquisition Boundary

A full simulation is one PF-controlled causal acquisition through the shared
runtime API:

1. PF submits one next-station pose and shield program.
2. The runtime acquires and durably records a complete station.
3. PF validates and assimilates that exact station.
4. The updated posterior selects the next action or stops.
5. At termination, PF binds its sealed posterior to the finalized immutable
   MeasurementLog digest.

This repository must not add finalized-log batch inference, prefix replay,
surrogate observation generation, or another simulator entry point.

## Likelihood and Observation Integrity

- Ingest only authenticated, unit-weight, nonnegative integer full spectra with
  the exact energy axis and generative-model contract declared by the runtime.
- Use the source-resolved joint full-spectrum generative model directly. Do not
  replace it with isotope-count extraction, peak windows, NNLS, unconstrained
  continuum fitting, expected-count observations, deterministic background
  smoothing, or an additional likelihood derived from the same spectrum.
- PF and DSS must call the same immutable prediction, sampling, and likelihood
  contract. Neither may add independent count, contrast, ratio, or covariance
  evidence from the already-consumed spectrum.
- Preserve `intensity_cps_1m` as expected pre-dead-time detector count rate at
  1 m for the configured detector and spectral processing. Do not reinterpret
  it as total isotropic gamma emission or an already dead-time-suppressed rate.
- When environment obstacles are active, include source-detector obstacle
  attenuation in PF likelihood and planning response calculations.
- Use the shared spherical-octant Fe/Pb geometry, dimensions, orientation
  identities, and pair ordering. A fixed-slab substitute is not a production
  likelihood.
- Exact-RJ scheduling, caches, and proposal heuristics may accelerate target
  evaluation but may not screen history, discard bins/views, change support,
  or alter the MH/RJ ratio.

## Truth Isolation

Private source truth, scene variants, scene seeds, and source RNG provenance
must not enter PF frames, particles, proposal guidance, planner inputs,
MeasurementLog, checkpoints, or PF result artifacts. Truth may be joined only
after a run has completed, using matching run identity, for private evaluation
or an asynchronous runtime-owned renderer endpoint that is inaccessible to PF.

## Generalization

Do not implement calibration, response correction, proposal behavior, stopping
criteria, or quality gates selected to pass a particular run, seed, scene,
source index, shield pair, or tail case. A failed run is diagnostic evidence,
not training data unless it was designated before evaluation.

Accuracy-motivated model changes require a new independent environment for
acceptance. Shield-sensitive changes require all 64 Fe/Pb pose pairs per new
validation scenario unless the user explicitly authorizes a smaller diagnostic
scope.

## Fidelity-Preserving Performance Work

Allowed work includes batched GPU/CPU execution, deterministic chunking,
bounded exact caches, process-parallel independent analyses, and reuse of
unchanged geometry or transport components. These changes must follow the
[PF compute policy](compute.md) and retain identical scientific semantics.

Reducing particles, predictive samples, acquired views, spectrum bins,
histories, response fidelity, or numerical precision is not an acceptable
fallback for a production full simulation.

## Full-Run Discipline

- Launch multi-hour live runs in a persistent session such as `tmux`, write a
  timestamped log, retain the session/PID identity, and monitor from a separate
  command. Relay any CUI URL immediately.
- Do not tune inference or planning from early stations. Mid-run changes are
  allowed only for independently demonstrated crashes, invalid wiring,
  invariant violations, or physics/model-contract mismatches.
- After terminal completion, verify durable artifacts and report acquisition
  fidelity, geometry and shield-pair diversity, posterior/cardinality,
  localization and strength results, limitations, and exact artifact paths.
- End that run cycle after analysis. Do not modify the method or start another
  full simulation without explicit user direction.

## Verification

Run `uv run pytest` after every PF ingestion, likelihood, planning, diagnostic,
or publication change. Add focused regression tests for any new option that
could weaken observation, target, geometry, truth-isolation, or causal-ordering
contracts. Runtime-owned changes must be made and tested in the sibling runtime
repository rather than copied here.

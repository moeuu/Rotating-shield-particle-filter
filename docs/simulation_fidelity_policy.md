# Simulation Fidelity Policy

Runtime simulation must preserve physical fidelity by default. Performance
work is acceptable only when it keeps the same physics, geometry, source-rate
definition, statistics, and observation path.

## Standard Full Simulation Entry Point

- "Full simulation" means `uv run python main.py --full-simulation`.
- The default `uv run python main.py` entry point and `--cui` alias must resolve
  to the standard no-GUI Geant4/PF runtime:
  `--mode geant4-cui` with
  `configs/geant4/variance_reduction_external_no_isaac_32threads.json`.
- Python analytic CUI is available only through the explicit `--python-cui` or
  `--mode python-cui` options.

## Prohibited Runtime Shortcuts

- Do not use surrogate transport in Geant4 runtime modes.
- Do not bypass spectrum generation with expected-count observations.
- Do not cap, downsample, or weight Geant4 histories to shorten runtime.
  Standard Python/Geant4 runtime and validation entry points must reject
  `primary_sampling_fraction` values other than `1.0` before transport starts.
- Do not silently reinterpret `intensity_cps_1m` as total isotropic source
  emission. Runtime defaults use the explicit `detector_cps_1m` source-rate
  model: `intensity_cps_1m` is the expected net detector count rate at 1 m for
  the configured detector and spectral processing.
- Do not use detector-directed transport when a configuration explicitly asks
  for an isotropic total-emission source model. Detector-directed transport is
  allowed only under the explicit `detector_cps_1m` source-rate model.
- Do not use deterministic background smoothing in runtime observations.
- Do not use `theory_tvl` attenuation or synthetic scatter gain in runtime
  Geant4 configurations.
- Do not use peak-window or unconstrained full-spectrum-continuum count
  extraction as a runtime default; runtime PF observation ingestion must use
  the calibrated full-spectrum Poisson response regression that reports
  observation covariance to the PF likelihood. Photopeak NNLS remains a
  diagnostic/calibration method, not a standard runtime input.
- Do not reinterpret crosstalk, low-SNR, or fit-quality classification flags as
  calibrated PF covariance. Heuristic diagnostic variance inflation must be
  opt-in; standard runtime covariance must keep formal extraction, counting,
  transport, and spectrum-model components separate.
- Do not make CUI mode a low-fidelity mode. CUI only means no Isaac Sim GUI.
- Do not ignore concrete/environment obstacles in PF observation likelihoods
  when an obstacle layout or generated environment is active.
- PF shielding likelihoods must use the same spherical-octant shell geometry
  and Pb/Fe dimensions exported to Geant4, not a lower-fidelity fixed-slab
  shortcut.
- Do not implement calibration, response correction, PF observation logic, or
  quality-gate changes that are selected to pass a specific validation run,
  random seed, environment, source layout, shield-pair set, or tail case.
- Do not use case-specific, run-specific, seed-specific, source-index-specific,
  or shield-pair-specific empirical corrections unless the parameter is
  physically defined before evaluation and fitted only from a designated
  training set.

## Allowed Performance Work

- CPU multithreading inside Geant4.
- GPU acceleration for PF or planning math when it preserves the same model.
- Batched or process-parallel PF structure updates when they preserve the same
  likelihood, model-order tests, and source-state semantics.
- Batched obstacle attenuation, shield-orientation scoring, spectrum-response
  regression, and candidate-source evaluation when they are numerically
  equivalent to the scalar equations.
- Geometry caching that does not change exported geometry or materials.
- Detector-equivalent Geant4 source sampling under `source_rate_model =
  detector_cps_1m`, where primary histories represent detector-count-rate
  histories rather than total gamma/s activity.
- Visualization sampling, as long as it affects only rendered tracks and not
  spectra, counts, or PF observations.
- Separate planning heuristics, if they do not replace the runtime spectrum or
  transport calculation.

## Required Checks

- Run `uv run pytest` after any simulation change.
- Keep `tests/test_geant4_fidelity_config.py` and
  `tests/test_simulation_fidelity_shortcuts.py` passing.
- Add a regression test before introducing any new simulation option that could
  lower runtime fidelity.
- For any accuracy-motivated calibration or observation-model change, use failed
  holdouts only for diagnosis. Final acceptance must be measured on a new random
  Geant4 environment that was not used to design, fit, or select the change.
- For shield-sensitive changes, independent validation must cover all 64 Fe/Pb
  shield-pose pairs for each new scenario unless the user explicitly requests a
  smaller diagnostic run.
- For PF, planning, spectrum, obstacle, or Geant4 orchestration code that spans
  particles, candidates, source slots, orientations, spectrum bins, or obstacle
  components, follow `docs/compute_parallelism_policy.md` and add a
  serial-vs-parallel equivalence test or a runtime-path selection test.

## Mid-Run Interpretation

- Do not use the first few poses or early station windows of a full simulation
  to decide new accuracy-improvement implementation changes. Let the run finish
  before proposing tuning, rescue, planning, or estimator-behavior changes.
- Mid-run intervention is allowed only for clear implementation mistakes,
  crashes, logically inconsistent behavior, invalid configuration wiring, or
  physics/model mismatches that can be demonstrated independently of early
  accuracy metrics.

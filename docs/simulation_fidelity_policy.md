# Simulation Fidelity Policy

Runtime simulation must preserve physical fidelity by default. Performance
work is acceptable only when it keeps the same physics, geometry, emission
statistics, and observation path.

## Prohibited Runtime Shortcuts

- Do not use surrogate transport in Geant4 runtime modes.
- Do not bypass spectrum generation with expected-count observations.
- Do not cap, downsample, or weight Geant4 histories to shorten runtime.
- Do not emit gammas only toward the detector; source emission must remain
  isotropic unless a validated physical source model explicitly requires
  anisotropy.
- Do not use deterministic background smoothing in runtime observations.
- Do not use `theory_tvl` attenuation or synthetic scatter gain in runtime
  Geant4 configurations.
- Do not use peak-window or unconstrained full-spectrum-continuum count
  extraction as a runtime default; use calibrated full-energy photopeak
  decomposition or a calibrated full-spectrum Poisson response regression that
  reports observation covariance to the PF likelihood.
- Do not make CUI mode a low-fidelity mode. CUI only means no Isaac Sim GUI.
- Do not ignore concrete/environment obstacles in PF observation likelihoods
  when an obstacle layout or generated environment is active.
- PF shielding likelihoods must use the same spherical-octant shell geometry
  and Pb/Fe dimensions exported to Geant4, not a lower-fidelity fixed-slab
  shortcut.

## Allowed Performance Work

- CPU multithreading inside Geant4.
- GPU acceleration for PF or planning math when it preserves the same model.
- Geometry caching that does not change exported geometry or materials.
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

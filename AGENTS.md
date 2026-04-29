# AGENTS.md

## Tech stack

- Use Python 3.x
- This project uses `uv` for environment management.
- To run tests, use `uv run pytest`.
- To add dependencies, use `uv add <package>` instead of `pip install`.


## Code style

- Implement everything in Python.
- Follow PEP8.
- Every function must have a docstring.
- Comments and explanations should be written in English.


## Testing

- After changing code, always run `pytest` and ensure tests pass.


## Simulation fidelity policy

- Runtime simulation fidelity must take priority over speed shortcuts.
- Before changing simulation, Geant4, spectrum-generation, or PF observation
  ingestion code, read `docs/simulation_fidelity_policy.md`.
- Do not introduce runtime shortcuts that lower physical fidelity for speed:
  surrogate transport, expected-count observations, weighted or capped Geant4
  histories, deterministic background smoothing, detector-directed emissions,
  `theory_tvl` runtime attenuation, non-Geant4 scatter synthesis, or
  peak-window or full-spectrum-continuum runtime count extraction.
- A calibrated full-spectrum Poisson response regression is acceptable when it
  uses the detector response model directly and propagates count covariance to
  the PF likelihood; do not replace it with unconstrained continuum fitting.
- CUI mode means "run without the Isaac Sim GUI." It must not mean lower
  transport fidelity.
- GPU or CPU multithreading is allowed only when it preserves the same physics
  and statistical meaning as the high-fidelity runtime.
- When an obstacle grid/environment geometry is active, PF observation models
  must include source-detector obstacle attenuation; the grid must not be used
  only for visualization or path planning.
- PF shield likelihoods must use the shared spherical-octant Pb/Fe geometry
  and dimensions used by Geant4, not a lower-fidelity fixed-slab shortcut.
- If an approximate method is needed for a benchmark, planning heuristic, or
  legacy comparison, keep it out of runtime simulation defaults, name it
  explicitly as approximate, and add tests that prevent it from being selected
  by `main.py` runtime modes.

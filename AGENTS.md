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


## RA-L paper ablation plan

- Before running RA-L paper ablations, read
  `docs/ral_minimal_ablation_plan.md`.
- The current RA-L paper subset is 13 full-simulation runs:
  `proposed`, `baseline_passive_no_shield`, `round_robin_shield`, and
  `one_step_path` for all three cases, plus `no_residual_birth` only for
  `case03_mixed_cardinality`.
- Use `uv run python scripts/build_ral_paper_subset.py` to regenerate
  `results/ral_ablation/ral_paper_subset_manifest.csv` and
  `results/ral_ablation/run_paper_subset.sh` from the exhaustive manifest.


## Compute parallelism policy

- Before adding PF, planning, spectrum-processing, obstacle-attenuation, or
  Geant4 orchestration features, read `docs/compute_parallelism_policy.md`.
- Implement compute-heavy features in a batched, GPU, Geant4-threaded, or
  process-parallel form from the first version when the operation spans PF
  particles, source slots, candidate locations, shield postures, spectrum bins,
  response-matrix columns, or obstacle components.
- Do not add scalar Python runtime loops over particles, candidates, source
  slots, orientations, or obstacle components and leave them for later
  parallelization. A scalar version may exist only as a small deterministic test
  oracle or explicit debug fallback.
- Parallelization must preserve the same physics, geometry, likelihood,
  source-rate semantics, and statistical meaning. It may only change execution
  schedule and wall time.
- Every new batched/parallel runtime path must include a serial-vs-parallel
  equivalence test, or a test proving that the standard runtime selects the
  batched/parallel path.


## Simulation fidelity policy

- Runtime simulation fidelity must take priority over speed shortcuts.
- "Full simulation" means the standard no-GUI Geant4/PF runtime. Use
  `uv run python main.py --full-simulation` or `uv run python main.py` unless
  the user explicitly asks for another mode. This resolves to
  `--mode geant4-cui` with
  `configs/geant4/variance_reduction_external_no_isaac_32threads.json`.
- Do not use `--cui` to mean the Python analytic model. `--cui` is an alias for
  the standard Geant4 CUI full simulation; use `--python-cui` only when the user
  explicitly asks for the Python analytic mode.
- Before changing simulation, Geant4, spectrum-generation, or PF observation
  ingestion code, read `docs/simulation_fidelity_policy.md`.
- Do not introduce runtime shortcuts that lower physical fidelity for speed:
  surrogate transport, expected-count observations, weighted or capped Geant4
  histories, deterministic background smoothing, unlabelled source-rate
  reinterpretation, `theory_tvl` runtime attenuation, non-Geant4 scatter
  synthesis, or peak-window or full-spectrum-continuum runtime count extraction.
- `intensity_cps_1m` means expected net detector count rate at 1 m for the
  configured detector and spectral processing, not total isotropic gamma/s
  source emission. Detector-directed Geant4 histories are allowed only under the
  explicit `source_rate_model = detector_cps_1m` source-rate model.
- A calibrated full-spectrum Poisson response regression is acceptable when it
  uses the detector response model directly and propagates count covariance to
  the PF likelihood; this is the runtime PF count-ingestion standard. Do not
  replace it with unconstrained continuum fitting.
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
- When starting a simulation that exposes a CUI/split-view progress URL, relay
  that URL in the chat immediately. Do not require the user to inspect terminal
  logs to find the progress view.

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
- Before changing RA-L manuscript content, acknowledgments, figures, tables, or
  layout, read `docs/ral_manuscript_policy.md`.
- Before changing RA-L manuscript figures or layout, read
  `docs/ral_figure_quality_policy.md`. The manuscript should use the eighth
  page effectively while staying within the eight-page RA-L limit; do not leave
  the final page mostly blank when necessary explanation can be restored.
- The anonymous RA-L manuscript must keep the masked sponsor footnote
  `This work was in part supported by XXX.` as a first-page `\thanks`
  acknowledgment. Do not remove it or move it into the main text unless the
  user explicitly asks for a camera-ready/non-anonymous version.
- The current RA-L paper subset is four full-simulation runs on
  `mix9_multi_isotope_cardinality`:
  `proposed`, `baseline_passive_equal_time_no_shield`, `round_robin_shield`,
  and `eig_only_path`. The task uses `4 Cs-137 + 3 Co-60 + 2 Eu-154` sources.
  Estimator-only ablations such as `no_residual_birth` and `no_verification`
  should be run as logged replay unless the user explicitly requests extra
  full simulations.
- Use `uv run python scripts/build_ral_paper_subset.py` to regenerate
  `results/ral_ablation/ral_paper_subset_manifest.csv` and
  `results/ral_ablation/run_paper_subset.sh` from the exhaustive manifest.


## RA-L figure quality policy

- Before creating or modifying RA-L manuscript figures, read
  `docs/ral_figure_quality_policy.md`.
- After generating any manuscript figure, inspect the rendered image itself
  before reporting completion. Do not rely only on code, LaTeX source, or file
  existence.
- Figures must be revised before completion if text, legends, panel labels,
  axes, markers, or arrows overlap; if metric aspect ratios are distorted; or
  if a panel is logically unclear or implies a method/physics behavior that is
  not actually used.
- Generated RA-L figure text must be readable at final paper size. Use at least
  7 pt for labels, ticks, and legends unless a stronger venue rule is known,
  use a consistent body font size across adjacent result tables, and avoid
  `\scriptsize` in main-paper result tables unless the table has first been
  simplified.
- If a manuscript result claims 3-D localization accuracy, the main result
  figure must show a 3-D view or paired projections that make height errors
  visible. Do not show only an x-y map unless the text explicitly says the
  figure is only a floor-plan diagnostic.
- Do not use the same rendered scene as the primary panel of multiple figures.
  Assign each figure a distinct role, such as problem setting, shield mechanism,
  or experimental result.
- When using `scripts/build_ral_figures.py`, inspect the PNG review copies in
  `results/ral_figure_review/` and inspect the compiled PDF page when the figure
  is included in the paper.


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
- Long full-simulation runs must be launched in a persistent background
  session such as `tmux` when the user is connected over SSH or asks for log
  monitoring. Do not run multi-hour RA-L/Geant4 simulations only as a foreground
  command attached to the current Codex/SSH terminal, because SSH disconnects or
  tool-session interruptions can terminate `main.py` while leaving Geant4
  sidecars or CUI servers orphaned.
- For persistent runs, write stdout/stderr to a timestamped log file, save the
  PID/session name, and monitor that log from a separate command. If a CUI URL is
  exposed, relay it immediately and keep the server process tied to the
  persistent session.
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

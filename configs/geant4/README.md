# Geant4 Runtime Configs

Use these top-level configs for current simulations:

Standard top-level configs use `primary_sampling_fraction=1.0`. The Python
Geant4 application rejects fractional primary-history sampling unless the
nonstandard accelerated mode is selected explicitly, and it checks native
response provenance before returning any observation to the PF.

- `variance_reduction_external_no_isaac_32threads.json`: standard no-GUI
  Geant4/PF full simulation config. This is the default for `main.py`,
  `--cui`, and `--full-simulation`. It loads the repo-stable PF
  transport-response calibration in `calibration/` for the
  incident-gamma-energy variance-reduction runtime and uses a conservative
  response-Poisson line-basis BIC margin for count extraction. The
  transport-response calibration uses optical-depth feature caps to prevent
  polynomial extrapolation in very rare high-shield-depth aperture rays.
- `variance_reduction_external_gui_32threads.json`: standard Geant4/PF full
  simulation config plus an Isaac Sim sidecar for visualization. It inherits
  the no-GUI config; the intended runtime difference is Isaac Sim startup only.
- `high_fidelity_external_no_isaac.json`: full-transport verification config.
  It is intentionally slower, does not inherit the variance-reduction
  transport-response calibration, and should be selected explicitly.
- `accelerated_weighted_external_no_isaac_32threads.json`: explicit,
  nonstandard, user-authorized acceleration overlay. It inherits the standard
  no-GUI config and targets at most 1.5 million sampled histories per
  Geant4 transport invocation. A fixed-dwell measurement is one invocation;
  dynamic-budget sampling is rejected with adaptive dwell because the number
  of independent chunks is not known to DSS before acquisition. Dim
  observations retain all histories, while the effective
  fraction is reduced for brighter scenes and each sampled history receives its
  reciprocal weight. The budget is selected from the declared ten-second target
  using measured full-history and fixed startup runtimes, rather than from
  localization results. The spectrum tally remains mean-unbiased, while its
  Monte Carlo variance is higher. Native `sum(w^2)`
  and dead-time provenance are propagated to count extraction and the PF; this
  history weight is also used by DSS for future-observation variance. Derived
  shield contrast/view-ratio likelihoods are disabled under the complete
  covariance semantics so the same weighted counts are not used twice. This
  mode is not the standard full-history result and must be labelled as weighted
  thinning in reports.
- `external_gui_scene.json`: explicit USD-backed Manchester Drum Store scene.
- `shield_validation_scene.json`: material/shield validation config.

Older duplicated or scene-specific configs are isolated under `legacy/`.
They are archival inputs and some still document the former unlabelled
fractional-history mode. The standard Python runtime rejects those values.
Migrate a legacy config to full-history sampling, or use the explicit
accelerated overlay and its uncertainty semantics; do not promote a legacy
config back to the top-level runtime set unchanged.

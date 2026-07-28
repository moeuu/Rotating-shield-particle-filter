# Geant4 Runtime Configs

Use these top-level configs for current simulations:

Standard top-level configs use `primary_sampling_fraction=1.0`. The Python
Geant4 application rejects fractional primary-history sampling unless the
nonstandard accelerated mode is selected explicitly, and it checks native
response provenance before returning any observation to the PF.

- `variance_reduction_external_no_isaac_32threads.json`: standard no-GUI
  Geant4/PF full simulation config. This is the default for `main.py`,
  `--cui`, and `--full-simulation`. It requires one authenticated
  geometry-conditioned joint full-spectrum model with a globally fitted,
  nonnegative additive noncollided transport component.
- `variance_reduction_external_gui_32threads.json`: standard Geant4/PF full
  simulation config plus an Isaac Sim sidecar for visualization. It inherits
  the no-GUI config; the intended runtime difference is Isaac Sim startup only.
- `high_fidelity_external_no_isaac.json`: full-transport verification config.
  It is intentionally slower and should be selected explicitly.
- `external_gui_scene.json`: explicit USD-backed Manchester Drum Store scene.
- `shield_validation_scene.json`: material/shield validation config.

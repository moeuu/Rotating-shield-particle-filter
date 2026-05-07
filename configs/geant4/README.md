# Geant4 Runtime Configs

Use these top-level configs for current simulations:

- `variance_reduction_external_no_isaac_32threads.json`: standard no-GUI
  Geant4/PF full simulation config. This is the default for `main.py`,
  `--cui`, and `--full-simulation`.
- `high_fidelity_external_no_isaac.json`: full-transport verification config.
  It is intentionally slower and should be selected explicitly.
- `external_gui_scene.json`: Geant4 transport with Isaac Sim sidecar/GUI scene
  integration.
- `shield_validation_scene.json`: material/shield validation config.

Older duplicated or scene-specific configs are isolated under `legacy/`.
Do not select those from standard runtime entry points. If a legacy config is
needed for a benchmark or reproduction, keep it explicit in the command and do
not promote it back to the top-level runtime set.

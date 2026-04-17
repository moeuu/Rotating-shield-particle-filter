# Rotating-shield-particle-filter

Demo simulations assume strong sources (≈20,000 cps at 1 m) for Cs-137, Co-60, and Eu-154 to mimic high-dose environments. Core APIs remain unchanged; only example defaults are scaled.

## Environment setup (uv)

This repository uses `uv` for environment and dependency management.

```
uv sync
```

To add dependencies:

```
uv add <package>
```

To run tests:

```
uv run pytest
```

## GPU acceleration (optional)

To enable CUDA acceleration for EIG, rotation rollouts, and PF updates, install torch:

```
uv add torch
```

The demo automatically enables GPU when CUDA is available. Set `use_gpu=False` in
`RotatingShieldPFConfig` to force CPU mode.

## Real-time visualization

To run the real-time particle filter visualization demo:

```
uv run python main.py
```

The run shows robot trajectory, shield orientations, particle clouds, and estimated sources; `results/result_pf.png` is written at the end, alongside the final spectrum as `results/result_spectrum.png`.

## Isaac Sim bridge workflow

This repository now supports a simulator backend boundary for observation generation.

Default analytic backend:

```bash
uv run python main.py --sim-backend analytic
```

Bridge-based Isaac Sim backend:

```bash
# Terminal 1: start the sidecar bridge in mock mode.
# The default config points at the Manchester Drum_Store USD, but mock mode
# keeps using the lightweight fake stage for fast smoke checks.
uv run python scripts/run_isaacsim_bridge.py --mock

# Terminal 2: run the PF loop against the bridge.
uv run python main.py --sim-backend isaacsim
```

Real Isaac Sim sidecar with the default Manchester Drum_Store USD:

```bash
# Terminal 1: start Isaac Sim headless and load the Manchester Drum_Store stage.
uv run python scripts/run_isaacsim_bridge.py --config configs/isaacsim/real_scene.json

# Terminal 2: run the PF loop against the live sidecar.
uv run python main.py --sim-backend isaacsim
```

The real sidecar opens the configured USD stage, authors helper prims for the robot, detector, shields, obstacles, and source markers, then streams observations back through the bridge. Observation generation now uses scene geometry for obstacle and shield attenuation by tracing the source-to-detector segment through authored solid prims. Path-prefix material rules are supported, prim-level `simbridge:material` overrides them, and standard USD `material:binding` is also resolved. Bound materials can override attenuation either with direct `simbridge_mu_*` inputs, with `simbridge_density_g_cm3` plus `simbridge_mass_att_*_cm2_g`, or with a `simbridge_material_preset` / `simbridge_composition` fallback when explicit isotope coefficients are missing. The real backend now applies attenuation per gamma line energy before response-matrix synthesis, so multi-line nuclides can harden spectrally through dense materials even though the detector itself is still modeled analytically rather than with a native Isaac radiation sensor.

## Manchester nuclear environment assets

The University of Manchester dataset is distributed as Gazebo SDF plus Collada
meshes/textures, so it must be converted to USD before Isaac Sim can load it.
The helper below downloads one Figshare ZIP, verifies its MD5, extracts it, runs
Blender to export USD, and writes an Isaac Sim bridge config. The default Isaac
Sim configs now use the converted Drum_Store USD at
`data/manchester_nuclear_assets/usd/drum_store.usda`; the older
`configs/isaacsim/demo_room.usda` is kept only as a lightweight smoke-test
asset.

```bash
# Show available assets and their Figshare download URLs.
uv run python scripts/prepare_manchester_dataset.py --list

# Prepare the generic drum-store room environment.
uv run python scripts/prepare_manchester_dataset.py --asset Drum_Store

# Terminal 1: start Isaac Sim with the converted Manchester USD.
uv run python scripts/run_isaacsim_bridge.py --config configs/isaacsim/manchester_drum_store.json

# Terminal 2: run the PF loop against that scene.
uv run python main.py --sim-backend isaacsim
```

The prepared files are written under `data/manchester_nuclear_assets/`, which is
ignored by git because the full dataset is large. Blender must be installed and
available on `PATH`, or passed with
`--blender-executable /home/moeu/.local/bin/blender`.

## Geant4 bridge workflow

This repository now also supports a `geant4` backend that keeps the existing PF loop unchanged and returns a detector spectrum through the same sidecar protocol.

Default Geant4 surrogate backend with a mock stage:

```bash
# Terminal 1: start the Geant4 sidecar.
uv run python scripts/run_geant4_bridge.py --config configs/geant4/default_scene.json

# Terminal 2: run the PF loop against the Geant4 bridge.
uv run python main.py --sim-backend geant4 --sim-config configs/geant4/default_scene.json
```

USD-backed Geant4 sidecar using the Manchester Drum_Store mesh:

```bash
# Terminal 1: start the Geant4 sidecar against the USD stage.
# Manual starts must use Isaac Sim Python so USD mesh traversal is available.
/home/moeu/.local/isaacsim/5.1.0/python.sh scripts/run_geant4_bridge.py --config configs/geant4/real_scene.json

# Terminal 2: run the PF loop against the Geant4 bridge.
uv run python main.py --sim-backend geant4 --sim-config configs/geant4/real_scene.json
```

When `main.py` auto-starts the sidecar, the Geant4 configs use their
`sidecar_python` setting to launch the bridge with Isaac Sim Python. The Geant4
sidecar reuses the Manchester USD stage as the scene source of truth, exports
mesh geometry under `/World/Environment`, and generates a detector spectrum from
that exported geometry. The current implementation includes Poisson source
emission, stage/shield attenuation, scatter-continuum synthesis,
detector-volume scaling, and dead-time scaling inside a surrogate engine so the
runtime contract is ready before a native Geant4 executable is connected.

Native Geant4 executable path:

```bash
# Build the native sidecar after installing Geant4 and exposing geant4-config.
uv run python scripts/build_geant4_sidecar.py

# Terminal 1: start the Python bridge against the native executable and USD mesh.
/home/moeu/.local/isaacsim/5.1.0/python.sh scripts/run_geant4_bridge.py --config configs/geant4/external_scene.json

# Terminal 2: run the PF loop against the Geant4 bridge.
uv run python main.py --sim-backend geant4 --sim-config configs/geant4/external_scene.json
```

The native Geant4 source lives in `native/geant4_sidecar/geant4_sidecar.cpp`. The external engine now uses a file-based protocol between Python and the native executable so the Geant4 code does not need a JSON dependency.

### Geant4 net-response calibration

The PF likelihood expects isotope-wise counts in the same measurement space as
the spectrum unfolding output. Analytic Python/surrogate runs use
`spectrum_count_method: "response_matrix"` so mixed-isotope spectra are unfolded
with the same detector response matrix that generated them. This avoids the
non-conservative cross-talk that occurs when peak-window net areas are compared
directly with inverse-square/shield theory. Native-external Geant4 configs point to
`configs/geant4/net_response_calibration.json`, which stores isotope-specific
response factors mapping ideal inverse-square/shield counts into spectrum-net
counts. Regenerate the factors and shield-validation report with:

```bash
PYTHONPATH=src uv run python scripts/calibrate_geant4_net_response.py --dwell-time-s 30.0
```

The validation layout in `source_layouts/shield_validation.json` places the
sources in the octant that is blocked by `fe_index=7` / `pb_index=7`, while
`configs/geant4/shield_validation_scene.json` runs an obstacle-free native
Geant4 scene with `physics_profile: "theory_tvl"`. In that profile, the native
sidecar applies the same Beer-Lambert TVL attenuation to the source term and
disables the physical shield volume to avoid double attenuation. The validation
script uses the sidecar's source-equivalent tally for the transport-scale
comparison and keeps spectrum-derived peak-window counts as diagnostics in
`results/geant4_net_response_validation.csv`.

## Birth move smoke test

Use these commands to verify birth is disabled/enabled:

```
# Birth disabled (default): r_mean stays at 1 and births remain 0.
uv run python main.py --max-steps 20

# Birth enabled: with max_sources=3, births should become >0 within a few steps.
uv run python main.py --birth --max-sources 3 --max-steps 20

# Stabilized expected-count run (reduce tempering resamples, skip roughening on temper-resample).
uv run python main.py --count expected --temper-max-resamples 0 --no-roughen-on-temper-resample
```

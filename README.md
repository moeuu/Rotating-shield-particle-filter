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

## Running simulations

`main.py` runs the particle-filter loop. The `--sim-backend` option controls
where observations come from:

| Backend | Command role | External requirements |
| --- | --- | --- |
| `analytic` | In-process Python spectrum simulator. | None beyond `uv sync`. |
| `isaacsim` | TCP client that talks to `scripts/run_isaacsim_bridge.py`. | Isaac Sim only for real mode; mock mode works without it. |
| `geant4` | TCP client that talks to `scripts/run_geant4_bridge.py`. | Surrogate mode needs no native Geant4; external mode needs a built Geant4 executable. |

All runs write final plots under `results/` by default:
`results/result_pf*.png`, `results/result_estimates*.png`, and
`results/result_spectrum*.png`. Use `--output-tag <name>` to keep results from
different runs separate.

Common options:

```bash
--headless                 # disable the interactive Matplotlib window
--max-steps 5              # stop after a fixed number of measurements
--max-poses 3              # stop after a fixed number of robot poses
--source-config PATH       # load sources from JSON, default source_layouts/Ex1.json
--obstacle-config PATH     # load blocked cells from JSON, default obstacle_layouts/Ex1_obstacles.json
--environment-mode random  # generate a fresh obstacle map for this run
--obstacle-seed 7          # make fixed generation or random mode reproducible
--robot-speed 0.5          # nominal travel speed used in mission-time accounting
--rotation-overhead-s 0.5  # fixed shield actuation overhead per measurement
```

### Analytic backend

Use this for the fastest smoke tests and for development that does not need a
live simulator process.

```bash
uv run python main.py --sim-backend analytic --headless --max-steps 3
```

Interactive visualization is enabled by default when `--headless` is omitted:

```bash
uv run python main.py --sim-backend analytic
```

Random obstacle layouts can also be generated at startup. If Blender is
available, a matching USD environment is written under
`results/blender_environments/`.

```bash
uv run python main.py \
  --sim-backend analytic \
  --environment-mode random \
  --obstacle-seed 7 \
  --headless \
  --max-steps 3
```

### Isaac Sim backend

The Isaac Sim backend is a two-process workflow: first start the sidecar, then
run `main.py` as a TCP client. Mock mode is the lightest check and does not
start a real Isaac Sim Kit application.

```bash
# Terminal 1: mock sidecar on 127.0.0.1:5555.
uv run python scripts/run_isaacsim_bridge.py --mock

# Terminal 2: PF loop using that sidecar.
uv run python main.py \
  --sim-backend isaacsim \
  --sim-config configs/isaacsim/default_scene.json \
  --headless \
  --max-steps 3
```

For a real Isaac Sim stage, run the sidecar with Isaac Sim's Python so the
`isaacsim`, `omni`, and `pxr` packages are available. Replace the path below
with your local Isaac Sim `python.sh` if needed.

```bash
# Terminal 1: real headless Isaac Sim sidecar.
/home/moeu/.local/isaacsim/5.1.0/python.sh \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/real_scene.json

# Terminal 2: PF loop using the same host/port/timeouts.
uv run python main.py \
  --sim-backend isaacsim \
  --sim-config configs/isaacsim/real_scene.json \
  --headless \
  --max-steps 5
```

Use the GUI config when you want the Isaac Sim viewport to stay open and show
robot motion, shields, sources, and radiation visualization:

```bash
# Terminal 1: GUI sidecar. This config keeps the sidecar alive after main exits.
/home/moeu/.local/isaacsim/5.1.0/python.sh \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/gui_scene.json

# Terminal 2: drive it from the PF loop.
uv run python main.py \
  --sim-backend isaacsim \
  --sim-config configs/isaacsim/gui_scene.json \
  --headless \
  --max-steps 10
```

The real sidecar opens the configured USD stage, authors helper prims for the
robot, detector, Fe/Pb shields, obstacles, and source markers, then streams
observations back through the bridge. Obstacle and shield attenuation are
computed by tracing the source-to-detector segment through authored solids.
Material rules can be assigned by path prefix, `simbridge:material`, standard
USD material bindings, or explicit attenuation metadata.

### Manchester nuclear environment assets

The University of Manchester dataset is distributed as Gazebo SDF plus Collada
meshes/textures, so it must be converted to USD before `real_scene.json` and
`gui_scene.json` can load the Drum_Store environment.

```bash
# Show available Figshare assets.
uv run python scripts/prepare_manchester_dataset.py --list

# Download, verify, extract, convert Drum_Store to USD, and write an Isaac config.
uv run python scripts/prepare_manchester_dataset.py \
  --asset Drum_Store \
  --blender-executable /home/moeu/.local/bin/blender
```

Prepared files are written under `data/manchester_nuclear_assets/`, which is
ignored by git because the full dataset is large. The default converted USD path
is `data/manchester_nuclear_assets/usd/drum_store.usda`, and the generated
config is `configs/isaacsim/manchester_drum_store.json`.

Run the converted Manchester scene like this:

```bash
# Terminal 1
/home/moeu/.local/isaacsim/5.1.0/python.sh \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/manchester_drum_store.json

# Terminal 2
uv run python main.py \
  --sim-backend isaacsim \
  --sim-config configs/isaacsim/manchester_drum_store.json \
  --headless \
  --max-steps 5
```

### Geant4 backend

The Geant4 backend uses the same sidecar protocol as Isaac Sim. `main.py`
auto-starts the Geant4 sidecar when no server is already listening on the
configured port. Sidecar logs go to `results/sidecars/`.

For a pure headless smoke test, use the surrogate engine with a mock stage and
no Isaac Sim companion:

```bash
uv run python main.py \
  --sim-backend geant4 \
  --sim-config configs/geant4/surrogate_no_isaac.json \
  --headless \
  --max-steps 3
```

Use `configs/geant4/default_scene.json` when you want the surrogate Geant4
observation model paired with an Isaac Sim sidecar for robot motion and
visualization. Start the Isaac sidecar with Isaac Sim Python first, then let
`main.py` auto-start the Geant4 sidecar and reuse the running Isaac sidecar.

```bash
# Terminal 1
/home/moeu/.local/isaacsim/5.1.0/python.sh \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/real_scene.json

# Terminal 2
uv run python main.py \
  --sim-backend geant4 \
  --sim-config configs/geant4/default_scene.json \
  --headless \
  --max-steps 5
```

For a real USD-backed surrogate run against the Manchester Drum_Store mesh,
prepare the Manchester assets first, then use:

```bash
uv run python main.py \
  --sim-backend geant4 \
  --sim-config configs/geant4/real_scene.json \
  --headless \
  --max-steps 5
```

`configs/geant4/real_scene.json` uses the surrogate Geant4 engine while reading
scene geometry from the USD stage. It includes Poisson source emission,
stage/shield attenuation, scatter-continuum synthesis, detector-volume scaling,
and dead-time scaling.

### Native Geant4 executable

Install Geant4 first and make sure `geant4-config` is on `PATH`, then build the
native sidecar executable:

```bash
uv run python scripts/build_geant4_sidecar.py
```

Run the PF loop with the external engine:

```bash
uv run python main.py \
  --sim-backend geant4 \
  --sim-config configs/geant4/external_scene.json \
  --headless \
  --max-steps 5
```

The native source is `native/geant4_sidecar/geant4_sidecar.cpp`. The Python
bridge communicates with it through a file-based request/response protocol, so
the C++ executable does not need a JSON dependency. GUI external-engine configs
are available as `configs/geant4/external_gui_scene.json`,
`configs/geant4/demo_room_external_gui.json`, and
`configs/geant4/demo_room_scatter_attenuation_external_gui.json`.

### Geant4 net-response calibration

The PF likelihood expects isotope-wise counts in the same measurement space as
the spectrum unfolding output. Analytic Python and surrogate Geant4 runs use
`spectrum_count_method: "response_matrix"` so mixed-isotope spectra are unfolded
with the same detector response matrix that generated them. Native external
Geant4 configs use `configs/geant4/net_response_calibration.json` to map ideal
inverse-square/shield counts into spectrum-net counts.

Regenerate the calibration and validation report with:

```bash
PYTHONPATH=src uv run python scripts/calibrate_geant4_net_response.py --dwell-time-s 30.0
```

The validation layout is `source_layouts/shield_validation.json`, and the
validation config is `configs/geant4/shield_validation_scene.json`. The script
writes diagnostics to `results/geant4_net_response_validation.csv` and updates
the JSON calibration payload.

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

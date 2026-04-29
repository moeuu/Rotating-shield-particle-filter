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
| `geant4` | TCP client that talks to `scripts/run_geant4_bridge.py`. | Requires the native Geant4 sidecar executable. |

All runs write final plots under `results/` by default:
`results/result_pf*.png`, `results/result_estimates*.png`, and
`results/result_spectrum*.png`. Use `--output-tag <name>` to keep results from
different runs separate.

Common options:

```bash
--mode python-gui          # Python spectrum model + Isaac Sim GUI
--mode geant4-isaacsim-gui # Geant4 spectrum model + Isaac Sim GUI
--mode python-cui          # Python spectrum model, no simulator GUI
--mode geant4-cui          # Geant4 spectrum model, no Isaac Sim GUI
--headless                 # force a non-GUI simulator mode
--matplotlib-live          # optional Matplotlib live plot
--max-steps 5              # stop after a fixed number of measurements
--max-poses 3              # stop after a fixed number of robot poses
--source-config PATH       # load sources from JSON, default source_layouts/demo_sources.json
--obstacle-config PATH     # load blocked cells from JSON, default obstacle_layouts/demo_obstacles.json
--environment-mode random  # generate a fresh obstacle map for this run
--obstacle-seed 7          # make fixed generation or random mode reproducible
--passage-width-m 1.0      # reserve a robot-passable corridor in random maps
--robot-radius-m 0.35      # clearance radius used for 2D traversability maps
--robot-speed 0.5          # nominal travel speed used in mission-time accounting
--rotation-overhead-s 0.5  # fixed shield actuation overhead per measurement
```

## Optional piplup notifications

Simulation notifications are opt-in and use the existing piplup-notify
`/api/events` endpoint. If no token is configured, simulations run normally and
only print a short warning when `--notify` or `--notify-spectrum` was requested.

The implementation sends only start, final, and failure events by default.
Add `--notify-spectrum` to enable notifications and send per-measurement
spectrum payloads that can be plotted by the Railway app. Spectrum events
include bin energies, counts, isotope-wise extracted counts, pose, and
shield-orientation indices.

```bash
export PIPLUP_NOTIFY_TOKEN=...  # same token as piplup EVENT_API_TOKEN

uv run python main.py \
  --mode geant4-cui \
  --sim-config configs/geant4/random_external_no_isaac.json \
  --max-poses 10 \
  --rotations-per-pose 4 \
  --max-steps 40 \
  --notify-spectrum
```

Useful overrides:

```bash
--notify-url https://piplup-notify-production.up.railway.app
--notify-account lab_rdp
--notify-run-id demo-001
--notify-spectrum
--notify-spectrum-every 1
--notify-spectrum-max-bins 800
--no-notify
```

For longer Geant4 runs, `--rotations-per-pose 4 --max-poses 10 --max-steps 40`
forces four shield measurements at each of ten robot poses before IG early
stopping can move to the next pose. Add `--min-rotations-per-pose 1` if you
want the old early-stop behavior under a four-rotation cap. Add
`--init-grid-spacing-m 0` when `--num-particles` should control the PF size
instead of the default 1 m grid initialization.

The `configs/geant4/random_external_no_isaac.json` external sidecar config uses
the balanced Geant4 physics profile with explicit shield and obstacle geometry.
It does not cap histories or use deterministic background smoothing. For a pure
source-only high-fidelity run, use
`configs/geant4/high_fidelity_external_no_isaac.json`. The main
`result_spectrum*.png` output uses the highest-count measurement as the
representative spectrum; the last measurement is also saved as
`result_spectrum_last*.png`.

Environment variables are also supported:
`PIPLUP_NOTIFY_ENABLED=1`, `PIPLUP_NOTIFY_TOKEN`, `PIPLUP_NOTIFY_URL`,
`PIPLUP_NOTIFY_ACCOUNT`, `PIPLUP_NOTIFY_RUN_ID`, and
`PIPLUP_NOTIFY_TIMEOUT_S`.

### Analytic backend

Use this for the fastest smoke tests and for development that does not need a
live simulator process.

```bash
uv run python main.py --mode python-cui --max-steps 3
```

The Python GUI mode uses the same PF loop, but asks an Isaac Sim sidecar to show
the scene and generate observations with the Python observation model:

```bash
uv run python main.py --mode python-gui --max-steps 3
```

Random obstacle layouts can also be generated at startup. If Blender is
available, a matching USD environment is written under
`results/blender_environments/`. Random layouts reserve a connected corridor
from the initial robot cell toward a far corner, so the generated obstacle cells
cannot fully block robot motion. The startup order is:

1. Generate the random 3D USD environment.
2. Project its obstacle volumes onto the floor and write a 2D robot traversable map.
3. Start/reset the selected simulator with the USD and map paths.

The traversability JSON and PNG are written next to the generated USD as
`*.traversability.json` and `*.traversability.png`. The JSON can be loaded with
`planning.traversability.TraversabilityMap.load(...)` and passed as `map_api`
to future path-planning code. When a base USD such as Manchester Drum_Store is
used, Blender extracts occupancy from the imported USD meshes and the generated
obstacle boxes before the simulator is reset.

```bash
uv run python main.py \
  --mode python-cui \
  --environment-mode random \
  --obstacle-seed 7 \
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
  --mode python-gui \
  --sim-config configs/isaacsim/default_scene.json \
  --max-steps 3
```

For a real Isaac Sim stage, run the sidecar with Isaac Sim's Python so the
`isaacsim`, `omni`, and `pxr` packages are available. Set `ISAACSIM_PYTHON`
once for your machine:

```bash
export ISAACSIM_PYTHON=/path/to/isaacsim/python.sh

# Terminal 1: real headless Isaac Sim sidecar.
"$ISAACSIM_PYTHON" \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/real_scene.json

# Terminal 2: PF loop using the same host/port/timeouts.
uv run python main.py \
  --mode python-gui \
  --sim-config configs/isaacsim/real_scene.json \
  --max-steps 5
```

Use the GUI config when you want the Isaac Sim viewport to stay open and show
robot motion, shields, sources, and radiation visualization:

```bash
# Terminal 1: GUI sidecar. This config keeps the sidecar alive after main exits.
"$ISAACSIM_PYTHON" \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/gui_scene.json

# Terminal 2: drive it from the PF loop.
uv run python main.py \
  --mode python-gui \
  --sim-config configs/isaacsim/gui_scene.json \
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
# If Blender is not on PATH, set BLENDER=/path/to/blender first.
uv run python scripts/prepare_manchester_dataset.py \
  --asset Drum_Store \
  --blender-executable "${BLENDER:-blender}"
```

Prepared files are written under `data/manchester_nuclear_assets/`, which is
ignored by git because the full dataset is large. The default converted USD path
is `data/manchester_nuclear_assets/usd/drum_store.usda`, and the generated
config is `configs/isaacsim/manchester_drum_store.json`.

Run the converted Manchester scene like this:

```bash
# Terminal 1
"$ISAACSIM_PYTHON" \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/manchester_drum_store.json

# Terminal 2
uv run python main.py \
  --mode python-gui \
  --sim-config configs/isaacsim/manchester_drum_store.json \
  --max-steps 5
```

The same Manchester USD can be used as the base for random layouts. In random
mode, `main.py` reads `usd_path` from the simulation config, imports that USD in
Blender, adds the generated obstacle layout, writes a 2D traversability map
from the combined 3D scene, and only then starts/resets the simulator:

```bash
uv run python main.py \
  --mode python-gui \
  --sim-config configs/isaacsim/manchester_drum_store.json \
  --environment-mode random \
  --obstacle-seed 7 \
  --passage-width-m 1.0 \
  --robot-radius-m 0.35 \
  --max-steps 5
```

### Geant4 backend

The Geant4 backend uses the same sidecar protocol as Isaac Sim. `main.py`
auto-starts the Geant4 sidecar when no server is already listening on the
configured port. Sidecar logs go to `results/sidecars/`.

For a headless/no-GUI Geant4 run, use `geant4-cui`. This still uses the native
Geant4 executable; it only avoids starting the Isaac Sim GUI companion:

```bash
uv run python main.py \
  --mode geant4-cui \
  --max-steps 3
```

Use `configs/geant4/external_gui_scene.json` when you want native Geant4
transport paired with an Isaac Sim sidecar for robot motion and visualization.
Start the Isaac sidecar with Isaac Sim Python first, then let `main.py`
auto-start the Geant4 sidecar and reuse the running Isaac sidecar.

```bash
# Terminal 1
"$ISAACSIM_PYTHON" \
  scripts/run_isaacsim_bridge.py \
  --config configs/isaacsim/real_scene.json

# Terminal 2
uv run python main.py \
  --mode geant4-isaacsim-gui \
  --sim-config configs/geant4/external_gui_scene.json \
  --max-steps 5
```

For a real USD-backed native Geant4 run against the Manchester Drum_Store mesh,
prepare the Manchester assets first, then use:

```bash
uv run python main.py \
  --mode geant4-cui \
  --sim-config configs/geant4/real_scene.json \
  --max-steps 5
```

`configs/geant4/real_scene.json` uses the native external Geant4 engine while
reading scene geometry from the USD stage. It includes Poisson source emission,
explicit stage/shield geometry, Geant4 EM interactions, detector response
smearing, and dead-time scaling.

### Native Geant4 executable

Install Geant4 first and make sure `geant4-config` is on `PATH`, then build the
native sidecar executable:

```bash
uv run python scripts/build_geant4_sidecar.py
```

Run the PF loop with the external engine:

```bash
export ISAACSIM_PYTHON=/path/to/isaacsim/python.sh

uv run python main.py \
  --mode geant4-cui \
  --sim-config configs/geant4/external_scene.json \
  --max-steps 5
```

The native source is `native/geant4_sidecar/geant4_sidecar.cpp`. The Python
bridge communicates with it through a file-based request/response protocol, so
the C++ executable does not need a JSON dependency. GUI external-engine configs
are available as `configs/geant4/external_gui_scene.json`,
`configs/geant4/demo_room_external_gui.json`, and
`configs/geant4/demo_room_scatter_attenuation_external_gui.json`.

The native sidecar emits gamma rays isotropically and uses Geant4 geometry for
shield, obstacle, and stage attenuation. When Geant4 is built with
multithreading support, `thread_count` selects the worker count.

### Geant4 spectrum-count validation

The PF likelihood consumes isotope-wise source-equivalent counts extracted from
measured spectra. Geant4 runtime configs use
`spectrum_count_method: "photopeak_nnls"`: local full-energy peak ROIs are
fitted with nonnegative isotope peak columns and nuisance continuum terms, so
Compton scatter and room background are not converted into source strength.

Run the validation report after changing detector geometry, materials, transport
settings, or peak-efficiency calibration:

```bash
PYTHONPATH=src uv run python scripts/calibrate_geant4_net_response.py --dwell-time-s 30.0
```

The validation layout is `source_layouts/shield_validation.json`, and the
validation config is `configs/geant4/shield_validation_scene.json`. The script
writes diagnostics to `results/geant4_net_response_validation.csv` and a
diagnostic count-extraction check under `results/`; runtime PF configs do not
load a net-response calibration JSON.

## Birth move smoke test

Use these commands to verify birth is disabled/enabled:

```
# Birth disabled (default): r_mean stays at 1 and births remain 0.
uv run python main.py --max-steps 20

# Birth enabled: with max_sources=3, births should become >0 within a few steps.
uv run python main.py --birth --max-sources 3 --max-steps 20

```

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
| `analytic` | Approximate in-process Python spectrum simulator for smoke tests only. | None beyond `uv sync`. |
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
--mode python-cui          # Python analytic model, no simulator GUI
--mode geant4-cui          # Standard Geant4 full simulation, no Isaac Sim GUI
--full-simulation          # Alias for the standard Geant4 full simulation
--cui                      # Alias for the standard Geant4 full simulation
--headless                 # force a non-GUI simulator mode
--matplotlib-live          # optional Matplotlib live plot
--max-steps 5              # stop after a fixed number of measurements
--max-poses 3              # stop after a fixed number of robot poses
--source-config PATH       # load sources from JSON, default source_layouts/demo_sources.json
--obstacle-config PATH     # load blocked cells from JSON, default obstacle_layouts/no_obstacles.json
--environment-mode random  # generate a fresh obstacle map for this run
--obstacle-seed 7          # make fixed generation or random mode reproducible
--passage-width-m 1.0      # reserve a robot-passable corridor in random maps
--robot-radius-m 0.35      # clearance radius used for 2D traversability maps
--robot-speed 0.5          # nominal travel speed used in mission-time accounting
--rotation-overhead-s 0.5  # fixed shield actuation overhead per measurement
```

Saved runs start the URL-served CUI progress view by default. The run prints a
line like `CUI split visualization URL: http://<host>:8877/latest/index.html`
and continuously updates the robot 2D view, PF 3D view, and spectrum view under
`results/cui_view/latest/`. Set `cui_split_view: false` in a runtime config only
when this progress page is intentionally not needed.

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
  --full-simulation \
  --environment-mode random \
  --max-poses 10 \
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

For longer Geant4 runs, the standard full-simulation config already uses the
balanced Geant4 physics profile, 32 native/Python workers, response-Poisson
spectrum counting, eight shield measurements per station, and the no-Isaac CUI
runtime. Add `--init-grid-spacing-m 0` only when `--num-particles` should
control the PF size instead of the default grid initialization.

The standard full-simulation config is
`configs/geant4/variance_reduction_external_no_isaac_32threads.json`. It uses
explicit shield and obstacle geometry, does not cap histories, and does not use
deterministic background smoothing. For detector/secondary-transport validation
only, use `configs/geant4/high_fidelity_external_no_isaac.json`. The main
`result_spectrum*.png` output uses the highest-count measurement as the
representative spectrum; the last measurement is also saved as
`result_spectrum_last*.png`.

Environment variables are also supported:
`PIPLUP_NOTIFY_ENABLED=1`, `PIPLUP_NOTIFY_TOKEN`, `PIPLUP_NOTIFY_URL`,
`PIPLUP_NOTIFY_ACCOUNT`, `PIPLUP_NOTIFY_RUN_ID`, and
`PIPLUP_NOTIFY_TIMEOUT_S`.

### Approximate analytic backend

Use this for the fastest smoke tests and for development that does not need a
live simulator process. It is isolated from standard runtime simulation and is
not a substitute for Geant4/PF full simulation.

```bash
uv run python main.py --mode python-cui --max-steps 3
```

The Python GUI mode uses the same PF loop, but asks an Isaac Sim sidecar to show
the scene and generate observations with the approximate Python observation
model:

```bash
uv run python main.py --mode python-gui --max-steps 3
```

Random obstacle layouts can also be generated at startup. If Blender is
available, a matching USD environment is written under
`results/blender_environments/`. Random layouts always reserve a connected
whole-room exploration backbone before obstacle cells are sampled. This keeps
reachable lanes across the room instead of merely opening one narrow passage,
so random obstacles cannot isolate large unobservable regions from the robot.
Blocked cells are not exported as fully solid concrete cubes. Each random cell
is replaced by a known Manchester-style clutter asset, such as a hollow steel
cabinet, pipe rack, partly filled drums, concrete barrier, or aluminum frame.
The robot traversability map uses each asset footprint, while Geant4 and the PF
expected-count model use the known component boxes, materials, and hollow
internal structure for partial attenuation.
When `--environment-mode random` is used without an explicit `--source-config`,
the source layout is also generated randomly, but sources are constrained to
physical surfaces: floor, ceiling, room walls, exposed obstacle sides, or
obstacle tops. Random source generation never places sources in open air or
inside obstacle volumes. The startup order is:

1. Generate the random 3D USD environment.
2. Project its obstacle volumes onto the floor and write a 2D robot traversable map.
3. Start/reset the selected simulator with the USD and map paths.

The traversability JSON and PNG are written next to the generated USD as
`*.traversability.json` and `*.traversability.png`. The JSON can be loaded with
`planning.traversability.TraversabilityMap.load(...)` and passed as `map_api`
to future path-planning code. When a base USD such as Manchester Drum_Store is
used, Blender extracts occupancy from the imported USD meshes and the generated
known obstacle assets before the simulator is reset.

```bash
uv run python main.py \
  --mode python-cui \
  --environment-mode random \
  --obstacle-seed 7 \
  --max-steps 3
```

To keep a fixed source layout while randomizing only the obstacles, pass the
source JSON explicitly:

```bash
uv run python main.py \
  --full-simulation \
  --environment-mode random \
  --source-config source_layouts/demo_sources.json
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
Blender, adds the generated known obstacle assets, writes a 2D traversability
map from the combined 3D scene, and only then starts/resets the simulator:

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
  --full-simulation \
  --max-steps 3
```

The repository default is the same standard full simulation, so
`uv run python main.py` also selects
`configs/geant4/variance_reduction_external_no_isaac_32threads.json`.
Use `--python-cui` only when you explicitly want the analytic Python model.

Use `--mode geant4-isaacsim-gui` when you want the standard native Geant4
transport paired with an Isaac Sim sidecar for robot motion and visualization.
The default GUI config is
`configs/geant4/variance_reduction_external_gui_32threads.json`, which inherits
the standard no-GUI full-simulation config and only adds Isaac Sim sidecar
startup settings.

```bash
uv run python main.py \
  --mode geant4-isaacsim-gui \
  --max-steps 5
```

For a real USD-backed native Geant4 run against the Manchester Drum_Store mesh,
prepare the Manchester assets first, then use the GUI-capable Geant4 config:

```bash
uv run python main.py \
  --mode geant4-isaacsim-gui \
  --sim-config configs/geant4/external_gui_scene.json \
  --max-steps 5
```

`configs/geant4/external_gui_scene.json` remains available for explicit
USD-backed Manchester Drum Store runs. It is not the default GUI mode because
the default GUI/CUI pair should share the same generated environment and Geant4
runtime settings.

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
  --full-simulation \
  --max-steps 5
```

The native source is `native/geant4_sidecar/geant4_sidecar.cpp`. The Python
bridge communicates with it through a file-based request/response protocol, so
the C++ executable does not need a JSON dependency. Older Geant4 configs that
are not part of the standard runtime are isolated under `configs/geant4/legacy/`.

Under the standard `source_rate_model = detector_cps_1m` model, the native
sidecar uses detector-equivalent source sampling and Geant4 geometry for shield,
obstacle, and stage attenuation. When Geant4 is built with multithreading
support, `thread_count` selects the worker count.

### Geant4 spectrum-count validation

The PF likelihood consumes isotope-wise source-equivalent counts extracted from
measured spectra. Geant4 runtime configs use
`spectrum_count_method: "response_poisson"`: a calibrated full-spectrum
Poisson response regression extracts isotope counts and propagates count
covariance to the PF likelihood. `photopeak_nnls` remains available only for
diagnostic and calibration checks; `peak_window` and `response_matrix` are not
runtime count-ingestion methods.

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

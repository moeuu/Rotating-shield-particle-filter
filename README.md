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

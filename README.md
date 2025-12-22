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

## Real-time visualization

To run the real-time particle filter visualization demo:

```
uv run python main.py
```

The run shows robot trajectory, shield orientations, particle clouds, and estimated sources; `results/result_pf.png` is written at the end, alongside the final spectrum as `results/result_spectrum.png`.

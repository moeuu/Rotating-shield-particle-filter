# Rotating-shield-particle-filter

Demo simulations assume strong sources (≈20,000 cps at 1 m) for Cs-137, Co-60, and Eu-154 to mimic high-dose environments. Core APIs remain unchanged; only example defaults are scaled.

## Real-time visualization

To run the real-time particle filter visualization demo:

```
python main.py --scenario configs/example_scenario.yaml --output result.png
```

The scenario argument is optional in the current demo (built-in synthetic path/sources are used if omitted). The run shows robot trajectory, shield orientations, particle clouds, and estimated sources; `result.png` is written at the end.

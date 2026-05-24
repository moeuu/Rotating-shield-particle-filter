# RA-L Minimal Ablation Plan

This file is the shared project note for the RA-L paper ablation scope. New
Codex sessions should use this plan unless the user explicitly changes it.

## Decision

Use a single deterministic seed for the paper table:

- `seed = 2026050901`

Run the core four variants for every RA-L case:

- `proposed`
- `baseline_passive_no_shield`
- `round_robin_shield`
- `one_step_path`

Run one extra residual-birth ablation only for the mixed-cardinality case:

- `case03_mixed_cardinality` + `no_residual_birth`

This gives 13 full-simulation runs:

- `case01_multi_isotope`: 4 variants
- `case02_three_cs`: 4 variants
- `case03_mixed_cardinality`: 5 variants

## Rationale

- `proposed` is the reference method.
- `baseline_passive_no_shield` shows the gain over ordinary unshielded mobile
  nondirectional measurements.
- `round_robin_shield` keeps the rotating shield but removes adaptive shield
  selection, isolating the value of information-driven shield programs.
- `one_step_path` keeps the same estimator and shield machinery but replaces
  DSS-PP with greedy planning, isolating the value of the planner.
- `no_residual_birth` is included only for `case03_mixed_cardinality`, where
  residual birth is most directly tied to mixed-cardinality multi-source
  discovery.

The full 162-run manifest remains available for supplementary or extended
experiments, but the RA-L paper plan is the 13-run subset above.

## Generated Files

Regenerate the paper subset after changing the full ablation manifest:

```bash
uv run python scripts/build_ral_paper_subset.py
```

The generated files are:

- `results/ral_ablation/ral_paper_subset_manifest.csv`
- `results/ral_ablation/run_paper_subset.sh`

Run the selected experiments with:

```bash
bash results/ral_ablation/run_paper_subset.sh
```

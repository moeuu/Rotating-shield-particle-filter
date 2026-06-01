# RA-L Minimal Ablation Plan

This file is the shared project note for the RA-L paper ablation scope. New
Codex sessions should use this plan unless the user explicitly changes it.

## Decision

Use a single deterministic seed for the paper table:

- `seed = 2026050901`

Use one main RA-L task. Past Case01/Case02/Case03 paper cases are not part of
the standard RA-L ablation implementation:

- `mix9_multi_isotope_cardinality`
- ground truth source cardinality: `4 Cs-137 + 3 Co-60 + 2 Eu-154`
- source support: walls, floor, ceiling, and exposed obstacle surfaces
- random placement uses the standard visibility, ceiling-count, and preferred
  height constraints from the Geant4 runtime config

Run only four closed-loop full-simulation variants for the main paper table:

- `proposed`
- `baseline_passive_equal_time_no_shield`
- `round_robin_shield`
- `eig_only_path`

Use the same source-count support across all variants:

- `max_sources = 5` per isotope

This is a method-level search support, not a value inferred from the known
ground-truth source count in the task.

## Rationale

The paper claim is multi-isotope source-term estimation with isotope-wise source
cardinality, 3-D localization, strength estimation, and same-isotope ambiguity
inside multiple radionuclide channels. The single MIX-9 task exercises all of
these mechanisms in one expensive run.

- `proposed` is the reference method.
- `baseline_passive_equal_time_no_shield` disables shield coding while
  preserving the same per-station physical live-time budget. It tests whether
  longer nondirectional dwell alone is sufficient.
- `round_robin_shield` keeps the Fe/Pb shield and the same posture budget but
  removes posterior-adaptive shield-program selection. It tests whether the
  hardware alone is sufficient.
- `eig_only_path` keeps active planning and shield programs but removes the
  explicit same-isotope signature, response-correlation, obstacle-shadow, and
  elevation terms from DSS-PP. It tests whether ordinary information-driven
  planning is sufficient.

Estimator-side ablations should be run as logged replay when the proposed
measurement log is available:

- `no_residual_birth`
- `no_verification`

These replay ablations isolate source-cardinality birth and final
verification/refit-after-remove logic without paying for additional Geant4
closed-loop transport. Do not include them in the default full-simulation run
script unless the user explicitly accepts the extra runtime.

## Generated Files

Regenerate the exhaustive manifest and then the compact paper subset:

```bash
PYTHONPATH=src uv run python -m baselines.ral_ablation.cli --seeds 2026050901
uv run python scripts/build_ral_paper_subset.py
```

The paper subset files are:

- `results/ral_ablation/ral_paper_subset_manifest.csv`
- `results/ral_ablation/run_paper_subset.sh`

Run the selected full simulations with:

```bash
bash results/ral_ablation/run_paper_subset.sh
```

Regenerate the RA-L manuscript figures after paper-scope results are available:

```bash
uv run python scripts/build_ral_figures.py
```

The experiment figure policy is recorded in
`docs/ral_experiment_figure_policy.md`. Before marking any figure ready, apply
the visual and logical QA checklist in `docs/ral_figure_quality_policy.md` and
inspect the generated review PNGs.

## Current Result Notes

The current PDF may contain placeholder or old representative values while new
MIX-9 experiments are pending. Treat those values as replaceable table entries;
do not change the agreed experiment scope just to match an old result table.

Shield-systematic strength handling is documented in
`docs/shield_strength_systematics.md`. It is reversible through
`response_poisson_shield_systematic_variance_enable=false` and must not be used
to artificially degrade no-shield baselines.

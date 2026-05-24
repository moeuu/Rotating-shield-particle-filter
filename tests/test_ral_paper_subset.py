"""Tests for the compact RA-L paper ablation subset."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_ral_paper_subset.py"
)
SPEC = importlib.util.spec_from_file_location("build_ral_paper_subset", MODULE_PATH)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
DEFAULT_SEED = MODULE.DEFAULT_SEED
select_paper_subset = MODULE.select_paper_subset


def _manifest_row(case: str, variant: str, seed: str = DEFAULT_SEED) -> dict[str, str]:
    """Return one minimal manifest row for subset tests."""
    tag = f"{case}_{variant}_seed_{seed}"
    return {
        "case": case,
        "variant": variant,
        "seed": seed,
        "config_path": f"results/ral_ablation/configs/{tag}.json",
        "source_path": f"results/ral_ablation/sources/{case}_seed_{seed}.json",
        "command": f"uv run python main.py --output-tag {tag}",
    }


def test_select_paper_subset_uses_thirteen_run_plan() -> None:
    """The RA-L paper subset should keep 4 core variants plus Case03 birth."""
    cases = (
        "case01_multi_isotope",
        "case02_three_cs",
        "case03_mixed_cardinality",
    )
    variants = (
        "proposed",
        "baseline_passive_no_shield",
        "round_robin_shield",
        "one_step_path",
        "no_residual_birth",
        "no_shield",
    )
    rows = [_manifest_row(case, variant) for case in cases for variant in variants]

    subset = select_paper_subset(rows)
    selected_pairs = {(row["case"], row["variant"]) for row in subset}

    assert len(subset) == 13
    for case in cases:
        assert (case, "proposed") in selected_pairs
        assert (case, "baseline_passive_no_shield") in selected_pairs
        assert (case, "round_robin_shield") in selected_pairs
        assert (case, "one_step_path") in selected_pairs
    assert ("case03_mixed_cardinality", "no_residual_birth") in selected_pairs
    assert ("case01_multi_isotope", "no_residual_birth") not in selected_pairs
    assert ("case02_three_cs", "no_residual_birth") not in selected_pairs
    assert all(row["seed"] == DEFAULT_SEED for row in subset)

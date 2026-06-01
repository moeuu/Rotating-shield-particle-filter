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


def test_select_paper_subset_uses_mix9_four_run_plan() -> None:
    """The RA-L paper subset should keep the four closed-loop MIX-9 runs."""
    cases = (
        "mix9_multi_isotope_cardinality",
        "legacy_case_not_selected",
    )
    variants = (
        "proposed",
        "baseline_passive_equal_time_no_shield",
        "round_robin_shield",
        "eig_only_path",
        "no_residual_birth",
        "no_verification",
        "no_shield",
    )
    rows = [_manifest_row(case, variant) for case in cases for variant in variants]

    subset = select_paper_subset(rows)
    selected_pairs = {(row["case"], row["variant"]) for row in subset}

    assert len(subset) == 4
    assert (
        "mix9_multi_isotope_cardinality",
        "proposed",
    ) in selected_pairs
    assert (
        "mix9_multi_isotope_cardinality",
        "baseline_passive_equal_time_no_shield",
    ) in selected_pairs
    assert (
        "mix9_multi_isotope_cardinality",
        "round_robin_shield",
    ) in selected_pairs
    assert (
        "mix9_multi_isotope_cardinality",
        "eig_only_path",
    ) in selected_pairs
    assert all(case == "mix9_multi_isotope_cardinality" for case, _ in selected_pairs)
    assert ("mix9_multi_isotope_cardinality", "no_residual_birth") not in selected_pairs
    assert ("mix9_multi_isotope_cardinality", "no_verification") not in selected_pairs
    assert all(row["seed"] == DEFAULT_SEED for row in subset)

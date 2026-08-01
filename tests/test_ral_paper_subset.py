"""Tests for the compact RA-L paper ablation subset."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_ral_paper_subset.py"
)
SPEC = importlib.util.spec_from_file_location("build_ral_paper_subset", MODULE_PATH)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
select_paper_subset = MODULE.select_paper_subset
TEST_SEED = "246813579"


def _manifest_row(case: str, variant: str, seed: str = TEST_SEED) -> dict[str, str]:
    """Return one minimal manifest row for subset tests."""
    tag = f"{case}_{variant}_seed_{seed}"
    return {
        "case": case,
        "variant": variant,
        "seed": seed,
        "source_seed": str(int(seed) + 17),
        "seed_policy": "fresh_per_batch",
        "config_path": f"results/ral_ablation/configs/{tag}.json",
        "source_path": f"results/ral_ablation/sources/{case}_seed_{seed}.json",
        "command": (
            "uv run python main.py --full-simulation "
            f"--sim-config results/ral_ablation/configs/{tag}.json "
            "--source-config "
            f"results/ral_ablation/sources/{case}_seed_{seed}.json "
            f"--output-tag {tag}"
        ),
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
    assert all(row["seed"] == TEST_SEED for row in subset)


def test_select_paper_subset_requires_seed_for_multi_batch_manifest() -> None:
    """Implicit selection must not silently choose among independent scenes."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant, seed)
        for seed in ("100", "200")
        for variant in MODULE.CORE_VARIANTS
    ]

    with pytest.raises(ValueError, match="exactly one scene seed"):
        select_paper_subset(rows)


@pytest.mark.parametrize(
    "replacement",
    [
        "--python-cui",
        "--full-simulation --sim-backend analytic",
        "--mode python-cui",
        "",
    ],
)
def test_select_paper_subset_rejects_non_geant4_commands(
    replacement: str,
) -> None:
    """A modified manifest cannot relabel an analytic run as a paper trial."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant)
        for variant in MODULE.CORE_VARIANTS
    ]
    rows[0]["command"] = rows[0]["command"].replace(
        "--full-simulation",
        replacement,
    )

    with pytest.raises(ValueError, match="full-simulation|conflicting"):
        select_paper_subset(rows)

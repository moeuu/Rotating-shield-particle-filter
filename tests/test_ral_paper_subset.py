"""Tests for the current shared-runtime RA-L paper subset."""

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
    """Return one current-schema manifest row for subset tests."""
    root = Path(__file__).resolve().parents[1]
    runtime_root = root.parent / "Rotating-shield-simulation-runtime"
    tag = f"{case}_{variant}_seed_{seed}"
    scenario = runtime_root / "private_runs" / "ral_ablation" / f"{tag}.json"
    log_path = root / "results" / "ral_ablation" / "measurement_logs" / tag
    pf_config = root / "results" / "ral_ablation" / "configs" / f"{tag}.json"
    runtime_config = (
        runtime_root
        / "private_runs"
        / "ral_ablation"
        / "runtime_configs"
        / f"{tag}.json"
    )
    pf_output = root / "results" / "ral_ablation" / "runs" / tag
    source_profile = "ral-mix9"
    return {
        "case": case,
        "variant": variant,
        "seed": seed,
        "pf_seed": seed,
        "seed_policy": "fresh_per_batch",
        "source_profile": source_profile,
        "pf_config_path": pf_config.as_posix(),
        "runtime_config_path": runtime_config.as_posix(),
        "scenario_path": scenario.as_posix(),
        "measurement_log_path": log_path.as_posix(),
        "pf_output_dir": pf_output.as_posix(),
        "scenario_command": (
            f"uv run --directory {runtime_root} rotating-shield-sim "
            f"generate-ral-scenario {scenario} "
            f"--measurement-log-output {log_path} --run-id {tag} "
            f"--runtime-config {runtime_config} --scene-seed {seed} "
            f"--source-profile {source_profile}"
        ),
        "pf_command": (
            f"uv run --directory {root} rotating-shield-pf-live "
            f"--scenario {scenario} "
            f"--runtime-root {runtime_root} --config {pf_config} "
            f"--output-dir {pf_output} --profile pf_strict --seed {seed} "
            f"--private-scene-profile {source_profile}"
        ),
    }


def test_select_paper_subset_uses_mix9_four_run_plan() -> None:
    """The paper subset should retain exactly the four causal MIX-9 runs."""
    cases = ("mix9_multi_isotope_cardinality", "legacy_case_not_selected")
    variants = (*MODULE.CORE_VARIANTS, "no_shield")
    rows = [_manifest_row(case, variant) for case in cases for variant in variants]

    subset = select_paper_subset(rows)

    assert [row["variant"] for row in subset] == list(MODULE.CORE_VARIANTS)
    assert all(row["case"] == "mix9_multi_isotope_cardinality" for row in subset)
    assert all("generate-ral-scenario" in row["scenario_command"] for row in subset)
    assert all("rotating-shield-pf-live" in row["pf_command"] for row in subset)


def test_select_paper_subset_requires_seed_for_multi_batch_manifest() -> None:
    """Implicit selection must not choose among independent scenes."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant, seed)
        for seed in ("100", "200")
        for variant in MODULE.CORE_VARIANTS
    ]
    with pytest.raises(ValueError, match="exactly one scene seed"):
        select_paper_subset(rows)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("scenario_command", "rotating-shield-sim", "scenario command"),
        ("scenario_command", "--scene-seed 999", "scene-seed"),
        ("pf_command", "python main.py --full-simulation", "PF command"),
        ("pf_command", "--profile legacy", "--profile"),
    ],
)
def test_select_paper_subset_rejects_obsolete_or_mismatched_commands(
    field: str,
    replacement: str,
    message: str,
) -> None:
    """Manifest commands must match the current runtime-to-PF boundary."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant)
        for variant in MODULE.CORE_VARIANTS
    ]
    if field == "scenario_command" and replacement == "rotating-shield-sim":
        rows[0][field] = replacement
    elif field == "scenario_command":
        rows[0][field] = rows[0][field].replace(
            f"--scene-seed {TEST_SEED}", replacement
        )
    elif replacement.startswith("python"):
        rows[0][field] = replacement
    else:
        rows[0][field] = rows[0][field].replace("--profile pf_strict", replacement)
    with pytest.raises(ValueError, match=message):
        select_paper_subset(rows)

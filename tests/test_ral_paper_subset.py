"""Tests for the current shared-runtime RA-L paper subset."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from baselines.ral_ablation.config_factory import (
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_RUNTIME_ROOT,
    build_ablation_plan,
    write_ablation_plan,
)

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
TEST_PF_SEED = "975318642"
TEST_TRANSPORT_SEED = "864297531"


def _manifest_row(
    case: str,
    variant: str,
    seed: str = TEST_SEED,
    *,
    batch_id: str = "opaque001",
) -> dict[str, str]:
    """Return one current-schema manifest row for subset tests."""
    root = Path(__file__).resolve().parents[1]
    runtime_root = root.parent / "Rotating-shield-simulation-runtime"
    tag = f"ral_{batch_id}_{variant}"
    scenario = runtime_root / "private_runs" / "ral_ablation" / f"{tag}.json"
    truth_manifest = (
        runtime_root
        / "private_runs"
        / "ral_ablation"
        / "truth_manifests"
        / f"{tag}.json"
    )
    log_path = root / "results" / "ral_ablation" / "measurement_logs" / tag
    pf_config = root / "results" / "ral_ablation" / "configs" / f"{tag}.json"
    control_policy = (
        root / "results" / "ral_ablation" / "control_policies" / f"{tag}.json"
    )
    runtime_config = (
        runtime_root
        / "private_runs"
        / "ral_ablation"
        / "runtime_configs"
        / f"{tag}.json"
    )
    pf_output = root / "results" / "ral_ablation" / "runs" / tag
    return {
        "case": case,
        "experiment_profile_id": MODULE.RAL_EXPERIMENT_PROFILE_ID,
        "scene_variant_id": MODULE.RAL_SCENE_VARIANT_ID,
        "variant": variant,
        "batch_id": batch_id,
        "scene_seed": seed,
        "pf_seed": TEST_PF_SEED,
        "transport_seed": TEST_TRANSPORT_SEED,
        "seed_policy": "fresh_per_batch",
        "run_id": tag,
        "pf_config_path": pf_config.as_posix(),
        "control_policy_path": control_policy.as_posix(),
        "control_policy_sha256": "a" * 64,
        "runtime_config_path": runtime_config.as_posix(),
        "scenario_path": scenario.as_posix(),
        "truth_manifest_path": truth_manifest.as_posix(),
        "measurement_log_path": log_path.as_posix(),
        "pf_output_dir": pf_output.as_posix(),
        "scenario_command": (
            f"uv run --directory {runtime_root} rotating-shield-sim "
            f"generate-scenario {scenario} "
            f"--truth-manifest-output {truth_manifest} "
            f"--measurement-log-output {log_path} --run-id {tag} "
            f"--runtime-config {runtime_config} --scene-seed {seed} "
            f"--experiment-profile {MODULE.RAL_EXPERIMENT_PROFILE_ID} "
            f"--scene-variant {MODULE.RAL_SCENE_VARIANT_ID}"
        ),
        "session_command": (
            f"uv run --directory {root} python -m "
            f"baselines.ral_ablation.session_runner "
            f"--runtime-root {runtime_root} --scenario {scenario} "
            f"--truth-manifest {truth_manifest} "
            f"--pf-config {pf_config} --control-policy {control_policy} "
            f"--expected-control-policy-sha256 {'a' * 64} "
            f"--pf-output-dir {pf_output} --pf-seed {TEST_PF_SEED}"
        ),
    }


def test_select_paper_subset_uses_mix9_four_run_plan() -> None:
    """The paper subset should retain exactly the four causal MIX-9 runs."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant)
        for variant in MODULE.CORE_VARIANTS
    ]

    subset = select_paper_subset(rows)

    assert [row["variant"] for row in subset] == list(MODULE.CORE_VARIANTS)
    assert all(row["case"] == "mix9_multi_isotope_cardinality" for row in subset)
    assert all("generate-scenario" in row["scenario_command"] for row in subset)
    assert all(
        "baselines.ral_ablation.session_runner" in row["session_command"]
        for row in subset
    )


def test_select_paper_subset_requires_id_for_multi_batch_manifest() -> None:
    """Implicit selection must not choose among opaque comparison batches."""
    rows = [
        _manifest_row(
            "mix9_multi_isotope_cardinality",
            variant,
            batch_id=batch_id,
        )
        for batch_id in ("batch100", "batch200")
        for variant in MODULE.CORE_VARIANTS
    ]
    with pytest.raises(ValueError, match="exactly one batch_id"):
        select_paper_subset(rows)


def test_select_paper_subset_rejects_extra_row_in_selected_batch() -> None:
    """A selected batch cannot hide an undeclared legacy trial."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant)
        for variant in MODULE.CORE_VARIANTS
    ]
    rows.append(_manifest_row("legacy_case", "legacy_variant"))
    with pytest.raises(ValueError, match="undeclared case/variant"):
        select_paper_subset(rows)


def test_select_paper_subset_rejects_noncanonical_batch_id() -> None:
    """Batch identifiers must be safe exact artifact identifiers."""
    rows = [
        _manifest_row(
            "mix9_multi_isotope_cardinality",
            variant,
            batch_id="bad batch",
        )
        for variant in MODULE.CORE_VARIANTS
    ]
    with pytest.raises(ValueError, match="ASCII letters"):
        select_paper_subset(rows)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("scene_seed", "111", "one exact scene_seed"),
        ("pf_seed", "222", "one exact pf_seed"),
        ("transport_seed", "333", "one exact transport_seed"),
        (
            "experiment_profile_id",
            "wrong-profile",
            "one exact experiment_profile_id",
        ),
        ("scene_variant_id", "cs4-co3-eu0", "one exact scene_variant_id"),
    ],
)
def test_select_paper_subset_rejects_cross_bound_batch_rows(
    field: str,
    replacement: str,
    message: str,
) -> None:
    """All four variants must bind to one environment and truth identity."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant)
        for variant in MODULE.CORE_VARIANTS
    ]
    rows[-1][field] = replacement
    if field == "scene_seed":
        rows[-1]["scenario_command"] = rows[-1]["scenario_command"].replace(
            f"--scene-seed {TEST_SEED}",
            f"--scene-seed {replacement}",
        )
    with pytest.raises(ValueError, match=message):
        select_paper_subset(rows)


@pytest.mark.parametrize("field", ("pf_seed", "transport_seed"))
def test_select_paper_subset_rejects_seed_aliasing(field: str) -> None:
    """PF and transport randomness must not alias private truth generation."""
    rows = [
        _manifest_row("mix9_multi_isotope_cardinality", variant)
        for variant in MODULE.CORE_VARIANTS
    ]
    for row in rows:
        row[field] = TEST_SEED
        if field == "pf_seed":
            row["session_command"] = row["session_command"].replace(
                f"--pf-seed {TEST_PF_SEED}",
                f"--pf-seed {TEST_SEED}",
            )
    with pytest.raises(ValueError, match="pairwise independent"):
        select_paper_subset(rows)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("scenario_command", "rotating-shield-sim", "scenario command"),
        ("scenario_command", "--scene-seed 999", "scene-seed"),
        ("scenario_command", "--scene-variant cs4-co3-eu0", "scene-variant"),
        ("session_command", "python main.py --full-simulation", "session command"),
        ("session_command", "--pf-seed 999", "--pf-seed"),
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
        original = (
            f"--scene-seed {TEST_SEED}"
            if replacement.startswith("--scene-seed")
            else f"--scene-variant {MODULE.RAL_SCENE_VARIANT_ID}"
        )
        rows[0][field] = rows[0][field].replace(original, replacement)
    elif replacement.startswith("python"):
        rows[0][field] = replacement
    else:
        rows[0][field] = rows[0][field].replace(
            f"--pf-seed {TEST_PF_SEED}", replacement
        )
    with pytest.raises(ValueError, match=message):
        select_paper_subset(rows)


def test_manifest_reader_requires_exact_current_header(tmp_path: Path) -> None:
    """Extra or reordered CSV fields must not enter the paper run script."""
    row = _manifest_row("mix9_multi_isotope_cardinality", MODULE.CORE_VARIANTS[0])
    malformed = tmp_path / "manifest.csv"
    malformed.write_text(
        ",".join((*MODULE.MANIFEST_FIELDS, "legacy_field"))
        + "\n"
        + ",".join((*(row[field] for field in MODULE.MANIFEST_FIELDS), "ignored"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly match"):
        MODULE._read_manifest(malformed)


def test_generated_manifest_round_trips_through_strict_subset_builder(
    tmp_path: Path,
) -> None:
    """The production factory and strict subset schema must share one contract."""
    private_root = tmp_path / "private"
    entries = build_ablation_plan(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        runtime_config_path=DEFAULT_RUNTIME_CONFIG,
        output_dir=tmp_path / "results",
        private_root=private_root,
        seeds=(int(TEST_SEED),),
        pf_seeds=(int(TEST_PF_SEED),),
        transport_seeds=(int(TEST_TRANSPORT_SEED),),
        batch_ids=("opaque001",),
    )
    manifest_path, _ = write_ablation_plan(entries, private_root=private_root)
    selected = MODULE.build_subset(
        manifest_path,
        private_root / "subset.csv",
        private_root / "run_subset.sh",
        batch_id="opaque001",
    )

    assert len(selected) == 4
    assert {row["batch_id"] for row in selected} == {"opaque001"}
    assert all(
        f"--experiment-profile {MODULE.RAL_EXPERIMENT_PROFILE_ID}"
        in row["scenario_command"]
        for row in selected
    )
    assert all(
        f"--scene-variant {MODULE.RAL_SCENE_VARIANT_ID}" in row["scenario_command"]
        for row in selected
    )


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("runtime", "no-op runtime intervention"),
        ("control", "variant-policy digest"),
        ("pf", "inactive planner fields"),
    ],
)
def test_subset_builder_rejects_variant_artifact_cross_binding(
    tmp_path: Path,
    artifact: str,
    message: str,
) -> None:
    """A valid artifact from another method cannot satisfy a variant row."""
    private_root = tmp_path / "private"
    entries = build_ablation_plan(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        runtime_config_path=DEFAULT_RUNTIME_CONFIG,
        output_dir=tmp_path / "results",
        private_root=private_root,
        seeds=(int(TEST_SEED),),
        pf_seeds=(int(TEST_PF_SEED),),
        transport_seeds=(int(TEST_TRANSPORT_SEED),),
        batch_ids=("opaque001",),
    )
    by_variant = {entry.variant: entry for entry in entries}
    proposed = by_variant["proposed"]
    if artifact == "runtime":
        payload = json.loads(proposed.runtime_config_path.read_text(encoding="utf-8"))
        payload["shield_transmission_target"] = 1.0
        proposed.runtime_config_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    elif artifact == "control":
        proposed.control_policy_path.write_text(
            by_variant["round_robin_shield"].control_policy_path.read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
    else:
        by_variant["round_robin_shield"].pf_config_path.write_text(
            by_variant["eig_only_path"].pf_config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    manifest_path, _ = write_ablation_plan(entries, private_root=private_root)

    with pytest.raises(ValueError, match=message):
        MODULE.build_subset(
            manifest_path,
            private_root / "subset.csv",
            private_root / "run_subset.sh",
            batch_id="opaque001",
        )

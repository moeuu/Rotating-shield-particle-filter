"""Tests for exact authored RA-L comparison-batch sealing."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
from runtime.experiment_profiles import CS_CO_SURFACE_SEARCH_PROFILE

from baselines.ral_ablation.batch_contract import seal_authored_batch
from baselines.ral_ablation.config_factory import (
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_RUNTIME_ROOT,
    RAL_SCENE_VARIANT_ID,
    build_ablation_plan,
    write_ablation_plan,
)


def _write_json(path: Path, payload: object) -> None:
    """Write one deterministic JSON object for a private test artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _authored_batch(tmp_path: Path) -> tuple[Path, list[object]]:
    """Create four manifest-bound authored scenarios with identical truth."""
    private_root = tmp_path / "private"
    entries = build_ablation_plan(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        runtime_config_path=DEFAULT_RUNTIME_CONFIG,
        output_dir=tmp_path / "results",
        private_root=private_root,
        seeds=(1234,),
        pf_seeds=(5678,),
        transport_seeds=(8765,),
        batch_ids=("batch001",),
    )
    provenance = {
        "schema_version": 1,
        "bit_generator": "PCG64",
        "derivation_method": "test",
        "root_seed": 1234,
        "streams": {},
    }
    sources = [
        {
            "isotope": isotope,
            "position": [float(index + 1), float(2 * index + 1), 1.0],
            "intensity_cps_1m": 500000.0 + 10000.0 * index,
        }
        for index, isotope in enumerate(
            ("Cs-137",) * 4 + ("Co-60",) * 3
        )
    ]
    acquisition = {
        "schema_version": 1,
        **asdict(CS_CO_SURFACE_SEARCH_PROFILE.acquisition),
    }
    environment = {
        "experiment_profile_id": CS_CO_SURFACE_SEARCH_PROFILE.profile_id,
        "acquisition_contract": acquisition,
        "size_x": 10.0,
        "size_y": 15.0,
        "size_z": 5.0,
        "obstacle_grid": {"contract": "same"},
    }
    metadata = {
        "experiment_profile_id": CS_CO_SURFACE_SEARCH_PROFILE.profile_id,
        "private_scene_variant_id": RAL_SCENE_VARIANT_ID,
        "scene_seed": 1234,
        "scene_rng_provenance": provenance,
        "scenario_family": "test",
    }
    for entry in entries:
        _write_json(
            entry.truth_manifest_path,
            {
                "schema_version": 1,
                "run_id": entry.run_id,
                "experiment_profile_id": entry.experiment_profile_id,
                "scene_variant_id": entry.scene_variant_id,
                "scene_seed": entry.scene_seed,
                "scene_rng_provenance": provenance,
                "sources": sources,
            },
        )
        _write_json(
            entry.scenario_path,
            {
                "schema_version": 1,
                "run_id": entry.run_id,
                "backend": "geant4",
                "runtime_config_path": entry.runtime_config_path.as_posix(),
                "output_dir": entry.measurement_log_path.as_posix(),
                "isotopes": ["Co-60", "Cs-137"],
                "environment": environment,
                "obstacle_layout_path": None,
                "scene": {"sources": sources, "geometry": "same"},
                "metadata": metadata,
            },
        )
    manifest_path, _ = write_ablation_plan(entries, private_root=private_root)
    return manifest_path, entries


def test_authored_batch_contract_seals_identical_environment_and_sources(
    tmp_path: Path,
) -> None:
    """All variants must bind one exact scene/source comparison payload."""
    manifest_path, _entries = _authored_batch(tmp_path)
    output = tmp_path / "private" / "batch_contracts" / "batch001.json"

    contract = seal_authored_batch(
        manifest_path,
        output,
        batch_id="batch001",
    )

    assert output.is_file()
    assert contract["source_count_by_isotope"] == {"Co-60": 3, "Cs-137": 4}
    assert len(contract["comparison_contract_sha256"]) == 64
    assert len(contract["private_truth_contract_sha256"]) == 64
    assert set(contract["per_variant"]) == {
        "proposed",
        "no_shield_native_path",
        "round_robin_shield",
        "eig_only_path",
    }


def test_authored_batch_contract_rejects_one_variant_source_change(
    tmp_path: Path,
) -> None:
    """A same-seed declaration cannot hide different authored source values."""
    manifest_path, entries = _authored_batch(tmp_path)
    changed = entries[-1]
    payload = json.loads(changed.truth_manifest_path.read_text(encoding="utf-8"))
    payload["sources"][0]["intensity_cps_1m"] = 500001.0
    _write_json(changed.truth_manifest_path, payload)

    with pytest.raises(ValueError, match="scenario sources differ"):
        seal_authored_batch(
            manifest_path,
            tmp_path / "contract.json",
            batch_id="batch001",
        )


def test_authored_batch_contract_rejects_one_variant_environment_change(
    tmp_path: Path,
) -> None:
    """A variant may not silently receive a different physical environment."""
    manifest_path, entries = _authored_batch(tmp_path)
    changed = entries[-1]
    payload = json.loads(changed.scenario_path.read_text(encoding="utf-8"))
    payload["environment"]["obstacle_grid"] = {"contract": "different"}
    _write_json(changed.scenario_path, payload)

    with pytest.raises(ValueError, match="do not share the exact authored"):
        seal_authored_batch(
            manifest_path,
            tmp_path / "contract.json",
            batch_id="batch001",
        )

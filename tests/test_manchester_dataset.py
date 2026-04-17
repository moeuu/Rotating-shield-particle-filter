"""Tests for Manchester nuclear dataset preparation helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import zipfile

import pytest

from sim.manchester_dataset import (
    asset_slug,
    convert_manchester_sdf_to_usd,
    extract_manchester_asset,
    find_manchester_sdf_path,
    resolve_manchester_asset,
    write_isaacsim_config,
)


def test_resolve_manchester_asset_accepts_human_names() -> None:
    """Asset lookup should accept spaces, underscores, and zip filenames."""
    assert resolve_manchester_asset("Drum Store").key == "Drum_Store"
    assert resolve_manchester_asset("drum_store.zip").key == "Drum_Store"
    assert resolve_manchester_asset("500l drum store").filename == "500L_Drum_Store.zip"
    assert asset_slug("500L Drum Store") == "500l_drum_store"


def test_extract_manchester_asset_returns_top_level_directory(tmp_path: Path) -> None:
    """Extraction should return the model root containing model.sdf."""
    zip_path = tmp_path / "asset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Barrel/model.sdf", "<sdf version='1.5'></sdf>")
        archive.writestr("Barrel/Barrel/Barrel.dae", "<COLLADA></COLLADA>")

    asset_root = extract_manchester_asset(zip_path, tmp_path / "raw")

    assert asset_root == (tmp_path / "raw" / "Barrel").resolve()
    assert find_manchester_sdf_path(asset_root) == asset_root / "model.sdf"


def test_extract_manchester_asset_rejects_zip_slip(tmp_path: Path) -> None:
    """Extraction should reject archive members outside the target root."""
    zip_path = tmp_path / "asset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "bad")

    with pytest.raises(ValueError, match="Unsafe zip member"):
        extract_manchester_asset(zip_path, tmp_path / "raw")


def test_convert_manchester_sdf_to_usd_invokes_blender(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The conversion helper should call Blender with the SDF converter script."""
    commands: list[list[str]] = []
    sdf_path = tmp_path / "Barrel" / "model.sdf"
    sdf_path.parent.mkdir()
    sdf_path.write_text("<sdf version='1.5'></sdf>", encoding="utf-8")
    output_path = tmp_path / "usd" / "barrel.usda"

    def _fake_resolve_blender_executable(blender_executable: str | None = None) -> str:
        """Return a fake Blender executable path."""
        assert blender_executable is None
        return "/usr/bin/blender"

    def _fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Record the Blender command and create the expected USD output."""
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("#usda 1.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "sim.manchester_dataset.resolve_blender_executable",
        _fake_resolve_blender_executable,
    )
    monkeypatch.setattr("sim.manchester_dataset.subprocess.run", _fake_run)

    result = convert_manchester_sdf_to_usd(
        sdf_path,
        output_path,
        asset_root=sdf_path.parent,
    )

    assert result == output_path.resolve()
    assert commands
    assert commands[0][:3] == ["/usr/bin/blender", "--background", "--python"]
    assert "--model-root" in commands[0]


def test_write_isaacsim_config_uses_relative_usd_path(tmp_path: Path) -> None:
    """Generated Isaac Sim config should reference nearby USD paths relatively."""
    usd_path = tmp_path / "data" / "drum_store.usda"
    usd_path.parent.mkdir()
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "isaacsim" / "manchester_drum_store.json"

    result = write_isaacsim_config(config_path, usd_path=usd_path)

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["usd_path"] == "../../data/drum_store.usda"
    assert payload["stage_material_rules"][0]["path_prefix"] == "/World/Environment"


def test_default_isaacsim_configs_use_manchester_drum_store() -> None:
    """Default Isaac Sim configs should point at the converted Manchester USD."""
    root = Path(__file__).resolve().parents[1]
    expected_usd = "../../data/manchester_nuclear_assets/usd/drum_store.usda"

    for config_name in ("default_scene.json", "real_scene.json", "manchester_drum_store.json"):
        config_path = root / "configs" / "isaacsim" / config_name
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        assert payload["usd_path"] == expected_usd
        assert payload["author_obstacle_prims"] is False
        assert payload["stage_material_rules"] == [
            {"path_prefix": "/World/Environment", "material": "concrete"}
        ]


def test_usd_backed_geant4_configs_use_manchester_drum_store() -> None:
    """USD-backed Geant4 configs should export the converted Manchester mesh."""
    root = Path(__file__).resolve().parents[1]
    expected_usd = "../../data/manchester_nuclear_assets/usd/drum_store.usda"

    for config_name in ("real_scene.json", "external_scene.json", "external_gui_scene.json"):
        config_path = root / "configs" / "geant4" / config_name
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        assert payload["usd_path"] == expected_usd
        assert payload["use_mock_stage"] is False
        assert payload["author_obstacle_prims"] is False
        assert "sidecar_python" not in payload
        assert payload["sidecar_python_env"] == "ISAACSIM_PYTHON"
        assert payload["stage_material_rules"] == [
            {"path_prefix": "/World/Environment", "material": "concrete"}
        ]

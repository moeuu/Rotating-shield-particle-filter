"""CLI regression tests for the real-time demo entry point."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from measurement.source_boundary import surface_emission_policy_sha256
from sim.runtime import load_runtime_config


def _load_main_module() -> object:
    """Load the repository main.py module for CLI tests."""
    module_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("main", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load main.py module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_passes_environment_mode_to_runtime(monkeypatch) -> None:
    """The CLI should forward the requested environment mode into run_live_pf."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--environment-mode",
            "random",
            "--no-obstacles",
            "--passage-width-m",
            "1.4",
            "--robot-radius-m",
            "0.42",
            "--robot-speed",
            "0.8",
            "--rotation-overhead-s",
            "1.25",
            "--measurement-time-s",
            "12",
            "--num-particles",
            "600",
            "--rotations-per-pose",
            "4",
            "--planning-eig-samples",
            "12",
            "--notify-spectrum",
            "--notify-spectrum-every",
            "2",
            "--notify-spectrum-max-bins",
            "256",
        ],
    )

    module.main()

    assert captured["environment_mode"] == "random"
    assert captured["obstacle_layout_path"] is None
    assert captured["passage_width_m"] == 1.4
    assert captured["robot_radius_m"] == 0.42
    assert captured["nominal_motion_speed_m_s"] == 0.8
    assert captured["rotation_overhead_s"] == 1.25
    assert captured["measurement_time_s"] == 12.0
    assert captured["num_particles"] == 600
    assert captured["pf_config_overrides"]["orientation_k"] == 4
    assert captured["pf_config_overrides"]["min_rotations_per_pose"] == 4
    assert captured["pf_config_overrides"]["planning_eig_samples"] == 12
    assert captured["notify_spectrum"] is True
    assert captured["notify_spectrum_every"] == 2
    assert captured["notify_spectrum_max_bins"] == 256
    assert captured["notification_config"].enabled is True
    assert captured["source_generation_mode"] == "surface_random"


def test_main_no_notify_overrides_spectrum_notifications(monkeypatch) -> None:
    """The explicit no-notify flag should disable Railway spectrum delivery."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--notify-spectrum",
            "--no-notify",
        ],
    )

    module.main()

    assert captured["notify_spectrum"] is True
    assert captured["notification_config"].enabled is False


def test_main_default_max_poses_uses_runtime_config(monkeypatch) -> None:
    """The standard CLI should use runtime pose cap and random obstacles by default."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    module.main()

    assert captured["max_poses"] is None
    assert captured["environment_mode"] == "random"
    assert str(captured["obstacle_layout_path"]).endswith(
        "obstacle_layouts/Ex5_obstacles.json"
    )
    assert captured["source_generation_mode"] == "surface_random"
    assert "max_sources" not in captured["pf_config_overrides"]
    measurement_log_output = Path(str(captured["measurement_log_output"]))
    assert measurement_log_output.parent == Path("results/measurement_logs")
    assert measurement_log_output.name.startswith("full_simulation_")


def test_main_explicit_measurement_log_output_is_forwarded(monkeypatch) -> None:
    """An explicit MeasurementLog target should be forwarded unchanged."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    target = "logs/pure_pf/explicit_run"
    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--measurement-log-output", target],
    )

    module.main()

    assert captured["measurement_log_output"] == target


def test_measurement_log_output_preserves_configured_target() -> None:
    """Configured RAL-style targets should remain owned by the runtime config."""
    module = _load_main_module()

    resolved = module._resolve_measurement_log_output(
        None,
        {
            "measurement_log_output_dir": (
                "results/ral_ablation/measurement_logs/proposed"
            )
        },
        output_tag="ignored",
        repository_root=Path.cwd(),
    )

    assert resolved is None


def test_measurement_log_output_is_unique_and_sanitizes_output_tag(
    tmp_path: Path,
) -> None:
    """Automatic targets should be unique and contain one safe tag component."""
    module = _load_main_module()

    first = module._resolve_measurement_log_output(
        None,
        {},
        output_tag="../../RAL run/α",
        repository_root=tmp_path,
    )
    second = module._resolve_measurement_log_output(
        None,
        {},
        output_tag="../../RAL run/α",
        repository_root=tmp_path,
    )

    assert first is not None
    assert second is not None
    assert first != second
    first_path = Path(first)
    assert first_path.parent == Path("results/measurement_logs")
    assert first_path.name.startswith("RAL_run_")
    assert ".." not in first_path.parts


def test_main_explicit_max_sources_overrides_runtime_config(monkeypatch) -> None:
    """An explicit source-count cap should be forwarded as a PF override."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--max-sources", "3"],
    )

    module.main()

    assert captured["pf_config_overrides"]["max_sources"] == 3


def test_main_explicit_source_config_keeps_fixed_sources_in_random_environment(
    monkeypatch,
) -> None:
    """Explicit source configs should override random-environment source generation."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--environment-mode",
            "random",
            "--source-config",
            "source_layouts/demo_sources.json",
        ],
    )

    module.main()

    source_path = Path(__file__).resolve().parents[1] / "source_layouts/demo_sources.json"
    assert captured["source_generation_mode"] == "provided_file"
    assert captured["sources"] is not None
    assert captured["source_config_provenance"] == {
        "provided_file_path": "source_layouts/demo_sources.json",
        "provided_file_path_kind": "repository_relative",
        "provided_file_bytes_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "provided_file_declared_metadata": {},
    }


@pytest.mark.parametrize("legacy_field", ["intensity", "strength_cps_1m"])
def test_source_config_rejects_removed_intensity_aliases(
    tmp_path: Path,
    legacy_field: str,
) -> None:
    """Explicit truth files must use the single detector-cps intensity field."""
    module = _load_main_module()
    source_path = tmp_path / "legacy_source.json"
    source_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "isotope": "Cs-137",
                        "position": [1.0, 2.0, 3.0],
                        legacy_field: 1000.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="removed intensity field"):
        module.load_sources_from_json(
            source_path,
            repository_root=Path(__file__).resolve().parents[1],
        )


def test_source_config_provenance_retains_external_path_hash_and_metadata(
    tmp_path: Path,
) -> None:
    """The parsed source truth must retain its exact external-file identity."""
    module = _load_main_module()
    source_path = tmp_path / "provided_sources.json"
    declared_metadata = {
        "sampling_measure": "predeclared_area_uniform",
        "source_seed": 20260728,
    }
    raw_bytes = json.dumps(
        {
            "metadata": declared_metadata,
            "sources": [
                {
                    "isotope": "Co-60",
                    "position": [4.0, 5.0, 6.0],
                    "intensity_cps_1m": 250000.0,
                }
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    source_path.write_bytes(raw_bytes)

    loaded = module.load_sources_from_json(
        source_path,
        repository_root=Path(__file__).resolve().parents[1],
    )

    assert len(loaded.sources) == 1
    assert loaded.provenance == {
        "provided_file_path": source_path.resolve().as_posix(),
        "provided_file_path_kind": "resolved_absolute",
        "provided_file_bytes_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "provided_file_declared_metadata": declared_metadata,
    }


@pytest.mark.parametrize(
    "raw_json",
    (
        '{"sources": [], "sources": []}',
        '{"metadata": {"seed": 1, "seed": 2}, "sources": []}',
        (
            '{"sources": [{"isotope": "Cs-137", "isotope": "Co-60", '
            '"position": [1, 2, 3], "intensity_cps_1m": 300000}]}'
        ),
    ),
)
def test_source_config_rejects_duplicate_json_object_keys(
    tmp_path: Path,
    raw_json: str,
) -> None:
    """No truth or provenance field may be silently replaced by last-wins JSON."""
    module = _load_main_module()
    source_path = tmp_path / "duplicate_keys.json"
    source_path.write_text(raw_json, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        module.load_sources_from_json(
            source_path,
            repository_root=Path(__file__).resolve().parents[1],
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {"sources": [], "unexpected": "silently ignored"},
            "unsupported top-level fields",
        ),
        ({"metadata": {}}, "explicit 'sources' list"),
    ),
)
def test_source_config_rejects_ambiguous_top_level_schema(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    """Object-form truth files must use only the declared explicit schema."""
    module = _load_main_module()
    source_path = tmp_path / "ambiguous_top_level.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.load_sources_from_json(
            source_path,
            repository_root=Path(__file__).resolve().parents[1],
        )


def test_source_config_preserves_authoritative_surface_coordinates(
    tmp_path: Path,
) -> None:
    """A complete chart/UV truth entry must not be reduced back to XYZ only."""
    module = _load_main_module()
    source_path = tmp_path / "surface_sources.json"
    policy_hash = surface_emission_policy_sha256()
    source_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "isotope": "Cs-137",
                        "position": [0.0, 0.5, 0.5],
                        "transport_position": [1.0e-6, 0.5, 0.5],
                        "intensity_cps_1m": 300_000.0,
                        "surface_chart_id": 7,
                        "surface_uv": [0.25, 0.75],
                        "surface_normal": [1.0, 0.0, 0.0],
                        "surface_emission_policy_sha256": policy_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = module.load_sources_from_json(
        source_path,
        repository_root=Path(__file__).resolve().parents[1],
    )

    source = loaded.sources[0]
    assert source.surface_chart_id == 7
    assert source.surface_uv == (0.25, 0.75)
    assert source.surface_normal == (1.0, 0.0, 0.0)
    assert source.transport_position == (1.0e-6, 0.5, 0.5)
    assert source.surface_emission_policy_sha256 == policy_hash


def test_main_rejects_missing_explicit_source_config(monkeypatch, tmp_path: Path) -> None:
    """A requested truth layout must never fall back to unrelated demo sources."""
    module = _load_main_module()
    missing_path = tmp_path / "missing_sources.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--source-config", missing_path.as_posix()],
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2


def test_main_random_source_token_uses_surface_source_generation(monkeypatch) -> None:
    """The random source-config token should request surface source generation."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--source-config",
            "random",
            "--source-seed",
            "11",
            "--random-source-count",
            "5",
            "--random-source-intensity-cps-1m",
            "45000",
        ],
    )

    module.main()

    assert captured["source_generation_mode"] == "surface_random"
    assert captured["random_source_seed"] == 11
    assert captured["random_source_count"] == 5
    assert captured["random_source_intensity_cps_1m"] == 45000.0


def test_main_allows_min_rotations_override(monkeypatch) -> None:
    """The CLI should allow early stopping below the orientation cap."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--rotations-per-pose",
            "4",
            "--min-rotations-per-pose",
            "1",
        ],
    )

    module.main()

    assert captured["pf_config_overrides"]["orientation_k"] == 4
    assert captured["pf_config_overrides"]["min_rotations_per_pose"] == 1


@pytest.mark.parametrize(
    "arguments",
    (
        ("--rotations-per-pose", "0"),
        ("--min-rotations-per-pose", "-1"),
        ("--planning-eig-samples", "0"),
        ("--max-temper-steps", "0"),
        ("--min-delta-beta", "0"),
        ("--target-ess-ratio", "1"),
        ("--max-sources", "0"),
    ),
)
def test_main_rejects_invalid_pf_overrides(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, str],
) -> None:
    """Invalid CLI values must not be clamped into a different PF model."""
    module = _load_main_module()
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])

    with pytest.raises(SystemExit):
        module.main()


@pytest.mark.parametrize(
    ("mode", "backend", "config_suffix"),
    [
        ("python-gui", "isaacsim", "configs/isaacsim/demo_room_gui.json"),
        (
            "geant4-isaacsim-gui",
            "geant4",
            "configs/geant4/variance_reduction_external_gui_32threads.json",
        ),
        ("python-cui", "analytic", "configs/python/high_fidelity_no_isaac.json"),
        (
            "geant4-cui",
            "geant4",
            "configs/geant4/variance_reduction_external_no_isaac_32threads.json",
        ),
    ],
)
def test_main_modes_select_expected_runtime(
    monkeypatch,
    mode: str,
    backend: str,
    config_suffix: str | None,
) -> None:
    """High-level modes should select the intended simulator runtime."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", mode])

    module.main()

    assert captured["sim_backend"] == backend
    if config_suffix is None:
        assert captured["sim_config_path"] is None
    else:
        assert str(captured["sim_config_path"]).endswith(config_suffix)
    assert captured["live"] is False


def test_main_default_selects_standard_geant4_full_simulation(monkeypatch) -> None:
    """The CLI default should be the standard no-GUI Geant4 full simulation."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    module.main()

    assert captured["sim_backend"] == "geant4"
    assert str(captured["sim_config_path"]).endswith(
        "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    assert captured["live"] is False
    assert captured["variable_cardinality"] is True
    runtime_config = load_runtime_config(Path(str(captured["sim_config_path"])))
    assert runtime_config["pure_pf_schema_version"] == 1
    assert runtime_config["variable_cardinality"] is True
    assert runtime_config["pf_max_sources"] == 5
    assert runtime_config["structural_cardinality_prior_mean"] == pytest.approx(2.0)
    assert runtime_config["estimator_profile"] == "pf_strict"
    assert float(runtime_config["pf_strength_prior_min_cps_1m"]) >= 0.0
    assert (
        float(runtime_config["pf_strength_prior_max_cps_1m"])
        > float(runtime_config["pf_strength_prior_min_cps_1m"])
    )
    assert float(runtime_config["structural_rj_move_probability"]) > 0.0
    assert float(runtime_config["structural_rj_birth_probability"]) > 0.0
    assert float(runtime_config["structural_rj_death_probability"]) > 0.0


def test_main_backend_override_without_mode_keeps_matching_default_config(
    monkeypatch,
) -> None:
    """Explicit backend overrides should not inherit the Geant4 config blindly."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(sys, "argv", ["main.py", "--sim-backend", "analytic"])

    module.main()

    assert captured["sim_backend"] == "analytic"
    assert str(captured["sim_config_path"]).endswith(
        "configs/python/high_fidelity_no_isaac.json"
    )
    assert captured["live"] is False


def test_main_gui_alias_selects_geant4_isaacsim(monkeypatch) -> None:
    """The GUI alias should select the Geant4 plus Isaac Sim mode."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(sys, "argv", ["main.py", "--gui"])

    module.main()

    assert captured["sim_backend"] == "geant4"
    assert str(captured["sim_config_path"]).endswith(
        "configs/geant4/variance_reduction_external_gui_32threads.json"
    )
    assert captured["live"] is False


@pytest.mark.parametrize(
    "alias",
    ["--cui", "--full-simulation", "--standard-geant4-full"],
)
def test_main_standard_full_aliases_select_geant4_cui(
    monkeypatch,
    alias: str,
) -> None:
    """Full-simulation aliases should select the standard Geant4 CUI config."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(sys, "argv", ["main.py", alias])

    module.main()

    assert captured["sim_backend"] == "geant4"
    assert str(captured["sim_config_path"]).endswith(
        "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    assert captured["live"] is False
    assert captured["variable_cardinality"] is True


@pytest.mark.parametrize(
    ("mode_flag", "backend"),
    [
        ("--full-simulation", "analytic"),
        ("--cui", "isaacsim"),
        ("--python-cui", "geant4"),
    ],
)
def test_main_rejects_mode_backend_contradictions(
    monkeypatch,
    mode_flag: str,
    backend: str,
) -> None:
    """A named runtime mode must not be relabeled with another transport."""
    module = _load_main_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", mode_flag, "--sim-backend", backend],
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2


def test_main_rejects_config_backend_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A Geant4 mode must not ingest a config declaring analytic transport."""
    module = _load_main_module()
    config_path = tmp_path / "wrong_backend.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "analytic",
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--full-simulation",
            "--sim-config",
            config_path.as_posix(),
        ],
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2


def test_main_can_explicitly_select_fixed_cardinality(monkeypatch) -> None:
    """The fixed-cardinality CLI override must win over the exact RJ config."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--full-simulation", "--fixed-cardinality"],
    )

    module.main()

    assert captured["variable_cardinality"] is False
    runtime_config = load_runtime_config(Path(str(captured["sim_config_path"])))
    assert runtime_config["variable_cardinality"] is True


def test_main_matplotlib_live_can_be_requested(monkeypatch) -> None:
    """The Matplotlib live plot should be opt-in for simulator modes."""
    module = _load_main_module()
    captured: dict[str, object] = {}

    def _fake_run_live_pf(**kwargs: object) -> None:
        """Capture CLI arguments without running the full simulation."""
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_live_pf", _fake_run_live_pf)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--mode", "python-cui", "--matplotlib-live"],
    )

    module.main()

    assert captured["live"] is True


def test_main_rejects_conflicting_gui_and_headless(monkeypatch) -> None:
    """The CLI should reject contradictory simulator GUI and headless requests."""
    module = _load_main_module()
    monkeypatch.setattr(sys, "argv", ["main.py", "--gui", "--headless"])

    with pytest.raises(SystemExit):
        module.main()

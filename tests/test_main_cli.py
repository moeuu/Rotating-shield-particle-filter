"""CLI regression tests for the real-time demo entry point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
            "--adaptive-dwell",
            "--adaptive-dwell-chunk-s",
            "1.5",
            "--adaptive-min-dwell-s",
            "3",
            "--adaptive-ready-min-counts",
            "250",
            "--adaptive-ready-min-isotopes",
            "2",
            "--adaptive-ready-min-snr",
            "1.5",
            "--no-adaptive-strength-prior",
            "--adaptive-strength-prior-steps",
            "5",
            "--adaptive-strength-prior-min-counts",
            "4.5",
            "--adaptive-strength-prior-log-sigma",
            "0.25",
            "--pose-min-observation-counts",
            "6.5",
            "--pose-min-observation-penalty-scale",
            "1.7",
            "--pose-min-observation-aggregate",
            "mean",
            "--num-particles",
            "600",
            "--rotations-per-pose",
            "4",
            "--init-grid-spacing-m",
            "0",
            "--planning-eig-samples",
            "12",
            "--planning-rollout-particles",
            "48",
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
    assert captured["adaptive_dwell"] is True
    assert captured["adaptive_dwell_chunk_s"] == 1.5
    assert captured["adaptive_min_dwell_s"] == 3.0
    assert captured["adaptive_ready_min_counts"] == 250.0
    assert captured["adaptive_ready_min_isotopes"] == 2
    assert captured["adaptive_ready_min_snr"] == 1.5
    assert captured["adaptive_strength_prior"] is False
    assert captured["adaptive_strength_prior_steps"] == 5
    assert captured["adaptive_strength_prior_min_counts"] == 4.5
    assert captured["adaptive_strength_prior_log_sigma"] == 0.25
    assert captured["pose_min_observation_counts"] == 6.5
    assert captured["pose_min_observation_penalty_scale"] == 1.7
    assert captured["pose_min_observation_aggregate"] == "mean"
    assert captured["num_particles"] == 600
    assert captured["pf_config_overrides"]["orientation_k"] == 4
    assert captured["pf_config_overrides"]["min_rotations_per_pose"] == 4
    assert captured["pf_config_overrides"]["init_grid_spacing_m"] is None
    assert captured["pf_config_overrides"]["planning_eig_samples"] == 12
    assert captured["pf_config_overrides"]["planning_rollout_particles"] == 48
    assert captured["notify_spectrum"] is True
    assert captured["notify_spectrum_every"] == 2
    assert captured["notify_spectrum_max_bins"] == 256
    assert captured["notification_config"].enabled is True


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
    ("mode", "backend", "config_suffix"),
    [
        ("python-gui", "isaacsim", "configs/isaacsim/demo_room_gui.json"),
        (
            "geant4-isaacsim-gui",
            "geant4",
            "configs/geant4/external_gui_scene.json",
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
        "configs/geant4/external_gui_scene.json"
    )
    assert captured["live"] is False


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

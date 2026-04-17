"""CLI regression tests for the real-time demo entry point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_main_passes_environment_mode_to_runtime(monkeypatch) -> None:
    """The CLI should forward the requested environment mode into run_live_pf."""
    module_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("main", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load main.py module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
            "--robot-speed",
            "0.8",
            "--rotation-overhead-s",
            "1.25",
        ],
    )

    module.main()

    assert captured["environment_mode"] == "random"
    assert captured["obstacle_layout_path"] is None
    assert captured["nominal_motion_speed_m_s"] == 0.8
    assert captured["rotation_overhead_s"] == 1.25

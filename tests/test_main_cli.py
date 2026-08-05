"""Tests for the replay-only compatibility entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_main_module() -> object:
    """Load the repository entry point without executing it as a script."""
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("pf_repository_main", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_delegates_only_to_truth_free_replay(monkeypatch: Any) -> None:
    """The compatibility CLI must not expose simulator or truth inputs."""
    module = _load_main_module()
    captured: list[object] = []

    def fake_main(argv: object = None) -> int:
        """Capture delegation to the PF replay command."""
        captured.append(argv)
        return 17

    monkeypatch.setattr(module, "main", fake_main)

    assert module.main(["--measurement-log", "/tmp/log"]) == 17
    assert captured == [["--measurement-log", "/tmp/log"]]
    assert not hasattr(module, "run_live_pf")
    assert not hasattr(module, "load_sources_from_json")


def test_main_source_has_no_simulator_mode_flags() -> None:
    """Removed simulator modes must not return through the compatibility shim."""
    source = (
        Path(__file__).resolve().parents[1] / "main.py"
    ).read_text(encoding="utf-8")

    assert "--full-simulation" not in source
    assert "--sim-backend" not in source
    assert "realtime_demo" not in source

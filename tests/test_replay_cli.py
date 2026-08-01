"""CLI ownership tests for the PF-only repository."""

from __future__ import annotations

from pf.replay import main


def test_replay_cli_help_exposes_only_artifact_inputs(capsys: object) -> None:
    """The PF CLI must consume a log and never expose a simulator backend."""
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "--measurement-log" in captured.out
    assert "--config" in captured.out
    assert "--sim-backend" not in captured.out
    assert "--full-simulation" not in captured.out

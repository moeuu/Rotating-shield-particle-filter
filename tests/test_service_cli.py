"""Tests for the optional independent-service bootstrap."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pf import service_cli


def test_service_cli_delegates_arguments_to_installed_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lightweight entry point must preserve the service adapter CLI."""
    calls: list[object] = []

    def service_main(arguments: object) -> int:
        """Record delegated command arguments."""
        calls.append(arguments)
        return 7

    def installed_service(name: str) -> object:
        """Return a stand-in for the installed service implementation."""
        assert name == "pf.service"
        return SimpleNamespace(main=service_main)

    monkeypatch.setattr(service_cli, "import_module", installed_service)

    assert service_cli.main(("capabilities", "--response", "/tmp/out.json")) == 7
    assert calls == [("capabilities", "--response", "/tmp/out.json")]


def test_service_cli_reports_actionable_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A core-only install must explain how to enable the optional service."""

    def missing_contract(name: str) -> object:
        """Simulate importing the adapter without its optional wire package."""
        raise ModuleNotFoundError(
            "No module named 'radiation_estimator_service_contracts'",
            name="radiation_estimator_service_contracts",
        )

    monkeypatch.setattr(service_cli, "import_module", missing_contract)

    assert service_cli.main(()) == 69
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "rotating-shield-particle-filter[service]" in captured.err


def test_service_cli_does_not_hide_unrelated_import_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing core dependencies and adapter defects must retain their traceback."""

    def missing_runtime(name: str) -> object:
        """Simulate one unrelated import failure from the real adapter."""
        raise ModuleNotFoundError(
            "No module named 'runtime'",
            name="runtime",
        )

    monkeypatch.setattr(service_cli, "import_module", missing_runtime)

    with pytest.raises(ModuleNotFoundError, match="runtime"):
        service_cli.main(())

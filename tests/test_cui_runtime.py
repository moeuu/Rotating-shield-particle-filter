"""Tests for the PF-owned shared CUI server lifecycle."""

from __future__ import annotations

import errno
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pf import cui_runtime


@pytest.mark.parametrize("value", ("false", 0, 1, None))
def test_cui_enable_flag_rejects_truthy_coercion(value: object) -> None:
    """Only literal JSON booleans may control the live CUI server."""
    with pytest.raises(TypeError, match="cui_split_view must be a boolean"):
        cui_runtime.resolve_cui_split_view_enabled(
            {"cui_split_view": value},
        )


@pytest.mark.parametrize(
    ("layout", "expected_root", "expected_index"),
    (
        ("default", "static", Path("index.html")),
        ("nested", "static", Path("run-001") / "index.html"),
        ("outside", "output", Path("index.html")),
    ),
)
def test_cui_wrapper_selects_a_root_containing_the_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    expected_root: str,
    expected_index: Path,
) -> None:
    """Default, nested, and external output directories must yield valid routes."""
    static_root = tmp_path / "static"
    if layout == "default":
        output_dir = static_root
    elif layout == "nested":
        output_dir = static_root / "run-001"
    else:
        output_dir = tmp_path / "custom-output"
    captured: list[dict[str, Any]] = []

    def fake_start_cui_server(
        root: Path,
        *,
        index_path: Path,
        config: object,
    ) -> SimpleNamespace:
        """Capture the shared server request and return one owning handle."""
        captured.append(
            {
                "root": Path(root),
                "index_path": Path(index_path),
                "config": config,
            }
        )
        return SimpleNamespace(
            url="http://127.0.0.1:8877/index.html",
            close=lambda: None,
        )

    monkeypatch.setattr(cui_runtime, "start_cui_server", fake_start_cui_server)

    handle = cui_runtime.start_cui_view_server(
        output_dir,
        host="127.0.0.1",
        port=8877,
        public_host="127.0.0.1",
        static_root=static_root,
    )

    assert handle.url == "http://127.0.0.1:8877/index.html"
    assert len(captured) == 1
    assert captured[0]["root"] == (
        static_root.resolve() if expected_root == "static" else output_dir.resolve()
    )
    assert captured[0]["index_path"] == expected_index
    assert output_dir.is_dir()
    assert not hasattr(cui_runtime, "ensure_cui_view_server")
    assert not hasattr(cui_runtime, "_CUI_SERVER_HANDLES")


@pytest.mark.parametrize("port", (0, -1, 65536, True, "8877"))
def test_cui_wrapper_preserves_nonzero_integer_port_contract(
    tmp_path: Path,
    port: object,
) -> None:
    """The PF wrapper must retain its strict nonzero TCP port setting."""
    error = TypeError if isinstance(port, (bool, str)) else ValueError
    with pytest.raises(error):
        cui_runtime.start_cui_view_server(
            tmp_path / "output",
            host="127.0.0.1",
            port=port,  # type: ignore[arg-type]
            public_host="127.0.0.1",
            static_root=tmp_path / "static",
        )


def test_cui_wrapper_fails_when_the_configured_port_is_occupied(tmp_path: Path) -> None:
    """The production wrapper must never publish on an unconfigured port."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "index.html").write_text("PF dashboard", encoding="utf-8")
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    requested_port = int(occupied.getsockname()[1])
    try:
        with pytest.raises(OSError) as error:
            cui_runtime.start_cui_view_server(
                output_dir,
                host="127.0.0.1",
                port=requested_port,
                public_host="127.0.0.1",
                static_root=output_dir,
            )
        assert error.value.errno == errno.EADDRINUSE
    finally:
        occupied.close()


def test_cui_wrapper_closes_a_handle_that_has_no_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed server result must be closed before startup fails."""
    closed: list[bool] = []
    handle = SimpleNamespace(url=None, close=lambda: closed.append(True))
    monkeypatch.setattr(
        cui_runtime,
        "start_cui_server",
        lambda *args, **kwargs: handle,
    )

    with pytest.raises(RuntimeError, match="did not provide"):
        cui_runtime.start_cui_view_server(
            tmp_path / "output",
            host="127.0.0.1",
            port=8877,
            public_host="127.0.0.1",
            static_root=tmp_path / "static",
        )

    assert closed == [True]

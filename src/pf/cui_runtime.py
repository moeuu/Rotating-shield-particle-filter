"""Compatibility wrappers for the shared runtime CUI server API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from runtime.cui import (
    CUIDashboardConfig,
    CUIServerHandle,
    resolve_cui_public_host,
    start_cui_server,
)

from pf.runtime_defaults import (
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUI_VIEW_DIR = ROOT / DEFAULT_CUI_SPLIT_VIEW_DIR

_CUIServerKey = tuple[Path, Path, str, int, str]
_CUI_SERVER_HANDLES: dict[_CUIServerKey, CUIServerHandle] = {}


def resolve_cui_split_view_enabled(
    runtime_config: Mapping[str, object],
    *,
    save_outputs: bool,
) -> bool:
    """Return whether the URL-served CUI progress view should run."""
    if "cui_split_view" in runtime_config:
        return bool(runtime_config["cui_split_view"])
    return bool(save_outputs)


def _nonzero_cui_port(value: object) -> int:
    """Return a valid legacy PF CUI port without accepting coercion or zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("CUI split visualization port must be an integer.")
    if value < 1 or value > 65535:
        raise ValueError(
            "CUI split visualization port must be between 1 and 65535."
        )
    return int(value)


def _server_root_and_index(
    output_dir: Path,
    static_root: Path,
) -> tuple[Path, Path]:
    """Return a serving root and relative index that contain the PF output."""
    output_path = Path(output_dir).expanduser().resolve()
    static_path = Path(static_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        relative_output = output_path.relative_to(static_path)
    except ValueError:
        return output_path, Path("index.html")
    static_path.mkdir(parents=True, exist_ok=True)
    if relative_output == Path("."):
        return static_path, Path("index.html")
    return static_path, relative_output / "index.html"


def _close_cui_server_handles() -> None:
    """Close every shared server retained by this compatibility module."""
    unique_handles = {id(handle): handle for handle in _CUI_SERVER_HANDLES.values()}
    _CUI_SERVER_HANDLES.clear()
    for handle in unique_handles.values():
        handle.close()


def ensure_cui_view_server(
    output_dir: Path,
    *,
    host: str = DEFAULT_CUI_SPLIT_VIEW_HOST,
    port: int = DEFAULT_CUI_SPLIT_VIEW_PORT,
    public_host: str | None = None,
    static_root: Path = DEFAULT_CUI_VIEW_DIR,
) -> str:
    """Start or reuse the shared server while preserving the PF string API."""
    root, index_path = _server_root_and_index(output_dir, static_root)
    resolved_public_host = resolve_cui_public_host(host, public_host)
    config = CUIDashboardConfig(
        serve=True,
        host=host,
        port=_nonzero_cui_port(port),
        public_host=resolved_public_host,
    )
    resolved_index = (root / index_path).resolve()
    server_key = (
        root,
        resolved_index,
        config.host,
        config.port,
        str(config.public_host),
    )
    existing = _CUI_SERVER_HANDLES.get(server_key)
    if existing is not None and not getattr(existing, "_closed", False):
        if existing.url is None:
            raise RuntimeError("Retained CUI server has no dashboard URL.")
        return existing.url
    _CUI_SERVER_HANDLES.pop(server_key, None)
    handle = start_cui_server(
        root,
        index_path=index_path,
        config=config,
    )
    _CUI_SERVER_HANDLES[server_key] = handle
    if handle.url is None:
        raise RuntimeError("Shared runtime did not provide a CUI dashboard URL.")
    return handle.url


__all__ = ["ensure_cui_view_server", "resolve_cui_split_view_enabled"]

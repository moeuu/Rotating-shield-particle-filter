"""PF-owned lifecycle for the shared runtime CUI server API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from runtime.cui import (
    CUIDashboardConfig,
    CUIServerHandle,
    resolve_cui_public_host,
    start_cui_server,
)

from runtime.defaults import (
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUI_VIEW_DIR = ROOT / DEFAULT_CUI_SPLIT_VIEW_DIR


def resolve_cui_split_view_enabled(
    runtime_config: Mapping[str, object],
) -> bool:
    """Return whether the CUI progress renderer should run."""
    value = runtime_config["cui_split_view"]
    if type(value) is not bool:
        raise TypeError("cui_split_view must be a boolean.")
    return value


def _nonzero_cui_port(value: object) -> int:
    """Return a valid PF CUI port without accepting coercion or zero."""
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


def start_cui_view_server(
    output_dir: Path,
    *,
    host: str = DEFAULT_CUI_SPLIT_VIEW_HOST,
    port: int = DEFAULT_CUI_SPLIT_VIEW_PORT,
    public_host: str | None = None,
    static_root: Path = DEFAULT_CUI_VIEW_DIR,
) -> CUIServerHandle:
    """Bind one fixed-port CUI server and return its sole owning handle."""
    root, index_path = _server_root_and_index(output_dir, static_root)
    resolved_public_host = resolve_cui_public_host(host, public_host)
    config = CUIDashboardConfig(
        serve=True,
        host=host,
        port=_nonzero_cui_port(port),
        public_host=resolved_public_host,
    )
    handle = start_cui_server(
        root,
        index_path=index_path,
        config=config,
    )
    if handle.url is None:
        error = RuntimeError("Shared runtime did not provide a CUI dashboard URL.")
        try:
            handle.close()
        except BaseException as close_error:
            error.add_note(
                "Secondary CUI server cleanup failure: "
                f"{type(close_error).__name__}: {close_error}"
            )
        raise error
    return handle


__all__ = ["resolve_cui_split_view_enabled", "start_cui_view_server"]

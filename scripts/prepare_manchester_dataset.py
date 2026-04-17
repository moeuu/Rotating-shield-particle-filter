"""Download and convert Manchester nuclear environment assets for Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.manchester_dataset import (
    DEFAULT_ASSET_ROOT,
    FIGSHARE_ARTICLE_URL,
    available_manchester_assets,
    asset_slug,
    convert_manchester_sdf_to_usd,
    download_manchester_asset,
    extract_manchester_asset,
    find_manchester_sdf_path,
    resolve_manchester_asset,
    write_isaacsim_config,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Prepare Manchester Gazebo nuclear assets for Isaac Sim."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known Figshare assets and exit.",
    )
    parser.add_argument(
        "--asset",
        default="Drum_Store",
        help="Asset key/name to prepare (default: Drum_Store).",
    )
    parser.add_argument(
        "--download-dir",
        default=(DEFAULT_ASSET_ROOT / "downloads").as_posix(),
        help="Directory used for downloaded ZIP files.",
    )
    parser.add_argument(
        "--extract-dir",
        default=(DEFAULT_ASSET_ROOT / "raw").as_posix(),
        help="Directory used for extracted Gazebo assets.",
    )
    parser.add_argument(
        "--usd-output",
        default=None,
        help="Output USD/USDA path (default: data/manchester_nuclear_assets/usd/<asset>.usda).",
    )
    parser.add_argument(
        "--config-output",
        default=None,
        help="Isaac Sim config path (default: configs/isaacsim/manchester_<asset>.json).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing ZIP in --download-dir instead of downloading.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Stop after downloading and verifying the ZIP.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Stop after extracting the ZIP and locating the SDF file.",
    )
    parser.add_argument(
        "--overwrite-extract",
        action="store_true",
        help="Replace any existing extracted asset directory.",
    )
    parser.add_argument(
        "--blender-executable",
        default=None,
        help="Blender executable path used for SDF to USD conversion.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Timeout for Blender conversion in seconds.",
    )
    return parser


def _format_size(size_bytes: int) -> str:
    """Format a byte count using binary units."""
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def _print_asset_list() -> None:
    """Print the known Manchester dataset files."""
    print(f"Manchester dataset: {FIGSHARE_ARTICLE_URL}")
    for asset in available_manchester_assets():
        print(
            f"{asset.key:28s} "
            f"{_format_size(asset.size_bytes):>10s} "
            f"{asset.download_url}"
        )


def _default_usd_output(asset_name: str) -> Path:
    """Return the default converted USD output path for an asset."""
    return DEFAULT_ASSET_ROOT / "usd" / f"{asset_slug(asset_name)}.usda"


def _default_config_output(asset_name: str) -> Path:
    """Return the default Isaac Sim config output path for an asset."""
    return ROOT / "configs" / "isaacsim" / f"manchester_{asset_slug(asset_name)}.json"


def _resolve_zip_path(asset_name: str, download_dir: Path, skip_download: bool) -> Path:
    """Return the local ZIP path, downloading it unless requested otherwise."""
    asset = resolve_manchester_asset(asset_name)
    if skip_download:
        zip_path = download_dir.expanduser().resolve() / asset.filename
        if not zip_path.exists():
            raise FileNotFoundError(f"Expected existing ZIP was not found: {zip_path}")
        return zip_path
    return download_manchester_asset(asset, download_dir)


def main() -> None:
    """Prepare the requested Manchester asset."""
    args = _build_parser().parse_args()
    if args.list:
        _print_asset_list()
        return
    asset = resolve_manchester_asset(args.asset)
    download_dir = Path(args.download_dir)
    extract_dir = Path(args.extract_dir)
    zip_path = _resolve_zip_path(asset.key, download_dir, bool(args.skip_download))
    print(f"ZIP ready: {zip_path}")
    if args.download_only:
        return
    asset_root = extract_manchester_asset(
        zip_path,
        extract_dir,
        overwrite=bool(args.overwrite_extract),
    )
    sdf_path = find_manchester_sdf_path(asset_root)
    print(f"SDF ready: {sdf_path}")
    if args.extract_only:
        return
    usd_output = (
        Path(args.usd_output)
        if args.usd_output
        else _default_usd_output(asset.key)
    )
    usd_path = convert_manchester_sdf_to_usd(
        sdf_path,
        usd_output,
        asset_root=asset_root,
        blender_executable=args.blender_executable,
        timeout_s=float(args.timeout_s),
    )
    print(f"USD ready: {usd_path}")
    config_output = (
        Path(args.config_output)
        if args.config_output
        else _default_config_output(asset.key)
    )
    config_path = write_isaacsim_config(config_output, usd_path=usd_path)
    print(f"Isaac Sim config ready: {config_path}")


if __name__ == "__main__":
    main()

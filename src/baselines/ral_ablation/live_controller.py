"""Run an RA-L controller adapter against an opaque runtime session socket."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from baselines.ral_ablation.control_policy import load_ral_control_policy_document
from pf.closed_loop import run_pf_closed_loop


RAL_MINIMUM_FINALIZABLE_STATIONS = 10


class RALStationStopRequest:
    """Poll one fail-closed sentinel at completed station boundaries."""

    def __init__(self, path: Path, *, minimum_stations: int) -> None:
        """Bind an initially absent regular-file request path."""
        self.path = Path(path).expanduser().resolve()
        if (
            isinstance(minimum_stations, bool)
            or not isinstance(minimum_stations, int)
            or minimum_stations < 1
        ):
            raise ValueError("minimum_stations must be a positive integer.")
        self.minimum_stations = int(minimum_stations)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(
                "RA-L station-stop request must be absent at controller start: "
                f"{self.path}"
            )

    def __call__(self, completed_stations: int) -> bool:
        """Return true once an empty regular sentinel is eligible."""
        if isinstance(completed_stations, bool) or not isinstance(
            completed_stations,
            int,
        ):
            raise TypeError("completed_stations must be an integer.")
        if completed_stations < 1:
            raise ValueError("completed_stations must be positive.")
        if not self.path.exists() and not self.path.is_symlink():
            return False
        if self.path.is_symlink() or not self.path.is_file():
            raise RuntimeError(
                "RA-L station-stop request must be a regular file."
            )
        if self.path.stat().st_size != 0:
            raise RuntimeError(
                "RA-L station-stop request must be an empty sentinel file."
            )
        return completed_stations >= self.minimum_stations


def main(argv: Sequence[str] | None = None) -> int:
    """Run one RA-L policy without exposing private scenario inputs to PF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-socket", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cui-truth-overlay-socket", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-policy", type=Path, required=True)
    parser.add_argument("--expected-control-policy-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--station-stop-request", type=Path, default=None)
    parser.add_argument("--profile", choices=("pf_strict",), default="pf_strict")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(None if argv is None else list(argv))
    policy_document = load_ral_control_policy_document(
        args.control_policy,
        expected_source_sha256=args.expected_control_policy_sha256,
    )
    stop_request = (
        None
        if args.station_stop_request is None
        else RALStationStopRequest(
            args.station_stop_request,
            minimum_stations=RAL_MINIMUM_FINALIZABLE_STATIONS,
        )
    )
    result = run_pf_closed_loop(
        args.session_socket,
        runtime_root=args.runtime_root,
        cui_truth_overlay_socket_path=args.cui_truth_overlay_socket,
        pf_config_path=args.config,
        output_dir=args.output_dir,
        profile=args.profile,
        seed=args.seed,
        control_policy=policy_document.policy(),
        station_boundary_stop_request=stop_request,
    )
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RAL_MINIMUM_FINALIZABLE_STATIONS",
    "RALStationStopRequest",
    "main",
]

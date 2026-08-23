"""Run an RA-L controller adapter against an opaque runtime session socket."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from baselines.ral_ablation.control_policy import load_ral_control_policy
from pf.closed_loop import run_pf_closed_loop


def main(argv: Sequence[str] | None = None) -> int:
    """Run one RA-L policy without exposing private scenario inputs to PF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-socket", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("pf_strict",), default="pf_strict")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(None if argv is None else list(argv))
    result = run_pf_closed_loop(
        args.session_socket,
        runtime_root=args.runtime_root,
        pf_config_path=args.config,
        output_dir=args.output_dir,
        profile=args.profile,
        seed=args.seed,
        control_policy=load_ral_control_policy(args.control_policy),
    )
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

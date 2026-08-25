"""CLI for generating RA-L ablation experiment configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.ral_ablation.config_factory import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PF_CONFIG,
    DEFAULT_PRIVATE_ROOT,
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_RUNTIME_ROOT,
    build_ablation_plan,
    write_ablation_plan,
)


def main() -> None:
    """Generate RA-L PF configs and shared-runtime command manifests."""
    parser = argparse.ArgumentParser(description="Generate RA-L ablation trials.")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
        help="Sibling shared-runtime repository root.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG,
        help="Canonical shared-runtime Geant4 config copied into each trial.",
    )
    parser.add_argument(
        "--pf-config",
        type=Path,
        default=DEFAULT_PF_CONFIG,
        help="Base pure-PF configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated PF configs, logs, runs, and manifests.",
    )
    parser.add_argument(
        "--private-root",
        type=Path,
        default=DEFAULT_PRIVATE_ROOT,
        help="Ignored sibling-runtime directory for truth-bearing scenarios.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Recorded environment seeds for repeating a live acquisition batch. "
            "When omitted, one fresh scene seed is generated."
        ),
    )
    parser.add_argument(
        "--pf-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Recorded independent PF seeds for exact replay. Required with "
            "--seeds and forbidden for a fresh batch."
        ),
    )
    parser.add_argument(
        "--transport-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Recorded independent transport seeds for exact replay. Required "
            "with --seeds and forbidden for a fresh batch."
        ),
    )
    parser.add_argument(
        "--batch-ids",
        nargs="+",
        default=None,
        help=(
            "Recorded opaque batch identifiers for exact replay. Required with "
            "--seeds; fresh batches generate opaque identifiers."
        ),
    )
    parser.add_argument(
        "--output-tag-suffix",
        default="",
        help="Optional safe suffix for isolated result and measurement-log paths.",
    )
    args = parser.parse_args()
    entries = build_ablation_plan(
        runtime_root=args.runtime_root,
        runtime_config_path=args.runtime_config,
        pf_config_path=args.pf_config,
        output_dir=args.output_dir,
        private_root=args.private_root,
        seeds=(None if args.seeds is None else tuple(int(seed) for seed in args.seeds)),
        pf_seeds=(
            None
            if args.pf_seeds is None
            else tuple(int(seed) for seed in args.pf_seeds)
        ),
        transport_seeds=(
            None
            if args.transport_seeds is None
            else tuple(int(seed) for seed in args.transport_seeds)
        ),
        batch_ids=(
            None
            if args.batch_ids is None
            else tuple(str(batch_id) for batch_id in args.batch_ids)
        ),
        output_tag_suffix=str(args.output_tag_suffix),
    )
    manifest_path, script_path = write_ablation_plan(
        entries, private_root=args.private_root
    )
    print(f"Wrote {len(entries)} ablation trials.")
    print(
        "Scene seeds: "
        + ", ".join(
            str(seed) for seed in sorted({entry.scene_seed for entry in entries})
        )
    )
    print(
        "Private scenario root: " + str(Path(args.private_root).expanduser().resolve())
    )
    print(f"Manifest: {manifest_path}")
    print(f"Run script: {script_path}")


if __name__ == "__main__":
    main()

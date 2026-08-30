"""Evaluate one completed PF run under the fixed cluster-accuracy policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.completed_run import evaluate_completed_pf_run  # noqa: E402
from runtime.artifacts import atomic_write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Return the fail-closed completed-run evaluation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--evaluation-input", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Join exact-run private truth and publish standardized accuracy metrics."""
    args = _parser().parse_args()
    evaluation = evaluate_completed_pf_run(
        result_path=args.result,
        posterior_path=args.posterior,
        evaluation_input_path=args.evaluation_input,
        truth_manifest_path=args.truth_manifest,
    )
    atomic_write_json(args.output, evaluation)


if __name__ == "__main__":
    main()

"""Summarize Eu-154 final count-model residuals from Geant4 run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _load_summary(path: Path) -> dict[str, Any]:
    """Load one JSON result summary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return payload


def _iter_summary_paths(paths: Sequence[str]) -> list[Path]:
    """Expand file and directory inputs into result-summary JSON paths."""
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            resolved.extend(sorted(path.glob("result_summary*.json")))
        else:
            resolved.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        abs_path = path.resolve()
        if abs_path in seen:
            continue
        seen.add(abs_path)
        unique.append(abs_path)
    return unique


def eu154_count_row(path: Path, summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return one Eu-154 count-residual row from a result summary."""
    diagnostics = summary.get("isotope_count_residual_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return None
    stats = diagnostics.get("Eu-154")
    if not isinstance(stats, dict):
        return None
    observed = float(stats.get("observed_total_counts", 0.0))
    predicted = float(stats.get("predicted_total_counts", 0.0))
    if observed > 0.0:
        rel_bias = (predicted - observed) / observed
        abs_rel_error = abs(rel_bias)
        underprediction_fraction = max(observed - predicted, 0.0) / observed
    else:
        rel_bias = 0.0
        abs_rel_error = 0.0
        underprediction_fraction = 0.0
    return {
        "path": path.as_posix(),
        "measurements_completed": int(summary.get("measurements_completed", 0)),
        "reported_source_count": int(stats.get("reported_source_count", 0)),
        "observed_total_counts": observed,
        "predicted_total_counts": predicted,
        "relative_bias": rel_bias,
        "absolute_relative_error": abs_rel_error,
        "underprediction_fraction": underprediction_fraction,
        "positive_residual_total_counts": float(
            stats.get("positive_residual_total_counts", 0.0)
        ),
        "negative_residual_total_counts": float(
            stats.get("negative_residual_total_counts", 0.0)
        ),
        "residual_chi2": float(stats.get("residual_chi2", 0.0)),
    }


def summarize_eu154_count_model(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Return Eu-154 count residual rows for the requested summaries."""
    rows: list[dict[str, Any]] = []
    for path in _iter_summary_paths(paths):
        summary = _load_summary(path)
        row = eu154_count_row(path, summary)
        if row is not None:
            rows.append(row)
    return rows


def _format_table(rows: Sequence[dict[str, Any]]) -> str:
    """Format Eu-154 residual rows as a compact text table."""
    header = (
        "path\tmeasurements\tsources\tobs_total\tpred_total\t"
        "rel_bias\tabs_rel_error\tunderprediction\tchi2"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["path"]),
                    str(row["measurements_completed"]),
                    str(row["reported_source_count"]),
                    f"{float(row['observed_total_counts']):.6g}",
                    f"{float(row['predicted_total_counts']):.6g}",
                    f"{float(row['relative_bias']):+.6f}",
                    f"{float(row['absolute_relative_error']):.6f}",
                    f"{float(row['underprediction_fraction']):.6f}",
                    f"{float(row['residual_chi2']):.6g}",
                ]
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Eu-154 count-model summary CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Eu-154 observed-vs-predicted count residuals from "
            "Geant4 result_summary JSON files."
        )
    )
    parser.add_argument("paths", nargs="+", help="Summary JSON files or directories.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON rows instead of a tab-separated table.",
    )
    parser.add_argument(
        "--fail-abs-rel-error",
        type=float,
        default=None,
        help="Exit nonzero when any Eu-154 absolute relative error exceeds this.",
    )
    args = parser.parse_args(argv)
    rows = summarize_eu154_count_model(args.paths)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(_format_table(rows))
    if args.fail_abs_rel_error is not None:
        threshold = max(float(args.fail_abs_rel_error), 0.0)
        if any(float(row["absolute_relative_error"]) > threshold for row in rows):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

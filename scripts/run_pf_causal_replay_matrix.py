"""Run predeclared PF-only causal comparisons on immutable measurement logs.

The command never changes observation data or fits a coefficient.  Every case
is run in a fresh process so CUDA state, caches, and random streams cannot leak
between comparisons.  Cases are deliberately run sequentially because
concurrent full-size PF replays on one GPU would change memory pressure and
confound wall-time comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _canonical_bytes(value: object) -> bytes:
    """Return deterministic strict JSON bytes for an authenticated report."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    """Return the SHA-256 digest of one immutable input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    """Load one strict top-level JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _required_string(
    payload: Mapping[str, object],
    key: str,
    *,
    location: str,
) -> str:
    """Return one required nonempty string from a matrix specification."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}.{key} must be a nonempty string.")
    return value


def _resolve_path(value: str, *, spec_path: Path) -> Path:
    """Resolve one path relative to the matrix specification."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    return path.resolve()


def _validated_rows(
    payload: Mapping[str, object],
    key: str,
    *,
    spec_path: Path,
) -> tuple[dict[str, object], ...]:
    """Validate and resolve labelled log or case rows."""
    raw_rows = payload.get(key)
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{key} must be a nonempty JSON list.")
    rows: list[dict[str, object]] = []
    labels: set[str] = set()
    path_key = "measurement_log" if key == "measurement_logs" else "config"
    for index, raw in enumerate(raw_rows):
        location = f"{key}[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{location} must be an object.")
        label = _required_string(raw, "label", location=location)
        if label in labels:
            raise ValueError(f"Duplicate {key} label {label!r}.")
        labels.add(label)
        input_path = _resolve_path(
            _required_string(raw, path_key, location=location),
            spec_path=spec_path,
        )
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        row: dict[str, object] = {
            "label": label,
            path_key: str(input_path),
            f"{path_key}_sha256": _sha256_path(input_path),
        }
        if key == "cases":
            seed = raw.get("seed", 0)
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(f"{location}.seed must be a nonnegative integer.")
            stop_after = raw.get("stop_after")
            if (
                stop_after is not None
                and (
                    isinstance(stop_after, bool)
                    or not isinstance(stop_after, int)
                    or stop_after < 0
                )
            ):
                raise ValueError(
                    f"{location}.stop_after must be null or nonnegative integer."
                )
            row.update({"seed": seed, "stop_after": stop_after})
        rows.append(row)
    return tuple(rows)


def _posterior_summary(output_dir: Path) -> dict[str, object]:
    """Extract compact cardinality and transition evidence from one replay."""
    posterior = _load_object(output_dir / "pf_posterior.json")
    diagnostics = _load_object(output_dir / "pf_diagnostics.json")
    isotope_rows = posterior.get("isotopes")
    if not isinstance(isotope_rows, Mapping):
        raise ValueError("Replay posterior lacks isotope rows.")
    cardinality: dict[str, object] = {}
    for isotope, raw in isotope_rows.items():
        if not isinstance(raw, Mapping):
            raise ValueError("Replay isotope posterior must be an object.")
        cardinality[str(isotope)] = {
            "representative_k": raw.get("map_cardinality"),
            "cardinality_probabilities": raw.get("cardinality_distribution"),
        }
    transition = diagnostics.get("structural_transition_provenance")
    return {
        "cardinality": cardinality,
        "structural_transition_provenance": transition,
        "final_state_sha256": diagnostics.get("final_state_sha256"),
        "resolved_config_sha256": diagnostics.get("resolved_config_sha256"),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the replay-matrix command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute each immutable log/config cell and publish one audit manifest."""
    args = _parse_args(argv)
    spec_path = args.spec.resolve()
    output_root = args.output_dir.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace {output_root}.")
    payload = _load_object(spec_path)
    if payload.get("schema_version") != 1:
        raise ValueError("Replay matrix schema_version must equal 1.")
    logs = _validated_rows(payload, "measurement_logs", spec_path=spec_path)
    cases = _validated_rows(payload, "cases", spec_path=spec_path)
    output_root.mkdir(parents=True)
    results: list[dict[str, object]] = []
    try:
        for log_row in logs:
            for case_row in cases:
                cell_name = f"{log_row['label']}__{case_row['label']}"
                cell_dir = output_root / cell_name
                log_path = output_root / f"{cell_name}.log"
                command = [
                    sys.executable,
                    "-m",
                    "pf.replay",
                    "--measurement-log",
                    str(log_row["measurement_log"]),
                    "--config",
                    str(case_row["config"]),
                    "--profile",
                    "pf_strict",
                    "--seed",
                    str(case_row["seed"]),
                    "--output-dir",
                    str(cell_dir),
                ]
                stop_after = case_row["stop_after"]
                if stop_after is not None:
                    command.extend(("--stop-after", str(stop_after)))
                with log_path.open("wb") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=REPOSITORY_ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                result: dict[str, object] = {
                    "cell": cell_name,
                    "measurement_log": dict(log_row),
                    "case": dict(case_row),
                    "exit_code": int(completed.returncode),
                    "process_log": str(log_path),
                }
                if completed.returncode == 0:
                    result["summary"] = _posterior_summary(cell_dir)
                results.append(result)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Replay matrix cell {cell_name} failed; see {log_path}."
                    )
        report = {
            "schema_version": 1,
            "diagnostic": "immutable_measurement_log_pf_causal_replay_matrix",
            "fit_or_tuning_performed": False,
            "acceptance_use": "causal_diagnosis_only",
            "execution_policy": "sequential_fresh_process_per_cell",
            "spec_path": str(spec_path),
            "spec_sha256": _sha256_path(spec_path),
            "results": results,
        }
        report_bytes = _canonical_bytes(report) + b"\n"
        (output_root / "causal_replay_matrix.json").write_bytes(report_bytes)
    except BaseException:
        failure = {
            "schema_version": 1,
            "diagnostic": "immutable_measurement_log_pf_causal_replay_matrix",
            "fit_or_tuning_performed": False,
            "completed": False,
            "results": results,
        }
        (output_root / "failed_matrix.json").write_bytes(
            _canonical_bytes(failure) + b"\n"
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

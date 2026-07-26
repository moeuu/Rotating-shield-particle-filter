"""Replay a logged run with the current target-preserving structural kernel.

This diagnostic keeps the immutable observations and the logged physical and
statistical PF settings. It replaces only the historical approximate
structural kernel and its incompatible state-changing heuristics. The
effective logged estimator block is removed from the in-memory log copy so the
standard replay safety check cannot silently restore the historical kernel.
No truth file is read by this runner.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

from pf.provenance import repository_commit, sha256_json
from pf.replay import (
    _logged_candidate_sources,
    _write_replay_outputs,
    build_replay_estimator,
    replay_records,
)
from runtime.measurement_log import MeasurementLog, load_measurement_log
from sim.runtime import load_runtime_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG = (
    REPOSITORY_ROOT
    / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
)
IMPLEMENTATION_FILES = (
    "main.py",
    "configs/geant4/variance_reduction_external_no_isaac_32threads.json",
    "src/measurement/surface_patches.py",
    "src/pf/estimator.py",
    "src/pf/likelihood.py",
    "src/pf/particle_filter.py",
    "src/pf/posterior.py",
    "src/pf/profiles.py",
    "src/pf/pure_estimator.py",
    "src/pf/replay.py",
    "src/pf/strength_prior.py",
    "src/pf/structural_rj.py",
    "src/realtime_demo.py",
    "uv.lock",
)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_source_hashes() -> dict[str, str]:
    """Bind the diagnostic to every implementation file that defines its target."""
    hashes: dict[str, str] = {}
    for relative_path in IMPLEMENTATION_FILES:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Required implementation file is missing: {relative_path}"
            )
        hashes[relative_path] = _sha256_file(path)
    hashes[
        "results/ral_ablation/diagnostic_runners/"
        "exact_rj_counterfactual_replay.py"
    ] = _sha256_file(Path(__file__).resolve())
    return hashes


def _write_output_hash_manifest(output_dir: Path) -> None:
    """Write hashes for every completed replay artifact except this manifest."""
    manifest_path = output_dir / "output_sha256.json"
    payload = {
        "schema_version": 1,
        "files": {
            path.relative_to(output_dir).as_posix(): _sha256_file(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path != manifest_path
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    """Parse the immutable log, current config, and new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG,
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _exact_config(
    log: MeasurementLog,
    current_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Return logged PF settings with only the exact kernel contract applied."""
    effective = log.runtime_config.get("effective_pf_replay")
    if not isinstance(effective, Mapping):
        raise RuntimeError("The diagnostic requires a logged effective PF block.")
    raw_pf = effective.get("pf_config")
    raw_grid = effective.get("candidate_grid")
    if not isinstance(raw_pf, Mapping) or not isinstance(raw_grid, Mapping):
        raise RuntimeError("The logged effective PF block is incomplete.")
    config = dict(raw_pf)
    max_sources = int(config["max_sources"])
    config.update(
        {
            "birth_enable": True,
            "structural_kernel_mode": "rj_mh",
            "structural_rj_patch_spacing_m": float(
                current_runtime["structural_rj_patch_spacing_m"]
            ),
            "structural_rj_move_probability": float(
                current_runtime["structural_rj_move_probability"]
            ),
            "structural_rj_birth_probability": float(
                current_runtime["structural_rj_birth_probability"]
            ),
            "structural_rj_death_probability": float(
                current_runtime["structural_rj_death_probability"]
            ),
            "structural_rj_position_move_probability": float(
                current_runtime["structural_rj_position_move_probability"]
            ),
            "structural_rj_local_position_move_probability": float(
                current_runtime.get(
                    "structural_rj_local_position_move_probability",
                    1.0,
                )
            ),
            "structural_rj_strength_move_probability": float(
                current_runtime["structural_rj_strength_move_probability"]
            ),
            "structural_cardinality_prior_probs": current_runtime.get(
                "structural_cardinality_prior_probs"
            ),
            "source_position_prior": "surface",
            "init_num_sources": [0, max_sources],
            "init_strength_prior": str(
                current_runtime["pf_init_strength_prior"]
            ),
            "init_strength_min": float(
                current_runtime["pf_init_strength_min_cps_1m"]
            ),
            "init_strength_max": float(
                current_runtime["pf_init_strength_max_cps_1m"]
            ),
            "split_prob": 0.0,
            "merge_prob": 0.0,
            "surface_rejuvenation_enable": False,
            "cardinality_preserving_resample": False,
            "mode_preserving_resample": False,
            "pseudo_source_verification_enable": False,
            "source_detector_exclusion_m": 0.0,
            "init_source_min_separation_m": 0.0,
        }
    )
    candidates = _logged_candidate_sources(log, raw_grid)
    config["replay_candidate_sources_xyz"] = candidates.tolist()
    return config


def _counterfactual_log(log: MeasurementLog) -> MeasurementLog:
    """Return an in-memory log that cannot restore its historical estimator."""
    runtime_config = dict(log.runtime_config)
    removed = runtime_config.pop("effective_pf_replay", None)
    if removed is None:
        raise RuntimeError("The log has no historical estimator block to replace.")
    return replace(log, runtime_config=runtime_config)


def _station_row(
    estimator: Any,
    *,
    station_id: int,
    record_index: int,
    elapsed_s: float,
) -> dict[str, Any]:
    """Return causal posterior and structural timing diagnostics for one station."""
    posterior = estimator.posterior_snapshot().to_dict()
    return {
        "station_id": int(station_id),
        "record_index": int(record_index),
        "elapsed_s": float(elapsed_s),
        "cardinality_distribution": {
            isotope: payload["cardinality_distribution"]
            for isotope, payload in posterior["isotopes"].items()
        },
        "structural_timing_s": {
            isotope: dict(filter_.last_structural_timing_s)
            for isotope, filter_ in estimator.filters.items()
        },
    }


def main() -> int:
    """Run and persist one truth-free exact-kernel counterfactual replay."""
    args = _parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace {args.output_dir}")
    original_log = load_measurement_log(args.measurement_log)
    current_runtime = load_runtime_config(args.runtime_config)
    config = _exact_config(original_log, current_runtime)
    replay_log = _counterfactual_log(original_log)
    estimator = build_replay_estimator(
        replay_log,
        config,
        profile="pf_strict",
        seed=int(args.seed),
    )
    started = time.perf_counter()
    station_rows: list[dict[str, Any]] = []

    def _report_station(
        current_estimator: Any,
        record: Any,
        record_index: int,
    ) -> None:
        """Record and print one completed station without reading future rows."""
        row = _station_row(
            current_estimator,
            station_id=int(record.station_id),
            record_index=int(record_index),
            elapsed_s=time.perf_counter() - started,
        )
        station_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    trace = replay_records(
        replay_log,
        estimator,
        station_complete_callback=_report_station,
    )
    _write_replay_outputs(
        args.output_dir,
        estimator=estimator,
        trace=trace,
        log=replay_log,
    )
    (args.output_dir / "counterfactual_pf_config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    contract = {
        "schema_version": 1,
        "diagnostic": "current_exact_rj_kernel_counterfactual_replay",
        "truth_read_by_runner": False,
        "observation_records_replaced": False,
        "physical_and_statistical_pf_settings_source": (
            "logged effective PF configuration"
        ),
        "historical_estimator_block_used": False,
        "structural_kernel_replaced": True,
        "disabled_incompatible_state_mutations": [
            "split_merge",
            "surface_rejuvenation",
            "cardinality_preserving_resample",
            "mode_preserving_resample",
            "pseudo_source_verification",
            "source_detector_exclusion",
            "initial_source_minimum_separation",
        ],
        "measurement_log_sha256": original_log.log_sha256,
        "measurement_log_resolved_config_sha256": (
            original_log.resolved_config_sha256
        ),
        "current_repository_commit": repository_commit(REPOSITORY_ROOT),
        "current_runtime_config_sha256": sha256_json(current_runtime),
        "counterfactual_pf_config_sha256": sha256_json(config),
        "structural_model_manifest": estimator.structural_model_manifest(),
        "implementation_source_sha256": _implementation_source_hashes(),
        "runtime_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "random_seed": int(args.seed),
        "record_count": int(len(trace)),
        "station_count": int(len(station_rows)),
        "elapsed_s": float(time.perf_counter() - started),
    }
    (args.output_dir / "counterfactual_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "station_diagnostics.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in station_rows
        ),
        encoding="utf-8",
    )
    _write_output_hash_manifest(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "record_count": len(trace),
                "station_count": len(station_rows),
                "elapsed_s": contract["elapsed_s"],
                "posterior": estimator.posterior_snapshot().to_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

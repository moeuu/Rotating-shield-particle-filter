"""Replay an opt-in PF with estimator-neutral external relocation directives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from pf.external_relocation import (
    ExternalRelocationSchedule,
    load_directive_schedule,
    pre_update_predictive_counts,
)
from pf.profiles import enforce_pure_runtime_settings
from pf.provenance import canonical_json_bytes, sha256_json
from pf.pure_estimator import PurePFEstimator
from pf.replay import (
    _sha256_bytes,
    _write_replay_outputs,
    build_replay_estimator,
    replay_records,
)
from runtime.measurement_log import MeasurementLog, load_measurement_log
from sim.runtime import load_runtime_config


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize deterministic compact JSON Lines."""
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_hybrid_outputs(
    output_dir: str | Path,
    *,
    estimator: PurePFEstimator,
    trace: Sequence[Mapping[str, Any]],
    log: Any,
    schedule: ExternalRelocationSchedule,
    predictive_rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically publish standard PF files plus explicit hybrid artifacts."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"Refusing to replace hybrid replay output {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.hybrid-tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Temporary hybrid replay output exists: {staging}.")
    try:
        _write_replay_outputs(
            staging,
            estimator=estimator,
            trace=trace,
            log=log,
        )
        schedule_payload = {
            "schema_version": 1,
            "directives": [item.to_dict() for item in schedule.directives],
        }
        receipts_payload = {
            "schema_version": 1,
            "receipts": schedule.receipts,
        }
        (staging / "external_directive_schedule.json").write_bytes(
            canonical_json_bytes(schedule_payload)
        )
        (staging / "external_relocation_receipts.json").write_bytes(
            canonical_json_bytes(receipts_payload)
        )
        receipt_directory = staging / "pf_directive_receipts"
        receipt_directory.mkdir()
        for detailed_receipt in schedule.receipts:
            contract_receipt = detailed_receipt["contract_receipt"]
            assert isinstance(contract_receipt, Mapping)
            receipt_id = str(contract_receipt["receipt_id"])
            (receipt_directory / f"{receipt_id}.json").write_bytes(
                canonical_json_bytes(contract_receipt)
            )
        (staging / "pf_pre_update_predictive.jsonl").write_bytes(
            _jsonl_bytes(predictive_rows)
        )
        diagnostics = {
            "schema_version": 1,
            "estimator_family": "particle_filter",
            "estimator_variant": "pf_external_relocation_mwg_v1",
            "base_estimator_variant": estimator.estimator_variant,
            "measurement_log_sha256": log.log_sha256,
            "resolved_config_sha256": estimator.resolved_config_hash,
            "directive_schedule_sha256": sha256_json(schedule_payload),
            "directives_declared": len(schedule.directives),
            "directives_applied": len(schedule.applied_directive_ids),
            "applied_directive_ids": list(schedule.applied_directive_ids),
            "pending_directive_ids": list(schedule.pending_directive_ids),
            "proposal_role": "target_preserving_mcmc_only",
            "proposal_kernel": "fixed_cardinality_metropolis_within_gibbs",
            "target_history": "records_0_through_each_directive_cutoff",
            "cardinality_changes_from_external_directives": False,
            "strength_changes_from_external_directives": False,
            "background_changes_from_external_directives": False,
            "arbitrary_particle_reweighting": False,
            "future_records_used_for_proposal_application": False,
            "surface_projection_used": False,
            "pre_update_predictive_record_count": len(predictive_rows),
        }
        (staging / "hybrid_diagnostics.json").write_bytes(
            canonical_json_bytes(diagnostics)
        )
        hybrid_posterior = {
            "schema_version": 1,
            "estimator_family": "particle_filter",
            "estimator_variant": "pf_external_relocation_mwg_v1",
            "base_estimator_variant": estimator.estimator_variant,
            "final_estimate_source": (
                "pf_posterior_after_target_preserving_external_relocation"
            ),
            "base_pf_posterior": estimator.posterior_snapshot().to_dict(),
            "applied_directive_ids": list(schedule.applied_directive_ids),
            "external_candidate_role": "mcmc_proposal_only",
            "cardinality_changed_by_external_directives": False,
            "particle_weights_changed_by_external_directives": False,
        }
        (staging / "hybrid_pf_posterior.json").write_bytes(
            canonical_json_bytes(hybrid_posterior)
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def replay_with_external_relocations(
    measurement_log: str | Path | MeasurementLog,
    config: str | Path | Mapping[str, Any],
    directive_schedule: str | Path | Mapping[str, Any],
    *,
    profile: str = "pf_strict",
    seed: int = 0,
    relocation_seed: int | None = None,
    stop_after: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[
    PurePFEstimator,
    tuple[dict[str, Any], ...],
    ExternalRelocationSchedule,
    tuple[dict[str, Any], ...],
]:
    """Run causal replay and apply bound proposal-only directives at cutoffs."""
    log = (
        measurement_log
        if isinstance(measurement_log, MeasurementLog)
        else load_measurement_log(measurement_log)
    )
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        config_hash = _sha256_bytes(config_path.read_bytes())
        resolved = load_runtime_config(config_path)
    else:
        raw_config = dict(config)
        config_hash = sha256_json(raw_config)
        resolved = raw_config
    resolved = enforce_pure_runtime_settings(resolved, profile=profile)
    estimator = build_replay_estimator(
        log,
        resolved,
        profile=profile,
        seed=int(seed),
        config_hash=config_hash,
        resolved_config_hash=None,
    )
    directives = load_directive_schedule(directive_schedule)
    schedule = ExternalRelocationSchedule(
        directives,
        log=log,
        estimator=estimator,
        base_seed=int(seed if relocation_seed is None else relocation_seed),
    )
    predictive_rows: list[dict[str, Any]] = []

    def _record_prediction(
        active_estimator: PurePFEstimator,
        record: Any,
        record_index: int,
        pose_idx: int,
    ) -> None:
        """Capture causal predictive moments before observing this row."""
        del pose_idx
        predictive_rows.append(
            pre_update_predictive_counts(
                active_estimator,
                record,
                record_index=record_index,
            )
        )

    trace = replay_records(
        log,
        estimator,
        stop_after=stop_after,
        pre_record_callback=_record_prediction,
        station_complete_callback=schedule.apply_at_boundary,
    )
    if output_dir is not None:
        _write_hybrid_outputs(
            output_dir,
            estimator=estimator,
            trace=trace,
            log=log,
            schedule=schedule,
            predictive_rows=predictive_rows,
        )
    return estimator, trace, schedule, tuple(predictive_rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the opt-in external-relocation replay command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--directive-schedule", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("pf_strict", "pf_profiled"),
        default="pf_strict",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relocation-seed", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(None if argv is None else list(argv))
    replay_with_external_relocations(
        args.measurement_log,
        args.config,
        args.directive_schedule,
        profile=args.profile,
        seed=args.seed,
        relocation_seed=args.relocation_seed,
        stop_after=args.stop_after,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

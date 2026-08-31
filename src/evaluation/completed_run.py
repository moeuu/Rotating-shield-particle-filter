"""Evaluate an exactly identified completed PF run against private truth."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.cluster_accuracy import (
    DEFAULT_CLUSTER_ACCURACY_CRITERIA,
    compute_cluster_accuracy_evaluation,
)
from evaluation.private_truth import (
    _load_json_object,
    load_private_truth_for_completed_result,
)


def evaluate_completed_pf_run(
    *,
    result_path: str | Path,
    posterior_path: str | Path,
    evaluation_input_path: str | Path,
    truth_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return standard metrics after an exact completed-run private-truth join."""
    result = _load_json_object(result_path, name="PF result")
    truth = load_private_truth_for_completed_result(
        result_path,
        truth_manifest_path,
    )
    posterior = _load_json_object(posterior_path, name="PF posterior")
    evaluation_input = _load_json_object(
        evaluation_input_path,
        name="PF post-run evaluation input",
    )
    if evaluation_input.get("source_run_id") != truth.run_id:
        raise ValueError("PF evaluation input run_id differs from private truth.")
    if evaluation_input.get("truth_read") is not False:
        raise ValueError("PF evaluation input must be truth-free.")
    provenance = posterior.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("PF posterior provenance is missing.")
    if provenance.get("measurement_log_sha256") != evaluation_input.get(
        "measurement_log_sha256"
    ):
        raise ValueError(
            "PF posterior and evaluation input use different MeasurementLogs."
        )
    truth_by_isotope: defaultdict[str, list[object]] = defaultdict(list)
    for source in truth.sources:
        isotope = source.get("isotope")
        if not isinstance(isotope, str) or not isotope:
            raise ValueError("Every private truth source must declare an isotope.")
        truth_by_isotope[isotope].append(source)
    evaluation = compute_cluster_accuracy_evaluation(
        truth_by_isotope,
        posterior,
        evaluation_input,
        criteria=DEFAULT_CLUSTER_ACCURACY_CRITERIA,
    )
    sampler_quality_status = result.get("sampler_quality_status")
    if sampler_quality_status not in {"pass", "warning", "failed"}:
        raise ValueError(
            "Completed PF result has an invalid sampler_quality_status."
        )
    hard_cap_failed = (
        evaluation["hard_cap_sampler_quality_status"] == "failed"
    )
    if hard_cap_failed != (sampler_quality_status == "failed"):
        raise ValueError(
            "Completed PF sampler quality contradicts hard-cap posterior evidence."
        )
    evaluation["execution_status"] = "complete"
    evaluation["sampler_quality_status"] = sampler_quality_status
    evaluation["run_identity"] = {
        "run_id": truth.run_id,
        "experiment_profile_id": truth.experiment_profile_id,
        "scene_variant_id": truth.scene_variant_id,
        "scene_seed": truth.scene_seed,
        "measurement_log_sha256": evaluation_input[
            "measurement_log_sha256"
        ],
    }
    return evaluation


__all__ = ["evaluate_completed_pf_run"]

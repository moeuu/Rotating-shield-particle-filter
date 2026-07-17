"""Recommend one hybrid DSS-PP action without mutating or actuating the PF.

The boundary accepts an estimator-neutral, collision-attested candidate set,
causally replays the pure PF and any target-preserving relocation directives
through one completed station, and evaluates the existing DSS-PP algorithm.
External modes are planning hypotheses only.  They never enter PF particles,
weights, source cardinality, strengths, or backgrounds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pf.external_relocation import covered_records_sha256
from pf.hybrid_replay import replay_with_external_relocations
from pf.provenance import canonical_json_bytes, json_safe, sha256_json
from pf.pure_estimator import PurePFEstimator
from planning.dss_pp import (
    DSSPPConfig,
    DSSPPResult,
    SignatureMode,
    _HYBRID_RECOMMENDATION_EXTERNAL_MODE_TOKEN,
    select_dss_pp_next_station,
)
from runtime.measurement_log import MeasurementLog, load_measurement_log


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "request_id",
    "source_run_id",
    "data_cutoff_step",
    "data_cutoff_station",
    "covered_records_sha256",
    "pf_resolved_config_sha256",
    "current_pose_xyz",
    "current_pair_id",
    "visited_poses_xyz",
    "candidate_poses_xyz",
    "candidate_attestation",
    "dsspp_config",
    "external_modes",
    "bounds_xyz",
    "continuous_height_bounds_m",
}


class HybridPlanningBoundaryError(RuntimeError):
    """Report an invalid cutoff, attestation, mode, or planning request."""


class ExternalModeVerificationState(StrEnum):
    """Name the verification state carried by an external planning mode."""

    PENDING = "pending"
    VERIFIED = "verified"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class ExternalPlanningMode:
    """Store estimator-neutral mode metadata used only by DSS-PP."""

    mode_id: str
    isotope: str
    position_xyz: tuple[float, float, float]
    strength_cps_1m: float
    weight: float
    spread_m: float
    verification_state: ExternalModeVerificationState
    source_snapshot_id: str | None = None

    @property
    def belief_source(self) -> str:
        """Return the exact planner-belief provenance label."""
        return f"external_mode_{self.verification_state.value}"

    def to_signature_mode(self) -> SignatureMode:
        """Convert planner metadata to the established DSS-PP mode shape."""
        return SignatureMode(
            isotope=self.isotope,
            position_xyz=np.asarray(self.position_xyz, dtype=float),
            strength_cps_1m=float(self.strength_cps_1m),
            weight=float(self.weight),
            spread_m=float(self.spread_m),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe planner-only mode payload."""
        return {
            "mode_id": self.mode_id,
            "isotope": self.isotope,
            "position_xyz": [float(value) for value in self.position_xyz],
            "strength_cps_1m": float(self.strength_cps_1m),
            "weight": float(self.weight),
            "spread_m": float(self.spread_m),
            "verification_state": self.verification_state.value,
            "belief_source": self.belief_source,
            "source_snapshot_id": self.source_snapshot_id,
            "planner_metadata_only": True,
        }


@dataclass(frozen=True)
class CandidateSetAttestation:
    """Bind an ordered candidate set to external workspace safety checks."""

    candidate_poses_sha256: str
    workspace_sha256: str
    planning_config_sha256: str
    collision_checked: bool
    reachability_filtered: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the exact attestation echoed by the recommendation."""
        return {
            "candidate_poses_sha256": self.candidate_poses_sha256,
            "workspace_sha256": self.workspace_sha256,
            "planning_config_sha256": self.planning_config_sha256,
            "collision_checked": self.collision_checked,
            "reachability_filtered": self.reachability_filtered,
        }


@dataclass(frozen=True)
class HybridPlanningRequest:
    """Hold one validated estimator-neutral DSS-PP recommendation request."""

    request_id: str
    source_run_id: str
    data_cutoff_step: int
    data_cutoff_station: int
    covered_records_sha256: str
    pf_resolved_config_sha256: str
    current_pose_xyz: tuple[float, float, float]
    current_pair_id: int | None
    visited_poses_xyz: tuple[tuple[float, float, float], ...]
    candidate_poses_xyz: tuple[tuple[float, float, float], ...]
    candidate_attestation: CandidateSetAttestation
    dsspp_config: DSSPPConfig
    external_modes: tuple[ExternalPlanningMode, ...]
    bounds_xyz: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        None
    )
    continuous_height_bounds_m: tuple[float, float] | None = None


class _ExternalPlanningEstimatorView:
    """Expose external hypotheses to DSS-PP while delegating pure PF reads."""

    def __init__(
        self,
        estimator: PurePFEstimator,
        modes: Sequence[ExternalPlanningMode],
    ) -> None:
        """Create a read-only planning view over one replayed estimator."""
        self._base_estimator = estimator
        included = tuple(
            mode
            for mode in modes
            if mode.verification_state
            in {
                ExternalModeVerificationState.PENDING,
                ExternalModeVerificationState.VERIFIED,
            }
        )
        grouped: dict[str, list[SignatureMode]] = {}
        for mode in included:
            grouped.setdefault(mode.isotope, []).append(mode.to_signature_mode())
        self._external_modes = {
            isotope: tuple(mode_list) for isotope, mode_list in grouped.items()
        }
        sources = ["pf_posterior", "pf_tentative"]
        for state in (
            ExternalModeVerificationState.PENDING,
            ExternalModeVerificationState.VERIFIED,
        ):
            if any(mode.verification_state is state for mode in included):
                sources.append(f"external_mode_{state.value}")
        self.planner_belief_sources = tuple(sources)

    def __getattr__(self, name: str) -> Any:
        """Delegate every established estimator read to the pure PF."""
        estimator = object.__getattribute__(self, "_base_estimator")
        return getattr(estimator, name)

    def planner_only_external_signature_modes(
        self,
    ) -> dict[str, tuple[SignatureMode, ...]]:
        """Return only pending/verified external hypotheses to DSS-PP."""
        return dict(self._external_modes)


def _sha256_field(value: object, field_name: str) -> str:
    """Validate and normalize one lowercase SHA-256 contract field."""
    normalized = str(value).strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise HybridPlanningBoundaryError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return normalized


def _nonempty_string(value: object, field_name: str) -> str:
    """Validate a required non-empty identifier."""
    normalized = str(value).strip()
    if not normalized:
        raise HybridPlanningBoundaryError(f"{field_name} must be non-empty.")
    return normalized


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    """Validate a non-boolean bounded integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise HybridPlanningBoundaryError(f"{field_name} must be an integer.")
    result = int(value)
    if result < int(minimum):
        raise HybridPlanningBoundaryError(
            f"{field_name} must be at least {int(minimum)}."
        )
    return result


def _finite_float(value: object, field_name: str, *, positive: bool = False) -> float:
    """Validate one finite floating-point contract value."""
    if isinstance(value, bool):
        raise HybridPlanningBoundaryError(f"{field_name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HybridPlanningBoundaryError(f"{field_name} must be numeric.") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise HybridPlanningBoundaryError(f"{field_name} must be {qualifier}.")
    return result


def _xyz(value: object, field_name: str) -> tuple[float, float, float]:
    """Validate one finite XYZ vector using exact float64 semantics."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HybridPlanningBoundaryError(
            f"{field_name} must contain three numeric coordinates."
        ) from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise HybridPlanningBoundaryError(
            f"{field_name} must contain three finite coordinates."
        )
    return tuple(float(item) for item in array)


def _xyz_rows(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[tuple[float, float, float], ...]:
    """Validate an ordered finite float64 XYZ matrix."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HybridPlanningBoundaryError(
            f"{field_name} must be an array of XYZ rows."
        ) from exc
    if array.size == 0 and allow_empty:
        return tuple()
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise HybridPlanningBoundaryError(
            f"{field_name} must be a finite array with shape (N, 3)."
        )
    if not allow_empty and array.shape[0] == 0:
        raise HybridPlanningBoundaryError(f"{field_name} must not be empty.")
    return tuple(tuple(float(item) for item in row) for row in array)


def candidate_poses_sha256(candidate_poses_xyz: object) -> str:
    """Hash the ordered candidate matrix after exact float64-list conversion."""
    rows = _xyz_rows(
        candidate_poses_xyz,
        "candidate_poses_xyz",
        allow_empty=False,
    )
    return sha256_json([list(row) for row in rows])


def _candidate_attestation_from_mapping(
    payload: object,
    *,
    candidate_rows: tuple[tuple[float, float, float], ...],
) -> CandidateSetAttestation:
    """Validate upstream collision and reachability claims without recreating them."""
    if not isinstance(payload, Mapping):
        raise HybridPlanningBoundaryError("candidate_attestation must be an object.")
    collision_checked = payload.get("collision_checked")
    reachability_filtered = payload.get("reachability_filtered")
    if collision_checked is not True or reachability_filtered is not True:
        raise HybridPlanningBoundaryError(
            "Candidate attestation requires collision_checked=true and "
            "reachability_filtered=true."
        )
    attestation = CandidateSetAttestation(
        candidate_poses_sha256=_sha256_field(
            payload.get("candidate_poses_sha256"),
            "candidate_attestation.candidate_poses_sha256",
        ),
        workspace_sha256=_sha256_field(
            payload.get("workspace_sha256"),
            "candidate_attestation.workspace_sha256",
        ),
        planning_config_sha256=_sha256_field(
            payload.get("planning_config_sha256"),
            "candidate_attestation.planning_config_sha256",
        ),
        collision_checked=True,
        reachability_filtered=True,
    )
    actual_digest = candidate_poses_sha256(candidate_rows)
    if attestation.candidate_poses_sha256 != actual_digest:
        raise HybridPlanningBoundaryError(
            "Candidate-set digest does not match the ordered float64 XYZ rows."
        )
    return attestation


def _external_mode_from_mapping(payload: object) -> ExternalPlanningMode:
    """Validate one external mode without granting it estimator authority."""
    if not isinstance(payload, Mapping):
        raise HybridPlanningBoundaryError("Each external mode must be an object.")
    try:
        state = ExternalModeVerificationState(
            str(payload.get("verification_state", "")).strip().lower()
        )
    except ValueError as exc:
        raise HybridPlanningBoundaryError(
            "external mode verification_state must be pending, verified, or "
            "quarantined."
        ) from exc
    spread = _finite_float(payload.get("spread_m", 0.0), "external_modes.spread_m")
    if spread < 0.0:
        raise HybridPlanningBoundaryError(
            "external_modes.spread_m must be non-negative."
        )
    snapshot_id_raw = payload.get("source_snapshot_id")
    snapshot_id = (
        None
        if snapshot_id_raw is None
        else _nonempty_string(snapshot_id_raw, "external_modes.source_snapshot_id")
    )
    return ExternalPlanningMode(
        mode_id=_nonempty_string(payload.get("mode_id"), "external_modes.mode_id"),
        isotope=_nonempty_string(payload.get("isotope"), "external_modes.isotope"),
        position_xyz=_xyz(
            payload.get("position_xyz"),
            "external_modes.position_xyz",
        ),
        strength_cps_1m=_finite_float(
            payload.get("strength_cps_1m"),
            "external_modes.strength_cps_1m",
            positive=True,
        ),
        weight=_finite_float(
            payload.get("weight"),
            "external_modes.weight",
            positive=True,
        ),
        spread_m=spread,
        verification_state=state,
        source_snapshot_id=snapshot_id,
    )


def _dsspp_config_from_mapping(payload: object) -> DSSPPConfig:
    """Resolve the existing DSS-PP config with attested-candidate safeguards."""
    if not isinstance(payload, Mapping):
        raise HybridPlanningBoundaryError("dsspp_config must be an object.")
    raw = dict(payload)
    if raw.get("augment_candidates") is not False:
        raise HybridPlanningBoundaryError(
            "Hybrid planning requires dsspp_config.augment_candidates=false; "
            "internally generated poses have no upstream collision attestation."
        )
    if raw.get("include_runtime_rescue_modes", False) is not False:
        raise HybridPlanningBoundaryError(
            "Hybrid planning cannot enable legacy runtime rescue modes."
        )
    if raw.get("include_global_surface_rescue_modes", False) is not False:
        raise HybridPlanningBoundaryError(
            "Hybrid planning cannot enable legacy surface rescue modes."
        )
    try:
        return DSSPPConfig(**raw)
    except TypeError as exc:
        raise HybridPlanningBoundaryError(
            f"Invalid DSS-PP configuration: {exc}"
        ) from exc


def planning_request_from_mapping(payload: Mapping[str, Any]) -> HybridPlanningRequest:
    """Parse and validate the generic hybrid planning request contract."""
    if not isinstance(payload, Mapping):
        raise HybridPlanningBoundaryError("Planning request must be an object.")
    if payload.get("schema_version") != 1:
        raise HybridPlanningBoundaryError("Planning request schema_version must be 1.")
    unknown = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise HybridPlanningBoundaryError(
            "Unknown planning request fields: " + ", ".join(unknown) + "."
        )
    candidate_rows = _xyz_rows(
        payload.get("candidate_poses_xyz"),
        "candidate_poses_xyz",
        allow_empty=False,
    )
    if len(set(candidate_rows)) != len(candidate_rows):
        raise HybridPlanningBoundaryError(
            "candidate_poses_xyz must not contain duplicate XYZ rows."
        )
    attestation = _candidate_attestation_from_mapping(
        payload.get("candidate_attestation"),
        candidate_rows=candidate_rows,
    )
    external_raw = payload.get("external_modes", [])
    if not isinstance(external_raw, list):
        raise HybridPlanningBoundaryError("external_modes must be an array.")
    external_modes = tuple(_external_mode_from_mapping(item) for item in external_raw)
    mode_ids = [mode.mode_id for mode in external_modes]
    if len(set(mode_ids)) != len(mode_ids):
        raise HybridPlanningBoundaryError("external mode_id values must be unique.")
    current_pair_raw = payload.get("current_pair_id")
    current_pair_id = (
        None
        if current_pair_raw is None
        else _integer(current_pair_raw, "current_pair_id")
    )
    if current_pair_id is not None and current_pair_id > 63:
        raise HybridPlanningBoundaryError("current_pair_id must be in 0..63 or null.")
    bounds_raw = payload.get("bounds_xyz")
    bounds = None
    if bounds_raw is not None:
        if not isinstance(bounds_raw, Mapping):
            raise HybridPlanningBoundaryError("bounds_xyz must be an object.")
        lower = _xyz(bounds_raw.get("min"), "bounds_xyz.min")
        upper = _xyz(bounds_raw.get("max"), "bounds_xyz.max")
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise HybridPlanningBoundaryError(
                "bounds_xyz.min must not exceed bounds_xyz.max."
            )
        bounds = (lower, upper)
    height_raw = payload.get("continuous_height_bounds_m")
    height_bounds = None
    if height_raw is not None:
        try:
            height_values = np.asarray(height_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise HybridPlanningBoundaryError(
                "continuous_height_bounds_m must contain two finite values."
            ) from exc
        if (
            height_values.shape != (2,)
            or not np.all(np.isfinite(height_values))
            or float(height_values[0]) > float(height_values[1])
        ):
            raise HybridPlanningBoundaryError(
                "continuous_height_bounds_m must be finite [minimum, maximum]."
            )
        height_bounds = (float(height_values[0]), float(height_values[1]))
    return HybridPlanningRequest(
        request_id=_nonempty_string(payload.get("request_id"), "request_id"),
        source_run_id=_nonempty_string(
            payload.get("source_run_id"),
            "source_run_id",
        ),
        data_cutoff_step=_integer(
            payload.get("data_cutoff_step"),
            "data_cutoff_step",
        ),
        data_cutoff_station=_integer(
            payload.get("data_cutoff_station"),
            "data_cutoff_station",
        ),
        covered_records_sha256=_sha256_field(
            payload.get("covered_records_sha256"),
            "covered_records_sha256",
        ),
        pf_resolved_config_sha256=_sha256_field(
            payload.get("pf_resolved_config_sha256"),
            "pf_resolved_config_sha256",
        ),
        current_pose_xyz=_xyz(payload.get("current_pose_xyz"), "current_pose_xyz"),
        current_pair_id=current_pair_id,
        visited_poses_xyz=_xyz_rows(
            payload.get("visited_poses_xyz", []),
            "visited_poses_xyz",
            allow_empty=True,
        ),
        candidate_poses_xyz=candidate_rows,
        candidate_attestation=attestation,
        dsspp_config=_dsspp_config_from_mapping(payload.get("dsspp_config")),
        external_modes=external_modes,
        bounds_xyz=bounds,
        continuous_height_bounds_m=height_bounds,
    )


def _load_json_mapping(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load one JSON object from a path or in-memory mapping."""
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HybridPlanningBoundaryError(
            f"Could not load JSON object {path}."
        ) from exc
    if not isinstance(payload, dict):
        raise HybridPlanningBoundaryError(f"{path} must contain one JSON object.")
    return payload


def _cutoff_record_index(log: MeasurementLog, request: HybridPlanningRequest) -> int:
    """Resolve and validate the exact writer-marked causal station boundary."""
    if request.source_run_id != log.run_id:
        raise HybridPlanningBoundaryError("Planning request source_run_id mismatch.")
    matching = [
        index
        for index, record in enumerate(log.records)
        if int(record.step_id) == request.data_cutoff_step
        and int(record.station_id) == request.data_cutoff_station
    ]
    if len(matching) != 1:
        raise HybridPlanningBoundaryError(
            "Planning cutoff must identify exactly one MeasurementLog record."
        )
    index = matching[0]
    record = log.records[index]
    if record.metadata.get("station_complete") is not True:
        raise HybridPlanningBoundaryError(
            "Hybrid planning is allowed only at a writer-marked completed station."
        )
    actual_digest = covered_records_sha256(log, index + 1)
    if request.covered_records_sha256 != actual_digest:
        raise HybridPlanningBoundaryError(
            "Planning request covered-record prefix digest mismatch."
        )
    return index


def _causal_directive_schedule(
    source: str | Path | Mapping[str, Any] | None,
    *,
    cutoff_record_index: int,
    cutoff_step: int,
) -> dict[str, Any]:
    """Drop future directives before validation so suffixes cannot affect planning."""
    if source is None:
        return {"schema_version": 1, "directives": []}
    payload = _load_json_mapping(source)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("directives"), list
    ):
        raise HybridPlanningBoundaryError(
            "Directive schedule must be a schema-v1 object with a directives array."
        )
    causal: list[dict[str, Any]] = []
    for item in payload["directives"]:
        if not isinstance(item, Mapping):
            raise HybridPlanningBoundaryError("Each directive must be an object.")
        if "apply_after_record_index" in item:
            apply_index = _integer(
                item["apply_after_record_index"],
                "directive.apply_after_record_index",
            )
            is_causal = apply_index <= int(cutoff_record_index)
        elif "apply_after_step" in item:
            apply_step = _integer(
                item["apply_after_step"], "directive.apply_after_step"
            )
            is_causal = apply_step <= int(cutoff_step)
        elif "data_cutoff_step" in item:
            apply_step = _integer(
                item["data_cutoff_step"],
                "directive.data_cutoff_step",
            )
            is_causal = apply_step <= int(cutoff_step)
        else:
            raise HybridPlanningBoundaryError(
                "Each directive requires an application record or step cutoff."
            )
        if is_causal:
            causal.append(dict(item))
    return {"schema_version": 1, "directives": causal}


def _validate_modes_for_estimator(
    estimator: PurePFEstimator,
    modes: Sequence[ExternalPlanningMode],
) -> None:
    """Reject unknown isotopes or external positions outside the PF source domain."""
    known_isotopes = {str(value) for value in estimator.isotopes}
    for mode in modes:
        if mode.verification_state is ExternalModeVerificationState.QUARANTINED:
            continue
        if mode.isotope not in known_isotopes:
            raise HybridPlanningBoundaryError(
                f"External mode {mode.mode_id!r} has unknown isotope {mode.isotope!r}."
            )
        filt = estimator.filters[mode.isotope]
        lower = np.asarray(filt.config.position_min, dtype=float).reshape(3)
        upper = np.asarray(filt.config.position_max, dtype=float).reshape(3)
        position = np.asarray(mode.position_xyz, dtype=float)
        if np.any(position < lower) or np.any(position > upper):
            raise HybridPlanningBoundaryError(
                f"External mode {mode.mode_id!r} lies outside the PF position domain."
            )


def _selected_candidate_index(
    candidates: NDArray[np.float64],
    selected: NDArray[np.float64],
) -> int:
    """Map the post-filter DSS-PP pose back to the attested ordered candidate set."""
    matches = np.flatnonzero(
        np.all(candidates == np.asarray(selected, dtype=np.float64), axis=1)
    )
    if matches.size != 1:
        raise HybridPlanningBoundaryError(
            "DSS-PP selected a pose outside the uniquely attested candidate set."
        )
    return int(matches[0])


def _recommendation_payload(
    *,
    request: HybridPlanningRequest,
    request_artifact_sha256: str,
    estimator: PurePFEstimator,
    planning_result: DSSPPResult,
    schedule: Any,
    causal_schedule: Mapping[str, Any],
    state_sha256_before: str,
    state_sha256_after: str,
    profile: str,
    seed: int,
    relocation_seed: int | None,
) -> dict[str, Any]:
    """Build one deterministic recommendation and complete causal provenance."""
    candidates = np.asarray(request.candidate_poses_xyz, dtype=np.float64)
    selected_index = _selected_candidate_index(candidates, planning_result.next_pose)
    included = tuple(
        mode
        for mode in request.external_modes
        if mode.verification_state
        in {
            ExternalModeVerificationState.PENDING,
            ExternalModeVerificationState.VERIFIED,
        }
    )
    excluded = tuple(
        mode
        for mode in request.external_modes
        if mode.verification_state is ExternalModeVerificationState.QUARANTINED
    )
    belief_sources = ["pf_posterior", "pf_tentative"]
    for state in (
        ExternalModeVerificationState.PENDING,
        ExternalModeVerificationState.VERIFIED,
    ):
        if any(mode.verification_state is state for mode in included):
            belief_sources.append(f"external_mode_{state.value}")
    selected_action = {
        "candidate_index": selected_index,
        "dsspp_filtered_pose_index": int(planning_result.next_pose_index),
        "pose_xyz": [float(value) for value in planning_result.next_pose],
        "detector_height_m": float(planning_result.next_pose[2]),
        "shield_program": {
            "name": str(planning_result.shield_program.name),
            "kind": str(planning_result.shield_program.kind),
            "pair_ids": [
                int(value) for value in planning_result.shield_program.pair_ids
            ],
        },
        "score": float(planning_result.score),
    }
    identity = {
        "schema_version": 1,
        "request_id": request.request_id,
        "causal_planning_request_sha256": request_artifact_sha256,
        "source_run_id": request.source_run_id,
        "data_cutoff_step": request.data_cutoff_step,
        "data_cutoff_station": request.data_cutoff_station,
        "covered_records_sha256": request.covered_records_sha256,
        "pf_resolved_config_sha256": request.pf_resolved_config_sha256,
        "candidate_poses_sha256": (
            request.candidate_attestation.candidate_poses_sha256
        ),
        "dsspp_config_sha256": sha256_json(asdict(request.dsspp_config)),
        "causal_directive_schedule_sha256": sha256_json(causal_schedule),
        "included_external_mode_ids": [mode.mode_id for mode in included],
        "excluded_quarantined_mode_ids": [mode.mode_id for mode in excluded],
        "pf_state_sha256": state_sha256_before,
        "selected_action": selected_action,
    }
    recommendation_id = "pf-dsspp-" + sha256_json(identity)[:24]
    result_diagnostics = dict(planning_result.diagnostics)
    result_diagnostics["planner_belief_sources"] = list(belief_sources)
    result_diagnostics["external_modes_included"] = len(included)
    result_diagnostics["external_modes_quarantined_excluded"] = len(excluded)
    return {
        "schema_version": 1,
        "recommendation_id": recommendation_id,
        "recommendation_kind": "algorithmic_dsspp_action_recommendation",
        "algorithmic_recommendation_only": True,
        "robot_actuation_authorized": False,
        "selected_action": selected_action,
        "sequence": json_safe(planning_result.sequence),
        "diagnostics": json_safe(result_diagnostics),
        "belief": {
            "planner_belief_sources": belief_sources,
            "external_modes_included": [mode.to_dict() for mode in included],
            "external_modes_quarantined_excluded": [
                mode.to_dict() for mode in excluded
            ],
            "included_external_mode_ids": [mode.mode_id for mode in included],
            "excluded_quarantined_mode_ids": [mode.mode_id for mode in excluded],
            "excluded_quarantined_mode_count": len(excluded),
            "external_strengths_and_weights_are_planner_metadata_only": True,
        },
        "candidate_attestation": request.candidate_attestation.to_dict(),
        "causal_boundary": {
            "source_run_id": request.source_run_id,
            "data_cutoff_step": request.data_cutoff_step,
            "data_cutoff_station": request.data_cutoff_station,
            "covered_records_sha256": request.covered_records_sha256,
            "pf_resolved_config_sha256": request.pf_resolved_config_sha256,
            "causal_identity_uses_record_prefix_only": True,
        },
        "external_relocation": {
            "causal_directive_schedule_sha256": sha256_json(causal_schedule),
            "causal_directives_declared": len(causal_schedule["directives"]),
            "applied_directive_ids": list(schedule.applied_directive_ids),
            "pending_directive_ids_at_cutoff": list(schedule.pending_directive_ids),
            "target_preserving_position_relocation_only": True,
        },
        "pf_state_integrity": {
            "state_sha256_before_planning": state_sha256_before,
            "state_sha256_after_planning": state_sha256_after,
            "pf_particles_or_weights_mutated_by_planning": False,
            "external_modes_mutated_pf": False,
        },
        "provenance": {
            "estimator_family": "particle_filter",
            "base_estimator_variant": estimator.estimator_variant,
            "planning_variant": "hybrid_dsspp_planner_only_modes_v1",
            "profile": profile,
            "repository_commit": estimator.repository_commit,
            "pf_resolved_config_sha256": estimator.resolved_config_hash,
            "dsspp_config_sha256": sha256_json(asdict(request.dsspp_config)),
            "causal_planning_request_sha256": request_artifact_sha256,
            "candidate_poses_sha256": (
                request.candidate_attestation.candidate_poses_sha256
            ),
            "workspace_sha256": request.candidate_attestation.workspace_sha256,
            "planning_config_sha256": (
                request.candidate_attestation.planning_config_sha256
            ),
            "pf_seed": int(seed),
            "relocation_seed": int(
                seed if relocation_seed is None else relocation_seed
            ),
            "map_api_used_by_pf_boundary": False,
            "upstream_collision_attestation_required": True,
        },
    }


def recommend_hybrid_dsspp_action(
    measurement_log: str | Path,
    config: str | Path | Mapping[str, Any],
    planning_request: str | Path | Mapping[str, Any],
    *,
    directive_schedule: str | Path | Mapping[str, Any] | None = None,
    profile: str = "pf_strict",
    seed: int = 0,
    relocation_seed: int | None = None,
) -> dict[str, Any]:
    """Replay one causal prefix and return a non-actuating DSS-PP recommendation."""
    raw_request = _load_json_mapping(planning_request)
    request_artifact_sha256 = sha256_json(raw_request)
    request = planning_request_from_mapping(raw_request)
    log = load_measurement_log(measurement_log)
    cutoff_index = _cutoff_record_index(log, request)
    causal_schedule = _causal_directive_schedule(
        directive_schedule,
        cutoff_record_index=cutoff_index,
        cutoff_step=request.data_cutoff_step,
    )
    prefix_log = log.prefix(cutoff_index + 1)
    estimator, _trace, schedule, _predictions = replay_with_external_relocations(
        prefix_log,
        config,
        causal_schedule,
        profile=profile,
        seed=int(seed),
        relocation_seed=relocation_seed,
    )
    if request.pf_resolved_config_sha256 != str(estimator.resolved_config_hash):
        raise HybridPlanningBoundaryError(
            "Planning request resolved PF config digest mismatch."
        )
    _validate_modes_for_estimator(estimator, request.external_modes)
    included_modes = tuple(
        mode
        for mode in request.external_modes
        if mode.verification_state
        in {
            ExternalModeVerificationState.PENDING,
            ExternalModeVerificationState.VERIFIED,
        }
    )
    planning_view = _ExternalPlanningEstimatorView(estimator, included_modes)
    state_before = sha256(estimator.serialized_state()).hexdigest()
    bounds = (
        None
        if request.bounds_xyz is None
        else (
            np.asarray(request.bounds_xyz[0], dtype=float),
            np.asarray(request.bounds_xyz[1], dtype=float),
        )
    )
    planning_result = select_dss_pp_next_station(
        planning_view,
        np.asarray(request.candidate_poses_xyz, dtype=np.float64),
        np.asarray(request.current_pose_xyz, dtype=np.float64),
        current_pair_id=request.current_pair_id,
        visited_poses_xyz=np.asarray(
            request.visited_poses_xyz, dtype=np.float64
        ).reshape(-1, 3),
        map_api=None,
        bounds_xyz=bounds,
        continuous_height_bounds_m=request.continuous_height_bounds_m,
        config=request.dsspp_config,
        _planner_only_external_mode_token=(_HYBRID_RECOMMENDATION_EXTERNAL_MODE_TOKEN),
    )
    state_after = sha256(estimator.serialized_state()).hexdigest()
    if state_before != state_after:
        raise HybridPlanningBoundaryError(
            "DSS-PP planning mutated the pure PF particle state."
        )
    return _recommendation_payload(
        request=request,
        request_artifact_sha256=request_artifact_sha256,
        estimator=estimator,
        planning_result=planning_result,
        schedule=schedule,
        causal_schedule=causal_schedule,
        state_sha256_before=state_before,
        state_sha256_after=state_after,
        profile=profile,
        seed=int(seed),
        relocation_seed=relocation_seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the opt-in hybrid DSS-PP recommendation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--planning-request", required=True, type=Path)
    parser.add_argument("--directive-schedule", type=Path)
    parser.add_argument(
        "--profile",
        choices=("pf_strict", "pf_profiled"),
        default="pf_strict",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relocation-seed", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(None if argv is None else list(argv))
    result = recommend_hybrid_dsspp_action(
        args.measurement_log,
        args.config,
        args.planning_request,
        directive_schedule=args.directive_schedule,
        profile=args.profile,
        seed=args.seed,
        relocation_seed=args.relocation_seed,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace recommendation {args.output}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

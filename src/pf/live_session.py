"""PF construction and record ingestion for one live runtime session."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from measurement.surface_atlas import ContinuousSurfaceAtlas
from measurement.surface_charts import build_surface_chart_geometry
from numpy.typing import NDArray
from runtime.adaptive_client import (
    AdaptiveCandidateSnapshot,
    AdaptiveStepRequest,
)
from runtime.contracts import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from runtime.artifacts import (
    AtomicBundlePublisher,
    atomic_write_bytes,
    build_artifact_inventory,
    publish_artifact_manifest,
)
from runtime.forward_context import ResolvedForwardContext
from runtime.experiment_profiles import acquisition_contract_from_environment
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    MeasurementLogValidationError,
    MeasurementLogView,
)
from runtime.prefix import measurement_records_digest
from runtime.provenance import DigestIdentity
from runtime.records import RunContext
from scipy.spatial import cKDTree

from pf.estimator_types import JointPlanningParticles
from pf.control_policy import PFControlPolicyProvenance
from pf.gpu_utils import (
    preflight_compute_backend,
    preflight_cuda_allocation_capacity,
)
from pf.joint_transport_cache import (
    JOINT_EXACT_MAX_STATIONS,
    JOINT_EXACT_MAX_VIEWS,
    JointTransportCache,
)
from pf.configuration import PFConfigDocument, load_pf_config
from pf.profiles import (
    apply_profile_to_config,
    enforce_pure_runtime_settings,
    production_pf_config_values,
)
from pf.provenance import (
    strict_canonical_json_bytes,
    strict_json_loads,
    strict_sha256_json,
)
from pf.pure_estimator import PurePFEstimator, RotatingShieldPFConfig


class PFLiveSessionError(RuntimeError):
    """Report an incompatible live context, setting, or observation."""


def _exact_nonempty_string(value: object, *, location: str) -> str:
    """Return one nonempty string without trimming or stringification."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location} must be an exact nonempty string.")
    return value


def _exact_isotope_order(
    value: object,
    *,
    location: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Return one exact unique tuple of canonical isotope strings."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{location} must be a sequence of isotope strings.")
    isotopes = tuple(
        _exact_nonempty_string(item, location=f"{location} item") for item in value
    )
    if (not allow_empty and not isotopes) or len(set(isotopes)) != len(isotopes):
        raise ValueError(f"{location} must contain unique nonempty strings.")
    return isotopes


def _strict_live_artifact_json_bytes(
    value: object,
    *,
    artifact_name: str,
) -> bytes:
    """Serialize one live artifact without any implicit value coercion."""
    try:
        return strict_canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError(
            f"{artifact_name} must contain only strict finite JSON values."
        ) from exc


def _strict_live_artifact_sha256(
    value: object,
    *,
    artifact_name: str,
) -> str:
    """Hash one live artifact without any implicit value coercion."""
    try:
        return strict_sha256_json(value)
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError(
            f"{artifact_name} must contain only strict finite JSON values."
        ) from exc


@dataclass(frozen=True, slots=True)
class PFLiveParticleSnapshot:
    """Expose one immutable truth-free particle generation for planning."""

    source_run_id: str
    record_count: int
    station_count: int
    covered_records_digest: DigestIdentity
    isotope_order: tuple[str, ...]
    weights_n: NDArray[np.float64]
    positions_nk3_by_isotope: Mapping[str, NDArray[np.float64]]
    surface_chart_ids_nk_by_isotope: Mapping[str, NDArray[np.int64]]
    surface_uv_nk2_by_isotope: Mapping[str, NDArray[np.float64]]
    strengths_nk_by_isotope: Mapping[str, NDArray[np.float64]]
    source_mask_nk_by_isotope: Mapping[str, NDArray[np.bool_]]
    original_particle_indices: NDArray[np.int64]
    posterior_summary_json: bytes
    posterior_summary_sha256: str

    def posterior_summary(self) -> dict[str, object]:
        """Return a detached JSON-compatible copy of the live PF summary."""
        payload = strict_json_loads(self.posterior_summary_json)
        if not isinstance(payload, dict):
            raise PFLiveSessionError("PF live posterior summary must be an object.")
        return payload


@dataclass(frozen=True, slots=True)
class PFExternalSurfaceGuidance:
    """Carry causal surface density used only as an exact-RJ proposal guide."""

    source_run_id: str
    record_count: int
    data_cutoff_step: int
    data_cutoff_station: int
    covered_records_digest: DigestIdentity
    isotope_order: tuple[str, ...]
    patch_centroids_xyz: NDArray[np.float64]
    density_by_isotope: NDArray[np.float64]
    proposal_mass: float
    bandwidth_m: float

    def __post_init__(self) -> None:
        """Validate lineage and freeze the estimator-neutral surface arrays."""
        run_id = _exact_nonempty_string(
            self.source_run_id,
            location="source_run_id",
        )
        record_count = _json_integer(
            self.record_count,
            location="record_count",
            minimum=1,
        )
        cutoff_step = _json_integer(
            self.data_cutoff_step,
            location="data_cutoff_step",
            minimum=0,
        )
        cutoff_station = _json_integer(
            self.data_cutoff_station,
            location="data_cutoff_station",
            minimum=0,
        )
        if not isinstance(self.covered_records_digest, DigestIdentity):
            raise TypeError("covered_records_digest must be a DigestIdentity.")
        isotope_order = _exact_isotope_order(
            self.isotope_order,
            location="isotope_order",
        )
        centroids = np.asarray(self.patch_centroids_xyz, dtype=np.float64)
        density = np.asarray(self.density_by_isotope, dtype=np.float64)
        if (
            centroids.ndim != 2
            or centroids.shape[0] < 1
            or centroids.shape[1] != 3
            or np.any(~np.isfinite(centroids))
        ):
            raise ValueError(
                "patch_centroids_xyz must be a non-empty finite (P, 3) array."
            )
        if (
            density.shape != (len(isotope_order), centroids.shape[0])
            or np.any(~np.isfinite(density))
            or np.any(density < 0.0)
        ):
            raise ValueError(
                "density_by_isotope must be finite, non-negative, and shaped (I, P)."
            )
        proposal_mass = _finite_real(self.proposal_mass, location="proposal_mass")
        bandwidth_m = _finite_real(self.bandwidth_m, location="bandwidth_m")
        if proposal_mass <= 0.0 or proposal_mass > 1.0:
            raise ValueError("proposal_mass must lie in (0, 1].")
        if bandwidth_m <= 0.0:
            raise ValueError("bandwidth_m must be positive.")
        immutable_centroids = np.frombuffer(
            np.ascontiguousarray(centroids, dtype=np.float64).tobytes(),
            dtype=np.float64,
        ).reshape(centroids.shape)
        immutable_density = np.frombuffer(
            np.ascontiguousarray(density, dtype=np.float64).tobytes(),
            dtype=np.float64,
        ).reshape(density.shape)
        object.__setattr__(self, "source_run_id", run_id)
        object.__setattr__(self, "record_count", record_count)
        object.__setattr__(self, "data_cutoff_step", cutoff_step)
        object.__setattr__(self, "data_cutoff_station", cutoff_station)
        object.__setattr__(self, "isotope_order", isotope_order)
        object.__setattr__(self, "patch_centroids_xyz", immutable_centroids)
        object.__setattr__(self, "density_by_isotope", immutable_density)
        object.__setattr__(self, "proposal_mass", proposal_mass)
        object.__setattr__(self, "bandwidth_m", bandwidth_m)

    @property
    def guidance_sha256(self) -> str:
        """Return a deterministic digest over lineage and all proposal inputs."""
        digest = sha256(b"pf_external_surface_guidance_v1\0")
        digest.update(self.source_run_id.encode("utf-8"))
        digest.update(
            np.asarray(
                [
                    self.record_count,
                    self.data_cutoff_step,
                    self.data_cutoff_station,
                ],
                dtype="<i8",
            ).tobytes()
        )
        digest.update(self.covered_records_digest.algorithm.encode("utf-8"))
        digest.update(self.covered_records_digest.sha256.encode("ascii"))
        for isotope in self.isotope_order:
            encoded = isotope.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        for values in (self.patch_centroids_xyz, self.density_by_isotope):
            array = np.ascontiguousarray(values, dtype="<f8")
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
        digest.update(
            np.asarray(
                [self.proposal_mass, self.bandwidth_m],
                dtype="<f8",
            ).tobytes()
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PFExternalSurfaceGuidanceReceipt:
    """Describe one MLE-style surface guide consumed by a PF station update."""

    guidance_sha256: str
    source_run_id: str
    record_count: int
    data_cutoff_step: int
    data_cutoff_station: int
    covered_records_digest: DigestIdentity
    proposal_mass: float
    bandwidth_m: float
    informative_isotopes: tuple[str, ...]
    evaluated_isotopes: tuple[str, ...]
    mapped_chart_count: int
    target_preserving: bool = True
    direct_weight_update: bool = False

    def __post_init__(self) -> None:
        """Validate the immutable audit receipt returned after PF evaluation."""
        guidance_sha256 = _sha256_string(
            self.guidance_sha256,
            location="guidance_sha256",
        )
        source_run_id = _exact_nonempty_string(
            self.source_run_id,
            location="source_run_id",
        )
        record_count = _json_integer(
            self.record_count,
            location="record_count",
            minimum=1,
        )
        cutoff_step = _json_integer(
            self.data_cutoff_step,
            location="data_cutoff_step",
            minimum=0,
        )
        cutoff_station = _json_integer(
            self.data_cutoff_station,
            location="data_cutoff_station",
            minimum=0,
        )
        if not isinstance(self.covered_records_digest, DigestIdentity):
            raise TypeError("covered_records_digest must be a DigestIdentity.")
        proposal_mass = _finite_real(self.proposal_mass, location="proposal_mass")
        bandwidth_m = _finite_real(self.bandwidth_m, location="bandwidth_m")
        if proposal_mass <= 0.0 or proposal_mass > 1.0:
            raise ValueError("proposal_mass must lie in (0, 1].")
        if bandwidth_m <= 0.0:
            raise ValueError("bandwidth_m must be positive.")
        informative = _exact_isotope_order(
            self.informative_isotopes,
            location="informative_isotopes",
            allow_empty=True,
        )
        evaluated = _exact_isotope_order(
            self.evaluated_isotopes,
            location="evaluated_isotopes",
        )
        if (
            not set(informative).issubset(evaluated)
        ):
            raise ValueError("Receipt isotope lists must be unique and consistent.")
        mapped_chart_count = _json_integer(
            self.mapped_chart_count,
            location="mapped_chart_count",
            minimum=1,
        )
        if self.target_preserving is not True or self.direct_weight_update is not False:
            raise ValueError(
                "Surface guidance must preserve the target and PF weights."
            )
        object.__setattr__(self, "guidance_sha256", guidance_sha256)
        object.__setattr__(self, "source_run_id", source_run_id)
        object.__setattr__(self, "record_count", record_count)
        object.__setattr__(self, "data_cutoff_step", cutoff_step)
        object.__setattr__(self, "data_cutoff_station", cutoff_station)
        object.__setattr__(self, "proposal_mass", proposal_mass)
        object.__setattr__(self, "bandwidth_m", bandwidth_m)
        object.__setattr__(self, "informative_isotopes", informative)
        object.__setattr__(self, "evaluated_isotopes", evaluated)
        object.__setattr__(self, "mapped_chart_count", mapped_chart_count)


@dataclass(frozen=True, slots=True)
class PFCompletedLiveState:
    """Seal a completed live posterior before MeasurementLog publication."""

    source_run_id: str
    runtime_config_sha256: str
    generative_contract_hash_sha256: str
    record_count: int
    station_count: int
    covered_step_ids: tuple[int, ...]
    covered_records_digest: DigestIdentity
    control_policy_provenance: PFControlPolicyProvenance
    checkpoint_state: bytes
    checkpoint_sha256: str
    diagnostics_json: bytes
    particle_snapshot: PFLiveParticleSnapshot


@dataclass(frozen=True, slots=True)
class PFBoundLiveState:
    """Provide immutable publication inputs after exact final-log binding."""

    completed: PFCompletedLiveState
    measurement_log_sha256: str
    posterior_json: bytes
    posterior_sha256: str

    @property
    def checkpoint_state(self) -> bytes:
        """Return the already-completed PF checkpoint without recomputation."""
        return self.completed.checkpoint_state

    @property
    def checkpoint_sha256(self) -> str:
        """Return the digest of the already-completed PF checkpoint."""
        return self.completed.checkpoint_sha256


@dataclass(frozen=True, slots=True)
class PFPublishedLiveResult:
    """Identify one package-owned PF result published after exact log binding."""

    root: Path
    posterior_path: Path
    diagnostics_path: Path
    checkpoint_path: Path
    checkpoint_state_path: Path
    particle_snapshot_path: Path
    post_run_evaluation_input_path: Path
    posterior_sha256: str
    checkpoint_sha256: str
    result_sha256: str


_SHA256_PATTERN = frozenset("0123456789abcdef")
_DEFAULT_SURFACE_DIAGNOSTIC_POINT_COUNT = 1024
_PRODUCTION_CONFIG_VALIDATION_TOKEN = object()
_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE = 1.0e-10


def validate_production_live_pf_config(
    config: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """Validate the complete production-live schema without filling defaults."""
    if not isinstance(config, Mapping) or any(
        not isinstance(key, str) for key in config
    ):
        raise PFLiveSessionError("Live PF configuration must be a string-keyed object.")
    try:
        validated = enforce_pure_runtime_settings(config, profile=profile)
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError(
            f"Production live PF configuration is incompatible: {exc}"
        ) from exc
    return validated


@dataclass(frozen=True, slots=True)
class ValidatedProductionPFConfig:
    """Bind validated production settings to the exact loaded source bytes."""

    document: PFConfigDocument
    profile: str
    _validation_token: object = field(repr=False, compare=False)
    _settings_json: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the document and freeze its resolved production settings."""
        if self._validation_token is not _PRODUCTION_CONFIG_VALIDATION_TOKEN:
            raise PFLiveSessionError(
                "ValidatedProductionPFConfig values may only be created by "
                "load_production_live_pf_config()."
            )
        if not isinstance(self.document, PFConfigDocument):
            raise TypeError("document must be a PFConfigDocument.")
        validated = validate_production_live_pf_config(
            self.document.config(),
            profile=self.profile,
        )
        object.__setattr__(
            self,
            "_settings_json",
            _strict_live_artifact_json_bytes(
                validated,
                artifact_name="Production live PF configuration",
            ),
        )

    @property
    def source_sha256(self) -> str:
        """Return the digest of the exact input file bytes."""
        return self.document.source_sha256

    def settings(self) -> dict[str, Any]:
        """Return a detached copy of the validated production settings."""
        payload = strict_json_loads(self._settings_json)
        if not isinstance(payload, dict):  # pragma: no cover - constructor invariant.
            raise PFLiveSessionError("Validated PF settings are not an object.")
        return payload


def load_production_live_pf_config(
    path: str | Path,
    *,
    profile: str,
) -> ValidatedProductionPFConfig:
    """Load and validate one complete production-live schema-v2 file."""
    try:
        document = load_pf_config(path)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(f"Cannot load live PF configuration: {exc}") from exc
    return ValidatedProductionPFConfig(
        document=document,
        profile=profile,
        _validation_token=_PRODUCTION_CONFIG_VALIDATION_TOKEN,
    )


def _sha256_string(value: object, *, location: str) -> str:
    """Return one exact lowercase SHA-256 string without coercion."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_PATTERN for character in value)
    ):
        raise PFLiveSessionError(
            f"{location} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _json_integer(
    value: object,
    *,
    location: str,
    minimum: int | None = None,
) -> int:
    """Return one exact JSON integer without boolean or float coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PFLiveSessionError(f"{location} must be a JSON integer.")
    if minimum is not None and value < minimum:
        raise PFLiveSessionError(f"{location} must be at least {minimum}.")
    return value


def _finite_real(value: object, *, location: str) -> float:
    """Return one exact finite real without boolean or string coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PFLiveSessionError(f"{location} must be a finite JSON number.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise PFLiveSessionError(f"{location} must be a finite JSON number.")
    return parsed


def _integer_keyed_cardinality_distribution(
    value: object,
    *,
    location: str,
) -> dict[str, float]:
    """Encode one scientific integer-keyed cardinality law as strict JSON."""
    if not isinstance(value, Mapping) or not value:
        raise PFLiveSessionError(f"{location} must be a nonempty object.")
    normalized: dict[str, float] = {}
    for raw_cardinality, raw_probability in value.items():
        cardinality = _json_integer(
            raw_cardinality,
            location=f"{location} cardinality",
            minimum=0,
        )
        normalized[str(cardinality)] = _finite_real(
            raw_probability,
            location=f"{location} probability {cardinality}",
        )
    return normalized


def _json_cardinality_distribution(
    value: object,
    *,
    location: str,
) -> dict[str, float]:
    """Validate one already JSON-keyed cardinality law without coercion."""
    if not isinstance(value, Mapping) or not value:
        raise PFLiveSessionError(f"{location} must be a nonempty object.")
    normalized: dict[str, float] = {}
    for cardinality, raw_probability in value.items():
        if (
            not isinstance(cardinality, str)
            or not cardinality.isascii()
            or not cardinality.isdigit()
            or cardinality != str(int(cardinality))
        ):
            raise PFLiveSessionError(
                f"{location} keys must be canonical nonnegative integer strings."
            )
        normalized[cardinality] = _finite_real(
            raw_probability,
            location=f"{location} probability {cardinality}",
        )
    return normalized


def _compact_pf_diagnostics(estimator: object) -> dict[str, object]:
    """Build compact final diagnostics without duplicating posterior provenance."""
    convergence_method = getattr(
        estimator,
        "posterior_convergence_diagnostics",
        None,
    )
    predictive_method = getattr(estimator, "posterior_predictive_check", None)
    if not callable(convergence_method) or not callable(predictive_method):
        raise PFLiveSessionError(
            "PF estimator does not expose final diagnostic methods."
        )
    raw_convergence = convergence_method()
    raw_predictive = predictive_method()
    if not isinstance(raw_convergence, Mapping) or not isinstance(
        raw_predictive,
        Mapping,
    ):
        raise PFLiveSessionError("PF final diagnostics must serialize objects.")
    convergence = dict(raw_convergence)
    if "sampler_health" not in convergence:
        raise PFLiveSessionError("PF convergence omits sampler_health.")
    sampler_health = convergence.pop("sampler_health")
    if not isinstance(sampler_health, Mapping):
        raise PFLiveSessionError("PF sampler health must be an object.")
    expected_sampler_health = {
        "smc_rejuvenation_wall_time_respected",
        "rejuvenation_mixing_complete",
        "structural_mixing_complete",
    }
    if set(sampler_health) != expected_sampler_health or any(
        type(sampler_health[name]) is not bool for name in expected_sampler_health
    ):
        raise PFLiveSessionError(
            "PF sampler health must contain exactly the three Boolean gates."
        )
    sampler_quality_reasons = sorted(
        name for name in expected_sampler_health if sampler_health[name] is not True
    )
    isotope_rows = convergence.get("isotopes")
    if not isinstance(isotope_rows, Mapping) or not isotope_rows:
        raise PFLiveSessionError("PF convergence requires nonempty isotope rows.")
    normalized_isotope_rows: dict[str, object] = {}
    hard_cap_failures: list[str] = []
    for isotope, raw_row in isotope_rows.items():
        if not isinstance(isotope, str) or not isotope:
            raise PFLiveSessionError(
                "PF convergence isotope keys must be nonempty strings."
            )
        if not isinstance(raw_row, Mapping):
            raise PFLiveSessionError(
                f"PF convergence isotope row {isotope} must be an object."
            )
        row = dict(raw_row)
        row["cardinality_distribution"] = (
            _integer_keyed_cardinality_distribution(
                row.get("cardinality_distribution"),
                location=f"PF convergence {isotope} cardinality distribution",
            )
        )
        hard_cap_mass = row.get("hard_cap_posterior_mass")
        hard_cap_limit = row.get("hard_cap_posterior_mass_limit")
        hard_cap_count = row.get("hard_cap_source_count")
        if (
            isinstance(hard_cap_count, bool)
            or not isinstance(hard_cap_count, int)
            or hard_cap_count <= 0
            or isinstance(hard_cap_mass, bool)
            or not isinstance(hard_cap_mass, (int, float))
            or isinstance(hard_cap_limit, bool)
            or not isinstance(hard_cap_limit, (int, float))
            or not np.isfinite(float(hard_cap_mass))
            or not np.isfinite(float(hard_cap_limit))
            or not 0.0 <= float(hard_cap_mass) <= 1.0 + 1.0e-12
            or not 0.0 <= float(hard_cap_limit) < 1.0
        ):
            raise PFLiveSessionError(
                f"PF convergence {isotope} has invalid hard-cap evidence."
            )
        if float(hard_cap_mass) > float(hard_cap_limit):
            hard_cap_failures.append(isotope)
        normalized_isotope_rows[isotope] = row
    convergence["isotopes"] = normalized_isotope_rows
    joint_gates = convergence.get("joint_gates")
    if isinstance(joint_gates, Mapping):
        if any(not isinstance(name, str) for name in joint_gates):
            raise PFLiveSessionError(
                "PF convergence joint-gate keys must be strings."
            )
        convergence["joint_gates"] = {
            name: value
            for name, value in joint_gates.items()
            if name not in sampler_health
        }
    sampler_quality_reasons.extend(
        f"hard_cap_posterior_mass_exceeded.{isotope}"
        for isotope in sorted(hard_cap_failures)
    )
    sampler_quality_status = (
        "failed"
        if hard_cap_failures
        else "warning"
        if sampler_quality_reasons
        else "pass"
    )
    return {
        "schema_version": 3,
        "estimator_family": "pure_particle_filter",
        "execution_status": "complete",
        "sampler_quality_status": sampler_quality_status,
        "sampler_quality_reasons": sorted(set(sampler_quality_reasons)),
        "posterior_convergence": convergence,
        "posterior_predictive_check": dict(raw_predictive),
        "sampler_health": dict(sampler_health),
    }


def _surface_atlas_diagnostic_points(
    environment: EnvironmentConfig,
    *,
    pf_config: RotatingShieldPFConfig,
    obstacle_grid: ObstacleGrid | None,
    obstacle_height_m: float,
) -> NDArray[np.float64]:
    """Build deterministic PF diagnostics from shared physical geometry."""
    point_count = _DEFAULT_SURFACE_DIAGNOSTIC_POINT_COUNT
    try:
        geometry = build_surface_chart_geometry(
            environment,
            obstacle_grid,
            max_edge_m=pf_config.structural_rj_surface_chart_max_edge_m,
            obstacle_height_m=obstacle_height_m,
        )
        if not geometry.obstacle_surfaces_available:
            raise ValueError("complete obstacle surfaces are unavailable")
        atlas = ContinuousSurfaceAtlas(geometry)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            "Cannot reconstruct the runtime continuous surface atlas."
        ) from exc
    quantiles = (np.arange(point_count, dtype=np.float64) + 0.5) / float(point_count)
    chart_ids = np.searchsorted(
        np.cumsum(atlas.chart_probabilities),
        quantiles,
        side="right",
    ).astype(np.int64)
    if np.any(chart_ids < 0) or np.any(chart_ids >= atlas.chart_count):
        raise PFLiveSessionError("Surface-atlas diagnostic chart IDs are invalid.")
    sequence = np.arange(point_count, dtype=np.float64) + 0.5
    uv = np.column_stack(
        (
            np.mod(sequence * ((np.sqrt(5.0) - 1.0) / 2.0), 1.0),
            np.mod(sequence * (np.sqrt(2.0) - 1.0), 1.0),
        )
    )
    return np.ascontiguousarray(atlas.positions_xyz(chart_ids, uv))


def _live_session_hash_payload(
    *,
    runtime_config_sha256: str,
    measurement_log_sha256: str,
    actual_pf: Mapping[str, object],
    random_seed: int,
    control_policy_provenance: PFControlPolicyProvenance,
) -> dict[str, object]:
    """Return the single canonical resolved-session hash input contract."""
    if not isinstance(control_policy_provenance, PFControlPolicyProvenance):
        raise PFLiveSessionError(
            "Live PF construction requires sealed control-policy provenance."
        )
    return {
        "measurement_runtime_config_sha256": runtime_config_sha256,
        "measurement_log_sha256": measurement_log_sha256,
        "pf_config": dict(actual_pf),
        "pf_random_seed": random_seed,
        "control_policy": control_policy_provenance.to_dict(),
    }


def _sealed_live_checkpoint_state(
    estimator: object,
    control_policy_provenance: PFControlPolicyProvenance,
) -> bytes:
    """Bind immutable control provenance into one serialized PF state."""
    raw_checkpoint = estimator.serialized_state()
    if not isinstance(raw_checkpoint, (bytes, bytearray, memoryview)):
        raise PFLiveSessionError("PF serialized_state() must return bytes.")
    try:
        payload = strict_json_loads(bytes(raw_checkpoint))
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError("PF serialized state must be strict JSON.") from exc
    if not isinstance(payload, dict):
        raise PFLiveSessionError("PF serialized state must be a JSON object.")
    estimator_schema_version = payload.get("schema_version")
    if type(estimator_schema_version) is not int or estimator_schema_version != 1:
        raise PFLiveSessionError(
            "PF estimator state must use exact internal schema version 1."
        )
    if "control_policy" in payload:
        raise PFLiveSessionError(
            "PF serialized state cannot replace session control provenance."
        )
    payload["schema_version"] = 2
    payload["estimator_state_schema_version"] = estimator_schema_version
    payload["control_policy"] = control_policy_provenance.to_dict()
    return _strict_live_artifact_json_bytes(
        payload,
        artifact_name="PF sealed checkpoint state",
    )


def _build_live_estimator_from_forward_context(
    forward: ResolvedForwardContext,
    config: ValidatedProductionPFConfig,
    *,
    profile: str,
    seed: int,
    measurement_log_schema_version: int,
    measurement_runtime_config_sha256: str,
    control_policy_provenance: PFControlPolicyProvenance,
) -> PurePFEstimator:
    """Build one live PF from an authenticated shared physical context."""
    if not isinstance(config, ValidatedProductionPFConfig):
        raise PFLiveSessionError(
            "Live estimator construction requires a validated config document."
        )
    settings = config.settings()
    if not isinstance(profile, str):
        raise PFLiveSessionError("PF profile must be a JSON string.")
    session_seed = _json_integer(seed, location="seed", minimum=0)
    schema_version = _json_integer(
        measurement_log_schema_version,
        location="measurement_log_schema_version",
        minimum=1,
    )
    runtime_config_sha256 = _sha256_string(
        measurement_runtime_config_sha256,
        location="measurement_runtime_config_sha256",
    )
    pending_log_digest = "unavailable"
    isotopes = tuple(forward.isotopes)
    if not isotopes or len(set(isotopes)) != len(isotopes):
        raise PFLiveSessionError(
            "Runtime candidate isotopes must be unique and nonempty."
        )
    _, upper = forward.bounds_xyz
    try:
        pure_config = validate_production_live_pf_config(settings, profile=profile)
        pf_config = RotatingShieldPFConfig(
            **production_pf_config_values(
                pure_config,
                position_max=tuple(float(value) for value in upper),
            )
        )
        apply_profile_to_config(pf_config)
        preflight_compute_backend(
            use_gpu=bool(pf_config.use_gpu),
            gpu_device=str(pf_config.gpu_device),
            gpu_dtype=str(pf_config.gpu_dtype),
        )
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError("External PF configuration is incompatible.") from exc
    observation_model = forward.observation_model
    acquisition_contract = acquisition_contract_from_environment(
        forward.environment_payload
    )
    if (
        acquisition_contract.max_stations > JOINT_EXACT_MAX_STATIONS
        or acquisition_contract.max_measurements > JOINT_EXACT_MAX_VIEWS
    ):
        raise PFLiveSessionError(
            "The runtime acquisition contract exceeds the fixed 16-station/"
            "128-view exact PF capacity."
        )
    source_capacity = pf_config.cardinality_capacity
    if (
        isinstance(source_capacity, bool)
        or not isinstance(source_capacity, int)
        or source_capacity <= 0
    ):
        raise PFLiveSessionError(
            "Production PF requires a positive fixed source-slot capacity."
        )
    model = forward.spectral_model
    required_cache_bytes = JointTransportCache.required_storage_bytes(
        particle_count=int(pf_config.num_particles),
        max_views=JOINT_EXACT_MAX_VIEWS,
        source_slots=len(isotopes) * source_capacity,
        line_count=len(tuple(model.line_identity)),
        feature_count=len(tuple(model.transport_feature_order)),
        max_stations=JOINT_EXACT_MAX_STATIONS,
        dtype_bytes=8,
    )
    reindex_scratch_bytes = JointTransportCache.reindex_scratch_bytes(
        particle_count=int(pf_config.num_particles),
        source_slots=len(isotopes) * source_capacity,
        line_count=len(tuple(model.line_identity)),
        feature_count=len(tuple(model.transport_feature_order)),
        max_stations=JOINT_EXACT_MAX_STATIONS,
        dtype_bytes=8,
    )
    minimum_state_chunk = min(32, int(pf_config.num_particles))
    minimum_likelihood_workspace_bytes = int(
        model.estimate_cross_likelihood_working_set_bytes(
            num_actions=acquisition_contract.max_stations,
            num_samples=1,
            num_particles=int(pf_config.num_particles),
            num_isotopes=len(isotopes) * source_capacity,
            num_views=acquisition_contract.views_per_station,
            state_chunk_size=minimum_state_chunk,
            dtype_bytes=8,
        )
    )
    minimum_overlay_scratch_bytes = int(
        acquisition_contract.max_stations
        * minimum_state_chunk
        * acquisition_contract.views_per_station
        * len(isotopes)
        * source_capacity
        * len(tuple(model.line_identity))
        * (2 + len(tuple(model.transport_feature_order)))
        * 8
    )
    minimum_overlay_workspace_bytes = (
        minimum_likelihood_workspace_bytes + minimum_overlay_scratch_bytes
    )
    required_live_cuda_bytes = (
        required_cache_bytes
        + reindex_scratch_bytes
        + 2 * minimum_overlay_workspace_bytes
    )
    try:
        cache_preflight = preflight_cuda_allocation_capacity(
            device=str(pf_config.gpu_device),
            required_bytes=required_live_cuda_bytes,
            allocation_name="fixed cache and minimum exact overlay workspace",
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            "The fixed joint exact transport cache failed CUDA preflight."
        ) from exc
    obstacle_grid = forward.obstacle_grid
    obstacle_enabled = forward.obstacle_attenuation_enabled
    if obstacle_grid is not None and not obstacle_enabled:
        raise PFLiveSessionError(
            "A runtime obstacle grid requires physical obstacle attenuation."
        )
    pf_obstacle_grid = obstacle_grid if obstacle_enabled else None
    surface_diagnostic_points = _surface_atlas_diagnostic_points(
        forward.environment,
        pf_config=pf_config,
        obstacle_grid=pf_obstacle_grid,
        obstacle_height_m=observation_model.obstacle_height_m,
    )
    actual_pf = asdict(pf_config)
    session_hash_payload = _live_session_hash_payload(
        runtime_config_sha256=runtime_config_sha256,
        measurement_log_sha256=pending_log_digest,
        actual_pf=actual_pf,
        random_seed=session_seed,
        control_policy_provenance=control_policy_provenance,
    )
    resolved_session_config_sha256 = _strict_live_artifact_sha256(
        session_hash_payload,
        artifact_name="PF resolved session identity",
    )
    input_config_sha256 = _sha256_string(
        config.source_sha256,
        location="config.source_sha256",
    )
    estimator = PurePFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=surface_diagnostic_points,
        shield_normals=generate_octant_orientations(),
        pf_config=pf_config,
        observation_model=observation_model,
        obstacle_grid=pf_obstacle_grid,
        full_spectrum_generative_model=forward.spectral_model,
        measurement_log_schema_version=schema_version,
        config_hash=input_config_sha256,
        resolved_config_hash=resolved_session_config_sha256,
        measurement_log_sha256=pending_log_digest,
        random_seed=session_seed,
    )
    estimator.joint_transport_cache_preflight = dict(cache_preflight)
    estimator.joint_transport_cache_preflight.update(
        {
            "cache_required_bytes": int(required_cache_bytes),
            "reindex_scratch_bytes": int(reindex_scratch_bytes),
            "minimum_state_chunk": int(minimum_state_chunk),
            "minimum_overlay_workspace_bytes": int(
                minimum_overlay_workspace_bytes
            ),
        }
    )
    environment = forward.environment
    assert environment.detector_position is not None
    initial_pose = np.asarray(environment.detector_position, dtype=np.float64)
    estimator.add_measurement_pose(initial_pose, reset_filters=False)
    try:
        allocated_cache_bytes = estimator.initialize_joint_exact_transport_cache()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            "The fixed joint exact transport cache could not be reserved "
            "before acquisition."
        ) from exc
    if allocated_cache_bytes != required_cache_bytes:
        raise PFLiveSessionError(
            "Allocated joint transport cache bytes differ from preflight."
        )
    estimator.joint_transport_cache_preflight["allocated_bytes"] = int(
        allocated_cache_bytes
    )
    remaining_workspace_bytes = (
        reindex_scratch_bytes + 2 * minimum_overlay_workspace_bytes
    )
    try:
        post_allocation_preflight = preflight_cuda_allocation_capacity(
            device=str(pf_config.gpu_device),
            required_bytes=remaining_workspace_bytes,
            allocation_name="post-cache exact PF workspace",
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            "The fixed cache left insufficient CUDA memory for exact PF "
            "workspace before acquisition."
        ) from exc
    estimator.joint_transport_cache_preflight[
        "post_allocation_workspace_preflight"
    ] = dict(post_allocation_preflight)
    return estimator


def build_live_estimator(
    context: RunContext,
    config: ValidatedProductionPFConfig,
    *,
    profile: str,
    seed: int,
    runtime_root: str | Path,
    control_policy_provenance: PFControlPolicyProvenance,
) -> PurePFEstimator:
    """Construct a PF from a truth-free live runtime handshake."""
    root = Path(runtime_root).expanduser().resolve()
    try:
        forward = ResolvedForwardContext.from_run_context(context, run_root=root)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            f"Cannot resolve the live runtime-authenticated forward context: {exc}"
        ) from exc
    return _build_live_estimator_from_forward_context(
        forward,
        config,
        profile=profile,
        seed=seed,
        measurement_log_schema_version=context.schema_version,
        measurement_runtime_config_sha256=context.runtime_config_sha256,
        control_policy_provenance=control_policy_provenance,
    )


def _validate_published_forward_context(log: MeasurementLog) -> None:
    """Validate the runtime-authenticated context of the published live log."""
    try:
        ResolvedForwardContext.from_log(log)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            f"Cannot validate the published runtime forward context: {exc}"
        ) from exc


def _bind_estimator_to_published_log(
    estimator: PurePFEstimator,
    log: MeasurementLog,
    *,
    live_records: Sequence[MeasurementLogRecord],
    control_policy_provenance: PFControlPolicyProvenance,
) -> None:
    """Bind a live PF to the immutable log after session identity checks."""
    logged_isotopes = tuple(log.run_manifest["isotopes"])
    active_isotopes = tuple(estimator.joint_isotope_order())
    if active_isotopes != logged_isotopes:
        raise PFLiveSessionError(
            "Published MeasurementLog isotope order differs from the live PF."
        )
    if len(estimator.measurements) != len(log.records):
        raise PFLiveSessionError(
            "Published MeasurementLog record count disagrees with the live PF."
        )
    if len(live_records) != len(log.records):
        raise PFLiveSessionError(
            "Published MeasurementLog record count disagrees with live ingestion."
        )
    try:
        live_records_digest = measurement_records_digest(tuple(live_records))
        published_records_digest = measurement_records_digest(log.records)
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError(
            "Cannot authenticate the ordered live MeasurementLog records."
        ) from exc
    if live_records_digest != published_records_digest:
        raise PFLiveSessionError(
            "Published MeasurementLog records differ from the ordered live records."
        )
    _validate_published_forward_context(log)
    actual_pf = asdict(estimator.pf_config)
    digest = log.log_sha256
    estimator.measurement_log_sha256 = digest
    session_hash_payload = _live_session_hash_payload(
        runtime_config_sha256=log.resolved_config_sha256,
        measurement_log_sha256=digest,
        actual_pf=actual_pf,
        random_seed=int(estimator.random_seed),
        control_policy_provenance=control_policy_provenance,
    )
    estimator.resolved_config_hash = _strict_live_artifact_sha256(
        session_hash_payload,
        artifact_name="PF resolved session identity",
    )


def _context_energy_bin_edges(context: RunContext) -> NDArray[np.float64]:
    """Return the exact full-spectrum bin edges declared by one handshake."""
    contract = context.runtime_config.get("full_spectrum_generative_model")
    if not isinstance(contract, Mapping):
        raise PFLiveSessionError(
            "Live runtime context lacks full_spectrum_generative_model."
        )
    bin_count = _json_integer(
        contract.get("energy_bin_count"),
        location="context full-spectrum energy_bin_count",
        minimum=1,
    )
    energy_min = _finite_real(
        contract.get("energy_min_keV"),
        location="context full-spectrum energy_min_keV",
    )
    energy_max = _finite_real(
        contract.get("energy_max_keV"),
        location="context full-spectrum energy_max_keV",
    )
    bin_width = _finite_real(
        contract.get("bin_width_keV"),
        location="context full-spectrum bin_width_keV",
    )
    if bin_width <= 0.0:
        raise PFLiveSessionError(
            "Context full-spectrum bin_width_keV must be positive."
        )
    axis = energy_min + np.arange(bin_count, dtype=np.float64) * bin_width
    if axis[-1] != energy_max:
        raise PFLiveSessionError(
            "Context full-spectrum energy dimensions are inconsistent."
        )
    edges = np.concatenate(
        (axis, np.asarray([axis[-1] + bin_width], dtype=np.float64))
    )
    edges.setflags(write=False)
    return edges


def _yaw_from_detector_quaternion_wxyz(
    quaternion_wxyz: Sequence[float],
) -> float:
    """Return Z-axis yaw from one finite nonzero WXYZ quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or np.any(~np.isfinite(quaternion)):
        raise PFLiveSessionError(
            "Prior detector quaternion must be one finite WXYZ vector."
        )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0 or not math.isfinite(norm):
        raise PFLiveSessionError("Prior detector quaternion must have positive norm.")
    w, x, y, z = quaternion / norm
    return float(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _expected_detector_quaternion_wxyz(
    request_candidates: AdaptiveCandidateSnapshot,
    *,
    candidate_index: int,
    previous_record: MeasurementLogRecord | None,
) -> NDArray[np.float64]:
    """Derive the runtime-commanded yaw quaternion for one exact request."""
    current = np.asarray(request_candidates.current_pose_xyz, dtype=np.float64)
    target = np.asarray(
        request_candidates.candidate_poses_xyz[candidate_index],
        dtype=np.float64,
    )
    if (
        current.shape != (3,)
        or target.shape != (3,)
        or np.any(~np.isfinite(current))
        or np.any(~np.isfinite(target))
    ):
        raise PFLiveSessionError(
            "Detector yaw binding requires finite current and target XYZ poses."
        )
    delta_x = float(target[0] - current[0])
    delta_y = float(target[1] - current[1])
    if delta_x == 0.0 and delta_y == 0.0:
        yaw = (
            0.0
            if previous_record is None
            else _yaw_from_detector_quaternion_wxyz(
                previous_record.detector_quat_wxyz
            )
        )
    else:
        yaw = math.atan2(delta_y, delta_x)
    half_yaw = 0.5 * yaw
    expected = np.asarray(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        dtype=np.float64,
    )
    expected.setflags(write=False)
    return expected


def _detector_quaternion_matches_command(
    quaternion_wxyz: Sequence[float],
    expected_wxyz: NDArray[np.float64],
) -> bool:
    """Return whether one WXYZ quaternion matches the commanded yaw up to sign."""
    observed = np.asarray(quaternion_wxyz, dtype=np.float64)
    if observed.shape != (4,) or np.any(~np.isfinite(observed)):
        return False
    if not np.isclose(
        float(np.linalg.norm(observed)),
        1.0,
        rtol=0.0,
        atol=_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE,
    ):
        return False
    return bool(
        np.allclose(
            observed,
            expected_wxyz,
            rtol=0.0,
            atol=_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE,
        )
        or np.allclose(
            observed,
            -expected_wxyz,
            rtol=0.0,
            atol=_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE,
        )
    )


def _initial_candidate_motion_contract(
    context: RunContext,
) -> tuple[tuple[float, float, float], float]:
    """Return the exact initial pose and shield speed from the live context."""
    environment = context.environment
    initial_pose = environment.get("detector_position")
    if not isinstance(initial_pose, (list, tuple)) or len(initial_pose) != 3:
        raise PFLiveSessionError(
            "Runtime context detector_position must be one three-number array."
        )
    if any(type(value) not in (int, float) for value in initial_pose):
        raise PFLiveSessionError(
            "Runtime context detector_position must contain exact JSON numbers."
        )
    pose = tuple(float(value) for value in initial_pose)
    if any(not math.isfinite(value) for value in pose):
        raise PFLiveSessionError("Runtime context detector_position must be finite.")
    adaptive = environment.get("adaptive_measurement")
    if not isinstance(adaptive, Mapping):
        raise PFLiveSessionError(
            "Runtime context must contain adaptive_measurement."
        )
    raw_speed = adaptive.get("shield_angular_speed_rad_s")
    if type(raw_speed) not in (int, float):
        raise PFLiveSessionError(
            "Runtime shield angular speed must be an exact JSON number."
        )
    speed = float(raw_speed)
    if not math.isfinite(speed) or speed <= 0.0:
        raise PFLiveSessionError(
            "Runtime shield angular speed must be finite and positive."
        )
    return pose, speed


def _require_refined_candidate_extension(
    previous: AdaptiveCandidateSnapshot,
    refined: AdaptiveCandidateSnapshot,
) -> None:
    """Require refinement to preserve physical state and every prior quote."""
    if not isinstance(previous, AdaptiveCandidateSnapshot) or not isinstance(
        refined,
        AdaptiveCandidateSnapshot,
    ):
        raise TypeError("Candidate refinement requires typed runtime snapshots.")
    if refined.current_pose_xyz != previous.current_pose_xyz:
        raise RuntimeError("Refined candidates changed the current detector pose.")
    if refined.current_pair_id != previous.current_pair_id:
        raise RuntimeError("Refined candidates changed the current shield state.")
    if refined.allowed_pair_ids != previous.allowed_pair_ids:
        raise RuntimeError("Refined candidates changed the allowed shield pairs.")
    if (
        refined.shield_angular_speed_rad_s
        != previous.shield_angular_speed_rad_s
    ):
        raise RuntimeError("Refined candidates changed the shield angular speed.")
    if len(refined.candidate_poses_xyz) <= len(previous.candidate_poses_xyz):
        raise RuntimeError("Candidate refinement did not add a new reachable pose.")
    refined_index = {
        pose: index for index, pose in enumerate(refined.candidate_poses_xyz)
    }
    for previous_index, pose in enumerate(previous.candidate_poses_xyz):
        if pose not in refined_index:
            raise RuntimeError("Candidate refinement removed a prior reachable pose.")
        new_index = refined_index[pose]
        for field_name in (
            "travel_costs",
            "horizontal_travel_times_s",
            "mast_vertical_times_s",
            "settling_times_s",
        ):
            old_values = getattr(previous, field_name)
            new_values = getattr(refined, field_name)
            if new_values[new_index] != old_values[previous_index]:
                raise RuntimeError(
                    "Candidate refinement changed a prior runtime motion quote: "
                    f"{field_name}."
                )


def measurement_record_to_station_input(
    record: MeasurementLogRecord,
) -> tuple[object, ...]:
    """Translate one live runtime record into the PF station contract."""
    spectrum = np.asarray(record.spectrum_counts)
    if spectrum.ndim != 1 or spectrum.dtype != np.int64 or np.any(spectrum < 0):
        raise PFLiveSessionError(
            "Live observations must contain raw nonnegative int64 spectra."
        )
    fe_index = _json_integer(
        record.fe_orientation_index,
        location="record.fe_orientation_index",
        minimum=0,
    )
    pb_index = _json_integer(
        record.pb_orientation_index,
        location="record.pb_orientation_index",
        minimum=0,
    )
    if fe_index > 7 or pb_index > 7:
        raise PFLiveSessionError("Live Fe/Pb orientation indices must lie in 0..7.")
    live_time_s = _finite_real(record.live_time_s, location="record.live_time_s")
    if live_time_s <= 0.0:
        raise PFLiveSessionError("record.live_time_s must be positive.")
    return (
        np.ascontiguousarray(spectrum),
        fe_index,
        pb_index,
        live_time_s,
    )


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    location: str,
) -> NDArray[Any]:
    """Return one exact-dtype immutable contiguous array copy."""
    array = np.asarray(value)
    if array.dtype != dtype:
        raise PFLiveSessionError(f"{location} must have exact dtype {dtype}.")
    copied = np.array(array, dtype=dtype, copy=True, order="C")
    immutable = np.frombuffer(copied.tobytes(order="C"), dtype=dtype).reshape(
        copied.shape
    )
    return immutable


def _readonly_isotope_arrays(
    values: Mapping[str, object],
    *,
    isotope_order: tuple[str, ...],
    dtype: np.dtype[Any],
    location: str,
) -> Mapping[str, NDArray[Any]]:
    """Copy an exact isotope-keyed array mapping into immutable storage."""
    if not isinstance(values, Mapping) or tuple(values) != isotope_order:
        raise PFLiveSessionError(
            f"{location} keys must exactly match the PF isotope order."
        )
    copied = {
        isotope: _readonly_array(
            values[isotope],
            dtype=dtype,
            location=f"{location}.{isotope}",
        )
        for isotope in isotope_order
    }
    return MappingProxyType(copied)


def _immutable_particle_snapshot(
    particles: JointPlanningParticles,
    *,
    source_run_id: str,
    record_count: int,
    station_count: int,
    covered_records_digest: DigestIdentity,
    posterior_summary_json: bytes,
) -> PFLiveParticleSnapshot:
    """Copy one estimator particle view into a read-only live DTO."""
    if not isinstance(particles, JointPlanningParticles):
        raise PFLiveSessionError(
            "PF planning_joint_particles() returned an incompatible contract."
        )
    isotope_order = tuple(particles.isotope_order)
    if not isotope_order or len(set(isotope_order)) != len(isotope_order):
        raise PFLiveSessionError(
            "PF planning isotope order must be unique and nonempty."
        )
    weights = _readonly_array(
        particles.weights_n,
        dtype=np.dtype(np.float64),
        location="planning.weights_n",
    )
    indices = _readonly_array(
        particles.original_particle_indices,
        dtype=np.dtype(np.int64),
        location="planning.original_particle_indices",
    )
    if weights.ndim != 1 or indices.shape != weights.shape or weights.size == 0:
        raise PFLiveSessionError(
            "PF planning weights and particle indices must be aligned vectors."
        )
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1.0e-12)
        or np.any(indices < 0)
    ):
        raise PFLiveSessionError(
            "PF planning weights/indices must describe a normalized generation."
        )
    positions = _readonly_isotope_arrays(
        particles.positions_nk3_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.float64),
        location="planning.positions_nk3_by_isotope",
    )
    chart_ids = _readonly_isotope_arrays(
        particles.surface_chart_ids_nk_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.int64),
        location="planning.surface_chart_ids_nk_by_isotope",
    )
    surface_uv = _readonly_isotope_arrays(
        particles.surface_uv_nk2_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.float64),
        location="planning.surface_uv_nk2_by_isotope",
    )
    strengths = _readonly_isotope_arrays(
        particles.strengths_nk_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.float64),
        location="planning.strengths_nk_by_isotope",
    )
    masks = _readonly_isotope_arrays(
        particles.source_mask_nk_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.bool_),
        location="planning.source_mask_nk_by_isotope",
    )
    for isotope in isotope_order:
        position = positions[isotope]
        chart = chart_ids[isotope]
        uv = surface_uv[isotope]
        strength = strengths[isotope]
        mask = masks[isotope]
        if (
            position.ndim != 3
            or position.shape[0] != weights.size
            or position.shape[2] != 3
            or chart.shape != position.shape[:2]
            or uv.shape != (*position.shape[:2], 2)
            or strength.shape != position.shape[:2]
            or mask.shape != position.shape[:2]
            or np.any(~np.isfinite(position))
            or np.any(~np.isfinite(uv))
            or np.any(~np.isfinite(strength))
            or np.any(strength < 0.0)
        ):
            raise PFLiveSessionError(
                f"PF planning particle arrays are misaligned for {isotope}."
            )
    return PFLiveParticleSnapshot(
        source_run_id=source_run_id,
        record_count=record_count,
        station_count=station_count,
        covered_records_digest=covered_records_digest,
        isotope_order=isotope_order,
        weights_n=weights,
        positions_nk3_by_isotope=positions,
        surface_chart_ids_nk_by_isotope=chart_ids,
        surface_uv_nk2_by_isotope=surface_uv,
        strengths_nk_by_isotope=strengths,
        source_mask_nk_by_isotope=masks,
        original_particle_indices=indices,
        posterior_summary_json=posterior_summary_json,
        posterior_summary_sha256=sha256(posterior_summary_json).hexdigest(),
    )


def _compact_live_point_estimate(
    payload: Mapping[str, object],
    *,
    isotope: str,
) -> dict[str, object]:
    """Keep only the posterior trajectory fields needed for station audit."""
    map_cardinality = _json_integer(
        payload.get("map_cardinality"),
        location=f"live posterior {isotope}.map_cardinality",
        minimum=0,
    )
    distribution = payload.get("cardinality_distribution")
    if not isinstance(distribution, Mapping):
        raise PFLiveSessionError(
            f"Live posterior {isotope} lacks a cardinality distribution."
        )
    selected_stratum_mass = _finite_real(
        payload.get("selected_stratum_mass"),
        location=f"live posterior {isotope}.selected_stratum_mass",
    )
    if not 0.0 <= selected_stratum_mass <= 1.0:
        raise PFLiveSessionError(
            f"Live posterior {isotope} selected-stratum mass is invalid."
        )
    raw_modes = payload.get("modes")
    if not isinstance(raw_modes, Sequence) or isinstance(raw_modes, (str, bytes)):
        raise PFLiveSessionError(f"Live posterior {isotope} modes are invalid.")
    if len(raw_modes) != map_cardinality:
        raise PFLiveSessionError(
            f"Live posterior {isotope} mode count and MAP cardinality disagree."
        )
    modes: list[dict[str, object]] = []
    for index, raw_mode in enumerate(raw_modes):
        if not isinstance(raw_mode, Mapping):
            raise PFLiveSessionError(
                f"Live posterior {isotope} mode {index} is invalid."
            )
        raw_position = raw_mode.get("position_medoid_xyz")
        if not isinstance(raw_position, Sequence) or isinstance(
            raw_position,
            (str, bytes),
        ):
            raise PFLiveSessionError(
                f"Live posterior {isotope} mode {index} lacks a position."
            )
        position = [
            _finite_real(
                value,
                location=f"live posterior {isotope} mode {index} position",
            )
            for value in raw_position
        ]
        if len(position) != 3:
            raise PFLiveSessionError(
                f"Live posterior {isotope} mode {index} position is not 3-D."
            )
        radius = _finite_real(
            raw_mode.get("credible_radius_95_m"),
            location=f"live posterior {isotope} mode {index} credible radius",
        )
        strength = _finite_real(
            raw_mode.get("strength_representative_cps_1m"),
            location=f"live posterior {isotope} mode {index} strength",
        )
        posterior_mass = _finite_real(
            raw_mode.get("posterior_mass"),
            location=f"live posterior {isotope} mode {index} mass",
        )
        if radius < 0.0 or strength < 0.0 or not 0.0 <= posterior_mass <= 1.0:
            raise PFLiveSessionError(
                f"Live posterior {isotope} mode {index} has invalid uncertainty."
            )
        modes.append(
            {
                "label_index": _json_integer(
                    raw_mode.get("label_index"),
                    location=f"live posterior {isotope} mode {index} label",
                    minimum=0,
                ),
                "position_medoid_xyz": position,
                "credible_radius_95_m": radius,
                "strength_representative_cps_1m": strength,
                "posterior_mass": posterior_mass,
            }
        )
    return {
        "map_cardinality": map_cardinality,
        "cardinality_distribution": _json_cardinality_distribution(
            distribution,
            location=f"live posterior {isotope} cardinality distribution",
        ),
        "selected_stratum_mass": selected_stratum_mass,
        "modes": modes,
    }


def live_posterior_summary(estimator: object) -> dict[str, object]:
    """Return a compact truth-free, non-publishable station summary."""
    method = getattr(estimator, "posterior_point_estimate", None)
    if not callable(method):
        raise PFLiveSessionError(
            "PF estimator does not expose posterior_point_estimate()."
        )
    raw = method()
    if not isinstance(raw, Mapping):
        raise PFLiveSessionError("PF live point estimates must be an isotope mapping.")
    isotopes: dict[str, object] = {}
    for isotope, estimate in raw.items():
        if not isinstance(isotope, str) or not isotope:
            raise PFLiveSessionError(
                "PF live point-estimate isotope keys must be nonempty strings."
            )
        to_dict = getattr(estimate, "to_dict", None)
        if not callable(to_dict):
            raise PFLiveSessionError(
                "Every PF live point estimate must be serializable."
            )
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise PFLiveSessionError(
                "Every PF live point estimate must serialize an object."
            )
        isotopes[isotope] = _compact_live_point_estimate(
            payload,
            isotope=isotope,
        )
    return {
        "schema_version": 2,
        "publishable": False,
        "isotopes": isotopes,
    }


def _post_run_evaluation_input_payload(
    estimator: object,
    *,
    source_run_id: str,
    measurement_log_sha256: str,
) -> dict[str, object]:
    """Build truth-free response signatures for standardized post-run scoring."""
    point_estimate_method = getattr(estimator, "posterior_point_estimate", None)
    signature_method = getattr(estimator, "source_response_signatures", None)
    config = getattr(estimator, "pf_config", None)
    if (
        not callable(point_estimate_method)
        or not callable(signature_method)
        or config is None
    ):
        raise PFLiveSessionError(
            "PF estimator cannot build standardized post-run evaluation input."
        )
    estimates = point_estimate_method()
    if not isinstance(estimates, Mapping) or not estimates:
        raise PFLiveSessionError("Post-run evaluation requires isotope estimates.")
    isotope_payload: dict[str, object] = {}
    for isotope, estimate in estimates.items():
        modes = tuple(getattr(estimate, "modes", ()))
        positions = np.asarray(
            [getattr(mode, "position_medoid_xyz") for mode in modes],
            dtype=np.float64,
        ).reshape(len(modes), 3)
        strengths = np.asarray(
            [getattr(mode, "strength_representative_cps_1m") for mode in modes],
            dtype=np.float64,
        )
        labels = np.asarray(
            [getattr(mode, "label_index") for mode in modes],
            dtype=np.int64,
        )
        signatures = np.asarray(
            signature_method(str(isotope), positions),
            dtype=np.float64,
        )
        if (
            signatures.ndim != 2
            or signatures.shape[1] != len(modes)
            or np.any(~np.isfinite(signatures))
            or np.any(signatures < 0.0)
            or strengths.shape != (len(modes),)
            or np.any(~np.isfinite(strengths))
            or np.any(strengths <= 0.0)
            or labels.shape != (len(modes),)
            or np.any(labels < 0)
            or np.unique(labels).size != labels.size
        ):
            raise PFLiveSessionError(
                f"Post-run response signatures are invalid for {isotope}."
            )
        signature_norms = np.linalg.norm(signatures, axis=0)
        if len(modes) and not np.allclose(
            signature_norms,
            1.0,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise PFLiveSessionError(
                f"Post-run response signatures are not normalized for {isotope}."
            )
        isotope_payload[str(isotope)] = {
            "mode_label_indices": [int(value) for value in labels],
            "mode_positions_xyz_m": positions.tolist(),
            "mode_strengths_cps_1m": strengths.tolist(),
            "normalized_response_signatures_measurement_by_mode": (
                signatures.tolist()
            ),
        }
    hard_max_sources = getattr(config, "cardinality_capacity", None)
    if isinstance(hard_max_sources, bool) or not isinstance(hard_max_sources, int):
        raise PFLiveSessionError("PF cardinality capacity is unavailable.")
    return {
        "schema_version": 1,
        "artifact_family": "pf_post_run_cluster_evaluation_input",
        "source_run_id": str(source_run_id),
        "measurement_log_sha256": str(measurement_log_sha256),
        "hard_max_sources_per_isotope": int(hard_max_sources),
        "response_signature_semantics": (
            "normalized_same_isotope_expected_count_by_completed_measurement"
        ),
        "truth_read": False,
        "isotopes": isotope_payload,
    }


def register_persisted_station_pose(
    estimator: PurePFEstimator,
    records: Sequence[MeasurementLogRecord],
    *,
    station_id: int,
) -> int:
    """Register one canonical single-pose station and return its pose index."""
    rows = tuple(records)
    if not rows:
        raise PFLiveSessionError("A PF station must contain at least one record.")
    if any(not isinstance(record, MeasurementLogRecord) for record in rows):
        raise PFLiveSessionError(
            "PF station ingestion requires MeasurementLogRecord values."
        )
    poses = np.asarray([record.detector_pose_xyz for record in rows], dtype=np.float64)
    quaternions = np.asarray(
        [record.detector_quat_wxyz for record in rows],
        dtype=np.float64,
    )
    if not np.all(poses == poses[0]) or not np.all(quaternions == quaternions[0]):
        raise PFLiveSessionError(
            "Every persisted PF station view must share one detector pose."
        )
    pose = poses[0]
    if station_id == 0 and not estimator.measurements and len(estimator.poses) == 1:
        estimator.poses[0] = pose.copy()
        estimator.kernel_cache = None
        return 0
    estimator.add_measurement_pose(pose, reset_filters=False)
    return len(estimator.poses) - 1


def _map_surface_guidance_to_pf_charts(
    estimator: PurePFEstimator,
    guidance: PFExternalSurfaceGuidance,
) -> tuple[dict[str, NDArray[np.float64]], tuple[str, ...], int]:
    """Map a surface grid to PF charts with one bounded vectorized neighbor query."""
    atlas = estimator.continuous_surface_atlas()
    chart_centers = np.asarray(
        atlas.geometry.centers_xyz,
        dtype=np.float64,
    ).reshape(-1, 3)
    if chart_centers.shape[0] < 1 or np.any(~np.isfinite(chart_centers)):
        raise PFLiveSessionError("PF continuous-surface atlas is invalid.")
    neighbor_count = min(8, int(guidance.patch_centroids_xyz.shape[0]))
    distances, indices = cKDTree(guidance.patch_centroids_xyz).query(
        chart_centers,
        k=neighbor_count,
        workers=-1,
    )
    distance_array = np.asarray(distances, dtype=np.float64).reshape(
        chart_centers.shape[0],
        neighbor_count,
    )
    index_array = np.asarray(indices, dtype=np.int64).reshape(
        chart_centers.shape[0],
        neighbor_count,
    )
    if (
        np.any(~np.isfinite(distance_array))
        or np.any(index_array < 0)
        or np.any(index_array >= guidance.patch_centroids_xyz.shape[0])
    ):
        raise PFLiveSessionError("Surface-guidance neighbor mapping is invalid.")
    log_kernel = -0.5 * np.square(
        distance_array / float(guidance.bandwidth_m)
    )
    log_kernel -= np.max(log_kernel, axis=1, keepdims=True)
    kernel = np.exp(log_kernel)
    normalizer = np.sum(kernel, axis=1, keepdims=True, dtype=np.float64)
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
        raise PFLiveSessionError("Surface-guidance interpolation has zero mass.")
    mapped_ic = np.einsum(
        "ick,ck->ic",
        guidance.density_by_isotope[:, index_array],
        kernel,
        optimize=True,
    ) / normalizer[:, 0][None, :]
    if (
        mapped_ic.shape != (len(guidance.isotope_order), chart_centers.shape[0])
        or np.any(~np.isfinite(mapped_ic))
        or np.any(mapped_ic < 0.0)
    ):
        raise PFLiveSessionError("Mapped surface guidance is invalid.")
    mapped: dict[str, NDArray[np.float64]] = {}
    informative: list[str] = []
    for isotope_index, isotope in enumerate(guidance.isotope_order):
        values = np.ascontiguousarray(mapped_ic[isotope_index], dtype=np.float64)
        values.setflags(write=False)
        mapped[isotope] = values
        if float(np.max(values, initial=0.0)) > 0.0:
            informative.append(isotope)
    return mapped, tuple(informative), int(chart_centers.shape[0])


def assimilate_persisted_station(
    estimator: PurePFEstimator,
    records: Sequence[MeasurementLogRecord],
    *,
    station_id: int,
    generative_contract_hash_sha256: str,
    surface_guidance: PFExternalSurfaceGuidance | None = None,
) -> PFExternalSurfaceGuidanceReceipt | None:
    """Assimilate one canonical single-pose station through the PF spectrum path."""
    rows = tuple(records)
    if any(record.station_id != station_id for record in rows):
        raise PFLiveSessionError(
            "Persisted PF station IDs must match the requested station."
        )
    if any(
        record.metadata.get("station_complete") is True for record in rows[:-1]
    ) or (rows and rows[-1].metadata.get("station_complete") is not True):
        raise PFLiveSessionError(
            "PF assimilation requires one final durable station marker."
        )
    pose_index = register_persisted_station_pose(
        estimator,
        rows,
        station_id=station_id,
    )
    receipt = None
    mapped: dict[str, NDArray[np.float64]] | None = None
    informative: tuple[str, ...] = ()
    chart_count = 0
    if surface_guidance is not None:
        mapped, informative, chart_count = _map_surface_guidance_to_pf_charts(
            estimator,
            surface_guidance,
        )
        estimator._joint_external_surface_guidance_by_isotope = mapped
        estimator._joint_external_surface_guidance_mass = float(
            surface_guidance.proposal_mass
        )
        estimator.last_external_surface_guidance_diagnostics = {}
        estimator.last_external_surface_guidance_evaluated_isotopes = set()
    try:
        estimator.update_spectrum_station(
            tuple(measurement_record_to_station_input(record) for record in rows),
            pose_idx=int(pose_index),
            generative_contract_hash_sha256=generative_contract_hash_sha256,
        )
    finally:
        estimator._joint_external_surface_guidance_by_isotope = None
        estimator._joint_external_surface_guidance_mass = 0.0
    if surface_guidance is not None:
        expected_isotopes = tuple(surface_guidance.isotope_order)
        evaluated_isotopes = tuple(
            isotope
            for isotope in expected_isotopes
            if isotope in estimator.last_external_surface_guidance_evaluated_isotopes
        )
        if evaluated_isotopes != expected_isotopes:
            raise RuntimeError(
                "PF station update did not evaluate external guidance for "
                "every isotope."
            )
        receipt = PFExternalSurfaceGuidanceReceipt(
            guidance_sha256=surface_guidance.guidance_sha256,
            source_run_id=surface_guidance.source_run_id,
            record_count=surface_guidance.record_count,
            data_cutoff_step=surface_guidance.data_cutoff_step,
            data_cutoff_station=surface_guidance.data_cutoff_station,
            covered_records_digest=surface_guidance.covered_records_digest,
            proposal_mass=surface_guidance.proposal_mass,
            bandwidth_m=surface_guidance.bandwidth_m,
            informative_isotopes=informative,
            evaluated_isotopes=evaluated_isotopes,
            mapped_chart_count=chart_count,
        )
    return receipt


class PFLiveSession:
    """Own one causal PF estimator from live handshake through final binding."""

    def __init__(
        self,
        context: RunContext,
        config: ValidatedProductionPFConfig,
        *,
        initial_candidates: AdaptiveCandidateSnapshot,
        profile: str,
        seed: int,
        runtime_root: str | Path,
        control_policy_provenance: PFControlPolicyProvenance,
    ) -> None:
        """Authenticate the runtime context and construct one live PF."""
        if not isinstance(context, RunContext):
            raise PFLiveSessionError("context must be a runtime RunContext.")
        if not isinstance(control_policy_provenance, PFControlPolicyProvenance):
            raise PFLiveSessionError(
                "Live PF session requires sealed control-policy provenance."
            )
        if not isinstance(initial_candidates, AdaptiveCandidateSnapshot):
            raise PFLiveSessionError(
                "Live PF session requires the exact handshake candidate snapshot."
            )
        initial_pose, shield_angular_speed_rad_s = (
            _initial_candidate_motion_contract(context)
        )
        if initial_pose != initial_candidates.current_pose_xyz:
            raise PFLiveSessionError(
                "Handshake candidates are not anchored to the runtime initial pose."
            )
        if initial_candidates.current_pair_id != 0:
            raise PFLiveSessionError(
                "Fresh handshake candidates must start at shield pair 0."
            )
        if initial_candidates.allowed_pair_ids != tuple(range(64)):
            raise PFLiveSessionError(
                "Fresh handshake candidates must expose all 64 shield pairs."
            )
        if (
            initial_candidates.shield_angular_speed_rad_s
            != shield_angular_speed_rad_s
        ):
            raise PFLiveSessionError(
                "Handshake candidate shield speed differs from the runtime context."
            )
        contract_hash = _sha256_string(
            context.runtime_config.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY),
            location=(
                "context.runtime_config."
                f"{FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY}"
            ),
        )
        estimator = build_live_estimator(
            context,
            config,
            profile=profile,
            seed=seed,
            runtime_root=runtime_root,
            control_policy_provenance=control_policy_provenance,
        )
        estimator_contract = getattr(
            getattr(estimator, "full_spectrum_generative_model", None),
            "contract_hash_sha256",
            None,
        )
        if estimator_contract != contract_hash:
            raise PFLiveSessionError(
                "Live PF generative model differs from its runtime handshake."
            )
        self._context = context
        self._estimator = estimator
        self._runtime_root = Path(runtime_root).expanduser().resolve()
        self._input_config = MappingProxyType(config.settings())
        self._profile = profile
        self._seed = seed
        self._control_policy_provenance = control_policy_provenance
        self._expected_candidates = initial_candidates
        self._generative_contract_hash_sha256 = contract_hash
        self._expected_energy_bin_edges_keV = _context_energy_bin_edges(context)
        self._records: list[MeasurementLogRecord] = []
        self._station_count = 0
        self._phase = "receiving"
        self._completed_state: PFCompletedLiveState | None = None
        self._bound_state: PFBoundLiveState | None = None
        self._pending_surface_guidance: PFExternalSurfaceGuidance | None = None
        self._last_surface_guidance_receipt: (
            PFExternalSurfaceGuidanceReceipt | None
        ) = None

    @property
    def context(self) -> RunContext:
        """Return the immutable truth-free runtime handshake."""
        return self._context

    @property
    def estimator(self) -> PurePFEstimator:
        """Return the estimator owned by this session for live diagnostics."""
        return self._estimator

    @property
    def records(self) -> tuple[MeasurementLogRecord, ...]:
        """Return the exact ordered records received from the runtime."""
        return tuple(self._records)

    @property
    def record_count(self) -> int:
        """Return the count of durably delivered runtime records."""
        return len(self._records)

    @property
    def station_count(self) -> int:
        """Return the count of stations already assimilated exactly once."""
        return self._station_count

    @property
    def phase(self) -> str:
        """Return the current receiving, completed, bound, or failed phase."""
        return self._phase

    @property
    def last_surface_guidance_receipt(
        self,
    ) -> PFExternalSurfaceGuidanceReceipt | None:
        """Return the most recent target-preserving surface-guidance receipt."""
        return self._last_surface_guidance_receipt

    def _ensure_receiving(self) -> None:
        """Reject observation delivery after completion or a failed update."""
        if self._phase != "receiving":
            raise PFLiveSessionError(
                f"PF live session cannot receive records while {self._phase}."
            )

    def receive_refined_candidates(
        self,
        refined: AdaptiveCandidateSnapshot,
    ) -> None:
        """Advance the owned candidate chain through one exact refinement event."""
        self._ensure_receiving()
        try:
            _require_refined_candidate_extension(
                self._expected_candidates,
                refined,
            )
        except (TypeError, ValueError, RuntimeError):
            self._phase = "failed"
            raise
        self._expected_candidates = refined

    def _validated_view(
        self,
        records: Sequence[MeasurementLogRecord],
    ) -> MeasurementLogView:
        """Validate one exact live prefix through the shared runtime view."""
        try:
            view = MeasurementLogView.from_records(self._context, tuple(records))
            view.station_view()
        except (TypeError, ValueError, MeasurementLogValidationError) as exc:
            raise PFLiveSessionError(
                f"Persisted PF records violate the runtime contract: {exc}"
            ) from exc
        for index, record in enumerate(view.records):
            if not np.array_equal(
                np.asarray(record.energy_bin_edges_keV, dtype=np.float64),
                self._expected_energy_bin_edges_keV,
            ):
                raise PFLiveSessionError(
                    "Persisted PF record energy axis differs from the runtime "
                    f"handshake at row {index}."
                )
            if (
                record.metadata.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY)
                != self._generative_contract_hash_sha256
            ):
                raise PFLiveSessionError(
                    "Persisted PF record generative contract differs from the "
                    f"runtime handshake at row {index}."
                )
        return view

    def receive_acquired(
        self,
        record: MeasurementLogRecord,
        *,
        request: AdaptiveStepRequest,
        request_candidates: AdaptiveCandidateSnapshot,
        next_candidates: AdaptiveCandidateSnapshot,
    ) -> bool:
        """Validate one complete action transaction before PF assimilation."""
        self._ensure_receiving()
        if not isinstance(record, MeasurementLogRecord):
            self._phase = "failed"
            raise PFLiveSessionError(
                "receive_acquired requires a MeasurementLogRecord."
            )
        if not isinstance(request, AdaptiveStepRequest):
            self._phase = "failed"
            raise PFLiveSessionError(
                "receive_acquired requires an AdaptiveStepRequest."
            )
        if not isinstance(request_candidates, AdaptiveCandidateSnapshot):
            self._phase = "failed"
            raise PFLiveSessionError(
                "receive_acquired requires the exact pre-action candidate snapshot."
            )
        if request_candidates != self._expected_candidates:
            self._phase = "failed"
            raise PFLiveSessionError(
                "request_candidates differs from the session-owned runtime "
                "candidate chain."
            )
        if not isinstance(next_candidates, AdaptiveCandidateSnapshot):
            self._phase = "failed"
            raise PFLiveSessionError(
                "receive_acquired requires an AdaptiveCandidateSnapshot."
            )
        candidate_index = request.candidate_index
        if candidate_index >= len(request_candidates.candidate_poses_xyz):
            self._phase = "failed"
            raise PFLiveSessionError(
                "Adaptive request.candidate_index lies outside its exact "
                "pre-action snapshot."
            )
        requested_pose = np.asarray(
            request_candidates.candidate_poses_xyz[candidate_index],
            dtype=np.float64,
        )
        if (
            requested_pose.dtype != np.dtype(np.float64)
            or requested_pose.shape != (3,)
            or np.any(~np.isfinite(requested_pose))
        ):
            self._phase = "failed"
            raise PFLiveSessionError(
                "Requested candidate pose must be one exact finite float64 XYZ."
            )
        try:
            expected_detector_quaternion = _expected_detector_quaternion_wxyz(
                request_candidates,
                candidate_index=candidate_index,
                previous_record=(self._records[-1] if self._records else None),
            )
        except (TypeError, ValueError, PFLiveSessionError):
            self._phase = "failed"
            raise
        expected_record_id = len(self._records)
        mismatches: list[str] = []
        if request.action_id != expected_record_id:
            mismatches.append("request.action_id")
        if record.step_id != request.action_id:
            mismatches.append("step_id")
        if record.action_id != request.action_id:
            mismatches.append("action_id")
        if record.station_id != request.station_id:
            mismatches.append("station_id")
        if tuple(record.detector_pose_xyz) != tuple(requested_pose.tolist()):
            mismatches.append("detector_pose_xyz")
        if not _detector_quaternion_matches_command(
            record.detector_quat_wxyz,
            expected_detector_quaternion,
        ):
            mismatches.append("detector_quat_wxyz")
        if record.fe_orientation_index != request.fe_orientation_index:
            mismatches.append("fe_orientation_index")
        if record.pb_orientation_index != request.pb_orientation_index:
            mismatches.append("pb_orientation_index")
        if record.live_time_s != request.dwell_time_s:
            mismatches.append("live_time_s")
        if (
            record.travel_time_s
            != request_candidates.travel_costs[candidate_index]
        ):
            mismatches.append("travel_time_s")
        requested_pair_id = (
            request.fe_orientation_index * 8 + request.pb_orientation_index
        )
        expected_shield_actuation_time_s = (
            request_candidates.quote_shield_program_time_s((requested_pair_id,))
        )
        if (
            record.shield_actuation_time_s
            != expected_shield_actuation_time_s
        ):
            mismatches.append("shield_actuation_time_s")
        station_complete = record.metadata.get("station_complete") is True
        if station_complete is not request.station_complete:
            mismatches.append("station_complete")
        if not np.array_equal(
            np.asarray(record.energy_bin_edges_keV, dtype=np.float64),
            self._expected_energy_bin_edges_keV,
        ):
            mismatches.append("energy_bin_edges_keV")
        if (
            record.metadata.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY)
            != self._generative_contract_hash_sha256
        ):
            mismatches.append(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY)
        if next_candidates.current_pair_id != requested_pair_id:
            mismatches.append("candidates.current_pair_id")
        if tuple(next_candidates.current_pose_xyz) != tuple(requested_pose.tolist()):
            mismatches.append("candidates.current_pose")
        if next_candidates.allowed_pair_ids != request_candidates.allowed_pair_ids:
            mismatches.append("candidates.allowed_pair_ids")
        if (
            next_candidates.shield_angular_speed_rad_s
            != request_candidates.shield_angular_speed_rad_s
        ):
            mismatches.append("candidates.shield_angular_speed_rad_s")
        if mismatches:
            self._phase = "failed"
            raise PFLiveSessionError(
                "Adaptive acquisition response differs from its exact request: "
                + ", ".join(mismatches)
                + "."
            )
        try:
            completed = self.receive_persisted(record)
        except BaseException:
            self._phase = "failed"
            raise
        self._expected_candidates = next_candidates
        return completed

    def stage_external_surface_guidance(
        self,
        guidance: PFExternalSurfaceGuidance,
    ) -> None:
        """Stage one exact-prefix surface proposal for the next PF station update."""
        self._ensure_receiving()
        if not isinstance(guidance, PFExternalSurfaceGuidance):
            raise TypeError("guidance must be a PFExternalSurfaceGuidance.")
        if self._pending_surface_guidance is not None:
            raise PFLiveSessionError("A surface-guidance proposal is already staged.")
        if self._records and (
            self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "Surface guidance may be staged only at a PF station boundary."
            )
        if guidance.source_run_id != self._context.run_id:
            raise PFLiveSessionError("Surface guidance belongs to another run_id.")
        if guidance.isotope_order != tuple(self._estimator.joint_isotope_order()):
            raise PFLiveSessionError(
                "Surface-guidance isotope order differs from the live PF."
            )
        if guidance.data_cutoff_station != self._station_count:
            raise PFLiveSessionError(
                "Surface guidance must cover exactly the next PF station."
            )
        if guidance.record_count <= len(self._records):
            raise PFLiveSessionError(
                "Surface guidance must extend the current PF record prefix."
            )
        self._pending_surface_guidance = guidance

    def receive_persisted(self, record: MeasurementLogRecord) -> bool:
        """Receive one durable record and assimilate only at its station marker."""
        self._ensure_receiving()
        if not isinstance(record, MeasurementLogRecord):
            raise PFLiveSessionError(
                "receive_persisted requires a MeasurementLogRecord."
            )
        if self._records and (
            record.station_id != self._records[-1].station_id
            and self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "A new PF station cannot begin before the prior marker is durable."
            )
        prospective = (*self._records, record)
        view = self._validated_view(prospective)
        guidance = self._pending_surface_guidance
        if record.metadata.get("station_complete") is True and guidance is not None:
            if (
                guidance.record_count != len(prospective)
                or guidance.data_cutoff_step != record.step_id
                or guidance.data_cutoff_station != record.station_id
                or guidance.covered_records_digest
                != measurement_records_digest(prospective)
            ):
                self._phase = "failed"
                raise PFLiveSessionError(
                    "Surface guidance does not bind the exact incoming PF prefix."
                )
        self._records.append(record)
        if record.metadata.get("station_complete") is not True:
            return False
        station = view.station_view().stations[-1]
        if station.station_id != self._station_count or not station.marked_complete:
            self._phase = "failed"
            raise PFLiveSessionError(
                "Persisted PF station sequence differs from completed assimilation."
            )
        try:
            receipt = assimilate_persisted_station(
                self._estimator,
                station.records,
                station_id=station.station_id,
                generative_contract_hash_sha256=(
                    self._generative_contract_hash_sha256
                ),
                surface_guidance=guidance,
            )
        except BaseException:
            self._phase = "failed"
            raise
        self._pending_surface_guidance = None
        self._last_surface_guidance_receipt = receipt
        self._station_count += 1
        if len(self._estimator.measurements) != len(self._records):
            self._phase = "failed"
            raise PFLiveSessionError(
                "PF estimator measurement history differs from live ingestion."
            )
        return True

    def receive_persisted_station(
        self,
        records: Sequence[MeasurementLogRecord],
    ) -> None:
        """Receive one complete durable station through the record API."""
        self._ensure_receiving()
        rows = tuple(records)
        if not rows:
            raise PFLiveSessionError("A persisted PF station cannot be empty.")
        if any(not isinstance(record, MeasurementLogRecord) for record in rows):
            raise PFLiveSessionError(
                "receive_persisted_station requires MeasurementLogRecord values."
            )
        if (
            self._records
            and self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "receive_persisted_station cannot continue a partially "
                "buffered station."
            )
        if len({record.station_id for record in rows}) != 1:
            raise PFLiveSessionError(
                "receive_persisted_station accepts exactly one station."
            )
        if any(
            record.metadata.get("station_complete") is True for record in rows[:-1]
        ) or rows[-1].metadata.get("station_complete") is not True:
            raise PFLiveSessionError(
                "A persisted station requires one final station_complete marker."
            )
        self._validated_view((*self._records, *rows))
        for record in rows:
            self.receive_persisted(record)

    def planning_particle_snapshot(
        self,
        *,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> PFLiveParticleSnapshot:
        """Return a copied truth-free particle generation at a station boundary."""
        if self._phase != "receiving":
            raise PFLiveSessionError(
                f"PF live session cannot plan while {self._phase}."
            )
        if not self._records or (
            self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "PF planning requires the latest durable station boundary."
            )
        particles = self._estimator.planning_joint_particles(
            max_particles=max_particles,
            method=method,
            rng=rng,
        )
        summary_json = _strict_live_artifact_json_bytes(
            live_posterior_summary(self._estimator),
            artifact_name="PF live posterior summary",
        )
        return _immutable_particle_snapshot(
            particles,
            source_run_id=self._context.run_id,
            record_count=len(self._records),
            station_count=self._station_count,
            covered_records_digest=measurement_records_digest(self._records),
            posterior_summary_json=summary_json,
        )

    def complete_live_state(
        self,
        *,
        diagnostics_extensions: Mapping[str, object] | None = None,
    ) -> PFCompletedLiveState:
        """Seal the already-assimilated live state before log publication."""
        if self._phase in {"completed", "bound"}:
            if diagnostics_extensions is not None:
                raise PFLiveSessionError(
                    "Completed PF diagnostics cannot be changed after sealing."
                )
            assert self._completed_state is not None
            return self._completed_state
        self._ensure_receiving()
        if not self._records:
            raise PFLiveSessionError(
                "A PF live session cannot complete without records."
            )
        view = self._validated_view(self._records)
        stations = view.station_view()
        if stations.complete_station_count != stations.station_count or (
            self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "PF live completion requires every station marker to be durable."
            )
        if self._station_count != stations.station_count or (
            len(self._estimator.measurements) != len(self._records)
        ):
            raise PFLiveSessionError(
                "PF live completion differs from its assimilated station history."
            )
        final_particles = self.planning_particle_snapshot()
        try:
            diagnostics = _compact_pf_diagnostics(self._estimator)
        except BaseException:
            self._phase = "failed"
            raise
        if diagnostics_extensions is not None:
            if not isinstance(diagnostics_extensions, Mapping):
                raise PFLiveSessionError(
                    "PF diagnostics extensions must be a mapping."
                )
            conflicts = sorted(set(diagnostics).intersection(diagnostics_extensions))
            if conflicts:
                raise PFLiveSessionError(
                    "PF diagnostics extensions cannot replace canonical fields: "
                    f"{conflicts}."
                )
            normalized_extensions = strict_json_loads(
                _strict_live_artifact_json_bytes(
                    dict(diagnostics_extensions),
                    artifact_name="PF diagnostics extensions",
                )
            )
            if not isinstance(normalized_extensions, dict):
                raise PFLiveSessionError(
                    "PF diagnostics extensions must serialize as an object."
                )
            diagnostics.update(normalized_extensions)
        diagnostics_json = _strict_live_artifact_json_bytes(
            diagnostics,
            artifact_name="PF diagnostics",
        )
        checkpoint = _sealed_live_checkpoint_state(
            self._estimator,
            self._control_policy_provenance,
        )
        digest = measurement_records_digest(self._records)
        completed = PFCompletedLiveState(
            source_run_id=self._context.run_id,
            runtime_config_sha256=self._context.runtime_config_sha256,
            generative_contract_hash_sha256=(
                self._generative_contract_hash_sha256
            ),
            record_count=len(self._records),
            station_count=self._station_count,
            covered_step_ids=tuple(record.step_id for record in self._records),
            covered_records_digest=digest,
            control_policy_provenance=self._control_policy_provenance,
            checkpoint_state=checkpoint,
            checkpoint_sha256=sha256(checkpoint).hexdigest(),
            diagnostics_json=diagnostics_json,
            particle_snapshot=final_particles,
        )
        self._completed_state = completed
        self._phase = "completed"
        return completed

    def bind_published_log(self, log: MeasurementLog) -> PFBoundLiveState:
        """Bind the sealed PF to its exact published log without assimilation."""
        if not isinstance(log, MeasurementLog) or log.path is None:
            raise PFLiveSessionError(
                "bind_published_log requires a published MeasurementLog."
            )
        if self._phase == "bound":
            assert self._bound_state is not None
            if self._bound_state.measurement_log_sha256 != log.log_sha256:
                raise PFLiveSessionError(
                    "PF live state is already bound to another MeasurementLog."
                )
            return self._bound_state
        if self._phase != "completed" or self._completed_state is None:
            raise PFLiveSessionError(
                "complete_live_state() must seal PF inference before log binding."
            )
        if log.run_id != self._context.run_id:
            raise PFLiveSessionError(
                "Published MeasurementLog belongs to another runtime run."
            )
        if log.context.to_payload() != self._context.to_payload():
            raise PFLiveSessionError(
                "Published MeasurementLog context differs from the live handshake."
            )
        published_digest = measurement_records_digest(log.records)
        if published_digest != self._completed_state.covered_records_digest:
            raise PFLiveSessionError(
                "Published MeasurementLog records differ from the completed PF state."
            )
        before_bind = self._completed_state.checkpoint_state
        _bind_estimator_to_published_log(
            self._estimator,
            log,
            live_records=self._records,
            control_policy_provenance=(
                self._completed_state.control_policy_provenance
            ),
        )
        after_bind = _sealed_live_checkpoint_state(
            self._estimator,
            self._completed_state.control_policy_provenance,
        )
        if after_bind != before_bind:
            self._phase = "failed"
            raise PFLiveSessionError(
                "Published-log binding changed the completed PF posterior state."
            )
        snapshot = self._estimator.posterior_snapshot()
        to_dict = getattr(snapshot, "to_dict", None)
        if not callable(to_dict):
            raise PFLiveSessionError("PF posterior snapshot is not serializable.")
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise PFLiveSessionError("PF posterior snapshot must serialize an object.")
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise PFLiveSessionError("Bound PF posterior lacks provenance.")
        log_digest = log.log_sha256
        if provenance.get("measurement_log_sha256") != log_digest or (
            payload.get("record_count") != len(self._records)
        ):
            raise PFLiveSessionError(
                "Bound PF posterior identity differs from the published log."
            )
        posterior_payload = dict(payload)
        posterior_provenance = dict(provenance)
        policy_provenance = (
            self._completed_state.control_policy_provenance.to_dict()
        )
        posterior_provenance["control_policy"] = policy_provenance
        posterior_payload["provenance"] = posterior_provenance
        posterior_json = _strict_live_artifact_json_bytes(
            posterior_payload,
            artifact_name="PF posterior",
        )
        bound = PFBoundLiveState(
            completed=self._completed_state,
            measurement_log_sha256=log_digest,
            posterior_json=posterior_json,
            posterior_sha256=sha256(posterior_json).hexdigest(),
        )
        self._bound_state = bound
        self._phase = "bound"
        return bound

    def publication_input(self) -> PFBoundLiveState:
        """Return bound result/checkpoint bytes without rerunning PF inference."""
        if self._phase != "bound" or self._bound_state is None:
            raise PFLiveSessionError(
                "PF publication input requires an exactly bound published log."
            )
        return self._bound_state

    def _publish_bound_result_into_staging(
        self,
        output_dir: str | Path,
    ) -> PFPublishedLiveResult:
        """Write canonical PF artifacts into one outer atomic staging root."""
        bound = self.publication_input()
        target = Path(output_dir).expanduser().resolve()
        if target.exists():
            if not target.is_dir() or target.is_symlink():
                raise FileExistsError(
                    f"PF result target is not a regular directory: {target}"
                )
        else:
            target.mkdir(parents=True)
        owned_names = {
            "pf_posterior.json",
            "pf_diagnostics.json",
            "pf_state.json",
            "pf_particles.npz",
            "pf_checkpoint.json",
            "pf_artifact_inventory.json",
            "pf_post_run_evaluation_input.json",
        }
        conflicts = sorted(name for name in owned_names if (target / name).exists())
        if conflicts:
            raise FileExistsError(
                "Refusing to replace package-owned PF artifacts: "
                f"{conflicts}."
            )
        posterior = strict_json_loads(bound.posterior_json)
        if not isinstance(posterior, dict):
            raise PFLiveSessionError("Bound PF posterior must be a JSON object.")
        provenance = posterior.get("provenance")
        if not isinstance(provenance, Mapping):
            raise PFLiveSessionError("Bound PF posterior lacks provenance.")
        expected_policy_provenance = (
            bound.completed.control_policy_provenance.to_dict()
        )
        if provenance.get("control_policy") != expected_policy_provenance:
            raise PFLiveSessionError(
                "Bound PF posterior control policy differs from the sealed session."
            )
        checkpoint_payload = strict_json_loads(bound.checkpoint_state)
        if not isinstance(checkpoint_payload, dict):
            raise PFLiveSessionError("PF checkpoint state must be a JSON object.")
        if checkpoint_payload.get("control_policy") != expected_policy_provenance:
            raise PFLiveSessionError(
                "PF checkpoint state control policy differs from the sealed session."
            )
        rng_states = checkpoint_payload.get("rng_states")
        if not isinstance(rng_states, Mapping):
            raise PFLiveSessionError("PF checkpoint state lacks RNG provenance.")

        posterior_path = atomic_write_bytes(
            target / "pf_posterior.json",
            bound.posterior_json,
        )
        diagnostics_path = atomic_write_bytes(
            target / "pf_diagnostics.json",
            bound.completed.diagnostics_json,
        )
        checkpoint_state_path = atomic_write_bytes(
            target / "pf_state.json",
            bound.checkpoint_state,
        )
        particle_buffer = BytesIO()
        snapshot = bound.completed.particle_snapshot
        particle_arrays: dict[str, object] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "isotope_names": np.asarray(snapshot.isotope_order),
            "weights_n": np.asarray(snapshot.weights_n, dtype=np.float64),
            "original_particle_indices": np.asarray(
                snapshot.original_particle_indices,
                dtype=np.int64,
            ),
        }
        for isotope_index, isotope in enumerate(snapshot.isotope_order):
            prefix = f"isotope_{isotope_index:03d}"
            particle_arrays[f"{prefix}_positions_nk3"] = np.asarray(
                snapshot.positions_nk3_by_isotope[isotope],
                dtype=np.float64,
            )
            particle_arrays[f"{prefix}_surface_chart_ids_nk"] = np.asarray(
                snapshot.surface_chart_ids_nk_by_isotope[isotope],
                dtype=np.int64,
            )
            particle_arrays[f"{prefix}_surface_uv_nk2"] = np.asarray(
                snapshot.surface_uv_nk2_by_isotope[isotope],
                dtype=np.float64,
            )
            particle_arrays[f"{prefix}_strengths_nk"] = np.asarray(
                snapshot.strengths_nk_by_isotope[isotope],
                dtype=np.float64,
            )
            particle_arrays[f"{prefix}_source_mask_nk"] = np.asarray(
                snapshot.source_mask_nk_by_isotope[isotope],
                dtype=np.bool_,
            )
        np.savez_compressed(particle_buffer, **particle_arrays)
        particle_snapshot_path = atomic_write_bytes(
            target / "pf_particles.npz",
            particle_buffer.getvalue(),
        )
        evaluation_input_path = atomic_write_bytes(
            target / "pf_post_run_evaluation_input.json",
            _strict_live_artifact_json_bytes(
                _post_run_evaluation_input_payload(
                    self._estimator,
                    source_run_id=bound.completed.source_run_id,
                    measurement_log_sha256=bound.measurement_log_sha256,
                ),
                artifact_name="PF post-run evaluation input",
            ),
        )
        if evaluation_input_path.parent != target:
            raise PFLiveSessionError(
                "PF post-run evaluation input escaped the publication root."
            )
        resolved_config_sha256 = provenance.get("resolved_config_sha256")
        estimator_commit = provenance.get("estimator_commit")
        random_seed = provenance.get("random_seed")
        if (
            not isinstance(resolved_config_sha256, str)
            or not isinstance(estimator_commit, str)
            or isinstance(random_seed, bool)
            or not isinstance(random_seed, int)
        ):
            raise PFLiveSessionError("Bound PF provenance is incomplete.")
        checkpoint_manifest = {
            "schema_version": 2,
            "checkpoint_family": "pure_pf_causal_state",
            "checkpoint_id": f"pf-live-{bound.checkpoint_sha256[:24]}",
            "source_run_id": bound.completed.source_run_id,
            "measurement_log_schema_version": self._context.schema_version,
            "data_cutoff_step": int(self._records[-1].step_id),
            "data_cutoff_station": int(self._records[-1].station_id),
            "covered_step_ids": list(bound.completed.covered_step_ids),
            "covered_records_sha256": (
                bound.completed.covered_records_digest.sha256
            ),
            "prefix_measurement_log_sha256": bound.measurement_log_sha256,
            "pf_repository_commit": estimator_commit,
            "resolved_config_sha256": resolved_config_sha256,
            "control_policy": (
                bound.completed.control_policy_provenance.to_dict()
            ),
            "random_seed": random_seed,
            "state_artifact": checkpoint_state_path.name,
            "state_schema_version": 2,
            "state_artifact_sha256": bound.checkpoint_sha256,
            "rng_state_sha256": _strict_live_artifact_sha256(
                dict(rng_states),
                artifact_name="PF checkpoint RNG state",
            ),
            "safety": {
                "prefix_causal": True,
                "truth_read": False,
                "batch_feedback_applied": False,
            },
        }
        checkpoint_path = atomic_write_bytes(
            target / "pf_checkpoint.json",
            _strict_live_artifact_json_bytes(
                checkpoint_manifest,
                artifact_name="PF checkpoint manifest",
            ),
        )
        payload_inventory = build_artifact_inventory(target)
        publish_artifact_manifest(
            target / "pf_artifact_inventory.json",
            payload_inventory,
            metadata={
                "artifact_family": "pure_pf_live_result",
                "source_run_id": bound.completed.source_run_id,
                "measurement_log_sha256": bound.measurement_log_sha256,
            },
        )
        inventory = build_artifact_inventory(target)
        return PFPublishedLiveResult(
            root=target,
            posterior_path=posterior_path,
            diagnostics_path=diagnostics_path,
            checkpoint_path=checkpoint_path,
            checkpoint_state_path=checkpoint_state_path,
            particle_snapshot_path=particle_snapshot_path,
            post_run_evaluation_input_path=evaluation_input_path,
            posterior_sha256=bound.posterior_sha256,
            checkpoint_sha256=_strict_live_artifact_sha256(
                checkpoint_manifest,
                artifact_name="PF checkpoint manifest",
            ),
            result_sha256=inventory.sha256,
        )

    def publish_bound_result(
        self,
        output_dir: str | Path,
    ) -> PFPublishedLiveResult:
        """Publish one new directory-atomic package-owned PF result."""
        target = Path(output_dir).expanduser().resolve()
        with AtomicBundlePublisher(target, policy="create") as publisher:
            staged = self._publish_bound_result_into_staging(
                publisher.staging_path
            )
            inventory = publisher.publish()
        return PFPublishedLiveResult(
            root=target,
            posterior_path=target / staged.posterior_path.name,
            diagnostics_path=target / staged.diagnostics_path.name,
            checkpoint_path=target / staged.checkpoint_path.name,
            checkpoint_state_path=target / staged.checkpoint_state_path.name,
            particle_snapshot_path=target / staged.particle_snapshot_path.name,
            post_run_evaluation_input_path=(
                target / staged.post_run_evaluation_input_path.name
            ),
            posterior_sha256=staged.posterior_sha256,
            checkpoint_sha256=staged.checkpoint_sha256,
            result_sha256=inventory.sha256,
        )


__all__ = [
    "PFBoundLiveState",
    "PFCompletedLiveState",
    "PFPublishedLiveResult",
    "PFLiveSessionError",
    "PFLiveParticleSnapshot",
    "PFLiveSession",
    "ValidatedProductionPFConfig",
    "assimilate_persisted_station",
    "build_live_estimator",
    "live_posterior_summary",
    "load_production_live_pf_config",
    "measurement_record_to_station_input",
    "register_persisted_station_pose",
    "validate_production_live_pf_config",
]

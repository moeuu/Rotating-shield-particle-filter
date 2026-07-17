"""Apply estimator-neutral external candidates as valid PF MCMC proposals.

This module deliberately implements a fixed-cardinality position relocation,
not source birth, deletion, reweighting, or an external-objective update.  The
kernel is a Metropolis-within-Gibbs step targeting the PF count posterior
through an exact causal cutoff.  A defensive truncated-Gaussian component
makes both forward and reverse proposal densities positive and evaluable.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri

from pf.likelihood import expected_counts_per_source
from pf.particle_filter import IsotopeParticleFilter, MeasurementData
from pf.provenance import canonical_json_bytes, sha256_json
from pf.pure_estimator import PurePFEstimator
from runtime.measurement_log import MeasurementLog, MeasurementLogRecord


class ExternalRelocationError(RuntimeError):
    """Report an invalid directive, cutoff, or unsupported PF target."""


def covered_records_sha256(log: MeasurementLog, record_count: int) -> str:
    """Hash an exact record prefix without incorporating any unseen suffix."""
    count = int(record_count)
    if count < 0 or count > len(log.records):
        raise ValueError("record_count is outside the MeasurementLog.")
    isotope_order = tuple(str(value) for value in log.run_manifest.get("isotopes", ()))

    def _covariance_mapping(record: MeasurementLogRecord) -> object:
        """Return the shared name-addressed covariance representation."""
        if record.isotope_count_covariance is None:
            return None
        covariance = np.asarray(record.isotope_count_covariance, dtype=float)
        expected = (len(isotope_order), len(isotope_order))
        if covariance.shape != expected:
            raise ValueError(f"Isotope covariance must have shape {expected}.")
        return {
            row_isotope: {
                column_isotope: float(covariance[row_index, column_index])
                for column_index, column_isotope in enumerate(isotope_order)
            }
            for row_index, row_isotope in enumerate(isotope_order)
        }

    records = [
        {
            "step_id": int(record.step_id),
            "action_id": int(record.action_id),
            "station_id": int(record.station_id),
            "detector_pose_xyz": list(record.detector_pose_xyz),
            "detector_quat_wxyz": list(record.detector_quat_wxyz),
            "fe_orientation_index": int(record.fe_orientation_index),
            "pb_orientation_index": int(record.pb_orientation_index),
            "live_time_s": float(record.live_time_s),
            "travel_time_s": float(record.travel_time_s),
            "shield_actuation_time_s": float(record.shield_actuation_time_s),
            "energy_bin_edges_keV": np.asarray(
                record.energy_bin_edges_keV, dtype=float
            ).tolist(),
            "spectrum_counts": np.asarray(record.spectrum_counts, dtype=float).tolist(),
            "spectrum_variance": (
                None
                if record.spectrum_variance is None
                else np.asarray(record.spectrum_variance, dtype=float).tolist()
            ),
            "isotope_counts": (
                None
                if record.isotope_counts is None
                else {
                    str(key): float(value)
                    for key, value in sorted(record.isotope_counts.items())
                }
            ),
            "isotope_count_covariance": (_covariance_mapping(record)),
            "metadata": dict(record.metadata),
        }
        for record in log.records[:count]
    ]
    return sha256_json(records)


@dataclass(frozen=True)
class GaussianCandidate:
    """Describe one diagonal Gaussian component from an external estimator."""

    mean_xyz: tuple[float, float, float]
    sigma_xyz_m: tuple[float, float, float]
    weight: float
    proposal_id: str = "candidate"
    snapshot_candidate_id: str = "candidate"


@dataclass(frozen=True)
class IsotopeCandidateMixture:
    """Hold the fixed external Gaussian mixture for one isotope."""

    candidates: tuple[GaussianCandidate, ...]


@dataclass(frozen=True)
class ExternalRelocationDirective:
    """Bind one proposal-only relocation to an exact PF causal boundary."""

    schema_version: int
    directive_id: str
    directive_kind: str
    proposal_source: str
    source_run_id: str
    covered_records_sha256: str
    source_measurement_log_sha256: str | None
    pf_resolved_config_sha256: str
    apply_after_record_index: int
    data_cutoff_step: int
    data_cutoff_station: int
    covered_step_ids: tuple[int, ...]
    defensive_weight: float
    defensive_sigma_xyz_m: tuple[float, float, float]
    isotopes: Mapping[str, IsotopeCandidateMixture]
    snapshot_id: str | None = None
    snapshot_sha256: str | None = None
    corroboration_min_step: int | None = None
    source_directive_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON representation of this directive."""
        return {
            "schema_version": int(self.schema_version),
            "directive_id": self.directive_id,
            "directive_kind": self.directive_kind,
            "proposal_source": self.proposal_source,
            "source_run_id": self.source_run_id,
            "covered_records_sha256": self.covered_records_sha256,
            "source_measurement_log_sha256": self.source_measurement_log_sha256,
            "pf_resolved_config_sha256": self.pf_resolved_config_sha256,
            "apply_after_record_index": int(self.apply_after_record_index),
            "data_cutoff_step": int(self.data_cutoff_step),
            "data_cutoff_station": int(self.data_cutoff_station),
            "covered_step_ids": [int(value) for value in self.covered_step_ids],
            "defensive_weight": float(self.defensive_weight),
            "defensive_sigma_xyz_m": [
                float(value) for value in self.defensive_sigma_xyz_m
            ],
            "isotopes": {
                isotope: {
                    "candidates": [
                        {
                            "mean_xyz": [float(value) for value in item.mean_xyz],
                            "sigma_xyz_m": [float(value) for value in item.sigma_xyz_m],
                            "weight": float(item.weight),
                            "proposal_id": item.proposal_id,
                            "snapshot_candidate_id": item.snapshot_candidate_id,
                        }
                        for item in mixture.candidates
                    ]
                }
                for isotope, mixture in sorted(self.isotopes.items())
            },
            "snapshot_id": self.snapshot_id,
            "snapshot_sha256": self.snapshot_sha256,
            "corroboration_min_step": self.corroboration_min_step,
        }


def _finite_xyz(
    value: object, *, name: str, positive: bool = False
) -> tuple[float, ...]:
    """Return a validated finite XYZ tuple."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ExternalRelocationError(f"{name} must be a finite XYZ vector.")
    if positive and np.any(array <= 0.0):
        raise ExternalRelocationError(f"{name} must be strictly positive.")
    return tuple(float(item) for item in array)


def directive_from_mapping(payload: Mapping[str, Any]) -> ExternalRelocationDirective:
    """Parse and fail-closed validate one external relocation directive."""
    if str(payload.get("directive_kind", "")) == "proposal_only_mh":
        return _orchestrator_directive_from_mapping(payload)
    try:
        schema_version = int(payload["schema_version"])
        directive_id = str(payload["directive_id"]).strip()
        directive_kind = str(payload["directive_kind"]).strip()
        proposal_source = str(payload["proposal_source"]).strip()
        source_run_id = str(payload["source_run_id"]).strip()
        covered_digest = str(payload["covered_records_sha256"]).strip().lower()
        raw_source_digest = payload.get("source_measurement_log_sha256")
        source_digest = (
            None
            if raw_source_digest is None
            else str(raw_source_digest).strip().lower()
        )
        config_digest = str(payload["pf_resolved_config_sha256"]).strip().lower()
        record_index = int(payload["apply_after_record_index"])
        cutoff_step = int(payload["data_cutoff_step"])
        cutoff_station = int(payload["data_cutoff_station"])
        covered_steps = tuple(int(value) for value in payload["covered_step_ids"])
        defensive_weight = float(payload["defensive_weight"])
        defensive_sigma = _finite_xyz(
            payload["defensive_sigma_xyz_m"],
            name="defensive_sigma_xyz_m",
            positive=True,
        )
        raw_isotopes = payload["isotopes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalRelocationError(
            "Malformed external relocation directive."
        ) from exc
    if schema_version != 1:
        raise ExternalRelocationError(
            "Only external directive schema version 1 is supported."
        )
    if not directive_id:
        raise ExternalRelocationError("directive_id must be non-empty.")
    if directive_kind != "fixed_cardinality_position_relocation":
        raise ExternalRelocationError(
            "directive_kind must be fixed_cardinality_position_relocation."
        )
    if not proposal_source:
        raise ExternalRelocationError("proposal_source must be non-empty.")
    if not source_run_id:
        raise ExternalRelocationError("source_run_id must be non-empty.")
    for name, digest in (
        ("covered_records_sha256", covered_digest),
        ("pf_resolved_config_sha256", config_digest),
    ):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ExternalRelocationError(f"{name} must be a lowercase SHA-256 digest.")
    if source_digest is not None and (
        len(source_digest) != 64
        or any(char not in "0123456789abcdef" for char in source_digest)
    ):
        raise ExternalRelocationError(
            "source_measurement_log_sha256 must be a lowercase SHA-256 digest."
        )
    if record_index < 0 or cutoff_step < 0 or cutoff_station < 0:
        raise ExternalRelocationError(
            "Directive cutoff identifiers must be non-negative."
        )
    if not covered_steps or tuple(sorted(set(covered_steps))) != covered_steps:
        raise ExternalRelocationError(
            "covered_step_ids must be a non-empty strictly increasing sequence."
        )
    if not 0.0 < defensive_weight <= 1.0:
        raise ExternalRelocationError("defensive_weight must lie in (0, 1].")
    if not isinstance(raw_isotopes, Mapping) or not raw_isotopes:
        raise ExternalRelocationError("isotopes must be a non-empty object.")
    isotopes: dict[str, IsotopeCandidateMixture] = {}
    for isotope, raw_mixture in raw_isotopes.items():
        if not isinstance(raw_mixture, Mapping):
            raise ExternalRelocationError("Each isotope mixture must be an object.")
        raw_candidates = raw_mixture.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(
            raw_candidates, (str, bytes)
        ):
            raise ExternalRelocationError("candidates must be an array.")
        candidates: list[GaussianCandidate] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                raise ExternalRelocationError("Each candidate must be an object.")
            try:
                mean = _finite_xyz(raw_candidate["mean_xyz"], name="mean_xyz")
                sigma = _finite_xyz(
                    raw_candidate["sigma_xyz_m"],
                    name="sigma_xyz_m",
                    positive=True,
                )
                weight = float(raw_candidate["weight"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ExternalRelocationError("Malformed Gaussian candidate.") from exc
            if not np.isfinite(weight) or weight <= 0.0:
                raise ExternalRelocationError("Candidate weights must be positive.")
            candidates.append(
                GaussianCandidate(
                    mean_xyz=(float(mean[0]), float(mean[1]), float(mean[2])),
                    sigma_xyz_m=(float(sigma[0]), float(sigma[1]), float(sigma[2])),
                    weight=weight,
                    proposal_id=str(raw_candidate.get("proposal_id", "candidate")),
                    snapshot_candidate_id=str(
                        raw_candidate.get("snapshot_candidate_id", "candidate")
                    ),
                )
            )
        if not candidates:
            raise ExternalRelocationError(
                "Every isotope requires at least one candidate."
            )
        isotope_name = str(isotope).strip()
        if not isotope_name:
            raise ExternalRelocationError("Isotope names must be non-empty.")
        isotopes[isotope_name] = IsotopeCandidateMixture(tuple(candidates))
    return ExternalRelocationDirective(
        schema_version=schema_version,
        directive_id=directive_id,
        directive_kind=directive_kind,
        proposal_source=proposal_source,
        source_run_id=source_run_id,
        covered_records_sha256=covered_digest,
        source_measurement_log_sha256=source_digest,
        pf_resolved_config_sha256=config_digest,
        apply_after_record_index=record_index,
        data_cutoff_step=cutoff_step,
        data_cutoff_station=cutoff_station,
        covered_step_ids=covered_steps,
        defensive_weight=defensive_weight,
        defensive_sigma_xyz_m=(
            float(defensive_sigma[0]),
            float(defensive_sigma[1]),
            float(defensive_sigma[2]),
        ),
        isotopes=isotopes,
        source_directive_sha256=sha256_json(payload),
    )


def _required_sha256(payload: Mapping[str, Any], name: str) -> str:
    """Return one required lowercase SHA-256 field."""
    value = str(payload.get(name, "")).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExternalRelocationError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _orchestrator_directive_from_mapping(
    payload: Mapping[str, Any],
) -> ExternalRelocationDirective:
    """Translate the orchestrator PFDirective v1 into the relocation kernel."""
    try:
        schema_version = int(payload["schema_version"])
        directive_id = str(payload["directive_id"]).strip()
        snapshot_id = str(payload["snapshot_id"]).strip()
        cutoff_step = int(payload["data_cutoff_step"])
        cutoff_station = int(payload["data_cutoff_station"])
        apply_after_step = int(payload["apply_after_step"])
        corroboration_min_step = int(payload["corroboration_min_step"])
        covered_steps = tuple(int(value) for value in payload["covered_step_ids"])
        proposals = payload["proposals"]
        safety = payload["safety_policy"]
        provenance = payload["provenance"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalRelocationError("Malformed orchestrator PFDirective v1.") from exc
    if schema_version != 1 or not directive_id or not snapshot_id:
        raise ExternalRelocationError("Invalid orchestrator PFDirective identity.")
    if payload.get("cutoff_station_complete") is not True:
        raise ExternalRelocationError("PFDirective cutoff station must be complete.")
    if apply_after_step != cutoff_step or corroboration_min_step != cutoff_step + 1:
        raise ExternalRelocationError(
            "PFDirective must apply at cutoff and corroborate strictly afterward."
        )
    if not isinstance(safety, Mapping) or any(
        (
            safety.get("direct_mle_objective_reweight") is not False,
            safety.get("hard_prune_authorized") is not False,
            safety.get("future_only_corroboration") is not True,
            safety.get("once_only_application") is not True,
            safety.get("requires_target_preserving_mh") is not True,
        )
    ):
        raise ExternalRelocationError(
            "PFDirective safety_policy is not proposal-only MH."
        )
    if not isinstance(provenance, Mapping):
        raise ExternalRelocationError("PFDirective provenance must be an object.")
    source_run_id = str(
        payload.get("source_run_id", provenance.get("source_run_id", ""))
    ).strip()
    if not source_run_id:
        raise ExternalRelocationError("PFDirective must bind source_run_id.")
    covered_digest_payload = {
        **dict(provenance),
        **{key: payload[key] for key in ("covered_records_sha256",) if key in payload},
    }
    covered_digest = _required_sha256(
        covered_digest_payload,
        "covered_records_sha256",
    )
    config_digest_payload = {
        **dict(provenance),
        **{
            key: payload[key]
            for key in ("pf_resolved_config_sha256",)
            if key in payload
        },
    }
    config_digest = _required_sha256(
        config_digest_payload,
        "pf_resolved_config_sha256",
    )
    raw_source_log_digest = payload.get("source_measurement_log_sha256")
    source_log_digest = (
        None
        if raw_source_log_digest is None
        else _required_sha256(payload, "source_measurement_log_sha256")
    )
    snapshot_digest = _required_sha256(payload, "snapshot_sha256")
    if not covered_steps or tuple(sorted(set(covered_steps))) != covered_steps:
        raise ExternalRelocationError(
            "PFDirective coverage must be strictly increasing."
        )
    if covered_steps[-1] != cutoff_step:
        raise ExternalRelocationError("PFDirective coverage must end at its cutoff.")
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise ExternalRelocationError("PFDirective proposals must be an array.")
    grouped: dict[str, list[GaussianCandidate]] = {}
    defensive_weights: set[float] = set()
    defensive_sigmas: set[tuple[float, float, float]] = set()
    for raw in proposals:
        if not isinstance(raw, Mapping):
            raise ExternalRelocationError("PFDirective proposal must be an object.")
        kernel = raw.get("proposal_kernel")
        if not isinstance(kernel, Mapping) or kernel.get("family") != (
            "defensive_truncated_gaussian_position"
        ):
            raise ExternalRelocationError(
                "PFDirective requires defensive_truncated_gaussian_position kernels."
            )
        sigma = _finite_xyz(
            kernel.get("position_sigma_xyz_m"),
            name="position_sigma_xyz_m",
            positive=True,
        )
        defensive_weight = float(kernel.get("defensive_weight", np.nan))
        candidate_weight = float(kernel.get("candidate_weight", np.nan))
        if not 0.0 < defensive_weight <= 1.0:
            raise ExternalRelocationError("Kernel defensive_weight must lie in (0, 1].")
        if not np.isfinite(candidate_weight) or candidate_weight <= 0.0:
            raise ExternalRelocationError("Kernel candidate_weight must be positive.")
        proposal_id = str(raw.get("proposal_id", "")).strip()
        snapshot_candidate_id = str(raw.get("snapshot_candidate_id", "")).strip()
        isotope = str(raw.get("isotope", "")).strip()
        if not proposal_id or not snapshot_candidate_id or not isotope:
            raise ExternalRelocationError(
                "PFDirective proposal identity is incomplete."
            )
        mean_payload = raw.get("candidate_mean_xyz", raw.get("position_xyz"))
        mean = _finite_xyz(mean_payload, name="candidate_mean_xyz")
        grouped.setdefault(isotope, []).append(
            GaussianCandidate(
                mean_xyz=(float(mean[0]), float(mean[1]), float(mean[2])),
                sigma_xyz_m=(float(sigma[0]), float(sigma[1]), float(sigma[2])),
                weight=candidate_weight,
                proposal_id=proposal_id,
                snapshot_candidate_id=snapshot_candidate_id,
            )
        )
        defensive_weights.add(defensive_weight)
        defensive_sigmas.add((float(sigma[0]), float(sigma[1]), float(sigma[2])))
    if not grouped:
        raise ExternalRelocationError("PFDirective requires at least one proposal.")
    if len(defensive_weights) != 1 or len(defensive_sigmas) != 1:
        raise ExternalRelocationError(
            "Relocation v1 requires one shared defensive weight and XYZ sigma."
        )
    sigma_xyz = next(iter(defensive_sigmas))
    return ExternalRelocationDirective(
        schema_version=1,
        directive_id=directive_id,
        directive_kind="proposal_only_mh",
        proposal_source="orchestrator_pf_directive_v1",
        source_run_id=source_run_id,
        covered_records_sha256=covered_digest,
        source_measurement_log_sha256=source_log_digest,
        pf_resolved_config_sha256=config_digest,
        apply_after_record_index=len(covered_steps) - 1,
        data_cutoff_step=cutoff_step,
        data_cutoff_station=cutoff_station,
        covered_step_ids=covered_steps,
        defensive_weight=next(iter(defensive_weights)),
        defensive_sigma_xyz_m=sigma_xyz,
        isotopes={
            isotope: IsotopeCandidateMixture(tuple(candidates))
            for isotope, candidates in grouped.items()
        },
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_digest,
        corroboration_min_step=corroboration_min_step,
        source_directive_sha256=sha256_json(payload),
    )


def load_directive_schedule(
    path_or_payload: str | Path | Mapping[str, Any],
) -> tuple[ExternalRelocationDirective, ...]:
    """Load a schema-v1 directive schedule and deduplicate identical IDs."""
    if isinstance(path_or_payload, (str, Path)):
        try:
            payload = json.loads(Path(path_or_payload).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalRelocationError(
                "Cannot load directive schedule JSON."
            ) from exc
    else:
        payload = dict(path_or_payload)
    if int(payload.get("schema_version", -1)) != 1:
        raise ExternalRelocationError(
            "Only directive schedule schema version 1 is supported."
        )
    raw_directives = (
        [payload] if "directive_kind" in payload else payload.get("directives")
    )
    if not isinstance(raw_directives, Sequence) or isinstance(
        raw_directives, (str, bytes)
    ):
        raise ExternalRelocationError("directives must be an array.")
    by_id: dict[str, ExternalRelocationDirective] = {}
    canonical_by_id: dict[str, bytes] = {}
    for raw in raw_directives:
        if not isinstance(raw, Mapping):
            raise ExternalRelocationError("Every directive must be an object.")
        directive = directive_from_mapping(raw)
        canonical = canonical_json_bytes(directive.to_dict())
        previous = canonical_by_id.get(directive.directive_id)
        if previous is not None and previous != canonical:
            raise ExternalRelocationError(
                f"Conflicting duplicate directive_id {directive.directive_id!r}."
            )
        by_id[directive.directive_id] = directive
        canonical_by_id[directive.directive_id] = canonical
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (item.apply_after_record_index, item.directive_id),
        )
    )


def _log_subtract_exp(
    log_x: NDArray[np.float64], log_y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return log(exp(log_x) - exp(log_y)) for log_x >= log_y."""
    delta = np.minimum(np.asarray(log_y) - np.asarray(log_x), -np.finfo(float).eps)
    return np.asarray(log_x) + np.log1p(-np.exp(delta))


def _log_truncated_normal_density(
    points: NDArray[np.float64],
    means: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate independent box-truncated Gaussian log densities."""
    x = np.asarray(points, dtype=float)
    mean = np.asarray(means, dtype=float)
    sigma = np.asarray(sigmas, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    standardized = (x - mean) / sigma
    log_pdf = -0.5 * standardized**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    alpha = (lo - mean) / sigma
    beta = (hi - mean) / sigma
    log_z = _log_subtract_exp(log_ndtr(beta), log_ndtr(alpha))
    result = np.sum(log_pdf - log_z, axis=-1)
    inside = np.all((x >= lo) & (x <= hi), axis=-1)
    return np.where(inside, result, -np.inf)


def _mixture_arrays(
    mixture: IsotopeCandidateMixture,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized means, diagonal sigmas, and weights."""
    means = np.asarray([item.mean_xyz for item in mixture.candidates], dtype=float)
    sigmas = np.asarray([item.sigma_xyz_m for item in mixture.candidates], dtype=float)
    weights = np.asarray([item.weight for item in mixture.candidates], dtype=float)
    weights /= np.sum(weights)
    return means, sigmas, weights


def log_defensive_mixture_density(
    points: NDArray[np.float64],
    current_centers: NDArray[np.float64],
    mixture: IsotopeCandidateMixture,
    *,
    defensive_weight: float,
    defensive_sigma_xyz_m: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> NDArray[np.float64]:
    """Evaluate q(points | current) for the defensive Gaussian mixture."""
    x = np.asarray(points, dtype=float).reshape(-1, 3)
    current = np.asarray(current_centers, dtype=float).reshape(-1, 3)
    if x.shape != current.shape:
        raise ValueError("points and current_centers must have matching shape.")
    lo = np.asarray(lower, dtype=float).reshape(1, 3)
    hi = np.asarray(upper, dtype=float).reshape(1, 3)
    local_sigma = np.broadcast_to(
        np.asarray(defensive_sigma_xyz_m, dtype=float).reshape(1, 3),
        x.shape,
    )
    local_log = _log_truncated_normal_density(x, current, local_sigma, lo, hi)
    means, sigmas, weights = _mixture_arrays(mixture)
    external_log_components = _log_truncated_normal_density(
        x[:, None, :],
        means[None, :, :],
        sigmas[None, :, :],
        lo[:, None, :],
        hi[:, None, :],
    )
    external_log = logsumexp(
        np.log(weights)[None, :] + external_log_components,
        axis=1,
    )
    defensive = float(defensive_weight)
    if defensive >= 1.0:
        return local_log
    return np.logaddexp(
        np.log(defensive) + local_log,
        np.log1p(-defensive) + external_log,
    )


def _sample_truncated_normal(
    means: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw independent box-truncated Gaussian vectors by inverse CDF."""
    alpha = (lower - means) / sigmas
    beta = (upper - means) / sigmas
    cdf_lo = ndtr(alpha)
    cdf_hi = ndtr(beta)
    span = np.maximum(cdf_hi - cdf_lo, np.finfo(float).tiny)
    uniforms = rng.random(means.shape)
    probabilities = cdf_lo + uniforms * span
    probabilities = np.clip(
        probabilities,
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )
    return np.clip(means + sigmas * ndtri(probabilities), lower, upper)


def sample_defensive_mixture(
    current_centers: NDArray[np.float64],
    mixture: IsotopeCandidateMixture,
    *,
    defensive_weight: float,
    defensive_sigma_xyz_m: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Draw all particle proposals in one vectorized mixture sample."""
    current = np.asarray(current_centers, dtype=float).reshape(-1, 3)
    count = int(current.shape[0])
    means, sigmas, weights = _mixture_arrays(mixture)
    component = np.full(count, -1, dtype=np.int64)
    external = rng.random(count) >= float(defensive_weight)
    if np.any(external):
        component[external] = rng.choice(
            means.shape[0],
            size=int(np.count_nonzero(external)),
            p=weights,
        )
    draw_means = current.copy()
    draw_sigmas = np.broadcast_to(
        np.asarray(defensive_sigma_xyz_m, dtype=float).reshape(1, 3),
        current.shape,
    ).copy()
    if np.any(external):
        draw_means[external] = means[component[external]]
        draw_sigmas[external] = sigmas[component[external]]
    lo = np.broadcast_to(np.asarray(lower, dtype=float).reshape(1, 3), current.shape)
    hi = np.broadcast_to(np.asarray(upper, dtype=float).reshape(1, 3), current.shape)
    return _sample_truncated_normal(draw_means, draw_sigmas, lo, hi, rng), component


def metropolis_log_acceptance(
    target_log_density_delta: NDArray[np.float64],
    log_proposal_correction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return min(0, target delta + log q(old|new) - log q(new|old))."""
    raw = np.asarray(target_log_density_delta, dtype=float) + np.asarray(
        log_proposal_correction, dtype=float
    )
    return np.where(np.isnan(raw), -np.inf, np.minimum(0.0, raw))


def _validate_supported_target(filt: IsotopeParticleFilter) -> None:
    """Reject targets whose complete density is not implemented by this kernel."""
    if filt._source_prior_is_surface():
        raise ExternalRelocationError(
            "Projected Gaussian relocation is invalid for a surface prior; "
            "hybrid relocation v1 supports only the continuous uniform-volume prior."
        )
    unsupported = {
        "shield_contrast_likelihood_enable": bool(
            filt.config.shield_contrast_likelihood_enable
        ),
        "shield_view_ratio_likelihood_enable": bool(
            filt.config.shield_view_ratio_likelihood_enable
        ),
        "station_view_covariance_enable": bool(
            filt.config.station_view_covariance_enable
        )
        and float(filt.config.station_view_correlated_spectrum_fraction) > 0.0,
    }
    active = sorted(name for name, enabled in unsupported.items() if enabled)
    if active:
        raise ExternalRelocationError(
            "Hybrid relocation v1 cannot yet evaluate the complete configured PF "
            f"target with: {', '.join(active)}."
        )


def batched_count_log_targets_for_relocations(
    filt: IsotopeParticleFilter,
    data: MeasurementData,
    particle_indices: NDArray[np.int64],
    source_slots: NDArray[np.int64],
    proposed_xyz: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate old/new full-history count targets in one response batch."""
    from pf import gpu_utils
    import torch

    indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
    slots = np.asarray(source_slots, dtype=np.int64).reshape(-1)
    proposed = np.asarray(proposed_xyz, dtype=float).reshape(-1, 3)
    if not (indices.size == slots.size == proposed.shape[0]):
        raise ValueError("Relocation arrays must have matching particle counts.")
    if indices.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty
    positions_t, strengths_t, backgrounds_t, mask_t = gpu_utils.pack_states(
        [particle.state for particle in filt.continuous_particles],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    old_positions = positions_t.detach().cpu().numpy()[indices]
    strengths = strengths_t.detach().cpu().numpy()[indices]
    mask = mask_t.detach().cpu().numpy()[indices]
    backgrounds = backgrounds_t.detach().cpu().numpy()[indices]
    if np.any(slots < 0) or np.any(slots >= old_positions.shape[1]):
        raise ValueError("source_slots contains an out-of-range slot.")
    if not np.all(mask[np.arange(indices.size), slots]):
        raise ValueError("source_slots must select active sources.")
    trial_positions = old_positions.copy()
    trial_positions[np.arange(indices.size), slots] = proposed
    source_count = int(old_positions.shape[1])
    combined_positions = np.concatenate([old_positions, trial_positions], axis=0)
    combined_strengths = np.concatenate([strengths * mask, strengths * mask], axis=0)
    flat_positions = combined_positions.reshape(-1, 3)
    flat_strengths = combined_strengths.reshape(-1)
    lambda_flat = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=data.detector_positions,
        sources=flat_positions,
        strengths=flat_strengths,
        live_times=data.live_times,
        fe_indices=data.fe_indices,
        pb_indices=data.pb_indices,
        source_scale=filt._measurement_source_scale_vector(
            data.fe_indices,
            data.pb_indices,
        ),
    )
    lambda_sources = np.asarray(lambda_flat, dtype=float).reshape(
        int(data.z_k.size),
        2 * int(indices.size),
        source_count,
    )
    combined_backgrounds = np.concatenate([backgrounds, backgrounds])
    lambda_total = np.sum(lambda_sources, axis=2) + (
        np.asarray(data.live_times, dtype=float)[:, None]
        * combined_backgrounds[None, :]
    )
    log_targets = filt._count_log_likelihood_matrix_np(
        data.z_k,
        lambda_total,
        observation_count_variance=data.observation_variances,
    )
    split = int(indices.size)
    return log_targets[:split], log_targets[split:]


def scalar_count_log_targets_for_relocations(
    filt: IsotopeParticleFilter,
    data: MeasurementData,
    particle_indices: NDArray[np.int64],
    source_slots: NDArray[np.int64],
    proposed_xyz: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate relocations serially as a small deterministic test oracle."""
    old_values: list[float] = []
    new_values: list[float] = []
    for particle_index, source_slot, proposed in zip(
        np.asarray(particle_indices, dtype=int),
        np.asarray(source_slots, dtype=int),
        np.asarray(proposed_xyz, dtype=float),
    ):
        state = filt.continuous_particles[int(particle_index)].state
        old_values.append(float(filt._trial_log_likelihood(state, data)))
        trial = state.copy()
        trial.positions[int(source_slot)] = np.asarray(proposed, dtype=float)
        new_values.append(float(filt._trial_log_likelihood(trial, data)))
    return np.asarray(old_values, dtype=float), np.asarray(new_values, dtype=float)


def _directive_seed(
    base_seed: int,
    directive: ExternalRelocationDirective,
    isotope: str,
) -> int:
    """Derive an order- and future-suffix-independent relocation RNG seed."""
    mixture = directive.isotopes[isotope]
    causal_identity = {
        "covered_records_sha256": directive.covered_records_sha256,
        "data_cutoff_step": directive.data_cutoff_step,
        "data_cutoff_station": directive.data_cutoff_station,
        "defensive_weight": directive.defensive_weight,
        "defensive_sigma_xyz_m": directive.defensive_sigma_xyz_m,
        "isotope": isotope,
        "candidates": [
            {
                "mean_xyz": candidate.mean_xyz,
                "sigma_xyz_m": candidate.sigma_xyz_m,
                "weight": candidate.weight,
            }
            for candidate in mixture.candidates
        ],
    }
    payload = canonical_json_bytes(
        {"base_seed": int(base_seed), "causal_identity": causal_identity}
    )
    return int.from_bytes(sha256(payload).digest()[:8], byteorder="big", signed=False)


def apply_relocation_to_filter(
    filt: IsotopeParticleFilter,
    data: MeasurementData,
    directive: ExternalRelocationDirective,
    isotope: str,
    *,
    base_seed: int,
) -> dict[str, Any]:
    """Apply one target-preserving fixed-cardinality relocation to a filter."""
    _validate_supported_target(filt)
    mixture = directive.isotopes[isotope]
    lower = np.asarray(filt.config.position_min, dtype=float)
    upper = np.asarray(filt.config.position_max, dtype=float)
    means, _sigmas, _weights = _mixture_arrays(mixture)
    if np.any(means < lower[None, :]) or np.any(means > upper[None, :]):
        raise ExternalRelocationError(
            f"Directive {directive.directive_id!r} has {isotope} means outside PF bounds."
        )
    source_counts = np.asarray(
        [particle.state.num_sources for particle in filt.continuous_particles],
        dtype=np.int64,
    )
    particle_indices = np.flatnonzero(source_counts > 0).astype(np.int64)
    if particle_indices.size == 0:
        return {
            "isotope": isotope,
            "eligible_particle_count": 0,
            "accepted_count": 0,
            "particle_indices": [],
            "source_slots": [],
            "proposal_components": [],
            "proposed_xyz": [],
            "target_log_density_delta": [],
            "log_proposal_correction": [],
            "log_acceptance_ratio": [],
            "log_uniform_draws": [],
            "accepted": [],
            "particle_log_weights_changed": False,
        }
    rng = np.random.default_rng(_directive_seed(base_seed, directive, isotope))
    selected_counts = source_counts[particle_indices]
    source_slots = np.floor(rng.random(particle_indices.size) * selected_counts).astype(
        np.int64
    )
    current = np.asarray(
        [
            filt.continuous_particles[int(particle_index)].state.positions[
                int(source_slot)
            ]
            for particle_index, source_slot in zip(particle_indices, source_slots)
        ],
        dtype=float,
    )
    proposed, components = sample_defensive_mixture(
        current,
        mixture,
        defensive_weight=directive.defensive_weight,
        defensive_sigma_xyz_m=directive.defensive_sigma_xyz_m,
        lower=lower,
        upper=upper,
        rng=rng,
    )
    old_target, new_target = batched_count_log_targets_for_relocations(
        filt,
        data,
        particle_indices,
        source_slots,
        proposed,
    )
    log_q_forward = log_defensive_mixture_density(
        proposed,
        current,
        mixture,
        defensive_weight=directive.defensive_weight,
        defensive_sigma_xyz_m=directive.defensive_sigma_xyz_m,
        lower=lower,
        upper=upper,
    )
    log_q_reverse = log_defensive_mixture_density(
        current,
        proposed,
        mixture,
        defensive_weight=directive.defensive_weight,
        defensive_sigma_xyz_m=directive.defensive_sigma_xyz_m,
        lower=lower,
        upper=upper,
    )
    target_delta = new_target - old_target
    q_correction = log_q_reverse - log_q_forward
    log_alpha = metropolis_log_acceptance(target_delta, q_correction)
    uniforms = np.maximum(rng.random(particle_indices.size), np.finfo(float).tiny)
    log_uniforms = np.log(uniforms)
    accepted = log_uniforms < log_alpha
    weights_before = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles], dtype=float
    )
    for particle_index, source_slot, proposed_position in zip(
        particle_indices[accepted],
        source_slots[accepted],
        proposed[accepted],
    ):
        filt.continuous_particles[int(particle_index)].state.positions[
            int(source_slot)
        ] = proposed_position
    weights_after = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles], dtype=float
    )
    if not np.array_equal(weights_before, weights_after):
        raise AssertionError("Relocation must never modify PF particle weights.")
    if np.any(accepted):
        filt.align_continuous_labels()
    return {
        "isotope": isotope,
        "eligible_particle_count": int(particle_indices.size),
        "accepted_count": int(np.count_nonzero(accepted)),
        "particle_indices": particle_indices.tolist(),
        "source_slots": source_slots.tolist(),
        "proposal_components": components.tolist(),
        "proposed_xyz": proposed.tolist(),
        "target_log_density_delta": target_delta.tolist(),
        "log_proposal_correction": q_correction.tolist(),
        "log_acceptance_ratio": log_alpha.tolist(),
        "log_uniform_draws": log_uniforms.tolist(),
        "accepted": accepted.tolist(),
        "particle_log_weights_changed": False,
    }


def aggregate_candidate_outcomes(
    mixture: IsotopeCandidateMixture,
    particle_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Summarize every particle-level MH result without choosing a representative.

    Each eligible particle draws exactly one component: the defensive component
    (encoded as ``-1``) or one external candidate.  Counts are candidate-local,
    so ``not_sampled_count`` includes defensive draws and draws assigned to the
    other candidates.  The legacy scalar MH evidence is emitted only when that
    candidate was attempted exactly once; a many-particle result must be read
    from the aggregate counts or the detailed particle receipt.
    """
    eligible_count = int(particle_receipt["eligible_particle_count"])
    components = np.asarray(particle_receipt["proposal_components"], dtype=int)
    accepted = np.asarray(particle_receipt["accepted"], dtype=bool)
    ratios = np.asarray(particle_receipt["log_acceptance_ratio"], dtype=float)
    draws = np.asarray(particle_receipt["log_uniform_draws"], dtype=float)
    expected_shape = (eligible_count,)
    for name, values in (
        ("proposal_components", components),
        ("accepted", accepted),
        ("log_acceptance_ratio", ratios),
        ("log_uniform_draws", draws),
    ):
        if values.shape != expected_shape:
            raise AssertionError(
                f"Particle receipt {name} must have shape {expected_shape}."
            )

    outcomes: list[dict[str, Any]] = []
    for component_index, candidate in enumerate(mixture.candidates):
        matching = np.flatnonzero(components == component_index)
        attempt_count = int(matching.size)
        accepted_count = int(np.count_nonzero(accepted[matching]))
        rejected_count = attempt_count - accepted_count
        not_sampled_count = eligible_count - attempt_count
        if attempt_count == 0:
            outcome = "not_applied"
        elif accepted_count == attempt_count:
            outcome = "mh_accepted"
        elif rejected_count == attempt_count:
            outcome = "mh_rejected"
        else:
            outcome = "mh_mixed"
        singleton = int(matching[0]) if attempt_count == 1 else None
        outcomes.append(
            {
                "proposal_id": candidate.proposal_id,
                "outcome": outcome,
                "mh_attempt_count": attempt_count,
                "mh_accepted_count": accepted_count,
                "mh_rejected_count": rejected_count,
                "not_sampled_count": not_sampled_count,
                "eligible_particle_count": eligible_count,
                "mh_log_acceptance_ratio": (
                    None if singleton is None else float(ratios[singleton])
                ),
                "mh_log_uniform_draw": (
                    None if singleton is None else float(draws[singleton])
                ),
            }
        )
    return outcomes


def pre_update_predictive_counts(
    estimator: PurePFEstimator,
    record: MeasurementLogRecord,
    *,
    record_index: int,
) -> dict[str, Any]:
    """Return observation-independent per-row PF predictive moments."""
    if estimator.kernel_cache is None:
        # This is the same lazy initialization that the immediately following
        # PF update performs.  It consumes no observation and keeps the random
        # stream identical because no stochastic operation occurs in between.
        estimator._ensure_kernel_cache()
    pose = np.asarray(record.detector_pose_xyz, dtype=float)
    isotope_payload: dict[str, Any] = {}
    for isotope, filt in sorted(estimator.filters.items()):
        expected = filt._continuous_expected_counts_pair_at_pose_cpu(
            detector_pos=pose,
            fe_index=int(record.fe_orientation_index),
            pb_index=int(record.pb_orientation_index),
            live_time_s=float(record.live_time_s),
        )
        weights = np.asarray(filt.continuous_weights, dtype=float)
        mean = float(np.sum(weights * expected)) if expected.size else 0.0
        epistemic = (
            float(np.sum(weights * (expected - mean) ** 2)) if expected.size else 0.0
        )
        isotope_payload[str(isotope)] = {
            "mean_counts": mean,
            "epistemic_variance_counts2": epistemic,
            "poisson_predictive_variance_counts2": mean + epistemic,
        }
    return {
        "schema_version": 1,
        "record_index": int(record_index),
        "step_id": int(record.step_id),
        "station_id": int(record.station_id),
        "action_id": int(record.action_id),
        "computed_before_observation_update": True,
        "uses_observed_isotope_counts": False,
        "isotopes": isotope_payload,
    }


class ExternalRelocationSchedule:
    """Validate, apply once, and receipt a deterministic directive schedule."""

    def __init__(
        self,
        directives: Sequence[ExternalRelocationDirective],
        *,
        log: MeasurementLog,
        estimator: PurePFEstimator,
        base_seed: int,
    ) -> None:
        """Bind every directive to this exact log and resolved PF config."""
        self.log = log
        self.estimator = estimator
        self.base_seed = int(base_seed)
        self.directives = tuple(directives)
        self._by_record: dict[int, list[ExternalRelocationDirective]] = {}
        self._applied_ids: set[str] = set()
        self.receipts: list[dict[str, Any]] = []
        for directive in self.directives:
            self._validate_binding(directive)
            self._by_record.setdefault(directive.apply_after_record_index, []).append(
                directive
            )

    def _validate_binding(self, directive: ExternalRelocationDirective) -> None:
        """Fail unless a directive names the exact causal station boundary."""
        if directive.source_run_id != self.log.run_id:
            raise ExternalRelocationError("Directive source_run_id mismatch.")
        if directive.pf_resolved_config_sha256 != str(
            self.estimator.resolved_config_hash
        ):
            raise ExternalRelocationError(
                "Directive resolved PF config digest mismatch."
            )
        index = int(directive.apply_after_record_index)
        if index >= len(self.log.records):
            raise ExternalRelocationError("Directive cutoff record is outside the log.")
        record = self.log.records[index]
        if not bool(record.metadata.get("station_complete", False)):
            raise ExternalRelocationError(
                "External relocation is allowed only at writer-marked station boundaries."
            )
        if int(record.step_id) != int(directive.data_cutoff_step):
            raise ExternalRelocationError("Directive data_cutoff_step mismatch.")
        if int(record.station_id) != int(directive.data_cutoff_station):
            raise ExternalRelocationError("Directive data_cutoff_station mismatch.")
        expected_steps = tuple(
            int(item.step_id) for item in self.log.records[: index + 1]
        )
        if directive.covered_step_ids != expected_steps:
            raise ExternalRelocationError(
                "covered_step_ids must equal the exact log prefix through the cutoff."
            )
        actual_covered_digest = covered_records_sha256(self.log, index + 1)
        if directive.covered_records_sha256 != actual_covered_digest:
            raise ExternalRelocationError(
                "Directive covered-record prefix digest mismatch."
            )
        configured_isotopes = set(str(value) for value in self.estimator.all_isotopes)
        unknown = sorted(set(directive.isotopes) - configured_isotopes)
        if unknown:
            raise ExternalRelocationError(
                f"Directive contains unknown PF isotopes: {', '.join(unknown)}."
            )

    def apply_at_boundary(
        self,
        estimator: PurePFEstimator,
        record: MeasurementLogRecord,
        record_index: int,
    ) -> None:
        """Apply all directives for one completed station exactly once."""
        if estimator is not self.estimator:
            raise ExternalRelocationError("Schedule received an unknown PF estimator.")
        if not bool(record.metadata.get("station_complete", False)):
            raise ExternalRelocationError(
                "Directive callback requires station completion."
            )
        for directive in self._by_record.get(int(record_index), ()):
            if directive.directive_id in self._applied_ids:
                continue
            for isotope in directive.isotopes:
                if isotope not in estimator.filters:
                    raise ExternalRelocationError(
                        f"PF isotope {isotope} is unavailable at the directive cutoff."
                    )
                _validate_supported_target(estimator.filters[isotope])
                source_counts = np.asarray(
                    [
                        particle.state.num_sources
                        for particle in estimator.filters[isotope].continuous_particles
                    ],
                    dtype=int,
                )
                if not np.any(source_counts > 0):
                    raise ExternalRelocationError(
                        "Fixed-cardinality relocation requires at least one active "
                        f"{isotope} source slot."
                    )
            state_before = sha256(estimator.serialized_state()).hexdigest()
            isotope_receipts: list[dict[str, Any]] = []
            for isotope in sorted(directive.isotopes):
                data = estimator._measurement_data_for_iso(isotope, window=None)
                if data is None or int(data.z_k.size) != int(record_index) + 1:
                    raise ExternalRelocationError(
                        "PF measurement history does not match directive cutoff."
                    )
                isotope_receipts.append(
                    apply_relocation_to_filter(
                        estimator.filters[isotope],
                        data,
                        directive,
                        isotope,
                        base_seed=self.base_seed,
                    )
                )
            accepted_count = sum(
                int(item["accepted_count"]) for item in isotope_receipts
            )
            state_after = sha256(estimator.serialized_state()).hexdigest()
            directive_digest = (
                directive.source_directive_sha256
                if directive.source_directive_sha256 is not None
                else sha256_json(directive.to_dict())
            )
            candidate_outcomes: list[dict[str, Any]] = []
            receipt_by_isotope = {
                str(item["isotope"]): item for item in isotope_receipts
            }
            for isotope, mixture in sorted(directive.isotopes.items()):
                detail = receipt_by_isotope[isotope]
                candidate_outcomes.extend(aggregate_candidate_outcomes(mixture, detail))
            contract_receipt_base = {
                "schema_version": 1,
                "directive_id": directive.directive_id,
                "directive_sha256": directive_digest,
                "directive_kind": directive.directive_kind,
                "consumer_family": "particle_filter",
                "consumer_variant": "pf_external_relocation_mwg_v1",
                "data_cutoff_step": int(record.step_id),
                "applied_after_step": int(record.step_id),
                "status": "applied",
                "pf_state_sha256_before": state_before,
                "pf_state_sha256_after": state_after,
                "candidate_outcomes": candidate_outcomes,
                "safety_evidence": {
                    "direct_mle_objective_reweight_performed": False,
                    "hard_prune_performed": False,
                    "target_preserving_mh_performed": True,
                    "reweighted_observation_step_ids": [],
                    "next_observation_min_step": int(record.step_id) + 1,
                },
                "provenance": {
                    "source_run_id": directive.source_run_id,
                    "covered_records_sha256": directive.covered_records_sha256,
                    "pf_resolved_config_sha256": (directive.pf_resolved_config_sha256),
                    "detailed_particle_receipt_in_same_artifact": True,
                },
            }
            receipt_id = (
                "receipt-"
                + sha256(canonical_json_bytes(contract_receipt_base)).hexdigest()[:20]
            )
            self.receipts.append(
                {
                    "schema_version": 1,
                    "contract_receipt": {
                        **contract_receipt_base,
                        "receipt_id": receipt_id,
                    },
                    "directive_id": directive.directive_id,
                    "directive_sha256": directive_digest,
                    "proposal_source": directive.proposal_source,
                    "application_record_index": int(record_index),
                    "data_cutoff_step": int(record.step_id),
                    "data_cutoff_station": int(record.station_id),
                    "applied_once": True,
                    "target": (
                        "full_count_likelihood_steps_0_to_cutoff_plus_"
                        "uniform_volume_position_prior"
                    ),
                    "proposal_role": "target_preserving_mcmc_only",
                    "cardinality_changed": False,
                    "strengths_changed": False,
                    "background_changed": False,
                    "particle_log_weights_changed": False,
                    "accepted_count": int(accepted_count),
                    "isotopes": isotope_receipts,
                }
            )
            self._applied_ids.add(directive.directive_id)
            if accepted_count:
                estimator._invalidate_report_cache()

    @property
    def applied_directive_ids(self) -> tuple[str, ...]:
        """Return sorted IDs that have changed or tested the PF state."""
        return tuple(sorted(self._applied_ids))

    @property
    def pending_directive_ids(self) -> tuple[str, ...]:
        """Return sorted directives whose cutoff has not yet been replayed."""
        return tuple(
            sorted(
                directive.directive_id
                for directive in self.directives
                if directive.directive_id not in self._applied_ids
            )
        )

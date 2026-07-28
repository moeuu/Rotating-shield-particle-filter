"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import TYPE_CHECKING, Callable, List, Tuple
import os
import time

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.kernels import MeasurementGeometry, ShieldParams
from measurement.continuous_kernels import (
    ContinuousKernel,
    LineTransportComponents,
)
from measurement.obstacles import ObstacleGrid
from measurement.source_boundary import surface_transport_positions
from measurement.shielding import (
    generate_octant_orientations,
    resolve_mu_values,
)
from measurement.surface_charts import (
    build_surface_chart_geometry,
    surface_chart_geometry_sha256,
)
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.diagnostics import build_source_event_record, reset_step_diagnostics
from pf.posterior import posterior_point_estimate_from_states
from pf.randomness import isotope_random_generator, normalize_pf_random_seed
from pf.state import IsotopeState
from pf.strength_prior import StrengthPrior
from spectrum.additive_scatter import AdditiveNoncollidedTransportResponse
from pf.structural_rj import (
    BirthDeathMoveProbabilities,
    CardinalityPrior,
    ContinuousStrengthProposal,
    ContinuousSurfacePositionProposal,
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    SplitMergeMoveProbabilities,
    continuous_birth_log_acceptance_ratio,
    continuous_death_log_acceptance_ratio,
    continuous_joint_position_strength_log_acceptance_ratio,
    continuous_merge_log_acceptance_ratio,
    continuous_position_log_acceptance_ratio,
    continuous_split_log_acceptance_ratio,
    split_fraction_bounds,
    truncated_poisson_cardinality_probabilities,
    validate_cardinality_prior_policy,
)
from pf.surface_atlas import ContinuousSurfaceAtlas

if TYPE_CHECKING:
    import torch


def _pf_debug_timing_enabled() -> bool:
    """Return True when verbose PF phase timing should be printed."""
    return os.environ.get("PF_DEBUG_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class TemperingIncrementRequiresRejuvenation(RuntimeError):
    """Signal that no configured positive beta increment preserves target ESS."""


def _canonical_sha256(value: object, *, name: str) -> str:
    """Return one lowercase SHA-256 string without coercion."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string.")
    return value


def _strict_config_boolean(value: object, *, name: str) -> bool:
    """Return one exact PF configuration boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _strict_config_integer(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int:
    """Return one exact PF configuration integer above a lower bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return resolved


def _strict_config_number(value: object, *, name: str) -> float:
    """Return one finite numeric PF configuration value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric.")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _extended_log_target_ratio(
    proposed_log_target: NDArray[np.float64],
    current_log_target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return an MH log-target ratio on the finite-or-minus-infinity domain.

    A proposed state with zero target mass is rejected, while a finite proposal
    can recover a row whose current state has zero target mass.  When both
    states have zero target mass the ratio is undefined mathematically, so the
    move is deterministically rejected instead of allowing ``-inf - -inf`` to
    produce a NaN.
    """
    proposed = np.asarray(proposed_log_target, dtype=np.float64)
    current = np.asarray(current_log_target, dtype=np.float64)
    if proposed.shape != current.shape:
        raise ValueError("Proposed and current log targets must be aligned.")
    if (
        np.any(np.isnan(proposed))
        or np.any(np.isnan(current))
        or np.any(np.isposinf(proposed))
        or np.any(np.isposinf(current))
    ):
        raise ValueError(
            "MH log targets may be finite or negative infinity, not NaN or "
            "positive infinity."
        )
    result = np.full(proposed.shape, float("-inf"), dtype=np.float64)
    proposed_finite = np.isfinite(proposed)
    current_finite = np.isfinite(current)
    both_finite = proposed_finite & current_finite
    result[both_finite] = proposed[both_finite] - current[both_finite]
    result[proposed_finite & ~current_finite] = float("inf")
    return result


@dataclass
class PFConfig:
    """Particle filter configuration (Sec. 3.4)."""

    num_particles: int = 200
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    variable_cardinality: bool = True
    structural_rj_surface_chart_max_edge_m: float = 1.0
    structural_rj_move_probability: float = 1.0
    structural_rj_birth_probability: float = 0.5
    structural_rj_death_probability: float = 0.5
    structural_rj_position_move_probability: float = 1.0
    structural_rj_position_proposal_prior_weight: float = 0.5
    structural_rj_strength_proposal_prior_weight: float = 0.5
    structural_rj_strength_proposal_sigma_fraction: float = 0.15
    structural_rj_strength_proposal_grid_size: int = 9
    structural_rj_proposal_chart_batch_size: int = 256
    structural_rj_proposal_score_cache_max_bytes: int = 268_435_456
    structural_rj_local_position_move_probability: float = 1.0
    structural_rj_local_position_sigma_m: float = 0.5
    structural_rj_strength_move_probability: float = 1.0
    structural_rj_split_merge_probability: float = 1.0
    structural_rj_split_probability: float = 0.5
    structural_rj_merge_probability: float = 0.5
    structural_cardinality_prior_policy: str = (
        TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY
    )
    structural_cardinality_prior_probs: tuple[float, ...] | None = None
    structural_cardinality_prior_mean: float = 2.0
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 256
    min_delta_beta: float = 1e-10
    # Continuous PF priors (Sec. 3.3.2)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: Tuple[int, int] = (
        0,
        DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    )
    strength_prior_min_cps_1m: float = 1.0
    strength_prior_max_cps_1m: float = 2_000_000.0
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float64"

    def __post_init__(self) -> None:
        """Normalize the exact surface-PF configuration and likelihood semantics."""
        for name, value, minimum in (
            ("num_particles", self.num_particles, 1),
            (
                "structural_rj_strength_proposal_grid_size",
                self.structural_rj_strength_proposal_grid_size,
                2,
            ),
            (
                "structural_rj_proposal_chart_batch_size",
                self.structural_rj_proposal_chart_batch_size,
                1,
            ),
            (
                "structural_rj_proposal_score_cache_max_bytes",
                self.structural_rj_proposal_score_cache_max_bytes,
                1,
            ),
            ("max_temper_steps", self.max_temper_steps, 1),
        ):
            _strict_config_integer(value, name=name, minimum=minimum)
        if self.max_sources is None:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        _strict_config_integer(
            self.max_sources,
            name="max_sources",
            minimum=1,
        )
        _strict_config_boolean(
            self.variable_cardinality,
            name="variable_cardinality",
        )
        _strict_config_boolean(self.use_gpu, name="use_gpu")
        if (
            not isinstance(self.init_num_sources, (tuple, list))
            or len(self.init_num_sources) != 2
        ):
            raise TypeError("init_num_sources must contain two integers.")
        for index, value in enumerate(self.init_num_sources):
            _strict_config_integer(
                value,
                name=f"init_num_sources[{index}]",
                minimum=0,
            )
        for name in (
            "structural_rj_surface_chart_max_edge_m",
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_position_proposal_prior_weight",
            "structural_rj_strength_proposal_prior_weight",
            "structural_rj_strength_proposal_sigma_fraction",
            "structural_rj_local_position_move_probability",
            "structural_rj_local_position_sigma_m",
            "structural_rj_strength_move_probability",
            "structural_rj_split_merge_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
            "structural_cardinality_prior_mean",
            "target_ess_ratio",
            "min_delta_beta",
            "strength_prior_min_cps_1m",
            "strength_prior_max_cps_1m",
        ):
            _strict_config_number(getattr(self, name), name=name)
        if self.structural_cardinality_prior_probs is not None:
            for index, value in enumerate(
                self.structural_cardinality_prior_probs
            ):
                _strict_config_number(
                    value,
                    name=f"structural_cardinality_prior_probs[{index}]",
                )
        position_max = np.asarray(self.position_max, dtype=object).reshape(-1)
        if position_max.shape != (3,):
            raise ValueError("position_max must contain three values.")
        for index, value in enumerate(position_max):
            if _strict_config_number(
                value,
                name=f"position_max[{index}]",
            ) <= 0.0:
                raise ValueError("position_max values must be positive.")
        if not isinstance(self.gpu_device, str) or not self.gpu_device.strip():
            raise TypeError("gpu_device must be a nonempty string.")
        if not isinstance(self.gpu_dtype, str):
            raise TypeError("gpu_dtype must be a string.")
        self.num_particles = int(self.num_particles)
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive.")
        if str(self.gpu_dtype).strip().lower() != "float64":
            raise ValueError(
                "Pure PF production kernels require gpu_dtype='float64'."
            )
        self.gpu_dtype = "float64"
        self.variable_cardinality = bool(self.variable_cardinality)
        self.structural_rj_surface_chart_max_edge_m = float(
            self.structural_rj_surface_chart_max_edge_m
        )
        if (
            not np.isfinite(self.structural_rj_surface_chart_max_edge_m)
            or self.structural_rj_surface_chart_max_edge_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_surface_chart_max_edge_m must be finite and positive."
            )
        self.structural_rj_local_position_sigma_m = float(
            self.structural_rj_local_position_sigma_m
        )
        if (
            not np.isfinite(self.structural_rj_local_position_sigma_m)
            or self.structural_rj_local_position_sigma_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_local_position_sigma_m must be finite and positive."
            )
        self.structural_rj_position_proposal_prior_weight = float(
            self.structural_rj_position_proposal_prior_weight
        )
        if (
            not np.isfinite(
                self.structural_rj_position_proposal_prior_weight
            )
            or self.structural_rj_position_proposal_prior_weight <= 0.0
            or self.structural_rj_position_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_position_proposal_prior_weight must lie in "
                "(0, 1]."
            )
        self.structural_rj_strength_proposal_prior_weight = float(
            self.structural_rj_strength_proposal_prior_weight
        )
        if (
            not np.isfinite(
                self.structural_rj_strength_proposal_prior_weight
            )
            or self.structural_rj_strength_proposal_prior_weight <= 0.0
            or self.structural_rj_strength_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_strength_proposal_prior_weight must lie in "
                "(0, 1]."
            )
        self.structural_rj_strength_proposal_sigma_fraction = float(
            self.structural_rj_strength_proposal_sigma_fraction
        )
        if (
            not np.isfinite(
                self.structural_rj_strength_proposal_sigma_fraction
            )
            or self.structural_rj_strength_proposal_sigma_fraction <= 0.0
        ):
            raise ValueError(
                "structural_rj_strength_proposal_sigma_fraction must be "
                "finite and positive."
            )
        self.structural_rj_strength_proposal_grid_size = int(
            self.structural_rj_strength_proposal_grid_size
        )
        if self.structural_rj_strength_proposal_grid_size < 2:
            raise ValueError(
                "structural_rj_strength_proposal_grid_size must be at least 2."
            )
        self.structural_rj_proposal_chart_batch_size = int(
            self.structural_rj_proposal_chart_batch_size
        )
        if self.structural_rj_proposal_chart_batch_size < 1:
            raise ValueError(
                "structural_rj_proposal_chart_batch_size must be positive."
            )
        self.structural_rj_proposal_score_cache_max_bytes = int(
            self.structural_rj_proposal_score_cache_max_bytes
        )
        if self.structural_rj_proposal_score_cache_max_bytes < 1:
            raise ValueError(
                "structural_rj_proposal_score_cache_max_bytes must be positive."
            )
        for probability_name in (
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_local_position_move_probability",
            "structural_rj_strength_move_probability",
            "structural_rj_split_merge_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
        ):
            probability = float(getattr(self, probability_name))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_name} must lie in [0, 1].")
            setattr(self, probability_name, probability)
        if self.max_sources is None or int(self.max_sources) < 1:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        self.max_sources = int(self.max_sources)
        strength_prior = StrengthPrior(
            minimum=self.strength_prior_min_cps_1m,
            maximum=self.strength_prior_max_cps_1m,
        )
        if strength_prior.minimum <= 0.0:
            raise ValueError(
                "Pure PF source states require a strictly positive strength "
                "prior minimum."
            )
        self.strength_prior_min_cps_1m = strength_prior.minimum
        self.strength_prior_max_cps_1m = strength_prior.maximum
        self.structural_cardinality_prior_policy = (
            validate_cardinality_prior_policy(
                self.structural_cardinality_prior_policy,
                has_explicit_probabilities=(
                    self.structural_cardinality_prior_probs is not None
                ),
            )
        )
        if self.structural_cardinality_prior_probs is not None:
            cardinality_prior = np.asarray(
                self.structural_cardinality_prior_probs,
                dtype=float,
            ).reshape(-1)
            if (
                cardinality_prior.size != self.max_sources + 1
                or np.any(~np.isfinite(cardinality_prior))
                or np.any(cardinality_prior <= 0.0)
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain "
                    "max_sources + 1 finite positive values."
                )
            cardinality_prior /= math.fsum(
                float(value) for value in cardinality_prior
            )
            self.structural_cardinality_prior_probs = tuple(
                float(value) for value in cardinality_prior
            )
        self.structural_cardinality_prior_mean = float(
            self.structural_cardinality_prior_mean
        )
        if (
            not np.isfinite(self.structural_cardinality_prior_mean)
            or self.structural_cardinality_prior_mean <= 0.0
        ):
            raise ValueError(
                "structural_cardinality_prior_mean must be finite and positive."
            )
        initial_lower, initial_upper = (
            int(self.init_num_sources[0]),
            int(self.init_num_sources[1]),
        )
        if self.variable_cardinality:
            if (
                self.structural_rj_move_probability <= 0.0
                or self.structural_rj_birth_probability <= 0.0
                or self.structural_rj_death_probability <= 0.0
            ):
                raise ValueError(
                    "Variable-cardinality pure PF requires positive structural "
                    "move, birth, and death proposal probabilities."
                )
            if initial_lower != 0 or initial_upper != self.max_sources:
                raise ValueError(
                    "Variable-cardinality pure PF initialization must cover "
                    "every cardinality from zero through max_sources."
                )
            if self.num_particles < self.max_sources + 1:
                raise ValueError(
                    "Variable-cardinality pure PF requires at least one initial "
                    "particle per cardinality from zero through max_sources."
                )
        elif (
            initial_lower != initial_upper
            or initial_lower < 0
            or initial_upper > self.max_sources
        ):
            raise ValueError(
                "Fixed-cardinality pure PF requires "
                "init_num_sources=(K, K) within zero through max_sources."
            )
        self.init_num_sources = (initial_lower, initial_upper)
        self.target_ess_ratio = float(self.target_ess_ratio)
        if (
            not np.isfinite(self.target_ess_ratio)
            or not 0.0 < self.target_ess_ratio < 1.0
        ):
            raise ValueError("target_ess_ratio must lie strictly between zero and one.")
        self.max_temper_steps = int(self.max_temper_steps)
        if self.max_temper_steps < 1:
            raise ValueError("max_temper_steps must be positive.")
        self.min_delta_beta = float(self.min_delta_beta)
        if (
            not np.isfinite(self.min_delta_beta)
            or not 0.0 < self.min_delta_beta <= 1.0
        ):
            raise ValueError("min_delta_beta must lie in (0, 1].")

@dataclass(frozen=True)
class JointRowIdentity:
    """Authenticate one immutable current row in the joint-isotope PF.

    The identity is independent of every isotope state.  Initial rows are
    rooted in one estimator contract digest.  A joint resample creates a new
    generation whose rows commit to their parent row and output ordinal, so
    repeated copies of one ancestor remain distinct while every isotope can
    verify the same row ordering.
    """

    root_sha256: str
    generation: int
    ordinal: int
    parent_row_sha256: str | None
    row_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate identity fields and derive the authenticated row digest."""
        root = _canonical_sha256(
            self.root_sha256,
            name="joint row identity root_sha256",
        )
        for name, value in (
            ("generation", self.generation),
            ("ordinal", self.ordinal),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(f"Joint row identity {name} must be an integer.")
            if int(value) < 0 or int(value) >= 1 << 64:
                raise ValueError(
                    f"Joint row identity {name} must lie in uint64 range."
                )
        parent = self.parent_row_sha256
        if int(self.generation) == 0:
            if parent is not None:
                raise ValueError(
                    "Initial joint row identities cannot have a parent."
                )
        elif parent is None:
            raise ValueError(
                "Resampled joint row identities require a parent digest."
            )
        if parent is not None:
            parent = _canonical_sha256(
                parent,
                name="joint row identity parent_row_sha256",
            )
        object.__setattr__(
            self,
            "root_sha256",
            root,
        )
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "parent_row_sha256", parent)
        object.__setattr__(self, "row_sha256", self._expected_sha256())

    def _expected_sha256(self) -> str:
        """Return the domain-separated digest implied by the identity fields."""
        digest = hashlib.sha256()
        digest.update(b"pure_pf_joint_row_identity_v1\0")
        digest.update(bytes.fromhex(self.root_sha256))
        digest.update(int(self.generation).to_bytes(8, "big", signed=False))
        digest.update(int(self.ordinal).to_bytes(8, "big", signed=False))
        if self.parent_row_sha256 is None:
            digest.update(b"\0")
        else:
            digest.update(b"\1")
            digest.update(bytes.fromhex(self.parent_row_sha256))
        return digest.hexdigest()

    def validate(self) -> str:
        """Return the row digest or fail if the immutable commitment is corrupt."""
        expected = self._expected_sha256()
        if self.row_sha256 != expected:
            raise RuntimeError(
                "Joint row identity digest does not match its lineage fields."
            )
        return expected

    @classmethod
    def initial(cls, *, root_sha256: str, ordinal: int) -> "JointRowIdentity":
        """Create one deterministic unique row in generation zero."""
        return cls(
            root_sha256=root_sha256,
            generation=0,
            ordinal=ordinal,
            parent_row_sha256=None,
        )

    def resampled_child(self, *, ordinal: int) -> "JointRowIdentity":
        """Create one unique next-generation child of this joint row."""
        self.validate()
        return JointRowIdentity(
            root_sha256=self.root_sha256,
            generation=self.generation + 1,
            ordinal=ordinal,
            parent_row_sha256=self.row_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical checkpoint/replay representation."""
        self.validate()
        return {
            "schema_version": 1,
            "root_sha256": self.root_sha256,
            "generation": self.generation,
            "ordinal": self.ordinal,
            "parent_row_sha256": self.parent_row_sha256,
            "row_sha256": self.row_sha256,
        }


@dataclass
class IsotopeParticle:
    """Continuous-state particle (Sec. 3.3.2)."""

    state: IsotopeState
    log_weight: float
    joint_row_identity: JointRowIdentity | None = None

    def __setattr__(self, name: str, value: object) -> None:
        """Keep joint row identity immutable while allowing state/MH updates."""
        if name == "joint_row_identity" and name in self.__dict__:
            raise AttributeError(
                "joint_row_identity is immutable; create a new particle row."
            )
        object.__setattr__(self, name, value)


@dataclass(frozen=True)
class StructuralGeometryBatch:
    """Store only geometry needed by continuous structural PF proposals."""

    detector_positions: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times: NDArray[np.float64]
    station_sequence_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate, copy, and freeze one aligned geometry batch."""
        detector_positions = np.array(
            self.detector_positions,
            dtype=np.float64,
            copy=True,
        )
        fe_indices = np.array(self.fe_indices, dtype=np.int64, copy=True).reshape(
            -1
        )
        pb_indices = np.array(self.pb_indices, dtype=np.int64, copy=True).reshape(
            -1
        )
        live_times = np.array(
            self.live_times,
            dtype=np.float64,
            copy=True,
        ).reshape(-1)
        station_ids = np.array(
            self.station_sequence_ids,
            dtype=np.int64,
            copy=True,
        ).reshape(-1)
        row_count = int(fe_indices.size)
        if (
            row_count == 0
            or detector_positions.shape != (row_count, 3)
            or pb_indices.size != row_count
            or live_times.size != row_count
            or station_ids.size != row_count
            or np.any(~np.isfinite(detector_positions))
            or np.any(~np.isfinite(live_times))
            or np.any(live_times <= 0.0)
            or np.any(fe_indices < 0)
            or np.any(pb_indices < 0)
            or np.any(station_ids < 0)
        ):
            raise ValueError(
                "Structural geometry must contain aligned finite detector, "
                "shield, positive-live-time, and station-ID rows."
            )
        for values in (
            detector_positions,
            fe_indices,
            pb_indices,
            live_times,
            station_ids,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "detector_positions", detector_positions)
        object.__setattr__(self, "fe_indices", fe_indices)
        object.__setattr__(self, "pb_indices", pb_indices)
        object.__setattr__(self, "live_times", live_times)
        object.__setattr__(self, "station_sequence_ids", station_ids)

    @property
    def row_count(self) -> int:
        """Return the number of aligned geometry rows."""
        return int(np.asarray(self.fe_indices).size)


@dataclass(frozen=True)
class TorchLineTransportComponents:
    """Store source-resolved line-rate components as Torch tensors."""

    total_kernel: "torch.Tensor"
    uncollided_kernel: "torch.Tensor"
    tau_fe: "torch.Tensor"
    tau_pb: "torch.Tensor"
    tau_obstacle: "torch.Tensor"
    distance_m: "torch.Tensor"


class IsotopeParticleFilter:
    """Per-isotope particle filter (continuous state is the primary mode)."""

    def __init__(
        self,
        isotope: str,
        kernel: MeasurementGeometry | None,
        config: PFConfig | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: dict[str, object] | None = None,
        additive_scatter_response: (
            AdditiveNoncollidedTransportResponse | None
        ) = None,
        random_seed: int = 0,
    ) -> None:
        """Initialize particle state, priors, and continuous measurement kernels."""
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
        self.obstacle_grid = obstacle_grid
        self.obstacle_height_m = float(obstacle_height_m)
        self.obstacle_mu_by_isotope = obstacle_mu_by_isotope
        self.obstacle_buildup_coeff = max(float(obstacle_buildup_coeff), 0.0)
        self.detector_radius_m = max(float(detector_radius_m), 0.0)
        if detector_aperture_radius_m is None:
            detector_aperture_radius_m = self.detector_radius_m
        self.detector_aperture_radius_m = max(float(detector_aperture_radius_m), 0.0)
        self.detector_aperture_samples = max(int(detector_aperture_samples), 1)
        self.detector_aperture_sampling = str(detector_aperture_sampling)
        self.source_extent_radius_m = max(float(source_extent_radius_m), 0.0)
        self.source_extent_samples = max(int(source_extent_samples), 1)
        self.line_mu_by_isotope = line_mu_by_isotope
        self.additive_scatter_response = additive_scatter_response
        self.random_seed = normalize_pf_random_seed(random_seed)
        self._random_generator = isotope_random_generator(
            self.random_seed,
            self.isotope,
        )
        self._strength_prior = self._build_strength_prior()
        self._structural_rj_surface_atlas: ContinuousSurfaceAtlas | None = None
        self._structural_rj_surface_atlas_sha256: str | None = None
        self._structural_rj_cardinality_prior_probs = (
            self._build_structural_cardinality_prior()
        )
        self._structural_rj_cardinality_prior: CardinalityPrior | None = None
        self._structural_rj_move_probabilities: BirthDeathMoveProbabilities | None = (
            None
        )
        self._structural_rj_split_merge_probabilities: (
            SplitMergeMoveProbabilities | None
        ) = None
        self._structural_rj_position_proposal: (
            ContinuousSurfacePositionProposal | None
        ) = None
        self._last_structural_rj_position_proposal: (
            ContinuousSurfacePositionProposal | None
        ) = None
        self._structural_rj_strength_proposal: (
            ContinuousStrengthProposal | None
        ) = None
        self._last_structural_rj_strength_proposal: (
            ContinuousStrengthProposal | None
        ) = None
        self.last_structural_rj_proposal_snapshot_sha256: str | None = None
        self._structural_rj_move_counts: dict[str, int | float] = {}
        self._structural_rj_tempering_start_row: int | None = None
        self._joint_target_evaluator: (
            Callable[..., NDArray[np.float64]] | None
        ) = None
        self._joint_proposal_evaluator: (
            Callable[
                ...,
                tuple[
                    NDArray[np.float64],
                    NDArray[np.float64],
                    bool,
                ],
            ]
            | None
        ) = None
        self._initialize_structural_rj_surface_support()
        mu_by_isotope = (
            getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        )
        shield_params = (
            getattr(kernel, "shield_params", ShieldParams())
            if kernel is not None
            else ShieldParams()
        )
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        self._continuous_kernel_physics_signature = (
            self._incoming_kernel_physics_signature(kernel)
        )
        self.continuous_particles: List[IsotopeParticle] = []
        self.last_ess: float | None = None
        self.last_ess_pre: float | None = None
        self.last_ess_post: float | None = None
        self.last_resample_ess = False
        self.last_resample_count = 0
        self.last_birth_count = 0
        self.last_death_count = 0
        self.last_temper_steps: list[dict[str, float]] = []
        self.last_temper_resample_count = 0
        self.last_temper_min_ess: float | None = None
        self.last_unique_ancestor_count: int | None = None
        self.last_station_unique_ancestor_count: int | None = None
        self.last_cumulative_unique_ancestor_count: int | None = None
        self.last_source_event_diagnostics: list[dict[str, object]] = []
        self.last_structural_timing_s: dict[str, float] = {}
        self.last_structural_transition_weight_mass: dict[str, float] = {}
        self.last_runtime_likelihood_route = "joint_full_spectrum_generative"
        self._resample_count_in_observation = 0
        self._init_continuous_particles()

    def _variable_cardinality_enabled(self) -> bool:
        """Return whether exact birth/death dimension changes are active."""
        return bool(self.config.variable_cardinality)

    def _build_strength_prior(self) -> StrengthPrior:
        """Build the normalized strength prior shared by initialization and moves."""
        return StrengthPrior(
            minimum=float(self.config.strength_prior_min_cps_1m),
            maximum=float(self.config.strength_prior_max_cps_1m),
        )

    def _build_structural_cardinality_prior(self) -> NDArray[np.float64]:
        """Return the normalized prior mass for cardinalities zero through max."""
        max_sources = self.config.max_sources
        if max_sources is None:
            return np.zeros(0, dtype=float)
        count = max(0, int(max_sources)) + 1
        configured = self.config.structural_cardinality_prior_probs
        if configured is None:
            return truncated_poisson_cardinality_probabilities(
                int(max_sources),
                float(self.config.structural_cardinality_prior_mean),
            )
        probabilities = np.asarray(configured, dtype=float).reshape(-1)
        if probabilities.size != count:
            raise ValueError(
                "structural_cardinality_prior_probs must have max_sources + 1 entries."
            )
        return probabilities / float(np.sum(probabilities))

    def _initialize_structural_rj_surface_support(self) -> None:
        """Build the rectangular atlas used by continuous surface states."""
        chart_geometry = build_surface_chart_geometry(
            self._source_prior_environment(),
            self.obstacle_grid,
            max_edge_m=float(
                self.config.structural_rj_surface_chart_max_edge_m
            ),
            obstacle_height_m=self.obstacle_height_m,
        )
        if not chart_geometry.obstacle_surfaces_available:
            warning = chart_geometry.obstacle_geometry_warning or (
                "Obstacle component surfaces are unavailable."
            )
            raise ValueError(
                f"rj_mh requires complete obstacle component geometry: {warning}"
            )
        self._structural_rj_surface_atlas = ContinuousSurfaceAtlas(
            chart_geometry
        )
        self._structural_rj_surface_atlas_sha256 = (
            surface_chart_geometry_sha256(chart_geometry)
        )
        max_sources = int(self.config.max_sources or 0)
        self._structural_rj_cardinality_prior = CardinalityPrior(
            self._structural_rj_cardinality_prior_probs
        )
        self._structural_rj_move_probabilities = BirthDeathMoveProbabilities(
            max_cardinality=max_sources,
            birth_weight=float(self.config.structural_rj_birth_probability),
            death_weight=float(self.config.structural_rj_death_probability),
        )
        self._structural_rj_split_merge_probabilities = (
            SplitMergeMoveProbabilities(
                max_cardinality=max_sources,
                split_weight=float(self.config.structural_rj_split_probability),
                merge_weight=float(self.config.structural_rj_merge_probability),
            )
        )

    @property
    def structural_rj_surface_atlas_sha256(self) -> str:
        """Return the immutable continuous-surface atlas contract digest."""
        value = self._structural_rj_surface_atlas_sha256
        if value is None:
            raise RuntimeError("Continuous surface atlas digest is unavailable.")
        return value

    def _surface_coordinates_for_state(
        self,
        state: IsotopeState,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return one validated authoritative chart/UV state."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        source_count = int(state.num_sources)
        strengths = np.asarray(state.strengths, dtype=float).reshape(-1)
        if strengths.size != source_count:
            raise ValueError("Surface state arrays must match num_sources.")
        chart_ids, surface_uv = atlas.validate_coordinates(
            state.surface_chart_ids,
            state.surface_uv,
        )
        if chart_ids.shape != (source_count,):
            raise ValueError("surface_chart_ids must contain one value per source.")
        return chart_ids, surface_uv

    def validate_continuous_surface_states(self) -> None:
        """Fail if any authoritative chart/UV/strength state is invalid.

        Validation never projects, reconstructs, or otherwise repairs state.
        Cartesian positions are absent from state and are derived only after
        this check.
        """
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        states = [
            particle.state for particle in self.continuous_particles
        ]
        if not states:
            return
        cardinalities = np.asarray(
            [int(state.num_sources) for state in states],
            dtype=np.int64,
        )
        maximum_cardinality = int(self.config.max_sources or 0)
        if np.any(cardinalities < 0) or np.any(
            cardinalities > maximum_cardinality
        ):
            raise ValueError(
                "PF state cardinalities must lie inside configured support."
            )
        for cardinality in np.unique(cardinalities).tolist():
            indices = np.flatnonzero(
                cardinalities == int(cardinality)
            )
            selected = [states[int(index)] for index in indices]
            strengths = np.stack(
                [
                    np.asarray(state.strengths, dtype=np.float64).reshape(
                        int(cardinality)
                    )
                    for state in selected
                ],
                axis=0,
            )
            if np.any(~np.asarray(
                self._strength_prior.in_support(strengths),
                dtype=bool,
            )):
                raise ValueError(
                    "PF source strengths must lie inside configured prior support."
                )
            chart_ids = np.stack(
                [
                    np.asarray(
                        state.surface_chart_ids,
                        dtype=np.int64,
                    ).reshape(int(cardinality))
                    for state in selected
                ],
                axis=0,
            )
            surface_uv = np.stack(
                [
                    np.asarray(
                        state.surface_uv,
                        dtype=np.float64,
                    ).reshape(int(cardinality), 2)
                    for state in selected
                ],
                axis=0,
            )
            validated_ids, validated_uv = atlas.validate_coordinates(
                chart_ids,
                surface_uv,
            )
            if (
                validated_ids.shape != (indices.size, int(cardinality))
                or validated_uv.shape
                != (indices.size, int(cardinality), 2)
            ):
                raise ValueError(
                    "PF chart/UV arrays do not match their source cardinality."
                )
            if int(cardinality) > 1:
                order = np.lexsort(
                    (
                        validated_uv[:, :, 1],
                        validated_uv[:, :, 0],
                        validated_ids,
                    ),
                    axis=1,
                )
                expected = np.broadcast_to(
                    np.arange(int(cardinality), dtype=np.int64),
                    order.shape,
                )
                if not np.array_equal(order, expected):
                    raise ValueError(
                        "PF source states must remain in canonical chart/UV order."
                    )

    def continuous_state_positions(
        self,
        state: IsotopeState,
    ) -> NDArray[np.float64]:
        """Derive one state's Cartesian XYZ solely from authoritative chart/UV."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        chart_ids, surface_uv = self._surface_coordinates_for_state(state)
        return np.asarray(
            atlas.positions_xyz(chart_ids, surface_uv),
            dtype=np.float64,
        )

    def _surface_transport_positions(
        self,
        anchors_xyz: NDArray[np.float64],
        *,
        chart_ids: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Map exact PF surface anchors to the shared air-side physics XYZ."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        anchors = np.asarray(anchors_xyz, dtype=np.float64)
        if anchors.shape[-1:] != (3,):
            raise ValueError("Surface anchors must have final dimension three.")
        if chart_ids is None:
            resolved_chart_ids, _ = atlas.locate_positions(anchors)
        else:
            raw_chart_ids = np.asarray(chart_ids)
            if raw_chart_ids.shape != anchors.shape[:-1]:
                raise ValueError("chart_ids must align with surface anchors.")
            if not np.issubdtype(raw_chart_ids.dtype, np.integer):
                raise TypeError("chart_ids must contain integers.")
            resolved_chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        normals = atlas.air_facing_normals_xyz(resolved_chart_ids)
        return surface_transport_positions(anchors, normals)

    def _packed_continuous_state_arrays(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:
        """Pack Cartesian states into the configured fixed source-slot layout."""
        positions, strengths, mask, _, _ = (
            self._packed_continuous_surface_state_arrays()
        )
        return positions, strengths, mask

    def _packed_continuous_surface_state_arrays(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.int64],
        NDArray[np.float64],
    ]:
        """Pack states and authoritative chart/UV into fixed source slots.

        A fixed ``max_sources`` slot axis is part of the aligned joint-isotope
        transport contract.  Shrinking this axis to the largest currently
        represented cardinality would shift isotope slot boundaries after
        resampling and make conditional RJ overwrite another isotope's
        components.
        """
        self.validate_continuous_surface_states()
        states = [
            particle.state for particle in self.continuous_particles
        ]
        particle_count = len(states)
        slot_count = int(self.config.max_sources or 0)
        chart_ids = np.zeros(
            (particle_count, slot_count),
            dtype=np.int64,
        )
        surface_uv = np.zeros(
            (particle_count, slot_count, 2),
            dtype=np.float64,
        )
        strengths = np.zeros(
            (particle_count, slot_count),
            dtype=np.float64,
        )
        mask = np.zeros(
            (particle_count, slot_count),
            dtype=bool,
        )
        for row, state in enumerate(states):
            cardinality = int(state.num_sources)
            if cardinality == 0:
                continue
            chart_ids[row, :cardinality] = state.surface_chart_ids
            surface_uv[row, :cardinality] = state.surface_uv
            strengths[row, :cardinality] = state.strengths
            mask[row, :cardinality] = True
        positions = np.zeros(
            (particle_count, slot_count, 3),
            dtype=np.float64,
        )
        if np.any(mask):
            positions[mask] = self._structural_rj_surface_atlas.positions_xyz(
                chart_ids[mask],
                surface_uv[mask],
            )
        return positions, strengths, mask, chart_ids, surface_uv

    def structural_surface_chart_coordinates(
        self,
        positions: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Resolve continuous physical-surface XYZ to chart identifiers and UV."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        return atlas.locate_positions(positions)

    def structural_surface_kinds(
        self,
        positions: NDArray[np.float64],
        *,
        strict: bool = True,
    ) -> NDArray[np.object_]:
        """Return authoritative physical-surface kinds for continuous positions."""
        if not bool(strict):
            raise ValueError(
                "Continuous PF surface labels require strict on-surface positions."
            )
        chart_ids, _ = self.structural_surface_chart_coordinates(positions)
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        return np.asarray(atlas.geometry.kinds, dtype=object)[chart_ids]

    def _canonicalize_structural_rj_state(
        self,
        state: IsotopeState,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Sort one state by chart/UV and return its continuous coordinates."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        chart_ids, surface_uv = self._surface_coordinates_for_state(state)
        if chart_ids.size <= 1:
            return chart_ids, surface_uv
        order = atlas.canonical_order(chart_ids, surface_uv)
        if not np.array_equal(order, np.arange(chart_ids.size)):
            state.surface_chart_ids = chart_ids[order]
            state.surface_uv = surface_uv[order]
            state.strengths = np.asarray(state.strengths, dtype=float)[order]
            chart_ids = state.surface_chart_ids
            surface_uv = state.surface_uv
        return (
            np.asarray(chart_ids, dtype=np.int64),
            np.asarray(surface_uv, dtype=np.float64),
        )

    def _build_continuous_kernel(
        self,
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
    ) -> ContinuousKernel:
        """Build the continuous kernel with the filter's environment attenuation settings."""
        kernel_kwargs: dict[str, object] = {}
        orientations = getattr(self.kernel, "orientations", None)
        if orientations is not None and len(orientations) > 1:
            kernel_kwargs["orientations"] = orientations
        return ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=bool(self.config.use_gpu),
            gpu_device=str(self.config.gpu_device),
            gpu_dtype=str(self.config.gpu_dtype),
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            additive_scatter_response=self.additive_scatter_response,
            **kernel_kwargs,
        )

    def _incoming_kernel_physics_signature(
        self,
        kernel: MeasurementGeometry | None,
    ) -> tuple[object, ...]:
        """Return canonical incoming physics that affects this isotope's kernel."""
        shield_params = (
            getattr(kernel, "shield_params", ShieldParams())
            if kernel is not None
            else ShieldParams()
        )
        mu_by_isotope = (
            getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        )
        mu_fe, mu_pb = resolve_mu_values(
            mu_by_isotope,
            self.isotope,
            default_fe=float(shield_params.mu_fe),
            default_pb=float(shield_params.mu_pb),
        )
        incoming_orientations = (
            getattr(kernel, "orientations", None) if kernel is not None else None
        )
        orientations = (
            generate_octant_orientations()
            if incoming_orientations is None or len(incoming_orientations) <= 1
            else np.asarray(incoming_orientations, dtype=np.float64)
        )
        orientation_array = np.asarray(
            orientations,
            dtype=np.float64,
        )
        canonical_orientations = np.ascontiguousarray(
            np.where(orientation_array == 0.0, 0.0, orientation_array),
            dtype="<f8",
        )
        shield_signature = (
            float(shield_params.mu_pb),
            float(shield_params.mu_fe),
            float(shield_params.thickness_pb_cm),
            float(shield_params.thickness_fe_cm),
            float(shield_params.inner_radius_fe_cm),
            float(shield_params.inner_radius_pb_cm),
            max(float(shield_params.buildup_fe_coeff), 0.0),
            max(float(shield_params.buildup_pb_coeff), 0.0),
            str(shield_params.shield_geometry_model),
            bool(shield_params.use_angle_attenuation),
        )
        return (
            (float(mu_fe), float(mu_pb)),
            shield_signature,
            canonical_orientations.shape,
            canonical_orientations.tobytes(order="C"),
        )

    def set_kernel(self, kernel: MeasurementGeometry) -> None:
        """Attach a discrete kernel and refresh only changed continuous physics."""
        incoming_signature = self._incoming_kernel_physics_signature(kernel)
        self.kernel = kernel
        if incoming_signature == self._continuous_kernel_physics_signature:
            return
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=getattr(kernel, "mu_by_isotope", None),
            shield_params=getattr(kernel, "shield_params", ShieldParams()),
        )
        self._continuous_kernel_physics_signature = incoming_signature
        self._structural_rj_position_proposal = None
        self._last_structural_rj_position_proposal = None
        self._structural_rj_strength_proposal = None
        self._last_structural_rj_strength_proposal = None

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by the source-position prior."""
        hi = np.array(self.config.position_max, dtype=float)
        if hi.shape != (3,):
            raise ValueError("position_max must be a 3-element vector.")
        if np.any(hi <= 0.0):
            raise ValueError("position_max must define positive room dimensions.")
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    def _sample_initial_strengths(
        self,
        shape: int | tuple[int, ...],
    ) -> NDArray[np.float64]:
        """Draw a batched initial-strength array from the configured PF prior."""
        return np.asarray(
            self._strength_prior.sample(
                shape,
                rng=self._random_generator,
            ),
            dtype=float,
        )

    def _exact_initial_cardinality_counts(
        self,
        particle_count: int,
    ) -> NDArray[np.int64]:
        """Allocate at least one exact-mode particle to every positive K prior."""
        count = max(1, int(particle_count))
        probabilities = np.asarray(
            self._structural_rj_cardinality_prior_probs,
            dtype=float,
        )
        positive = probabilities > 0.0
        positive_count = int(np.count_nonzero(positive))
        if count < positive_count:
            raise ValueError(
                "rj_mh needs at least one initial particle for every positive "
                "cardinality-prior entry."
            )
        allocation = np.zeros(probabilities.size, dtype=np.int64)
        allocation[positive] = 1
        remaining = count - positive_count
        if remaining <= 0:
            return allocation
        expected = remaining * probabilities
        extra = np.floor(expected).astype(np.int64)
        allocation += extra
        leftover = remaining - int(np.sum(extra))
        if leftover > 0:
            fractions = expected - extra
            order = np.argsort(fractions, kind="stable")[::-1]
            allocation[order[:leftover]] += 1
        return allocation

    def _init_exact_structural_particles(self) -> None:
        """Initialize exact particles from K, continuous-surface, and strength priors."""
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        if atlas is None or cardinality_prior is None:
            raise RuntimeError("rj_mh surface and cardinality priors are unavailable.")
        # Exact-mode particles are weighted Monte Carlo samples from the
        # normalized cardinality, surface-set, and strength priors.
        target_n = int(self.config.num_particles)
        allocation = self._exact_initial_cardinality_counts(target_n)
        particles: list[IsotopeParticle] = []
        for cardinality, cardinality_count in enumerate(allocation.tolist()):
            if cardinality_count <= 0:
                continue
            chart_ids_flat, surface_uv_flat, _ = atlas.sample(
                cardinality_count * cardinality,
                rng=self._random_generator,
            )
            chart_ids = chart_ids_flat.reshape(cardinality_count, cardinality)
            surface_uv = surface_uv_flat.reshape(
                cardinality_count,
                cardinality,
                2,
            )
            strengths = self._sample_initial_strengths((cardinality_count, cardinality))
            per_particle_mass = float(
                cardinality_prior.probabilities[cardinality]
            ) / float(cardinality_count)
            log_weight = float(np.log(per_particle_mass))
            for row in range(cardinality_count):
                state = IsotopeState(
                    num_sources=cardinality,
                    strengths=np.asarray(strengths[row], dtype=float).copy(),
                    surface_chart_ids=np.asarray(
                        chart_ids[row],
                        dtype=np.int64,
                    ).copy(),
                    surface_uv=np.asarray(
                        surface_uv[row],
                        dtype=np.float64,
                    ).copy(),
                )
                self._canonicalize_structural_rj_state(state)
                particles.append(IsotopeParticle(state=state, log_weight=log_weight))
        permutation = self._random_generator.permutation(len(particles))
        self.continuous_particles = [particles[int(index)] for index in permutation]
        self.N = len(self.continuous_particles)
        self.config.num_particles = self.N

    def _init_fixed_cardinality_particles(self) -> None:
        """Initialize a fixed-K PF from continuous surface and strength priors."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface support is unavailable.")
        cardinality = int(self.config.init_num_sources[0])
        particle_count = max(1, int(self.config.num_particles))
        chart_ids_flat, surface_uv_flat, _ = atlas.sample(
            particle_count * cardinality,
            rng=self._random_generator,
        )
        chart_ids = chart_ids_flat.reshape(particle_count, cardinality)
        surface_uv = surface_uv_flat.reshape(particle_count, cardinality, 2)
        strengths = self._sample_initial_strengths((particle_count, cardinality))
        log_weight = float(-np.log(particle_count))
        self.continuous_particles = []
        for row in range(particle_count):
            state = IsotopeState(
                num_sources=cardinality,
                strengths=np.asarray(strengths[row], dtype=float).copy(),
                surface_chart_ids=np.asarray(
                    chart_ids[row],
                    dtype=np.int64,
                ).copy(),
                surface_uv=np.asarray(
                    surface_uv[row],
                    dtype=np.float64,
                ).copy(),
            )
            self._canonicalize_structural_rj_state(state)
            self.continuous_particles.append(
                IsotopeParticle(state=state, log_weight=log_weight)
            )
        self.N = particle_count
        self.config.num_particles = particle_count

    def _init_continuous_particles(self) -> None:
        """Initialize exact variable-K or fixed-K particles from PF priors."""
        self.continuous_particles = []
        if self._variable_cardinality_enabled():
            self._init_exact_structural_particles()
        else:
            self._init_fixed_cardinality_particles()

    def reset_step_stats(self) -> None:
        """Reset per-step diagnostic counters."""
        reset_step_diagnostics(self)

    def _record_source_event(
        self,
        event: str,
        st: IsotopeState,
        source_idx: int,
        *,
        reason: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Record an accepted exact-RJ source birth or death event."""
        record = build_source_event_record(
            event=event,
            isotope=self.isotope,
            state=st,
            source_idx=int(source_idx),
            position_xyz=self.continuous_state_positions(st)[int(source_idx)],
            reason=reason,
            extra=extra,
        )
        if record is None:
            return
        self.last_source_event_diagnostics.append(record)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in PFConfig.")
        gpu_utils.require_torch_compute_device(
            str(self.config.gpu_device),
            str(self.config.gpu_dtype),
        )
        return True

    def _can_use_gpu(self) -> bool:
        """Select explicit NumPy mode or require the configured torch device."""
        if not self.config.use_gpu:
            return False
        return self._gpu_enabled()


    def _continuous_expected_line_transport_components_pair_sequence_torch(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
        positive_line_indices: NDArray[np.int64],
    ) -> TorchLineTransportComponents:
        """Return batched source-resolved line-rate transport components.

        Total and uncollided arrays have shape particle x view x source-slot x
        line and contain strength-scaled rates before branching fractions and
        live time.  Geometry features share that shape and are independent of
        source strength.  The full-spectrum model is the only component that
        converts rates to finite-live-time event counts.
        """
        from pf import gpu_utils
        import torch

        fe_arr = np.asarray(fe_indices, dtype=np.int64).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=np.int64).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=np.float64).reshape(-1)
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        if not (
            fe_arr.size == pb_arr.size == live_arr.size
            and fe_arr.size > 0
        ):
            raise ValueError(
                "Fe, Pb, and live-time arrays must have one common positive "
                "view count."
            )
        if np.any(~np.isfinite(live_arr)) or np.any(live_arr <= 0.0):
            raise ValueError("Full-spectrum live times must be positive.")
        if (
            line_indices.size == 0
            or np.unique(line_indices).size != line_indices.size
            or np.any(line_indices < 0)
        ):
            raise ValueError(
                "positive_line_indices must be nonempty, unique, and "
                "nonnegative."
            )
        if self.kernel is None:
            raise RuntimeError("Continuous line transport requires PF poses.")
        self.validate_continuous_surface_states()
        device = (
            gpu_utils.resolve_device(self.config.gpu_device)
            if self.config.use_gpu
            else torch.device("cpu")
        )
        (
            positions,
            strengths,
            mask,
            chart_ids,
            _surface_uv,
        ) = self._packed_continuous_surface_state_arrays()
        particle_count = int(positions.shape[0])
        slot_count = int(mask.shape[1])
        view_count = int(fe_arr.size)
        line_count = int(line_indices.size)
        output_shape = (
            particle_count,
            view_count,
            slot_count,
            line_count,
        )
        arrays = [
            np.zeros(output_shape, dtype=np.float64) for _ in range(6)
        ]
        if np.any(mask):
            active_positions = self._surface_transport_positions(
                positions[mask],
                chart_ids=chart_ids[mask],
            )
            active_strengths = strengths[mask]
            particle_ids, source_slots = np.nonzero(mask)
            unique_positions, inverse = np.unique(
                active_positions,
                axis=0,
                return_inverse=True,
            )
            detector_position = np.asarray(
                self.kernel.poses[int(pose_idx)],
                dtype=np.float64,
            )
            detector_positions = np.repeat(
                detector_position.reshape(1, 3),
                view_count,
                axis=0,
            )
            components = (
                self.continuous_kernel
                .line_transport_components_selected_pairs_for_detectors(
                    isotope=self.isotope,
                    detector_positions=detector_positions,
                    sources=unique_positions,
                    fe_indices=fe_arr,
                    pb_indices=pb_arr,
                    positive_line_indices=line_indices,
                )
            )
            total_active = np.transpose(
                components.total_kernel[:, inverse, :],
                (1, 0, 2),
            )
            uncollided_active = np.transpose(
                components.uncollided_kernel[:, inverse, :],
                (1, 0, 2),
            )
            rate_scale = active_strengths[:, None, None]
            arrays[0][particle_ids, :, source_slots, :] = (
                total_active * rate_scale
            )
            arrays[1][particle_ids, :, source_slots, :] = (
                uncollided_active * rate_scale
            )
            for output, values in zip(
                arrays[2:],
                (
                    components.tau_fe,
                    components.tau_pb,
                    components.tau_obstacle,
                    components.distance_m,
                ),
            ):
                output[particle_ids, :, source_slots, :] = np.transpose(
                    np.asarray(values, dtype=np.float64)[:, inverse, :],
                    (1, 0, 2),
                )
        tensors = [
            torch.as_tensor(value, dtype=torch.float64, device=device)
            for value in arrays
        ]
        return TorchLineTransportComponents(*tensors)


    def _current_log_weights_torch(self, device: "torch.device") -> "torch.Tensor":
        """Return log-weights as a float64 torch tensor on the requested device."""
        import torch

        return torch.as_tensor(
            [p.log_weight for p in self.continuous_particles],
            device=device,
            dtype=torch.float64,
        )

    def _normalized_log_weights_torch(self, logw: "torch.Tensor") -> "torch.Tensor":
        """Normalize valid log-weights or fail before posterior corruption."""
        import torch

        if logw.ndim != 1 or int(logw.numel()) <= 0:
            raise ValueError("Particle log weights must be a nonempty vector.")
        if bool(torch.any(torch.isnan(logw)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(logw) & (logw > 0.0)).detach().cpu().item()
        ):
            raise RuntimeError(
                "Particle log weights contain NaN or positive infinity."
            )
        if not bool(torch.any(torch.isfinite(logw)).detach().cpu().item()):
            raise RuntimeError(
                "All particle log weights are negative infinity; posterior "
                "normalization is undefined."
            )
        normalizer = torch.logsumexp(logw, dim=0)
        if not bool(torch.isfinite(normalizer).detach().cpu().item()):
            raise RuntimeError("Particle log-weight normalizer is non-finite.")
        normalized = logw - normalizer
        if bool(torch.any(torch.isnan(normalized)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(normalized) & (normalized > 0.0))
            .detach()
            .cpu()
            .item()
        ):
            raise RuntimeError("Normalized particle log weights are invalid.")
        return normalized

    def _ess_from_logw_torch(self, logw: "torch.Tensor") -> float:
        """Return the effective sample size from normalized log-weights."""
        import torch

        if logw.ndim != 1 or int(logw.numel()) <= 0:
            raise ValueError(
                "ESS requires a nonempty normalized log-weight vector."
            )
        if bool(torch.any(torch.isnan(logw)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(logw) & (logw > 0.0)).detach().cpu().item()
        ) or not bool(torch.any(torch.isfinite(logw)).detach().cpu().item()):
            raise RuntimeError("ESS received invalid particle log weights.")
        log_normalizer = float(
            torch.logsumexp(logw, dim=0).detach().cpu().item()
        )
        if (
            not np.isfinite(log_normalizer)
            or not np.isclose(
                log_normalizer,
                0.0,
                rtol=0.0,
                atol=1.0e-10,
            )
        ):
            raise ValueError("ESS requires already normalized log weights.")
        w = torch.exp(logw)
        denominator = float(torch.sum(w**2).detach().cpu().item())
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("ESS denominator must be finite and positive.")
        ess = 1.0 / denominator
        if not np.isfinite(ess) or not 1.0 - 1.0e-9 <= ess <= int(
            logw.numel()
        ) + 1.0e-9:
            raise RuntimeError("Effective sample size lies outside its support.")
        return float(ess)

    def _select_delta_beta(
        self,
        logw_prev: "torch.Tensor",
        ll_t: "torch.Tensor",
        remaining: float,
        target_ess: float,
    ) -> tuple[float, "torch.Tensor", float]:
        """
        Return the largest delta_beta that keeps ESS above the target.

        Returns (delta_beta, logw_new, ess).
        """
        import torch

        remaining = float(remaining)
        target_ess = float(target_ess)
        if (
            logw_prev.ndim != 1
            or ll_t.ndim != 1
            or tuple(logw_prev.shape) != tuple(ll_t.shape)
            or int(logw_prev.numel()) <= 0
        ):
            raise ValueError(
                "Tempering requires aligned nonempty log-weight and likelihood "
                "vectors."
            )
        if not np.isfinite(remaining) or not 0.0 < remaining <= 1.0:
            raise ValueError("Tempering remaining beta must lie in (0, 1].")
        if (
            not np.isfinite(target_ess)
            or target_ess <= 0.0
            or target_ess > int(logw_prev.numel()) + 1.0e-9
        ):
            raise ValueError("Tempering target ESS lies outside particle support.")
        self._ess_from_logw_torch(logw_prev)
        if bool(torch.any(torch.isnan(ll_t)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(ll_t) & (ll_t > 0.0))
            .detach()
            .cpu()
            .item()
        ):
            raise RuntimeError(
                "Tempering likelihood contains NaN or positive infinity."
            )
        if not bool(torch.any(torch.isfinite(ll_t)).detach().cpu().item()):
            raise RuntimeError(
                "All particle likelihoods are negative infinity; the "
                "observation is impossible under the PF model."
            )
        min_delta = float(self.config.min_delta_beta)
        if remaining <= min_delta:
            logw_new = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
            ess = self._ess_from_logw_torch(logw_new)
            if ess + 1.0e-9 < target_ess:
                raise TemperingIncrementRequiresRejuvenation(
                    "The final configured tempering increment would violate "
                    "the target ESS and requires rejuvenation at the current "
                    "intermediate target."
                )
            return remaining, logw_new, ess

        logw_full = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
        ess_full = self._ess_from_logw_torch(logw_full)
        if ess_full >= target_ess:
            return remaining, logw_full, ess_full

        logw_low = self._normalized_log_weights_torch(logw_prev + min_delta * ll_t)
        ess_low = self._ess_from_logw_torch(logw_low)
        if ess_low < target_ess:
            raise TemperingIncrementRequiresRejuvenation(
                "No configured positive tempering increment preserves target "
                "ESS before rejuvenation at the current intermediate target."
            )

        low = min_delta
        high = remaining
        logw_best = logw_low
        ess_best = ess_low
        for _ in range(48):
            mid = 0.5 * (low + high)
            logw_mid = self._normalized_log_weights_torch(logw_prev + mid * ll_t)
            ess_mid = self._ess_from_logw_torch(logw_mid)
            if ess_mid >= target_ess:
                low = mid
                logw_best = logw_mid
                ess_best = ess_mid
            else:
                high = mid
        return low, logw_best, ess_best

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized weights for continuous particles."""
        logw = np.asarray(
            [p.log_weight for p in self.continuous_particles], dtype=np.float64
        )
        if logw.size == 0:
            return np.zeros(0, dtype=float)
        if np.any(np.isnan(logw)) or np.any(np.isposinf(logw)):
            raise RuntimeError("Particle log weights contain NaN or positive infinity.")
        finite = np.isfinite(logw)
        if not np.any(finite):
            raise RuntimeError(
                "All particle log weights are negative infinity; posterior is invalid."
            )
        normalized = np.zeros(logw.size, dtype=np.float64)
        shifted = logw[finite] - float(np.max(logw[finite]))
        normalized[finite] = np.exp(shifted)
        total = float(np.sum(normalized))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(
                "Particle weights do not have a finite positive normalization."
            )
        return normalized / total

    def _build_continuous_rj_position_proposal(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float,
    ) -> ContinuousSurfacePositionProposal:
        """Build the sweep-fixed full-spectrum residual proposal.

        Chart centers are evaluated only to define a proposal density.  The
        accepted state remains continuous in chart ``(u, v)`` and every MH/RJ
        target evaluation uses its exact XYZ.  A positive area-prior mixture
        gives global support, while the estimator supplies batched
        full-spectrum residual evidence and a chart-conditional strength
        location.
        """
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        evaluator = self._joint_proposal_evaluator
        if evaluator is None:
            raise RuntimeError(
                "Continuous exact-RJ requires the estimator-owned "
                "full-spectrum residual proposal evaluator."
            )
        prior_probabilities = np.asarray(
            atlas.chart_probabilities,
            dtype=np.float64,
        )
        alignment_scores, strength_locations, informative = evaluator(
            filt=self,
            data=data,
            chart_centers_xyz=np.asarray(
                atlas.geometry.centers_xyz,
                dtype=np.float64,
            ),
            target_beta=target_beta,
        )
        alignment = np.asarray(
            alignment_scores,
            dtype=np.float64,
        ).reshape(-1)
        locations = np.asarray(
            strength_locations,
            dtype=np.float64,
        ).reshape(-1)
        if (
            alignment.shape != (atlas.chart_count,)
            or locations.shape != (atlas.chart_count,)
            or np.any(~np.isfinite(alignment))
            or np.any(alignment < 0.0)
            or np.any(~np.isfinite(locations))
            or np.any(
                ~np.asarray(self._strength_prior.in_support(locations), dtype=bool)
            )
        ):
            raise ValueError(
                "Full-spectrum residual proposal arrays are invalid."
            )
        proposal = ContinuousSurfacePositionProposal(
            area_prior_probabilities=prior_probabilities,
            alignment_scores=(
                alignment
                if bool(informative)
                else np.zeros_like(alignment)
            ),
            prior_component_probability=float(
                self.config.structural_rj_position_proposal_prior_weight
            ),
        )
        strength_proposal = ContinuousStrengthProposal(
            minimum=float(self._strength_prior.minimum),
            maximum=float(self._strength_prior.maximum),
            data_locations_by_chart=locations,
            data_sigma=float(
                self.config.structural_rj_strength_proposal_sigma_fraction
            )
            * (
                float(self._strength_prior.maximum)
                - float(self._strength_prior.minimum)
            ),
            prior_component_probability=float(
                self.config.structural_rj_strength_proposal_prior_weight
            ),
            data_informative=bool(informative),
        )
        self._last_structural_rj_position_proposal = proposal
        self._structural_rj_strength_proposal = strength_proposal
        self._last_structural_rj_strength_proposal = strength_proposal
        self.last_structural_rj_proposal_snapshot_sha256 = (
            self._continuous_rj_proposal_snapshot_sha256(
                proposal,
                strength_proposal,
            )
        )
        return proposal

    @staticmethod
    def _continuous_rj_proposal_snapshot_sha256(
        position_proposal: ContinuousSurfacePositionProposal,
        strength_proposal: ContinuousStrengthProposal,
    ) -> str:
        """Hash every frozen parameter used by birth/death proposal densities."""
        digest = hashlib.sha256(
            b"continuous_surface_birth_proposal_snapshot_v1\0"
        )
        arrays = (
            position_proposal.area_prior_probabilities,
            position_proposal.alignment_scores,
            position_proposal.chart_probabilities,
            strength_proposal.data_locations_by_chart,
            np.asarray(
                [
                    position_proposal.prior_component_probability,
                    strength_proposal.minimum,
                    strength_proposal.maximum,
                    strength_proposal.data_sigma,
                    strength_proposal.prior_component_probability,
                    float(strength_proposal.data_informative),
                ],
                dtype="<f8",
            ),
        )
        for value in arrays:
            array = np.ascontiguousarray(value, dtype="<f8")
            digest.update(
                np.asarray(array.shape, dtype="<i8").tobytes(order="C")
            )
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def _active_continuous_rj_position_proposal(
        self,
    ) -> ContinuousSurfacePositionProposal:
        """Return the proposal frozen at the start of this structural sweep."""
        proposal = self._structural_rj_position_proposal
        if proposal is None:
            raise RuntimeError(
                "Continuous RJ position proposal was not frozen for this sweep."
            )
        return proposal

    def _active_continuous_rj_strength_proposal(
        self,
    ) -> ContinuousStrengthProposal:
        """Return the strength proposal frozen with the current sweep."""
        proposal = self._structural_rj_strength_proposal
        if proposal is None:
            raise RuntimeError(
                "Continuous RJ strength proposal was not frozen for this sweep."
            )
        return proposal


    def _continuous_rj_line_transport_component_columns(
        self,
        data: StructuralGeometryBatch,
        positions: NDArray[np.float64],
        positive_line_indices: NDArray[np.int64],
        *,
        chart_ids: NDArray[np.int64] | None = None,
    ) -> LineTransportComponents:
        """Return view-by-source-by-line unit-strength rate components."""
        requested = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        requested_chart_ids: NDArray[np.int64] | None = None
        if chart_ids is not None:
            raw_chart_ids = np.asarray(chart_ids)
            if (
                not np.issubdtype(raw_chart_ids.dtype, np.integer)
                or raw_chart_ids.size != requested.shape[0]
            ):
                raise ValueError(
                    "Continuous line-component chart IDs must be integer and "
                    "align with source positions."
                )
            requested_chart_ids = np.asarray(
                raw_chart_ids,
                dtype=np.int64,
            ).reshape(-1)
        requested_transport = self._surface_transport_positions(
            requested,
            chart_ids=requested_chart_ids,
        )
        measurement_count = data.row_count
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        detector_positions = np.asarray(
            data.detector_positions,
            dtype=np.float64,
        )
        fe_indices = np.asarray(data.fe_indices, dtype=np.int64).reshape(-1)
        pb_indices = np.asarray(data.pb_indices, dtype=np.int64).reshape(-1)
        live_times = np.asarray(data.live_times, dtype=np.float64).reshape(-1)
        if (
            detector_positions.shape != (measurement_count, 3)
            or fe_indices.size != measurement_count
            or pb_indices.size != measurement_count
            or live_times.size != measurement_count
            or np.any(~np.isfinite(live_times))
            or np.any(live_times <= 0.0)
        ):
            raise ValueError(
                "Line-component measurement geometry and live times are "
                "invalid."
            )
        components = (
            self.continuous_kernel
            .line_transport_components_selected_pairs_for_detectors(
                isotope=self.isotope,
                detector_positions=detector_positions,
                sources=requested_transport,
                fe_indices=fe_indices,
                pb_indices=pb_indices,
                positive_line_indices=line_indices,
            )
        )
        return LineTransportComponents(
            total_kernel=np.asarray(
                components.total_kernel,
                dtype=np.float64,
            ),
            unattenuated_kernel=np.asarray(
                components.unattenuated_kernel,
                dtype=np.float64,
            ),
            uncollided_kernel=np.asarray(
                components.uncollided_kernel,
                dtype=np.float64,
            ),
            tau_fe=np.asarray(components.tau_fe, dtype=np.float64),
            tau_pb=np.asarray(components.tau_pb, dtype=np.float64),
            tau_obstacle=np.asarray(
                components.tau_obstacle,
                dtype=np.float64,
            ),
            tau_obstacle_compton=np.asarray(
                components.tau_obstacle_compton,
                dtype=np.float64,
            ),
            distance_m=np.asarray(components.distance_m, dtype=np.float64),
        )

    def _continuous_rj_group_arrays(
        self,
        particle_indices: NDArray[np.int64],
        cardinality: int,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return chart, UV, derived XYZ, and strength arrays for one K."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        source_count = int(cardinality)
        states = [
            self.continuous_particles[int(particle_index)].state
            for particle_index in indices
        ]
        cardinalities = np.fromiter(
            (int(state.num_sources) for state in states),
            dtype=np.int64,
            count=indices.size,
        )
        if np.any(cardinalities != source_count):
            raise ValueError("Continuous RJ group mixes cardinalities.")
        if indices.size == 0 or source_count == 0:
            return (
                np.zeros((indices.size, source_count), dtype=np.int64),
                np.zeros((indices.size, source_count, 2), dtype=np.float64),
                np.zeros((indices.size, source_count, 3), dtype=np.float64),
                np.zeros((indices.size, source_count), dtype=np.float64),
            )
        charts = np.stack(
            [
                np.asarray(state.surface_chart_ids, dtype=np.int64).reshape(
                    source_count
                )
                for state in states
            ],
            axis=0,
        )
        surface_uv = np.stack(
            [
                np.asarray(state.surface_uv, dtype=np.float64).reshape(
                    source_count,
                    2,
                )
                for state in states
            ],
            axis=0,
        )
        strengths = np.stack(
            [
                np.asarray(state.strengths, dtype=np.float64).reshape(
                    source_count
                )
                for state in states
            ],
            axis=0,
        )
        positions = self._structural_rj_surface_atlas.positions_xyz(
            charts,
            surface_uv,
        )
        canonical = self._continuous_rj_canonicalize_rows(
            charts,
            surface_uv,
            positions,
            strengths,
        )
        if not all(
            np.array_equal(actual, expected)
            for actual, expected in zip(
                (charts, surface_uv, positions, strengths),
                canonical,
            )
        ):
            raise RuntimeError(
                "Continuous RJ state sources must already be canonical."
            )
        if not np.all(self._strength_prior.in_support(strengths)):
            raise ValueError("Continuous RJ strength lies outside its prior.")
        return (
            charts.astype(np.int64, copy=False),
            surface_uv.astype(np.float64, copy=False),
            positions.astype(np.float64, copy=False),
            strengths.astype(np.float64, copy=False),
        )

    def _continuous_rj_group_log_likelihood(
        self,
        data: StructuralGeometryBatch,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        chart_ids: NDArray[np.int64],
        particle_indices: NDArray[np.int64] | None = None,
        target_beta: float = 1.0,
        tempering_start_row: int | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate a batched equal-K group at an optional intermediate target."""
        position_array = np.asarray(positions, dtype=np.float64)
        strength_array = np.asarray(strengths, dtype=np.float64)
        raw_chart_ids = np.asarray(chart_ids)
        active_tempering_start_row = self._structural_rj_tempering_start_row
        if (
            tempering_start_row is not None
            and active_tempering_start_row is not None
            and int(tempering_start_row) != int(active_tempering_start_row)
        ):
            raise ValueError(
                "Continuous RJ likelihood evaluation changed the active "
                "tempering station boundary."
            )
        resolved_tempering_start_row = (
            active_tempering_start_row
            if tempering_start_row is None
            else int(tempering_start_row)
        )
        if (
            position_array.ndim != 3
            or position_array.shape[2] != 3
            or strength_array.shape != position_array.shape[:2]
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != strength_array.shape
        ):
            raise ValueError(
                "Continuous RJ chart, position, and strength arrays must share "
                "particle/source axes."
            )
        chart_id_array = np.asarray(raw_chart_ids, dtype=np.int64)
        particle_count = int(strength_array.shape[0])
        if self._joint_target_evaluator is not None:
            if particle_indices is None:
                raise ValueError(
                    "Joint-target RJ evaluation requires aligned particle indices."
                )
            indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
            if (
                indices.size != particle_count
                or np.any(indices < 0)
                or np.any(indices >= len(self.continuous_particles))
            ):
                raise ValueError(
                    "Joint-target particle indices must match the candidate rows."
                )
            result = np.asarray(
                self._joint_target_evaluator(
                    filt=self,
                    data=data,
                    positions_pks=position_array,
                    chart_ids_pk=chart_id_array,
                    strengths_pk=strength_array,
                    particle_indices=indices,
                    target_beta=float(target_beta),
                    tempering_start_row=resolved_tempering_start_row,
                ),
                dtype=np.float64,
            ).reshape(-1)
            if (
                result.size != particle_count
                or np.any(np.isnan(result))
                or np.any(np.isposinf(result))
            ):
                raise ValueError(
                    "Joint-target evaluator must return one finite or "
                    "negative-infinity value per candidate particle."
                )
            return result
        raise RuntimeError(
            "Continuous exact-RJ moves require the estimator-owned full "
            "joint-isotope target evaluator."
        )

    def set_joint_target_evaluator(
        self,
        evaluator: Callable[..., NDArray[np.float64]] | None,
    ) -> None:
        """Attach the estimator-owned aligned multi-isotope MH target."""
        if evaluator is not None and not callable(evaluator):
            raise TypeError("Joint target evaluator must be callable or None.")
        self._joint_target_evaluator = evaluator

    def set_joint_proposal_evaluator(
        self,
        evaluator: Callable[
            ...,
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                bool,
            ],
        ]
        | None,
    ) -> None:
        """Attach the estimator-owned full-spectrum residual proposal."""
        if evaluator is not None and not callable(evaluator):
            raise TypeError(
                "Joint proposal evaluator must be callable or None."
            )
        self._joint_proposal_evaluator = evaluator

    def _continuous_rj_canonicalize_rows(
        self,
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Canonicalize batched source rows by chart, U, then V."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        charts, uv = atlas.validate_coordinates(chart_ids, surface_uv)
        xyz = np.asarray(positions, dtype=np.float64)
        q = np.asarray(strengths, dtype=np.float64)
        if (
            charts.ndim != 2
            or uv.shape != charts.shape + (2,)
            or xyz.shape != charts.shape + (3,)
            or q.shape != charts.shape
        ):
            raise ValueError("Continuous RJ canonical arrays have invalid shapes.")
        derived_xyz = atlas.positions_xyz(charts, uv)
        if not np.allclose(
            xyz,
            derived_xyz,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                "Transient RJ XYZ must equal the authoritative chart/UV image."
            )
        xyz = derived_xyz
        if charts.shape[1] <= 1:
            return charts, uv, xyz, q
        order = np.lexsort(
            (uv[:, :, 1], uv[:, :, 0], charts),
            axis=1,
        )
        return (
            np.take_along_axis(charts, order, axis=1),
            np.take_along_axis(uv, order[:, :, None], axis=1),
            np.take_along_axis(xyz, order[:, :, None], axis=1),
            np.take_along_axis(q, order, axis=1),
        )

    def _commit_continuous_rj_states(
        self,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_],
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> int:
        """Commit accepted continuous chart states without changing PF weights."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
        charts, uv, xyz, q = self._continuous_rj_canonicalize_rows(
            chart_ids,
            surface_uv,
            positions,
            strengths,
        )
        if acceptance.size != indices.size or charts.shape[0] != indices.size:
            raise ValueError("Continuous RJ commit arrays must share P.")
        accepted_rows = np.flatnonzero(acceptance)
        cardinality = int(charts.shape[1])
        # All numerical proposal and acceptance work is batched. This loop only
        # commits variable-length state objects for the accepted particle rows.
        for row in accepted_rows.tolist():
            self.continuous_particles[int(indices[row])].state = IsotopeState(
                num_sources=cardinality,
                surface_chart_ids=charts[row],
                surface_uv=uv[row],
                strengths=q[row],
            )
        return int(accepted_rows.size)

    def _continuous_rj_transition_mass(
        self,
        name: str,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_] | None = None,
    ) -> None:
        """Accumulate attempted/accepted posterior weight mass diagnostics."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        weights = np.asarray(self.continuous_weights, dtype=np.float64)
        if accepted is None:
            selected = indices
        else:
            acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
            if acceptance.size != indices.size:
                raise ValueError("accepted must match particle_indices.")
            selected = indices[acceptance]
        key = f"{name}_weight_mass"
        mass = float(np.sum(weights[selected], dtype=np.float64))
        self._structural_rj_move_counts[key] = float(
            self._structural_rj_move_counts.get(key, 0.0)
        ) + mass
        self.last_structural_transition_weight_mass[key] = float(
            self.last_structural_transition_weight_mass.get(key, 0.0)
        ) + mass

    def _apply_continuous_rj_birth_death(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply one exact continuous-surface birth/death attempt per particle."""
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_move_probabilities
        if atlas is None or cardinality_prior is None or move_probabilities is None:
            raise RuntimeError("Continuous RJ priors are unavailable.")
        position_proposal = self._active_continuous_rj_position_proposal()
        strength_proposal = self._active_continuous_rj_strength_proposal()
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = self._random_generator.random(particle_count) < float(
            self.config.structural_rj_move_probability
        )
        accepted_births = 0
        accepted_deaths = 0
        attempted_births = 0
        attempted_deaths = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            group_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            if group_indices.size == 0:
                continue
            birth_probability, _ = move_probabilities.probabilities(
                int(cardinality)
            )
            birth_move = self._random_generator.random(group_indices.size) < float(
                birth_probability
            )
            for is_birth in (True, False):
                selected_rows = np.flatnonzero(birth_move == is_birth)
                if selected_rows.size == 0:
                    continue
                selected_indices = group_indices[selected_rows]
                (
                    chart_ids,
                    surface_uv,
                    positions,
                    strengths,
                ) = self._continuous_rj_group_arrays(
                    selected_indices,
                    int(cardinality),
                )
                base_ll = self._continuous_rj_group_log_likelihood(
                    data,
                    positions,
                    strengths,
                    chart_ids=chart_ids,
                    particle_indices=selected_indices,
                    target_beta=target_beta,
                )
                if is_birth:
                    attempted_births += int(selected_indices.size)
                    self._continuous_rj_transition_mass(
                        "birth_attempted",
                        selected_indices,
                    )
                    new_chart_ids, new_uv, new_positions = atlas.sample(
                        selected_indices.size,
                        rng=self._random_generator,
                        chart_probabilities=(
                            position_proposal.chart_probabilities
                        ),
                    )
                    new_strengths = np.asarray(
                        strength_proposal.sample(
                            new_chart_ids,
                            rng=self._random_generator,
                        ),
                        dtype=np.float64,
                    )
                    proposed_chart_ids = np.concatenate(
                        (chart_ids, new_chart_ids[:, None]),
                        axis=1,
                    )
                    proposed_uv = np.concatenate(
                        (surface_uv, new_uv[:, None, :]),
                        axis=1,
                    )
                    proposed_positions = np.concatenate(
                        (positions, new_positions[:, None, :]),
                        axis=1,
                    )
                    proposed_strengths = np.concatenate(
                        (strengths, new_strengths[:, None]),
                        axis=1,
                    )
                    (
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    ) = self._continuous_rj_canonicalize_rows(
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    proposed_ll = self._continuous_rj_group_log_likelihood(
                        data,
                        proposed_positions,
                        proposed_strengths,
                        chart_ids=proposed_chart_ids,
                        particle_indices=selected_indices,
                        target_beta=target_beta,
                    )
                    log_position_density = atlas.log_chart_probabilities[
                        new_chart_ids
                    ]
                    log_position_proposal = position_proposal.log_density(
                        new_chart_ids
                    )
                    log_strength_prior_density = np.asarray(
                        self._strength_prior.log_prob(new_strengths),
                        dtype=np.float64,
                    )
                    log_strength_proposal_density = (
                        strength_proposal.log_density(
                            new_chart_ids,
                            new_strengths,
                        )
                    )
                    log_target_ratio = _extended_log_target_ratio(
                        proposed_ll,
                        base_ll,
                    )
                    log_ratio = continuous_birth_log_acceptance_ratio(
                        current_cardinality=int(cardinality),
                        log_likelihood_ratio=log_target_ratio,
                        cardinality_prior=cardinality_prior,
                        move_probabilities=move_probabilities,
                        log_position_prior_density=log_position_density,
                        log_strength_prior_density=(
                            log_strength_prior_density
                        ),
                        log_forward_position_proposal=(
                            log_position_proposal
                        ),
                        log_forward_strength_proposal=(
                            log_strength_proposal_density
                        ),
                        log_abs_jacobian=0.0,
                    )
                    accepted = np.log(
                        self._random_generator.random(selected_indices.size)
                    ) < np.minimum(log_ratio, 0.0)
                    accepted_births += self._commit_continuous_rj_states(
                        selected_indices,
                        accepted,
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    self._continuous_rj_transition_mass(
                        "birth_accepted",
                        selected_indices,
                        accepted,
                    )
                    for row in np.flatnonzero(accepted).tolist():
                        state = self.continuous_particles[
                            int(selected_indices[row])
                        ].state
                        matches = (
                            np.asarray(state.surface_chart_ids)
                            == int(new_chart_ids[row])
                        ) & np.all(
                            np.asarray(state.surface_uv)
                            == new_uv[row],
                            axis=1,
                        )
                        source_column = int(np.flatnonzero(matches)[0])
                        self._record_source_event(
                            "source_birth_accepted",
                            state,
                            source_column,
                            reason="continuous_rj_mh_birth",
                            extra={
                                "delta_ll": float(log_target_ratio[row]),
                                "log_acceptance_ratio": float(log_ratio[row]),
                                "surface_chart_id": int(new_chart_ids[row]),
                                "surface_uv": new_uv[row].tolist(),
                            },
                        )
                    continue

                attempted_deaths += int(selected_indices.size)
                self._continuous_rj_transition_mass(
                    "death_attempted",
                    selected_indices,
                )
                death_columns = self._random_generator.integers(
                    0,
                    int(cardinality),
                    size=selected_indices.size,
                    dtype=np.int64,
                )
                rows = np.arange(selected_indices.size, dtype=np.int64)
                removed_chart_ids = chart_ids[rows, death_columns]
                removed_uv = surface_uv[rows, death_columns]
                removed_strengths = strengths[rows, death_columns]
                keep = (
                    np.arange(int(cardinality))[None, :]
                    != death_columns[:, None]
                )
                proposed_chart_ids = chart_ids[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                )
                proposed_uv = surface_uv[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                    2,
                )
                proposed_positions = positions[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                    3,
                )
                proposed_strengths = strengths[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                )
                proposed_ll = self._continuous_rj_group_log_likelihood(
                    data,
                    proposed_positions,
                    proposed_strengths,
                    chart_ids=proposed_chart_ids,
                    particle_indices=selected_indices,
                    target_beta=target_beta,
                )
                log_position_density = atlas.log_chart_probabilities[
                    removed_chart_ids
                ]
                log_reverse_position_proposal = (
                    position_proposal.log_density(removed_chart_ids)
                )
                log_strength_prior_density = np.asarray(
                    self._strength_prior.log_prob(removed_strengths),
                    dtype=np.float64,
                )
                log_reverse_strength_proposal = (
                    strength_proposal.log_density(
                        removed_chart_ids,
                        removed_strengths,
                    )
                )
                log_target_ratio = _extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                )
                log_ratio = continuous_death_log_acceptance_ratio(
                    current_cardinality=int(cardinality),
                    log_likelihood_ratio=log_target_ratio,
                    cardinality_prior=cardinality_prior,
                    move_probabilities=move_probabilities,
                    log_removed_position_prior_density=log_position_density,
                    log_removed_strength_prior_density=(
                        log_strength_prior_density
                    ),
                    log_reverse_position_proposal=(
                        log_reverse_position_proposal
                    ),
                    log_reverse_strength_proposal=(
                        log_reverse_strength_proposal
                    ),
                    log_abs_reverse_jacobian=0.0,
                )
                accepted = np.log(
                    self._random_generator.random(selected_indices.size)
                ) < np.minimum(log_ratio, 0.0)
                for row in np.flatnonzero(accepted).tolist():
                    old_state = self.continuous_particles[
                        int(selected_indices[row])
                    ].state
                    self._record_source_event(
                        "source_removed",
                        old_state,
                        int(death_columns[row]),
                        reason="continuous_rj_mh_death",
                        extra={
                            "delta_ll": float(log_target_ratio[row]),
                            "log_acceptance_ratio": float(log_ratio[row]),
                            "surface_chart_id": int(removed_chart_ids[row]),
                            "surface_uv": removed_uv[row].tolist(),
                        },
                    )
                accepted_deaths += self._commit_continuous_rj_states(
                    selected_indices,
                    accepted,
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                self._continuous_rj_transition_mass(
                    "death_accepted",
                    selected_indices,
                    accepted,
                )
        self._structural_rj_move_counts.update(
            {
                "birth_attempted": int(attempted_births),
                "birth_accepted": int(accepted_births),
                "death_attempted": int(attempted_deaths),
                "death_accepted": int(accepted_deaths),
            }
        )
        return accepted_births, accepted_deaths

    def _apply_continuous_rj_global_position_moves(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply an exact joint global position-and-strength independence move.

        Full-spectrum location and activity are strongly correlated.  Drawing
        both from the same sweep-frozen, state-independent proposal lets the
        kernel cross that correlation without weakening the target: the exact
        reverse position and chart-conditional strength densities remain in
        the MH ratio.
        """
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        position_proposal = self._active_continuous_rj_position_proposal()
        strength_proposal = self._active_continuous_rj_strength_proposal()
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_position_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            (
                chart_ids,
                surface_uv,
                positions,
                strengths,
            ) = self._continuous_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._continuous_rj_group_log_likelihood(
                data,
                positions,
                strengths,
                chart_ids=chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            old_chart_ids = chart_ids[rows, source_columns].copy()
            old_strengths = strengths[rows, source_columns].copy()
            new_chart_ids, new_uv, new_positions = atlas.sample(
                particle_indices.size,
                rng=self._random_generator,
                chart_probabilities=position_proposal.chart_probabilities,
            )
            new_strengths = strength_proposal.sample(
                new_chart_ids,
                rng=self._random_generator,
            )
            proposed_chart_ids = chart_ids.copy()
            proposed_uv = surface_uv.copy()
            proposed_positions = positions.copy()
            proposed_strengths = strengths.copy()
            proposed_chart_ids[rows, source_columns] = new_chart_ids
            proposed_uv[rows, source_columns] = new_uv
            proposed_positions[rows, source_columns] = new_positions
            proposed_strengths[rows, source_columns] = new_strengths
            (
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            ) = self._continuous_rj_canonicalize_rows(
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            )
            proposed_ll = self._continuous_rj_group_log_likelihood(
                data,
                proposed_positions,
                proposed_strengths,
                chart_ids=proposed_chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            old_log_density = atlas.log_chart_probabilities[old_chart_ids]
            new_log_density = atlas.log_chart_probabilities[new_chart_ids]
            old_log_proposal = position_proposal.log_density(old_chart_ids)
            new_log_proposal = position_proposal.log_density(new_chart_ids)
            old_strength_log_prior = np.asarray(
                self._strength_prior.log_prob(old_strengths),
                dtype=np.float64,
            )
            new_strength_log_prior = np.asarray(
                self._strength_prior.log_prob(new_strengths),
                dtype=np.float64,
            )
            old_strength_log_proposal = strength_proposal.log_density(
                old_chart_ids,
                old_strengths,
            )
            new_strength_log_proposal = strength_proposal.log_density(
                new_chart_ids,
                new_strengths,
            )
            log_ratio = continuous_joint_position_strength_log_acceptance_ratio(
                log_likelihood_ratio=_extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                ),
                log_old_position_prior_density=old_log_density,
                log_new_position_prior_density=new_log_density,
                log_old_strength_prior_density=old_strength_log_prior,
                log_new_strength_prior_density=new_strength_log_prior,
                log_reverse_position_proposal_density=old_log_proposal,
                log_forward_position_proposal_density=new_log_proposal,
                log_reverse_strength_proposal_density=(
                    old_strength_log_proposal
                ),
                log_forward_strength_proposal_density=(
                    new_strength_log_proposal
                ),
                log_abs_jacobian=0.0,
            )
            accepted = np.log(
                self._random_generator.random(particle_indices.size)
            ) < np.minimum(log_ratio, 0.0)
            accepted_count += self._commit_continuous_rj_states(
                particle_indices,
                accepted,
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "global_position_attempted": attempted_count,
                "global_position_accepted": accepted_count,
            }
        )
        return accepted_count

    def _apply_continuous_rj_local_position_moves(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply exact symmetric tangent proposals across surface-chart portals."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_local_position_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        movable_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            (
                chart_ids,
                surface_uv,
                positions,
                strengths,
            ) = self._continuous_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._continuous_rj_group_log_likelihood(
                data,
                positions,
                strengths,
                chart_ids=chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            selected_chart_ids = chart_ids[rows, source_columns]
            old_uv = surface_uv[rows, source_columns]
            (
                new_chart_ids,
                new_uv,
                log_reverse_over_forward,
            ) = atlas.tangent_geodesic_portal_proposal(
                selected_chart_ids,
                old_uv,
                sigma_m=float(
                    self.config.structural_rj_local_position_sigma_m
                ),
                rng=self._random_generator,
            )
            new_positions = atlas.positions_xyz(new_chart_ids, new_uv)
            proposed_chart_ids = chart_ids.copy()
            proposed_uv = surface_uv.copy()
            proposed_positions = positions.copy()
            proposed_chart_ids[rows, source_columns] = new_chart_ids
            proposed_uv[rows, source_columns] = new_uv
            proposed_positions[rows, source_columns] = new_positions
            (
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            ) = self._continuous_rj_canonicalize_rows(
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                strengths,
            )
            proposed_ll = self._continuous_rj_group_log_likelihood(
                data,
                proposed_positions,
                proposed_strengths,
                chart_ids=proposed_chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            old_chart_log_density = atlas.log_chart_probabilities[
                selected_chart_ids
            ]
            new_chart_log_density = atlas.log_chart_probabilities[
                new_chart_ids
            ]
            zeros = np.zeros(particle_indices.size, dtype=np.float64)
            log_ratio = continuous_position_log_acceptance_ratio(
                log_likelihood_ratio=_extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                ),
                log_old_position_prior_density=old_chart_log_density,
                log_new_position_prior_density=new_chart_log_density,
                log_reverse_proposal_density=log_reverse_over_forward,
                log_forward_proposal_density=zeros,
                log_abs_jacobian=0.0,
            )
            moved = (new_chart_ids != selected_chart_ids) | np.any(
                new_uv != old_uv,
                axis=1,
            )
            movable_count += int(np.count_nonzero(moved))
            accepted = moved & (
                np.log(self._random_generator.random(particle_indices.size))
                < np.minimum(log_ratio, 0.0)
            )
            accepted_count += self._commit_continuous_rj_states(
                particle_indices,
                accepted,
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "local_position_attempted": attempted_count,
                "local_position_movable": movable_count,
                "local_position_accepted": accepted_count,
            }
        )
        return accepted_count

    def _apply_continuous_rj_strength_moves(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply exact prior-independence strength proposals in one batch per K."""
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_strength_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            (
                chart_ids,
                surface_uv,
                positions,
                strengths,
            ) = self._continuous_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._continuous_rj_group_log_likelihood(
                data,
                positions,
                strengths,
                chart_ids=chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            proposed_strengths = strengths.copy()
            proposed_strengths[
                np.arange(particle_indices.size),
                source_columns,
            ] = np.asarray(
                self._strength_prior.sample(
                    particle_indices.size,
                    rng=self._random_generator,
                ),
                dtype=np.float64,
            )
            proposed_ll = self._continuous_rj_group_log_likelihood(
                data,
                positions,
                proposed_strengths,
                chart_ids=chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            log_target_ratio = _extended_log_target_ratio(
                proposed_ll,
                base_ll,
            )
            accepted = np.log(
                self._random_generator.random(particle_indices.size)
            ) < np.minimum(log_target_ratio, 0.0)
            accepted_count += self._commit_continuous_rj_states(
                particle_indices,
                accepted,
                chart_ids,
                surface_uv,
                positions,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "strength_attempted": attempted_count,
                "strength_accepted": accepted_count,
            }
        )
        return accepted_count

    def _apply_continuous_rj_split_merge(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply exact strength-transfer split/merge RJ proposals.

        A split keeps one source position, draws a second position from the
        continuous surface prior, and partitions the original strength.  Its
        reverse merge deletes the ordered donor and transfers its strength to
        the ordered recipient.  The full strength-map Jacobian and truncated
        split-fraction density are included in the RJ ratio.
        """
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_split_merge_probabilities
        if atlas is None or cardinality_prior is None or move_probabilities is None:
            raise RuntimeError("Continuous split/merge priors are unavailable.")
        position_proposal = self._active_continuous_rj_position_proposal()
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        split_probabilities, merge_probabilities = (
            move_probabilities.probabilities(cardinalities)
        )
        direction_available = (
            np.asarray(split_probabilities, dtype=np.float64)
            + np.asarray(merge_probabilities, dtype=np.float64)
        ) > 0.0
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_split_merge_probability)
        ) & direction_available
        attempted_splits = 0
        accepted_splits = 0
        attempted_merges = 0
        accepted_merges = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            group_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            split_probability, _ = move_probabilities.probabilities(
                int(cardinality)
            )
            split_move = self._random_generator.random(group_indices.size) < float(
                split_probability
            )
            for is_split in (True, False):
                selected_rows = np.flatnonzero(split_move == is_split)
                if selected_rows.size == 0:
                    continue
                particle_indices = group_indices[selected_rows]
                (
                    chart_ids,
                    surface_uv,
                    positions,
                    strengths,
                ) = self._continuous_rj_group_arrays(
                    particle_indices,
                    int(cardinality),
                )
                base_ll = self._continuous_rj_group_log_likelihood(
                    data,
                    positions,
                    strengths,
                    chart_ids=chart_ids,
                    particle_indices=particle_indices,
                    target_beta=target_beta,
                )
                rows = np.arange(particle_indices.size, dtype=np.int64)
                if is_split:
                    attempted_splits += int(particle_indices.size)
                    self._continuous_rj_transition_mass(
                        "split_attempted",
                        particle_indices,
                    )
                    source_columns = self._random_generator.integers(
                        0,
                        int(cardinality),
                        size=particle_indices.size,
                        dtype=np.int64,
                    )
                    total_strength = strengths[rows, source_columns]
                    lower, upper, feasible = split_fraction_bounds(
                        total_strength,
                        minimum_strength=self._strength_prior.minimum,
                        maximum_strength=self._strength_prior.maximum,
                    )
                    width = upper - lower
                    safe_width = np.where(feasible, width, 1.0)
                    fraction = lower + safe_width * self._random_generator.random(
                        particle_indices.size
                    )
                    retained_strength = (1.0 - fraction) * total_strength
                    new_strength = fraction * total_strength
                    new_chart_ids, new_uv, new_positions = atlas.sample(
                        particle_indices.size,
                        rng=self._random_generator,
                        chart_probabilities=(
                            position_proposal.chart_probabilities
                        ),
                    )
                    base_split_strengths = strengths.copy()
                    base_split_strengths[
                        rows,
                        source_columns,
                    ] = retained_strength
                    proposed_chart_ids = np.concatenate(
                        (chart_ids, new_chart_ids[:, None]),
                        axis=1,
                    )
                    proposed_uv = np.concatenate(
                        (surface_uv, new_uv[:, None, :]),
                        axis=1,
                    )
                    proposed_positions = np.concatenate(
                        (positions, new_positions[:, None, :]),
                        axis=1,
                    )
                    proposed_strengths = np.concatenate(
                        (base_split_strengths, new_strength[:, None]),
                        axis=1,
                    )
                    (
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    ) = self._continuous_rj_canonicalize_rows(
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    proposed_ll = self._continuous_rj_group_log_likelihood(
                        data,
                        proposed_positions,
                        proposed_strengths,
                        chart_ids=proposed_chart_ids,
                        particle_indices=particle_indices,
                        target_beta=target_beta,
                    )
                    log_ratio = np.full(
                        particle_indices.size,
                        float("-inf"),
                        dtype=np.float64,
                    )
                    valid_rows = np.flatnonzero(feasible)
                    if valid_rows.size:
                        log_position_density = atlas.log_chart_probabilities[
                            new_chart_ids[valid_rows]
                        ]
                        log_position_proposal = (
                            position_proposal.log_density(
                                new_chart_ids[valid_rows]
                            )
                        )
                        log_ratio[valid_rows] = (
                            continuous_split_log_acceptance_ratio(
                                current_cardinality=int(cardinality),
                                total_strength=total_strength[valid_rows],
                                log_likelihood_ratio=(
                                    _extended_log_target_ratio(
                                        proposed_ll[valid_rows],
                                        base_ll[valid_rows],
                                    )
                                ),
                                cardinality_prior=cardinality_prior,
                                move_probabilities=move_probabilities,
                                log_new_position_prior_density=(
                                    log_position_density
                                ),
                                log_old_strength_prior_density=(
                                    self._strength_prior.log_prob(
                                        total_strength[valid_rows]
                                    )
                                ),
                                log_retained_strength_prior_density=(
                                    self._strength_prior.log_prob(
                                        retained_strength[valid_rows]
                                    )
                                ),
                                log_new_strength_prior_density=(
                                    self._strength_prior.log_prob(
                                        new_strength[valid_rows]
                                    )
                                ),
                                log_forward_position_proposal=(
                                    log_position_proposal
                                ),
                                log_forward_fraction_proposal=(
                                    -np.log(width[valid_rows])
                                ),
                            )
                        )
                    accepted = feasible & (
                        np.log(
                            self._random_generator.random(particle_indices.size)
                        )
                        < np.minimum(log_ratio, 0.0)
                    )
                    accepted_splits += self._commit_continuous_rj_states(
                        particle_indices,
                        accepted,
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    self._continuous_rj_transition_mass(
                        "split_accepted",
                        particle_indices,
                        accepted,
                    )
                    continue

                attempted_merges += int(particle_indices.size)
                self._continuous_rj_transition_mass(
                    "merge_attempted",
                    particle_indices,
                )
                delete_columns = self._random_generator.integers(
                    0,
                    int(cardinality),
                    size=particle_indices.size,
                    dtype=np.int64,
                )
                receiver_raw = self._random_generator.integers(
                    0,
                    int(cardinality) - 1,
                    size=particle_indices.size,
                    dtype=np.int64,
                )
                receiver_columns = receiver_raw + (
                    receiver_raw >= delete_columns
                ).astype(np.int64)
                deleted_chart_ids = chart_ids[rows, delete_columns]
                deleted_strengths = strengths[rows, delete_columns]
                retained_strengths = strengths[rows, receiver_columns]
                merged_strength = deleted_strengths + retained_strengths
                lower, upper, reverse_feasible = split_fraction_bounds(
                    merged_strength,
                    minimum_strength=self._strength_prior.minimum,
                    maximum_strength=self._strength_prior.maximum,
                )
                reverse_fraction = deleted_strengths / np.maximum(
                    merged_strength,
                    np.finfo(np.float64).tiny,
                )
                feasible = (
                    np.asarray(
                        self._strength_prior.in_support(merged_strength),
                        dtype=bool,
                    )
                    & reverse_feasible
                    & (reverse_fraction >= lower)
                    & (reverse_fraction <= upper)
                )
                merged_strengths = strengths.copy()
                merged_strengths[
                    rows,
                    receiver_columns,
                ] = merged_strength
                keep = (
                    np.arange(int(cardinality))[None, :]
                    != delete_columns[:, None]
                )
                proposed_chart_ids = chart_ids[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 1,
                )
                proposed_uv = surface_uv[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 1,
                    2,
                )
                proposed_positions = positions[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 1,
                    3,
                )
                proposed_strengths = merged_strengths[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 1,
                )
                (
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                ) = self._continuous_rj_canonicalize_rows(
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                proposed_ll = self._continuous_rj_group_log_likelihood(
                    data,
                    proposed_positions,
                    proposed_strengths,
                    chart_ids=proposed_chart_ids,
                    particle_indices=particle_indices,
                    target_beta=target_beta,
                )
                log_ratio = np.full(
                    particle_indices.size,
                    float("-inf"),
                    dtype=np.float64,
                )
                valid_rows = np.flatnonzero(feasible)
                if valid_rows.size:
                    log_position_density = atlas.log_chart_probabilities[
                        deleted_chart_ids[valid_rows]
                    ]
                    log_reverse_position_proposal = (
                        position_proposal.log_density(
                            deleted_chart_ids[valid_rows]
                        )
                    )
                    width = upper[valid_rows] - lower[valid_rows]
                    log_ratio[valid_rows] = continuous_merge_log_acceptance_ratio(
                        current_cardinality=int(cardinality),
                        merged_strength=merged_strength[valid_rows],
                        log_likelihood_ratio=(
                            _extended_log_target_ratio(
                                proposed_ll[valid_rows],
                                base_ll[valid_rows],
                            )
                        ),
                        cardinality_prior=cardinality_prior,
                        move_probabilities=move_probabilities,
                        log_deleted_position_prior_density=log_position_density,
                        log_deleted_strength_prior_density=(
                            self._strength_prior.log_prob(
                                deleted_strengths[valid_rows]
                            )
                        ),
                        log_retained_strength_prior_density=(
                            self._strength_prior.log_prob(
                                retained_strengths[valid_rows]
                            )
                        ),
                        log_merged_strength_prior_density=(
                            self._strength_prior.log_prob(
                                merged_strength[valid_rows]
                            )
                        ),
                        log_reverse_position_proposal=(
                            log_reverse_position_proposal
                        ),
                        log_reverse_fraction_proposal=-np.log(width),
                    )
                accepted = feasible & (
                    np.log(self._random_generator.random(particle_indices.size))
                    < np.minimum(log_ratio, 0.0)
                )
                accepted_merges += self._commit_continuous_rj_states(
                    particle_indices,
                    accepted,
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                self._continuous_rj_transition_mass(
                    "merge_accepted",
                    particle_indices,
                    accepted,
                )
        self._structural_rj_move_counts.update(
            {
                "split_attempted": attempted_splits,
                "split_accepted": accepted_splits,
                "merge_attempted": attempted_merges,
                "merge_accepted": accepted_merges,
            }
        )
        return accepted_splits, accepted_merges

    def _apply_exact_structural_rj_moves(
        self,
        evidence_data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
        tempering_start_row: int | None = None,
    ) -> None:
        """Apply continuous RJ/MH and always clear the tempered-target context."""
        self._structural_rj_tempering_start_row = tempering_start_row
        try:
            self._apply_exact_structural_rj_moves_impl(
                evidence_data,
                target_beta=target_beta,
            )
        finally:
            self._structural_rj_position_proposal = None
            self._structural_rj_strength_proposal = None
            self._structural_rj_tempering_start_row = None

    def _apply_exact_structural_rj_moves_impl(
        self,
        evidence_data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> None:
        """Apply continuous-surface RJ/MH at the requested tempered target."""
        structural_start = time.perf_counter()
        original_log_weights = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        self._structural_rj_move_counts = {
            "birth_attempted": 0,
            "birth_accepted": 0,
            "death_attempted": 0,
            "death_accepted": 0,
            "global_position_attempted": 0,
            "global_position_accepted": 0,
            "local_position_attempted": 0,
            "local_position_movable": 0,
            "local_position_accepted": 0,
            "strength_attempted": 0,
            "strength_accepted": 0,
            "split_attempted": 0,
            "split_accepted": 0,
            "merge_attempted": 0,
            "merge_accepted": 0,
        }
        response_start = time.perf_counter()
        self._structural_rj_position_proposal = (
            self._build_continuous_rj_position_proposal(
                evidence_data,
                target_beta=target_beta,
            )
        )
        response_elapsed = time.perf_counter() - response_start
        birth_count = 0
        death_count = 0
        birth_death_elapsed = 0.0
        if self._variable_cardinality_enabled():
            move_start = time.perf_counter()
            birth_count, death_count = self._apply_continuous_rj_birth_death(
                evidence_data,
                target_beta=target_beta,
            )
            birth_death_elapsed = time.perf_counter() - move_start
        position_start = time.perf_counter()
        position_count = self._apply_continuous_rj_global_position_moves(
            evidence_data,
            target_beta=target_beta,
        )
        position_elapsed = time.perf_counter() - position_start
        local_position_start = time.perf_counter()
        local_position_count = self._apply_continuous_rj_local_position_moves(
            evidence_data,
            target_beta=target_beta,
        )
        local_position_elapsed = time.perf_counter() - local_position_start
        strength_start = time.perf_counter()
        strength_count = self._apply_continuous_rj_strength_moves(
            evidence_data,
            target_beta=target_beta,
        )
        strength_elapsed = time.perf_counter() - strength_start
        split_merge_start = time.perf_counter()
        split_count = 0
        merge_count = 0
        if self._variable_cardinality_enabled():
            split_count, merge_count = self._apply_continuous_rj_split_merge(
                evidence_data,
                target_beta=target_beta,
            )
        split_merge_elapsed = time.perf_counter() - split_merge_start
        current_log_weights = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        outer_weight_array_equal = bool(
            np.array_equal(original_log_weights, current_log_weights)
        )
        with np.errstate(invalid="ignore"):
            outer_weight_differences = np.where(
                original_log_weights == current_log_weights,
                0.0,
                np.abs(original_log_weights - current_log_weights),
            )
        outer_weight_differences = np.where(
            np.isfinite(outer_weight_differences),
            outer_weight_differences,
            float("inf"),
        )
        outer_weight_max_abs_diff = (
            float(np.max(outer_weight_differences))
            if outer_weight_differences.size
            else 0.0
        )
        position_proposal = self._active_continuous_rj_position_proposal()
        strength_proposal = self._active_continuous_rj_strength_proposal()
        proposal_probabilities = np.asarray(
            position_proposal.chart_probabilities,
            dtype=np.float64,
        )
        proposal_entropy = -float(
            np.sum(
                proposal_probabilities
                * np.log(np.maximum(proposal_probabilities, 1.0e-300)),
                dtype=np.float64,
            )
        )
        self.last_birth_count += int(birth_count)
        self.last_death_count += int(death_count)
        self.last_structural_timing_s = {
            "total": float(time.perf_counter() - structural_start),
            "response_dictionary": float(response_elapsed),
            "rj_birth_death": float(birth_death_elapsed),
            "rj_position": float(position_elapsed),
            "rj_global_position": float(position_elapsed),
            "rj_local_position": float(local_position_elapsed),
            "rj_strength": float(strength_elapsed),
            "rj_split_merge": float(split_merge_elapsed),
            "target_beta": float(target_beta),
            "rj_birth_attempted": float(
                self._structural_rj_move_counts["birth_attempted"]
            ),
            "rj_birth_accepted": float(birth_count),
            "rj_death_attempted": float(
                self._structural_rj_move_counts["death_attempted"]
            ),
            "rj_death_accepted": float(death_count),
            "rj_global_position_attempted": float(
                self._structural_rj_move_counts["global_position_attempted"]
            ),
            "rj_global_position_accepted": float(position_count),
            "rj_position_attempted": float(
                self._structural_rj_move_counts["global_position_attempted"]
            ),
            "rj_position_accepted": float(position_count),
            "rj_local_position_attempted": float(
                self._structural_rj_move_counts["local_position_attempted"]
            ),
            "rj_local_position_movable": float(
                self._structural_rj_move_counts["local_position_movable"]
            ),
            "rj_local_position_accepted": float(local_position_count),
            "rj_strength_attempted": float(
                self._structural_rj_move_counts["strength_attempted"]
            ),
            "rj_strength_accepted": float(strength_count),
            "rj_split_attempted": float(
                self._structural_rj_move_counts["split_attempted"]
            ),
            "rj_split_accepted": float(split_count),
            "rj_merge_attempted": float(
                self._structural_rj_move_counts["merge_attempted"]
            ),
            "rj_merge_accepted": float(merge_count),
            "rj_birth_attempted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "birth_attempted_weight_mass",
                    0.0,
                )
            ),
            "rj_birth_accepted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "birth_accepted_weight_mass",
                    0.0,
                )
            ),
            "rj_death_attempted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "death_attempted_weight_mass",
                    0.0,
                )
            ),
            "rj_death_accepted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "death_accepted_weight_mass",
                    0.0,
                )
            ),
            "rj_position_proposal_prior_weight": float(
                position_proposal.prior_component_probability
            ),
            "rj_position_proposal_data_informative": float(
                position_proposal.data_informative
            ),
            "rj_position_proposal_max_chart_mass": float(
                np.max(proposal_probabilities)
            ),
            "rj_position_proposal_entropy": float(proposal_entropy),
            "rj_strength_proposal_prior_weight": float(
                strength_proposal.prior_component_probability
            ),
            "rj_strength_proposal_data_informative": float(
                strength_proposal.data_informative
            ),
            "rj_strength_proposal_sigma_cps_1m": float(
                strength_proposal.data_sigma
            ),
            "rj_strength_proposal_location_min_cps_1m": float(
                np.min(strength_proposal.data_locations_by_chart)
            ),
            "rj_strength_proposal_location_max_cps_1m": float(
                np.max(strength_proposal.data_locations_by_chart)
            ),
            "outer_log_weight_max_abs_diff": float(outer_weight_max_abs_diff),
            "outer_log_weight_array_equal": float(outer_weight_array_equal),
            "weights_preserved": float(outer_weight_array_equal),
        }
        if not outer_weight_array_equal:
            raise RuntimeError("rj_mh rejuvenation must not alter PF weights.")

    def apply_structural_moves(
        self,
        evidence_data: StructuralGeometryBatch | None,
        *,
        target_beta: float = 1.0,
        tempering_start_row: int | None = None,
    ) -> None:
        """Apply exact continuous-surface MH/RJ rejuvenation when evidence exists."""
        if not self.continuous_particles:
            return
        if evidence_data is None or evidence_data.row_count == 0:
            self.last_structural_timing_s = {
                "total": 0.0,
                "rj_mh_no_evidence": 1.0,
                "weights_preserved": 1.0,
            }
            return
        self._apply_exact_structural_rj_moves(
            evidence_data,
            target_beta=target_beta,
            tempering_start_row=tempering_start_row,
        )

    def estimate(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the canonical MAP-cardinality PF posterior projection."""
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        self.validate_continuous_surface_states()
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError(
                "Canonical PF projection requires the continuous surface atlas."
            )
        point_estimate = posterior_point_estimate_from_states(
            [particle.state for particle in self.continuous_particles],
            np.asarray(self.continuous_weights, dtype=float),
            max_cardinality=self.config.max_sources,
            positions_by_state=[
                self.continuous_state_positions(particle.state)
                for particle in self.continuous_particles
            ],
            surface_chart_ids_by_state=[
                np.asarray(
                    particle.state.surface_chart_ids,
                    dtype=np.int64,
                )
                for particle in self.continuous_particles
            ],
            surface_uv_by_state=[
                np.asarray(
                    particle.state.surface_uv,
                    dtype=np.float64,
                )
                for particle in self.continuous_particles
            ],
            surface_coordinate_path_distance=(
                atlas.surface_coordinate_path_distance_upper_bound_m
            ),
        )
        positions = np.asarray(
            [mode.position_medoid_xyz for mode in point_estimate.modes],
            dtype=float,
        ).reshape(-1, 3)
        strengths = np.asarray(
            [
                mode.strength_representative_cps_1m
                for mode in point_estimate.modes
            ],
            dtype=float,
        )
        return positions, strengths

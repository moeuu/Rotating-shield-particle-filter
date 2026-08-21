"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import MeasurementGeometry, ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.surface_atlas import ContinuousSurfaceAtlas
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.diagnostics import build_source_event_record, reset_step_diagnostics
from pf.particle_filter_math import (
    extended_log_target_ratio as _extended_log_target_ratio,  # noqa: F401
)
from pf.particle_filter_rj_basic import StructuralRJBasicMoveMixin
from pf.particle_filter_rj_block import StructuralRJBlockIndependenceMixin
from pf.particle_filter_rj_multi import StructuralRJMultiComponentMixin
from pf.particle_filter_rj_proposal import StructuralRJProposalMixin
from pf.particle_filter_rj_runtime import StructuralRJSweepMixin
from pf.particle_filter_rj_split_merge import StructuralRJSplitMergeMixin
from pf.particle_filter_rj_target import StructuralRJTargetMixin
from pf.particle_filter_surface import ParticleSurfaceMixin
from pf.particle_filter_tempering import (
    ParticleTemperingMixin,
    TemperingIncrementRequiresRejuvenation,  # noqa: F401
)
from pf.posterior import posterior_point_estimate_from_states
from pf.particle_types import (
    StructuralGeometryBatch,
    TorchLineTransportComponents,  # noqa: F401
)
from pf.randomness import isotope_random_generator, normalize_pf_random_seed
from pf.state import IsotopeState
from pf.strength_prior import StrengthPrior
from spectrum.additive_scatter import AdditiveNoncollidedTransportResponse
from pf.structural_rj import (
    POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
    BirthDeathMoveProbabilities,
    CardinalityPrior,
    ContinuousStrengthProposal,
    ContinuousSurfacePositionProposal,
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    SplitMergeMoveProbabilities,
    truncated_poisson_cardinality_probabilities,
    poisson_geometric_tail_cardinality_probabilities,
    validate_cardinality_prior_policy,
)

if TYPE_CHECKING:
    import torch


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


@dataclass
class PFConfig:
    """Particle filter configuration (Sec. 3.4)."""

    num_particles: int = 200
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    hard_max_sources: int | None = None
    variable_cardinality: bool = True
    structural_rj_surface_chart_max_edge_m: float = 1.0
    structural_rj_move_probability: float = 1.0
    structural_rj_birth_probability: float = 0.5
    structural_rj_death_probability: float = 0.5
    structural_rj_position_move_probability: float = 1.0
    structural_rj_position_proposal_prior_weight: float = 0.5
    structural_rj_strength_proposal_prior_weight: float = 0.5
    structural_rj_strength_proposal_sigma_fraction: float = 0.15
    structural_rj_strength_proposal_grid_size: int = 5
    structural_rj_proposal_chart_batch_size: int = 256
    structural_rj_proposal_score_cache_max_bytes: int = 268_435_456
    structural_rj_local_position_move_probability: float = 1.0
    structural_rj_local_position_sigma_m: float = 0.5
    structural_rj_strength_move_probability: float = 1.0
    structural_rj_split_merge_probability: float = 1.0
    structural_rj_block_independence_probability: float = 0.1
    structural_rj_multi_component_probability: float = 0.1
    structural_rj_multi_component_max_group_size: int = 4
    structural_rj_split_probability: float = 0.5
    structural_rj_merge_probability: float = 0.5
    structural_rj_split_global_position_probability: float = 0.1
    structural_rj_merge_uniform_pair_probability: float = 0.1
    structural_rj_merge_distance_sigma_m: float = 0.5
    structural_rj_merge_response_sigma: float = 0.05
    structural_cardinality_prior_policy: str = (
        TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY
    )
    structural_cardinality_prior_probs: tuple[float, ...] | None = None
    structural_cardinality_prior_mean: float = 2.0
    structural_cardinality_tail_ratio: float = 0.05
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 256
    min_delta_beta: float = 1e-10
    # Continuous PF priors (Sec. 3.3.2)
    position_max: tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: tuple[int, int] = (
        0,
        DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    )
    strength_prior_min_cps_1m: float = 1.0
    strength_prior_max_cps_1m: float = 2_000_000.0
    strength_prior_family: str = "bounded_uniform"
    strength_prior_gamma_shape: float = 2.0
    strength_prior_gamma_scale_cps_1m: float = 425_000.0
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
            (
                "structural_rj_multi_component_max_group_size",
                self.structural_rj_multi_component_max_group_size,
                3,
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
        if self.hard_max_sources is None:
            self.hard_max_sources = int(self.max_sources)
        _strict_config_integer(
            self.hard_max_sources,
            name="hard_max_sources",
            minimum=int(self.max_sources),
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
            "structural_rj_block_independence_probability",
            "structural_rj_multi_component_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
            "structural_rj_split_global_position_probability",
            "structural_rj_merge_uniform_pair_probability",
            "structural_rj_merge_distance_sigma_m",
            "structural_rj_merge_response_sigma",
            "structural_cardinality_prior_mean",
            "structural_cardinality_tail_ratio",
            "target_ess_ratio",
            "min_delta_beta",
            "strength_prior_min_cps_1m",
            "strength_prior_max_cps_1m",
            "strength_prior_gamma_shape",
            "strength_prior_gamma_scale_cps_1m",
        ):
            _strict_config_number(getattr(self, name), name=name)
        if self.structural_cardinality_prior_probs is not None:
            for index, value in enumerate(self.structural_cardinality_prior_probs):
                _strict_config_number(
                    value,
                    name=f"structural_cardinality_prior_probs[{index}]",
                )
        position_max = np.asarray(self.position_max, dtype=object).reshape(-1)
        if position_max.shape != (3,):
            raise ValueError("position_max must contain three values.")
        for index, value in enumerate(position_max):
            if (
                _strict_config_number(
                    value,
                    name=f"position_max[{index}]",
                )
                <= 0.0
            ):
                raise ValueError("position_max values must be positive.")
        if not isinstance(self.gpu_device, str) or not self.gpu_device.strip():
            raise TypeError("gpu_device must be a nonempty string.")
        if not isinstance(self.gpu_dtype, str):
            raise TypeError("gpu_dtype must be a string.")
        self.num_particles = int(self.num_particles)
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive.")
        if str(self.gpu_dtype).strip().lower() != "float64":
            raise ValueError("Pure PF production kernels require gpu_dtype='float64'.")
        self.gpu_dtype = "float64"
        self.variable_cardinality = bool(self.variable_cardinality)
        self.structural_cardinality_tail_ratio = float(
            self.structural_cardinality_tail_ratio
        )
        if (
            self.structural_cardinality_prior_policy
            == POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY
            and not 0.0 < self.structural_cardinality_tail_ratio < 1.0
        ):
            raise ValueError("structural_cardinality_tail_ratio must lie in (0, 1).")
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
            not np.isfinite(self.structural_rj_position_proposal_prior_weight)
            or self.structural_rj_position_proposal_prior_weight <= 0.0
            or self.structural_rj_position_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_position_proposal_prior_weight must lie in (0, 1]."
            )
        self.structural_rj_strength_proposal_prior_weight = float(
            self.structural_rj_strength_proposal_prior_weight
        )
        if (
            not np.isfinite(self.structural_rj_strength_proposal_prior_weight)
            or self.structural_rj_strength_proposal_prior_weight <= 0.0
            or self.structural_rj_strength_proposal_prior_weight > 1.0
        ):
            raise ValueError(
                "structural_rj_strength_proposal_prior_weight must lie in (0, 1]."
            )
        self.structural_rj_strength_proposal_sigma_fraction = float(
            self.structural_rj_strength_proposal_sigma_fraction
        )
        if (
            not np.isfinite(self.structural_rj_strength_proposal_sigma_fraction)
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
            "structural_rj_block_independence_probability",
            "structural_rj_multi_component_probability",
            "structural_rj_split_probability",
            "structural_rj_merge_probability",
            "structural_rj_split_global_position_probability",
            "structural_rj_merge_uniform_pair_probability",
        ):
            probability = float(getattr(self, probability_name))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_name} must lie in [0, 1].")
            setattr(self, probability_name, probability)
        for full_support_probability_name in (
            "structural_rj_split_global_position_probability",
            "structural_rj_merge_uniform_pair_probability",
        ):
            if getattr(self, full_support_probability_name) <= 0.0:
                raise ValueError(f"{full_support_probability_name} must lie in (0, 1].")
        self.structural_rj_merge_distance_sigma_m = float(
            self.structural_rj_merge_distance_sigma_m
        )
        if (
            not np.isfinite(self.structural_rj_merge_distance_sigma_m)
            or self.structural_rj_merge_distance_sigma_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_merge_distance_sigma_m must be finite and positive."
            )
        self.structural_rj_merge_response_sigma = float(
            self.structural_rj_merge_response_sigma
        )
        if (
            not np.isfinite(self.structural_rj_merge_response_sigma)
            or self.structural_rj_merge_response_sigma <= 0.0
        ):
            raise ValueError(
                "structural_rj_merge_response_sigma must be finite and positive."
            )
        self.structural_rj_multi_component_max_group_size = int(
            self.structural_rj_multi_component_max_group_size
        )
        if self.structural_rj_multi_component_max_group_size < 3:
            raise ValueError(
                "structural_rj_multi_component_max_group_size must be at least 3."
            )
        if self.max_sources is None or int(self.max_sources) < 1:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        self.max_sources = int(self.max_sources)
        strength_prior = StrengthPrior(
            minimum=self.strength_prior_min_cps_1m,
            maximum=self.strength_prior_max_cps_1m,
            family=self.strength_prior_family,
            gamma_shape=self.strength_prior_gamma_shape,
            gamma_scale=self.strength_prior_gamma_scale_cps_1m,
        )
        if strength_prior.minimum <= 0.0:
            raise ValueError(
                "Pure PF source states require a strictly positive strength "
                "prior minimum."
            )
        self.strength_prior_min_cps_1m = strength_prior.minimum
        self.strength_prior_max_cps_1m = strength_prior.maximum
        self.strength_prior_family = strength_prior.family
        self.strength_prior_gamma_shape = strength_prior.gamma_shape
        self.strength_prior_gamma_scale_cps_1m = strength_prior.gamma_scale
        self.structural_cardinality_prior_policy = validate_cardinality_prior_policy(
            self.structural_cardinality_prior_policy,
            has_explicit_probabilities=(
                self.structural_cardinality_prior_probs is not None
            ),
        )
        if self.structural_cardinality_prior_probs is not None:
            cardinality_prior = np.asarray(
                self.structural_cardinality_prior_probs,
                dtype=float,
            ).reshape(-1)
            if (
                cardinality_prior.size != int(self.hard_max_sources) + 1
                or np.any(~np.isfinite(cardinality_prior))
                or np.any(cardinality_prior <= 0.0)
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain "
                    "hard_max_sources + 1 finite positive values."
                )
            cardinality_prior /= math.fsum(float(value) for value in cardinality_prior)
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
        if not np.isfinite(self.min_delta_beta) or not 0.0 < self.min_delta_beta <= 1.0:
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
                raise ValueError(f"Joint row identity {name} must lie in uint64 range.")
        parent = self.parent_row_sha256
        if int(self.generation) == 0:
            if parent is not None:
                raise ValueError("Initial joint row identities cannot have a parent.")
        elif parent is None:
            raise ValueError("Resampled joint row identities require a parent digest.")
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


class IsotopeParticleFilter(
    ParticleSurfaceMixin,
    ParticleTemperingMixin,
    StructuralRJProposalMixin,
    StructuralRJTargetMixin,
    StructuralRJSweepMixin,
    StructuralRJBasicMoveMixin,
    StructuralRJMultiComponentMixin,
    StructuralRJBlockIndependenceMixin,
    StructuralRJSplitMergeMixin,
):
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
        additive_scatter_response: (AdditiveNoncollidedTransportResponse | None) = None,
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
        self._structural_rj_strength_proposal: ContinuousStrengthProposal | None = None
        self._last_structural_rj_strength_proposal: (
            ContinuousStrengthProposal | None
        ) = None
        self.last_structural_rj_proposal_snapshot_sha256: str | None = None
        self._structural_rj_move_counts: dict[str, int | float] = {}
        self._structural_rj_tempering_start_row: int | None = None
        self._structural_rj_current_target_log_likelihood: (
            NDArray[np.float64] | None
        ) = None
        self.last_structural_target_log_likelihood: NDArray[np.float64] | None = None
        self._joint_target_evaluator: Callable[..., NDArray[np.float64]] | None = None
        self._joint_strength_grid_target_evaluator: (
            Callable[..., NDArray[np.float64]] | None
        ) = None
        self._structural_rj_current_block_strength_centers: (
            NDArray[np.float64] | None
        ) = None
        self._structural_rj_current_block_strength_cardinalities: (
            NDArray[np.int64] | None
        ) = None
        self._structural_rj_device_state: dict[str, "torch.Tensor"] | None = None
        self.last_structural_device_diagnostics: dict[str, int | str] = {}
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
        self.continuous_particles: list[IsotopeParticle] = []
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
        self.last_structural_rejection_diagnostics: dict[str, object] = {}
        self._structural_mh_component_samples: dict[
            str,
            list[dict[str, NDArray[np.float64] | NDArray[np.bool_]]],
        ] = {}
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
            family=str(self.config.strength_prior_family),
            gamma_shape=float(self.config.strength_prior_gamma_shape),
            gamma_scale=float(self.config.strength_prior_gamma_scale_cps_1m),
        )

    def _build_structural_cardinality_prior(self) -> NDArray[np.float64]:
        """Return the normalized prior mass for cardinalities zero through max."""
        max_sources = self.config.hard_max_sources
        if max_sources is None:
            return np.zeros(0, dtype=float)
        count = max(0, int(max_sources)) + 1
        configured = self.config.structural_cardinality_prior_probs
        if configured is None:
            if (
                self.config.structural_cardinality_prior_policy
                == POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY
            ):
                return poisson_geometric_tail_cardinality_probabilities(
                    int(self.config.max_sources),
                    int(max_sources),
                    float(self.config.structural_cardinality_prior_mean),
                    float(self.config.structural_cardinality_tail_ratio),
                )
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




    def apply_structural_moves(
        self,
        evidence_data: StructuralGeometryBatch | None,
        *,
        target_beta: float = 1.0,
        tempering_start_row: int | None = None,
        current_target_log_likelihood: NDArray[np.float64] | None = None,
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
            current_target_log_likelihood=current_target_log_likelihood,
        )

    def estimate(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
            max_cardinality=self.config.hard_max_sources,
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
            [mode.strength_representative_cps_1m for mode in point_estimate.modes],
            dtype=float,
        )
        return positions, strengths

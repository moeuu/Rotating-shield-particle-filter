"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import MeasurementGeometry, ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.surface_atlas import ContinuousSurfaceAtlas
from pf.diagnostics import build_source_event_record, reset_step_diagnostics
from pf.estimator_config import RotatingShieldPFConfig
from pf.particle_filter_rj_basic import StructuralRJBasicMoveMixin
from pf.particle_filter_rj_block import StructuralRJBlockIndependenceMixin
from pf.particle_filter_rj_multi import StructuralRJMultiComponentMixin
from pf.particle_filter_rj_proposal import StructuralRJProposalMixin
from pf.particle_filter_rj_runtime import StructuralRJSweepMixin
from pf.particle_filter_rj_split_merge import StructuralRJSplitMergeMixin
from pf.particle_filter_rj_target import StructuralRJTargetMixin
from pf.particle_filter_rj_torch_basic import StructuralRJTorchBasicMoveMixin
from pf.particle_filter_rj_torch_block import (
    StructuralRJTorchBlockIndependenceMixin,
)
from pf.particle_filter_rj_torch_multi import (
    StructuralRJTorchMultiComponentMixin,
)
from pf.particle_filter_rj_torch_split_merge import (
    StructuralRJTorchSplitMergeMixin,
)
from pf.particle_filter_surface import ParticleSurfaceMixin
from pf.particle_filter_tempering import ParticleTemperingMixin
from pf.posterior import posterior_point_estimate_from_states
from pf.particle_types import StructuralGeometryBatch
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
    SplitMergeMoveProbabilities,
    truncated_poisson_cardinality_probabilities,
    poisson_geometric_tail_cardinality_probabilities,
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
        """Return the canonical checkpoint-and-restore representation."""
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
    StructuralRJTorchBasicMoveMixin,
    StructuralRJTorchBlockIndependenceMixin,
    StructuralRJTorchMultiComponentMixin,
    StructuralRJTorchSplitMergeMixin,
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
        config: RotatingShieldPFConfig | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        detector_impact_parameter_edges_fraction: object | None = None,
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: dict[str, object] | None = None,
        strict_catalog_line_contract: bool = False,
        dry_air_total_attenuation_contract_id: str | None = None,
        dry_air_total_attenuation_contract_sha256: str | None = None,
        additive_scatter_response: (AdditiveNoncollidedTransportResponse | None) = None,
        random_seed: int = 0,
    ) -> None:
        """Initialize particle state, priors, and continuous measurement kernels."""
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or RotatingShieldPFConfig()
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
        if detector_impact_parameter_edges_fraction is None:
            self.detector_impact_parameter_edges_fraction = None
        else:
            impact_edges = np.asarray(
                detector_impact_parameter_edges_fraction,
                dtype=np.float64,
            )
            if (
                impact_edges.ndim != 1
                or impact_edges.size < 2
                or np.any(~np.isfinite(impact_edges))
                or impact_edges[0] != 0.0
                or impact_edges[-1] != 1.0
                or np.any(np.diff(impact_edges) <= 0.0)
            ):
                raise ValueError(
                    "detector_impact_parameter_edges_fraction must be a "
                    "strict finite partition from zero to one."
                )
            self.detector_impact_parameter_edges_fraction = (
                np.ascontiguousarray(impact_edges)
            )
        self.source_extent_radius_m = max(float(source_extent_radius_m), 0.0)
        self.source_extent_samples = max(int(source_extent_samples), 1)
        self.line_mu_by_isotope = line_mu_by_isotope
        if not isinstance(strict_catalog_line_contract, bool):
            raise TypeError("strict_catalog_line_contract must be a boolean.")
        self.strict_catalog_line_contract = strict_catalog_line_contract
        self.dry_air_total_attenuation_contract_id = (
            dry_air_total_attenuation_contract_id
        )
        self.dry_air_total_attenuation_contract_sha256 = (
            dry_air_total_attenuation_contract_sha256
        )
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
        self._structural_rj_current_target_log_likelihood_device: (
            "torch.Tensor" | None
        ) = None
        self._structural_rj_current_station_log_likelihood_device: (
            "torch.Tensor" | None
        ) = None
        self._structural_rj_torch_generator: "torch.Generator" | None = None
        self._structural_rj_device_constants: dict[str, "torch.Tensor"] = {}
        self.last_structural_target_log_likelihood: NDArray[np.float64] | None = None
        self.last_structural_target_log_likelihood_device: (
            "torch.Tensor" | None
        ) = None
        self.last_structural_station_log_likelihood_device: (
            "torch.Tensor" | None
        ) = None
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
        self._structural_rj_device_state_authoritative = False
        self._structural_rj_device_state_dirty = False
        self.last_structural_device_diagnostics: dict[str, object] = {}
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
        self.last_station_unique_ancestor_count: int | None = None
        self.last_cumulative_unique_ancestor_count: int | None = None
        self.last_source_event_diagnostics: list[dict[str, object]] = []
        self.last_structural_timing_s: dict[str, float] = {}
        self.last_structural_transition_weight_mass: dict[str, float] = {}
        self.last_structural_full_support_accepted_mask: NDArray[np.bool_] = (
            np.zeros(0, dtype=np.bool_)
        )
        self.last_structural_rejection_diagnostics: dict[str, object] = {}
        self._structural_mh_component_samples: dict[
            str,
            list[dict[str, NDArray[np.float64] | NDArray[np.bool_]]],
        ] = {}
        self.last_runtime_likelihood_route = "joint_full_spectrum_generative"
        self._resample_count_in_observation = 0
        self._init_continuous_particles()
        self.last_structural_full_support_accepted_mask = np.zeros(
            len(self.continuous_particles),
            dtype=np.bool_,
        )

    def _variable_cardinality_enabled(self) -> bool:
        """Return whether exact birth/death dimension changes are active."""
        return bool(self.config.variable_cardinality)

    def _build_strength_prior(self) -> StrengthPrior:
        """Build the normalized strength prior shared by initialization and moves."""
        return self.config.build_strength_prior()

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
        current_target_log_likelihood: object | None = None,
        current_station_log_likelihood: object | None = None,
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
            current_station_log_likelihood=current_station_log_likelihood,
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

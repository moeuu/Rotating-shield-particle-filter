"""Module-boundary tests for extracted implementation ownership."""

from __future__ import annotations

import planning.dss_pp as dss_pp
import pf.particle_filter as particle_filter

from evaluation.source_normalization import Source as ExtractedSource
from evaluation_metrics import Source as PublicSource
from pf.estimator import (
    JointPlanningParticles as PublicJointPlanningParticles,
)
from pf.estimator import (
    JointStationObservation as PublicJointStationObservation,
)
from pf.estimator import (
    MeasurementRecord as PublicMeasurementRecord,
)
from pf.estimator_types import (
    JointPlanningParticles,
    JointStationObservation,
    MeasurementRecord,
)
from pf.particle_filter import (
    IsotopeParticleFilter,
    StructuralGeometryBatch as PublicStructuralGeometryBatch,
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
)
from pf.particle_types import StructuralGeometryBatch
from planning.dss_pp import (
    ShieldProgram as PublicShieldProgram,
)
from planning.dss_pp import __all__ as dss_public_exports
from planning.program_types import ShieldProgram
from visualization.frame import PFFrame
from visualization.realtime_viz import PFFrame as PublicPFFrame


def test_extracted_types_preserve_public_class_identity() -> None:
    """Existing import paths must resolve to the exact extracted classes."""
    assert PublicSource is ExtractedSource
    assert PublicPFFrame is PFFrame
    assert PublicMeasurementRecord is MeasurementRecord
    assert PublicJointStationObservation is JointStationObservation
    assert PublicJointPlanningParticles is JointPlanningParticles
    assert PublicStructuralGeometryBatch is StructuralGeometryBatch
    assert PublicShieldProgram is ShieldProgram


def test_dss_wildcard_exports_remain_public() -> None:
    """DSS wildcard imports must not expose private implementation helpers."""
    assert dss_public_exports
    assert all(not name.startswith("_") for name in dss_public_exports)
    assert not hasattr(dss_pp, "build_shield_program_library")
    assert not hasattr(dss_pp, "augment_candidate_stations")


def test_particle_filter_does_not_reexport_owned_internals() -> None:
    """Extracted helpers and types must be imported from their owner modules."""
    for name in (
        "_extended_log_target_ratio",
        "TemperingIncrementRequiresRejuvenation",
        "TorchLineTransportComponents",
    ):
        assert not hasattr(particle_filter, name)


def test_particle_filter_algorithms_are_inherited_without_wrappers() -> None:
    """The particle-filter facade should inherit each extracted algorithm."""
    assert (
        IsotopeParticleFilter.validate_continuous_surface_states
        is ParticleSurfaceMixin.validate_continuous_surface_states
    )
    assert (
        IsotopeParticleFilter._select_delta_beta
        is ParticleTemperingMixin._select_delta_beta
    )
    assert (
        IsotopeParticleFilter._build_continuous_rj_position_proposal
        is StructuralRJProposalMixin._build_continuous_rj_position_proposal
    )
    assert (
        IsotopeParticleFilter._continuous_rj_group_log_likelihood
        is StructuralRJTargetMixin._continuous_rj_group_log_likelihood
    )
    assert (
        IsotopeParticleFilter._apply_continuous_rj_birth_death
        is StructuralRJBasicMoveMixin._apply_continuous_rj_birth_death
    )
    assert (
        IsotopeParticleFilter._apply_continuous_rj_multi_component
        is StructuralRJMultiComponentMixin._apply_continuous_rj_multi_component
    )
    assert (
        IsotopeParticleFilter._apply_continuous_rj_block_independence
        is StructuralRJBlockIndependenceMixin._apply_continuous_rj_block_independence
    )
    assert (
        IsotopeParticleFilter._apply_continuous_rj_split_merge
        is StructuralRJSplitMergeMixin._apply_continuous_rj_split_merge
    )
    assert (
        IsotopeParticleFilter._apply_exact_structural_rj_moves
        is StructuralRJSweepMixin._apply_exact_structural_rj_moves
    )

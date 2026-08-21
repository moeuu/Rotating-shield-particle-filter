"""Compatibility tests for behavior-preserving giant-module extractions."""

from __future__ import annotations

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
    StructuralGeometryBatch as PublicStructuralGeometryBatch,
)
from pf.particle_filter import (
    TorchLineTransportComponents as PublicTorchLineTransportComponents,
)
from pf.particle_types import StructuralGeometryBatch, TorchLineTransportComponents
from planning.dss_pp import (
    ShieldProgram as PublicShieldProgram,
)
from planning.dss_pp import (
    build_shield_program_library as public_build_shield_program_library,
)
from planning.shield_programs import ShieldProgram, build_shield_program_library
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
    assert PublicTorchLineTransportComponents is TorchLineTransportComponents
    assert PublicShieldProgram is ShieldProgram


def test_shield_program_builder_preserves_public_function_identity() -> None:
    """DSS-PP should directly re-export the extracted batched builder."""
    assert public_build_shield_program_library is build_shield_program_library

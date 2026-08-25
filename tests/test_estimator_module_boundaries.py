"""Tests for estimator algorithm-unit extraction boundaries."""

from __future__ import annotations

import pf.estimator as estimator_module
from pf.estimator import (
    RotatingShieldPFConfig as PublicRotatingShieldPFConfig,
)
from pf.estimator import (
    RotatingShieldPFEstimator,
)
from pf.estimator import (
    SurfaceAtlasQuadrature as PublicSurfaceAtlasQuadrature,
)
from pf.estimator import (
    build_complete_surface_atlas_quadrature as public_build_surface_quadrature,
)
from pf.estimator import (
    _stratified_categorical_draws as public_stratified_categorical_draws,
)
from pf.estimator import (
    _stratified_joint_cardinality_draws as public_joint_cardinality_draws,
)
from pf.estimator_config import RotatingShieldPFConfig
from pf.estimator_likelihood import JointLikelihoodMixin
from pf.estimator_rejuvenation import JointRejuvenationMixin
from pf.estimator_reporting import EstimatorReportingMixin
from pf.estimator_sampling import (
    _stratified_categorical_draws,
    _stratified_joint_cardinality_draws,
)
from pf.estimator_structural import EstimatorStructuralProposalMixin
from pf.estimator_surface import SurfaceAtlasQuadrature
from pf.estimator_surface import build_complete_surface_atlas_quadrature
from pf.pure_estimator import PurePFEstimator


def test_estimator_module_exposes_only_owned_public_algorithm_types() -> None:
    """The estimator module must not facade unrelated algorithm utilities."""
    assert PublicRotatingShieldPFConfig is RotatingShieldPFConfig
    assert PublicSurfaceAtlasQuadrature is SurfaceAtlasQuadrature
    assert public_build_surface_quadrature is build_complete_surface_atlas_quadrature
    assert public_stratified_categorical_draws is _stratified_categorical_draws
    assert public_joint_cardinality_draws is _stratified_joint_cardinality_draws
    assert not hasattr(estimator_module, "posterior_point_estimate_from_states")
    assert not hasattr(estimator_module, "systematic_resample")


def test_estimator_inherits_each_algorithm_unit_directly() -> None:
    """The estimator must expose the exact extracted implementations."""
    assert (
        RotatingShieldPFEstimator._joint_station_from_spectrum_records
        is JointLikelihoodMixin._joint_station_from_spectrum_records
    )
    assert (
        RotatingShieldPFEstimator._joint_structural_target_evaluator
        is EstimatorStructuralProposalMixin._joint_structural_target_evaluator
    )
    assert (
        RotatingShieldPFEstimator._joint_tempered_station_update
        is JointRejuvenationMixin._joint_tempered_station_update
    )
    assert (
        RotatingShieldPFEstimator.posterior_convergence_diagnostics
        is EstimatorReportingMixin.posterior_convergence_diagnostics
    )


def test_pure_estimator_inherits_the_canonical_estimates_projection() -> None:
    """Pure-PF reporting must not duplicate the canonical projection method."""
    assert PurePFEstimator.estimates is EstimatorReportingMixin.estimates

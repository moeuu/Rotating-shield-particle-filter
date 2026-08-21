"""Tests for joint DSS planning and measurement geometry."""

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

import planning.candidate_generation as candidate_generation
import planning.dss_pp as dss_pp
from measurement.kernels import ShieldParams
from measurement.surface_charts import SurfaceChartGeometry
from pf.estimator import (
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
    build_complete_surface_atlas_quadrature,
)
from measurement.surface_atlas import ContinuousSurfaceAtlas
from pf.state import IsotopeState
from planning.dss_pp import (
    DSSPPConfig,
    _continuous_kernel_for_estimator,
    build_shield_program_library,
    estimate_lambda_cost,
    select_dss_pp_next_station,
)
from planning.candidate_generation import (
    expand_candidate_height_actions,
    generate_candidate_poses,
    resolve_detector_height_actions,
)
from pf.particle_filter import IsotopeParticle
from measurement.shielding import generate_octant_orientations
from pure_pf_test_support import approved_full_spectrum_model


def _state_on_filter(
    particle_filter: object,
    positions_xyz: np.ndarray,
    strengths: np.ndarray,
) -> IsotopeState:
    """Build a continuous-surface state from physical test positions."""
    positions = np.asarray(positions_xyz, dtype=float).reshape(-1, 3)
    strength_values = np.asarray(strengths, dtype=float).reshape(-1)
    chart_ids, surface_uv = particle_filter.structural_surface_chart_coordinates(  # type: ignore[attr-defined]
        positions
    )
    return IsotopeState(
        num_sources=int(strength_values.size),
        strengths=strength_values,
        surface_chart_ids=chart_ids,
        surface_uv=surface_uv,
    )


def test_runtime_candidate_values_remain_aligned_after_filtering() -> None:
    """Planner filtering must preserve runtime-owned candidate motion costs."""
    original = np.asarray(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
        dtype=float,
    )
    retained = original[[2, 0]]

    aligned = dss_pp._align_candidate_values(
        original,
        np.asarray([0.0, 3.0, 8.0]),
        retained,
    )

    assert np.array_equal(aligned, np.asarray([8.0, 0.0]))


def _encoded_surface_state(
    positions_xyz: np.ndarray,
    strengths: np.ndarray,
) -> IsotopeState:
    """Encode simple XYZ fixtures into chart IDs and unit-square coordinates."""
    positions = np.asarray(positions_xyz, dtype=float).reshape(-1, 3)
    if (
        np.any(positions[:, 0] < 0.0)
        or np.any(positions[:, 0] != np.floor(positions[:, 0]))
        or np.any(positions[:, 1:] < 0.0)
        or np.any(positions[:, 1:] > 1.0)
    ):
        raise ValueError("Encoded surface fixtures require integer x and unit yz.")
    strength_values = np.asarray(strengths, dtype=float).reshape(-1)
    return IsotopeState(
        num_sources=int(strength_values.size),
        strengths=strength_values,
        surface_chart_ids=positions[:, 0].astype(np.int64),
        surface_uv=positions[:, 1:],
    )


def _decode_encoded_surface_state(state: IsotopeState) -> np.ndarray:
    """Decode the unit-test chart encoding into XYZ positions."""
    return np.column_stack(
        (
            np.asarray(state.surface_chart_ids, dtype=float),
            np.asarray(state.surface_uv, dtype=float),
        )
    )


def _build_simple_estimator(
    *,
    canonical_octants: bool = False,
) -> RotatingShieldPFEstimator:
    """Build a minimal estimator with deterministic particles."""
    surface_diagnostic_points = np.array([[1.0, 0.0, 0.0]], dtype=float)
    normals = (
        np.asarray(generate_octant_orientations(), dtype=float)
        if canonical_octants
        else np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=float,
        )
    )
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        variable_cardinality=False,
        init_num_sources=(1, 1),
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        surface_diagnostic_points=surface_diagnostic_points,
        shield_normals=normals,
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=config,
        shield_params=ShieldParams(),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.array([0.0, 0.0, 0.0]))
    estimator._ensure_kernel_cache()
    particle_filter = estimator.filters["Cs-137"]
    particle_filter.continuous_particles = [
        IsotopeParticle(
            state=_state_on_filter(
                particle_filter,
                np.array([[0.0, 0.0, 0.0]]),
                np.array([10.0]),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=(
                particle_filter.continuous_particles[0].joint_row_identity
            ),
        ),
        IsotopeParticle(
            state=_state_on_filter(
                particle_filter,
                np.array([[0.0, 0.0, 0.0]]),
                np.array([1.0]),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=(
                particle_filter.continuous_particles[1].joint_row_identity
            ),
        ),
    ]
    return estimator


def _build_full_spectrum_planning_estimator() -> RotatingShieldPFEstimator:
    """Build a production-approved tiny estimator for DSS spectrum tests."""
    isotope = "Cs-137"
    model = approved_full_spectrum_model()
    line_mu = tuple(
        {
            "energy_keV": float(line["energy_keV"]),
            "weight": float(line["branching_weight"]),
            "fe": float(line["mu_fe_cm_inv"]),
            "pb": float(line["mu_pb_cm_inv"]),
        }
        for line in model.line_identity
        if str(line["isotope"]) == isotope
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=(isotope,),
        surface_diagnostic_points=np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        shield_normals=np.asarray([[0.0, 0.0, 1.0]], dtype=float),
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            planning_eig_samples=4,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        line_mu_by_isotope={isotope: line_mu},
        full_spectrum_generative_model=model,
        random_seed=9,
    )
    estimator.add_measurement_pose(np.array([1.0, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    particles = estimator.filters[isotope].continuous_particles
    particle_filter = estimator.filters[isotope]
    for index, particle in enumerate(particles):
        particle.state = _state_on_filter(
            particle_filter,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([float(10 + 10 * index)]),
        )
        particle.log_weight = float(np.log(0.5))
    return estimator


def test_dss_pp_uses_estimator_shared_continuous_kernel() -> None:
    """DSS-PP should not rebuild a divergent PF physics kernel by hand."""
    estimator = _build_simple_estimator()
    calls: list[int | None] = []
    sentinel = object()

    def fake_continuous_kernel(*, detector_aperture_samples=None, use_gpu=None):
        """Return a sentinel while recording kernel factory arguments."""
        del use_gpu
        calls.append(detector_aperture_samples)
        return sentinel

    estimator.continuous_kernel = fake_continuous_kernel  # type: ignore[method-assign]

    result = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=7,
    )

    assert result is sentinel
    assert calls == [7]


def test_dss_pp_default_aperture_samples_matches_pf_standard() -> None:
    """DSS-PP defaults should not fall back to the obsolete one-ray kernel."""
    assert DSSPPConfig().detector_aperture_samples == 121


def test_dss_full_spectrum_components_and_eig_share_pf_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSS must sample and score the same source-resolved spectrum model."""
    estimator = _build_full_spectrum_planning_estimator()
    joint = estimator.planning_joint_particles()
    program = dss_pp.ShieldProgram(
        name="one_view",
        pair_ids=(0,),
        kind="test",
    )
    components = dss_pp._full_spectrum_joint_program_components(
        estimator,
        np.array([[1.0, 0.0, 0.0]], dtype=float),
        [program],
        joint,
        live_time_s=3.0,
        detector_aperture_samples=121,
    )
    model = estimator.full_spectrum_generative_model
    mean = model.predict_mean_numpy(
        components.total_pnvsl,
        components.uncollided_pnvsl,
        components.features_pnvslf,
        components.live_times_v,
    )
    source_rate = np.sum(components.total_pnvsl, axis=(-2, -1))

    assert components.total_pnvsl.shape[:3] == (1, 2, 1)
    assert np.all(components.uncollided_pnvsl <= components.total_pnvsl + 1.0e-12)
    node_rates = source_rate[..., None] * model._rate_scale_nodes_j + float(
        model.background_rate_cps
    )
    expected_totals = np.sum(
        (3.0 * node_rates / (1.0 + node_rates * float(model.dead_time_tau_s)))
        * model._rate_scale_weights_j,
        axis=-1,
    )
    np.testing.assert_allclose(
        np.sum(mean, axis=-1),
        expected_totals,
        rtol=1.0e-10,
        atol=1.0e-8,
    )
    cross_calls = 0
    exact_cross_likelihood = model.cross_log_likelihood_numpy

    def _record_exact_cross_likelihood(
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        """Record use of the exact spectrum law while preserving its result."""
        nonlocal cross_calls
        cross_calls += 1
        return exact_cross_likelihood(*args, **kwargs)

    def _reject_mean_surrogate(*_args: object, **_kwargs: object) -> np.ndarray:
        """Reject accidental replacement of the exact law by mean scoring."""
        raise AssertionError("DSS EIG must not score a predictive-mean surrogate.")

    monkeypatch.setattr(
        model,
        "cross_log_likelihood_numpy",
        _record_exact_cross_likelihood,
    )
    monkeypatch.setattr(model, "predict_mean_numpy", _reject_mean_surrogate)
    gains = dss_pp._full_spectrum_information_gain(
        estimator,
        components,
        joint.weights_n,
        sample_count=4,
        rng=np.random.default_rng(17),
        use_gpu=False,
        gpu_device="cpu",
    )
    assert gains.shape == (1,)
    assert np.all(np.isfinite(gains))
    assert np.all(gains >= 0.0)
    assert cross_calls == 1
    pytest.importorskip("torch")
    torch_gains = dss_pp._full_spectrum_information_gain(
        estimator,
        components,
        joint.weights_n,
        sample_count=4,
        rng=np.random.default_rng(17),
        use_gpu=True,
        gpu_device="cpu",
    )
    np.testing.assert_allclose(
        torch_gains,
        gains,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_dss_transport_deduplicates_identical_pose_pair_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated program views must share one deterministic transport call."""
    estimator = _build_full_spectrum_planning_estimator()
    joint = estimator.planning_joint_particles()
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=121,
    )
    evaluate_components = kernel.line_transport_components_all_pairs_for_detectors
    evaluated_detector_counts: list[int] = []

    def _record_components(**kwargs: object) -> object:
        """Record unique detectors and preserve the physical result."""
        evaluated_detector_counts.append(
            int(np.asarray(kwargs["detector_positions"]).shape[0])
        )
        return evaluate_components(**kwargs)

    monkeypatch.setattr(
        kernel,
        "line_transport_components_all_pairs_for_detectors",
        _record_components,
    )
    monkeypatch.setattr(
        dss_pp,
        "_continuous_kernel_for_estimator",
        lambda *_args, **_kwargs: kernel,
    )
    repeated_program = dss_pp.ShieldProgram(
        name="repeated",
        pair_ids=(0,),
        kind="test",
    )
    components = dss_pp._full_spectrum_joint_program_components(
        estimator,
        np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        [repeated_program, repeated_program],
        joint,
        live_time_s=3.0,
        detector_aperture_samples=121,
    )

    assert evaluated_detector_counts == [1]
    np.testing.assert_array_equal(
        components.total_pnvsl[0],
        components.total_pnvsl[1],
    )
    np.testing.assert_array_equal(
        components.uncollided_pnvsl[0],
        components.uncollided_pnvsl[1],
    )
    np.testing.assert_array_equal(
        components.features_pnvslf[0],
        components.features_pnvslf[1],
    )


def test_dss_transport_skips_inactive_padded_source_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padded source slots must not enter the expensive physical kernel."""
    estimator = _build_full_spectrum_planning_estimator()
    original = estimator.planning_joint_particles()
    isotope = "Cs-137"
    particle_count = int(original.weights_n.size)
    padded_positions = np.zeros((particle_count, 2, 3), dtype=np.float64)
    padded_positions[:, :1] = original.positions_nk3_by_isotope[isotope]
    padded_chart_ids = np.zeros((particle_count, 2), dtype=np.int64)
    padded_chart_ids[:, :1] = original.surface_chart_ids_nk_by_isotope[isotope]
    padded_uv = np.zeros((particle_count, 2, 2), dtype=np.float64)
    padded_uv[:, :1] = original.surface_uv_nk2_by_isotope[isotope]
    padded_strengths = np.zeros((particle_count, 2), dtype=np.float64)
    padded_strengths[:, :1] = original.strengths_nk_by_isotope[isotope]
    padded_mask = padded_strengths > 0.0
    padded = dss_pp.JointPlanningParticles(
        isotope_order=original.isotope_order,
        weights_n=original.weights_n,
        positions_nk3_by_isotope={isotope: padded_positions},
        surface_chart_ids_nk_by_isotope={isotope: padded_chart_ids},
        surface_uv_nk2_by_isotope={isotope: padded_uv},
        strengths_nk_by_isotope={isotope: padded_strengths},
        source_mask_nk_by_isotope={isotope: padded_mask},
        original_particle_indices=original.original_particle_indices,
    )
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=121,
    )
    evaluate_components = kernel.line_transport_components_all_pairs_for_detectors
    evaluated_source_counts: list[int] = []

    def _record_components(**kwargs: object) -> object:
        """Record evaluated sources while preserving the physical result."""
        evaluated_source_counts.append(int(np.asarray(kwargs["sources"]).shape[0]))
        return evaluate_components(**kwargs)

    monkeypatch.setattr(
        kernel,
        "line_transport_components_all_pairs_for_detectors",
        _record_components,
    )
    monkeypatch.setattr(
        dss_pp,
        "_continuous_kernel_for_estimator",
        lambda *_args, **_kwargs: kernel,
    )
    program = dss_pp.ShieldProgram(
        name="one_view",
        pair_ids=(0,),
        kind="test",
    )
    components = dss_pp._full_spectrum_joint_program_components(
        estimator,
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        [program],
        padded,
        live_time_s=3.0,
        detector_aperture_samples=121,
    )

    assert evaluated_source_counts == [int(np.count_nonzero(padded_mask))]
    assert np.all(components.total_pnvsl[..., 1, :] == 0.0)
    assert np.all(components.uncollided_pnvsl[..., 1, :] == 0.0)
    assert np.all(components.features_pnvslf[..., 1, :, :] == 0.0)


def test_dss_full_spectrum_proxy_matches_direct_reduced_eig() -> None:
    """Proxy scheduling must preserve the exact joint spectrum calculation."""
    estimator = _build_full_spectrum_planning_estimator()
    joint = estimator.planning_joint_particles()
    detectors = np.asarray(
        [[1.0, 0.0, 0.0], [1.5, 0.25, 0.0]],
        dtype=np.float64,
    )
    program = dss_pp.ShieldProgram(
        name="one_view",
        pair_ids=(0,),
        kind="test",
    )
    config = DSSPPConfig(
        max_programs=1,
        program_length=1,
        forced_program_pair_ids=(0,),
        live_time_s=1.0,
        detector_aperture_samples=121,
        proxy_eig_samples=2,
        exact_eig_coverage_reserve=0,
        exact_eig_program_diversity_reserve=0,
    )
    proxy_diagnostics: dict[str, object] = {}
    proxy_scores = dss_pp._program_information_proxy_for_poses(
        estimator,
        detectors,
        [program],
        config=config,
        joint_particles=joint,
        rng=np.random.default_rng(91),
        eig_call_seed=1234,
        diagnostics=proxy_diagnostics,
    )
    direct_scores = dss_pp._program_information_gains_for_poses(
        estimator,
        detectors,
        [[program], [program]],
        config=config,
        joint_particles=joint,
        rng=np.random.default_rng(123),
        sample_count_override=2,
        eig_call_seed=1234,
        memory_budget_bytes_override=config.proxy_memory_budget_bytes,
    )

    assert proxy_scores.shape == (2, 1)
    np.testing.assert_allclose(
        proxy_scores,
        np.vstack(direct_scores),
        rtol=0.0,
        atol=0.0,
    )
    assert proxy_diagnostics["backend"] == "numpy"


def test_dss_exact_eig_is_action_batch_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical action RNG streams must remove batching-order dependence."""
    estimator = _build_full_spectrum_planning_estimator()
    joint = estimator.planning_joint_particles()
    detectors = np.asarray(
        [[1.0, 0.0, 0.0], [1.25, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    program = dss_pp.ShieldProgram(
        name="one_view",
        pair_ids=(0,),
        kind="test",
    )
    programs_by_pose = [[program] for _ in range(detectors.shape[0])]
    config = DSSPPConfig(
        max_programs=1,
        program_length=1,
        forced_program_pair_ids=(0,),
        live_time_s=1.0,
        exact_eig_coverage_reserve=0,
        exact_eig_program_diversity_reserve=0,
    )

    monkeypatch.setattr(
        dss_pp,
        "_dss_eig_action_batch_size",
        lambda *_args, **_kwargs: 1,
    )
    scalar_batches = dss_pp._program_information_gains_for_poses(
        estimator,
        detectors,
        programs_by_pose,
        config=config,
        rng=np.random.default_rng(991),
        joint_particles=joint,
    )
    monkeypatch.setattr(
        dss_pp,
        "_dss_eig_action_batch_size",
        lambda *_args, **_kwargs: 3,
    )
    one_batch = dss_pp._program_information_gains_for_poses(
        estimator,
        detectors,
        programs_by_pose,
        config=config,
        rng=np.random.default_rng(991),
        joint_particles=joint,
    )

    np.testing.assert_allclose(
        np.concatenate(one_batch),
        np.concatenate(scalar_batches),
        rtol=0.0,
        atol=0.0,
    )


def test_dss_state_chunk_respects_declared_memory_budget() -> None:
    """DSS must shrink its state chunk before rejecting a valid action."""
    estimator = _build_full_spectrum_planning_estimator()
    model = estimator.full_spectrum_generative_model
    state_chunk_size = dss_pp._dss_eig_state_chunk_size(
        model,
        action_count=32,
        particle_count=512,
        sample_count=50,
        source_slot_count=15,
        view_count=8,
        memory_budget_bytes=256 * 1024 * 1024,
    )

    assert state_chunk_size >= 1
    assert state_chunk_size < 512
    working_set_bytes = model.estimate_cross_likelihood_working_set_bytes(
        num_actions=32,
        num_samples=50,
        num_particles=512,
        num_isotopes=15,
        num_views=8,
        action_chunk_size=1,
        state_chunk_size=state_chunk_size,
    )
    assert working_set_bytes <= 128 * 1024 * 1024


def test_dss_likelihood_action_chunk_respects_declared_memory_budget() -> None:
    """DSS must batch actions without exceeding its likelihood workspace."""
    estimator = _build_full_spectrum_planning_estimator()
    model = estimator.full_spectrum_generative_model
    memory_budget_bytes = 256 * 1024 * 1024
    action_chunk_size = dss_pp._dss_eig_likelihood_action_chunk_size(
        model,
        action_count=89,
        particle_count=16,
        sample_count=2,
        source_slot_count=15,
        view_count=8,
        state_chunk_size=16,
        memory_budget_bytes=memory_budget_bytes,
    )

    assert action_chunk_size == 8
    working_set_bytes = model.estimate_cross_likelihood_working_set_bytes(
        num_actions=89,
        num_samples=2,
        num_particles=16,
        num_isotopes=15,
        num_views=8,
        action_chunk_size=action_chunk_size,
        state_chunk_size=16,
    )
    assert working_set_bytes <= memory_budget_bytes // 2


def test_dss_exact_eig_halves_and_retries_after_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recoverable action-batch OOM must preserve the exact EIG result."""
    estimator = _build_full_spectrum_planning_estimator()
    joint = estimator.planning_joint_particles()
    detectors = np.asarray(
        [[1.0, 0.0, 0.0], [1.25, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    program = dss_pp.ShieldProgram(
        name="one_view",
        pair_ids=(0,),
        kind="test",
    )
    programs_by_pose = [[program] for _ in range(detectors.shape[0])]
    config = DSSPPConfig(
        max_programs=1,
        program_length=1,
        forced_program_pair_ids=(0,),
        live_time_s=1.0,
        exact_eig_coverage_reserve=0,
        exact_eig_program_diversity_reserve=0,
    )
    exact_batch_size = dss_pp._dss_eig_action_batch_size
    monkeypatch.setattr(
        dss_pp,
        "_dss_eig_action_batch_size",
        lambda *_args, **_kwargs: 1,
    )
    expected = dss_pp._program_information_gains_for_poses(
        estimator,
        detectors,
        programs_by_pose,
        config=config,
        rng=np.random.default_rng(773),
        joint_particles=joint,
    )

    exact_information_gain = dss_pp._full_spectrum_information_gain
    attempted_action_counts: list[int] = []
    raised = False

    def fail_first_multi_action_batch(
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        """Raise once for a multi-action batch, then run the exact evaluator."""
        nonlocal raised
        components = args[1]
        action_count = int(np.asarray(components.total_pnvsl).shape[0])
        attempted_action_counts.append(action_count)
        if action_count > 1 and not raised:
            raised = True
            raise MemoryError("synthetic DSS action-batch OOM")
        return exact_information_gain(*args, **kwargs)

    def three_action_batch(*args: object, **kwargs: object) -> int:
        """Populate the real memory contract, then force one retry fixture."""
        exact_batch_size(*args, **kwargs)
        return 3

    monkeypatch.setattr(
        dss_pp,
        "_dss_eig_action_batch_size",
        three_action_batch,
    )
    monkeypatch.setattr(
        dss_pp,
        "_full_spectrum_information_gain",
        fail_first_multi_action_batch,
    )
    diagnostics: dict[str, object] = {}
    actual = dss_pp._program_information_gains_for_poses(
        estimator,
        detectors,
        programs_by_pose,
        config=config,
        rng=np.random.default_rng(773),
        joint_particles=joint,
        diagnostics=diagnostics,
    )

    assert attempted_action_counts[0] == 3
    assert all(count == 1 for count in attempted_action_counts[1:])
    assert diagnostics["oom_retry_count"] == 1
    assert diagnostics["cpu_fallback_used"] is False
    assert diagnostics["attempted_action_batch_sizes"] == [3, 1, 1, 1]
    assert diagnostics["successful_action_batch_sizes"] == [1, 1, 1]
    memory_contracts = diagnostics["memory_contracts"]
    assert isinstance(memory_contracts, list)
    assert len(memory_contracts) == 1
    assert memory_contracts[0]["model_working_set_bytes"] > 0
    assert memory_contracts[0]["persistent_per_action_bytes"] > 0
    np.testing.assert_allclose(
        np.concatenate(actual),
        np.concatenate(expected),
        rtol=0.0,
        atol=0.0,
    )


def test_candidate_generation_adds_map_cells_when_random_sampling_is_sparse() -> None:
    """Candidate generation should include deterministic free-cell centers."""

    class SparseRuntimeMap:
        """Expose one runtime-authored free cell through the planner contract."""

        origin = (0.0, 0.0)
        cell_size = 1.0
        traversable_cells = ((8, 8),)

        def is_free_batch(self, points: np.ndarray) -> np.ndarray:
            """Accept only points inside the declared runtime free cell."""
            values = np.asarray(points, dtype=float)
            cells = np.floor(values[:, :2]).astype(np.int64)
            return np.all(cells == np.asarray([8, 8]), axis=1)

    traversable = SparseRuntimeMap()

    candidates = generate_candidate_poses(
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        map_api=traversable,
        n_candidates=4,
        strategy="free_space_sobol",
        min_dist_from_visited=1.0,
        visited_poses_xyz=np.array([[0.5, 0.5, 0.5]], dtype=float),
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 0.5], dtype=float),
        ),
        rng=np.random.default_rng(7),
    )

    assert any(np.allclose(candidate, [8.5, 8.5, 0.5]) for candidate in candidates)


def test_candidate_generation_rejects_missing_or_inverted_bounds() -> None:
    """A hidden default cube or reordered bounds can exclude the true workspace."""
    common = {
        "current_pose_xyz": np.asarray([0.5, 0.5, 0.5], dtype=float),
        "map_api": None,
        "n_candidates": 4,
        "rng": np.random.default_rng(7),
    }
    with pytest.raises(ValueError, match="explicit environment bounds"):
        generate_candidate_poses(**common)
    with pytest.raises(ValueError, match="finite ordered"):
        generate_candidate_poses(
            **common,
            bounds_xyz=(
                np.asarray([2.0, 0.0, 0.0], dtype=float),
                np.asarray([1.0, 1.0, 1.0], dtype=float),
            ),
        )


def test_candidate_generation_rejects_unknown_map_capabilities() -> None:
    """Unknown maps must not silently mark obstacles or paths as free."""

    class UnknownMap:
        """Deliberately omit every free-space map contract."""

    with pytest.raises(TypeError, match="is_free_batch"):
        generate_candidate_poses(
            current_pose_xyz=np.asarray([0.5, 0.5, 0.5], dtype=float),
            map_api=UnknownMap(),
            n_candidates=4,
            bounds_xyz=(
                np.zeros(3, dtype=float),
                np.ones(3, dtype=float),
            ),
            rng=np.random.default_rng(7),
        )


def test_candidate_generation_replenishes_after_batch_reachability_filter() -> None:
    """Reachability loss must trigger native map-center replenishment."""

    class ReachabilityMap:
        """Expose one reachable native cell and reject sampled off-center poses."""

        origin = (0.0, 0.0)
        cell_size = 1.0
        grid_shape = (10, 10)
        traversable_cells = ((8, 8),)

        def __init__(self) -> None:
            """Initialize batched reachability accounting."""
            self.reachability_calls = 0

        def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
            """Return the center of one native map cell."""
            return float(cell[0]) + 0.5, float(cell[1]) + 0.5

        def is_free(self, _point: np.ndarray) -> bool:
            """Reject accidental use of the scalar free-space callback."""
            raise AssertionError("scalar free-space path must not be selected")

        def is_free_batch(self, points: np.ndarray) -> np.ndarray:
            """Mark all sampled endpoints free before reachability filtering."""
            return np.ones(np.asarray(points).shape[0], dtype=bool)

        def is_motion_reachable_batch(
            self,
            _start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Keep only the native center to make replenishment deterministic."""
            self.reachability_calls += 1
            goals = np.asarray(goals_xyz, dtype=float)
            return np.all(
                np.isclose(goals, np.array([8.5, 8.5, 0.5])[None, :]),
                axis=1,
            )

    planning_map = ReachabilityMap()
    candidates = generate_candidate_poses(
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        map_api=planning_map,
        n_candidates=4,
        strategy="free_space_sobol",
        min_dist_from_visited=1.0,
        visited_poses_xyz=np.array([[0.5, 0.5, 0.5]], dtype=float),
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 0.5], dtype=float),
        ),
        rng=np.random.default_rng(7),
        require_motion_reachable=True,
    )

    assert planning_map.reachability_calls >= 2
    np.testing.assert_allclose(candidates, np.array([[8.5, 8.5, 0.5]]))


def test_candidate_filter_prefers_batched_free_space_path() -> None:
    """Standard maps should filter candidate arrays without scalar callbacks."""

    class BatchPlanningMap:
        """Expose both paths while making an accidental scalar call fail."""

        def __init__(self) -> None:
            """Initialize batch-call accounting."""
            self.batch_calls = 0

        def is_free(self, _point: np.ndarray) -> bool:
            """Reject use of the compatibility-only scalar path."""
            raise AssertionError("scalar free-space path must not be selected")

        def is_free_batch(self, points: np.ndarray) -> np.ndarray:
            """Accept candidates whose x coordinate is at least one metre."""
            self.batch_calls += 1
            return np.asarray(points, dtype=float)[:, 0] >= 1.0

    planning_map = BatchPlanningMap()
    candidates = np.asarray(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 1.5]],
        dtype=float,
    )

    filtered = candidate_generation._filter_candidates(
        candidates,
        visited_poses_xyz=None,
        min_dist_from_visited=0.0,
        is_free_batch_fn=(
            candidate_generation._resolve_free_space_batch_checker(planning_map)
        ),
    )

    assert planning_map.batch_calls == 1
    assert np.allclose(filtered, candidates[1:])


def test_candidate_height_expansion_matches_scalar_oracle() -> None:
    """Discrete height actions should be a vectorized Cartesian expansion."""
    candidates = np.array(
        [[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]],
        dtype=float,
    )
    heights = resolve_detector_height_actions(
        [1.5, 0.5, 1.5],
        default_height_m=0.5,
        bounds_z=(0.0, 2.0),
    )

    expanded = expand_candidate_height_actions(candidates, heights)
    expected = np.asarray(
        [
            [candidate[0], candidate[1], height]
            for candidate in candidates
            for height in heights
        ],
        dtype=float,
    )

    assert np.allclose(expanded, expected)


def test_candidate_filter_uses_one_generic_3d_separation() -> None:
    """Vertical and lateral actions must obey the same Euclidean spacing."""
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [1.0, 1.0, 0.5001],
            [1.0, 1.0, 3.5],
            [4.0, 1.0, 0.5],
        ],
        dtype=float,
    )

    filtered = candidate_generation._filter_candidates(
        candidates,
        visited,
        3.0,
        is_free_batch_fn=lambda points: np.ones(
            np.asarray(points).shape[0],
            dtype=bool,
        ),
    )

    assert np.allclose(filtered, candidates[1:])


def test_dss_station_spacing_uses_one_generic_3d_separation() -> None:
    """DSS filtering and revisit penalties must use Euclidean XYZ distance."""
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 3.5],
            [4.0, 1.0, 0.5],
        ],
        dtype=float,
    )

    filtered, removed = dss_pp._filter_station_separation(
        candidates,
        visited,
        min_separation_m=3.0,
    )
    penalties = dss_pp._station_revisit_penalties_batch(
        candidates,
        visited,
        min_separation_m=3.0,
    )

    assert removed == 1
    assert np.allclose(filtered, candidates[1:])
    assert penalties[1] == pytest.approx(0.0)
    assert penalties[0] > 0.0


def test_dss_continuous_augmentation_preserves_base_and_uses_batch_filter() -> None:
    """Continuous augmentation should vary z and use batched free-space checks."""

    class BatchOnlyPlanningMap:
        """Expose only a usable batched free-space runtime path."""

        def __init__(self) -> None:
            """Initialize batch-call accounting."""
            self.batch_calls = 0

        def is_free(self, _point: np.ndarray) -> bool:
            """Reject accidental use of the scalar compatibility path."""
            raise AssertionError("scalar free-space path must not be selected")

        def is_free_batch(self, points: np.ndarray) -> np.ndarray:
            """Accept every in-bounds candidate in one batch."""
            self.batch_calls += 1
            return np.ones(np.asarray(points).shape[0], dtype=bool)

    planning_map = BatchOnlyPlanningMap()
    base = np.array(
        [[0.25, 0.25, 0.4], [0.75, 0.75, 1.4]],
        dtype=float,
    )
    current = np.array([0.5, 0.5, 0.5], dtype=float)
    bounds = (
        np.array([0.0, 0.0, 0.25], dtype=float),
        np.array([2.0, 2.0, 1.75], dtype=float),
    )
    config = DSSPPConfig(max_augmented_candidates=16)

    continuous = dss_pp.augment_candidate_stations(
        base,
        modes_by_isotope={},
        current_pose_xyz=current,
        visited_poses_xyz=None,
        map_api=planning_map,
        bounds_xyz=bounds,
        config=config,
        continuous_height_bounds_m=(0.25, 1.75),
        rng=np.random.default_rng(17),
    )
    legacy = dss_pp.augment_candidate_stations(
        base,
        modes_by_isotope={},
        current_pose_xyz=current,
        visited_poses_xyz=None,
        map_api=planning_map,
        bounds_xyz=bounds,
        config=config,
        rng=np.random.default_rng(17),
    )

    assert planning_map.batch_calls == 2
    assert np.allclose(continuous[: base.shape[0]], base)
    assert np.unique(np.round(continuous[base.shape[0] :, 2], 6)).size > 1
    assert np.all(
        (continuous[base.shape[0] :, 2] >= 0.25)
        & (continuous[base.shape[0] :, 2] <= 1.75)
    )
    assert np.allclose(legacy[: base.shape[0]], base)
    assert np.allclose(legacy[base.shape[0] :, 2], current[2])


def test_dss_augmented_geometry_matches_scalar_test_oracle() -> None:
    """Batched ring and cross-bearing construction must preserve geometry."""
    base = np.array([[9.0, 9.0, 0.5]], dtype=np.float64)
    current = np.array([5.0, 5.0, 0.5], dtype=np.float64)
    visited = np.array(
        [[2.0, 0.0, 0.5], [0.0, 3.0, 1.0]],
        dtype=np.float64,
    )
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            strength_cps_1m=100.0,
            weight=0.7,
            spread_m=0.2,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([4.0, 1.0, 1.0], dtype=np.float64),
            strength_cps_1m=50.0,
            weight=0.3,
            spread_m=0.4,
        ),
    ]
    config = DSSPPConfig(
        ring_radii_m=(1.0, 2.0),
        ring_angles=4,
        max_augmented_candidates=512,
    )

    actual = dss_pp.augment_candidate_stations(
        base,
        modes_by_isotope={"Cs-137": modes},
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        map_api=None,
        bounds_xyz=None,
        config=config,
        rng=np.random.default_rng(31),
    )

    scalar_points = [base[0].copy()]
    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    for mode in modes:
        for radius in config.ring_radii_m:
            for angle in angles:
                scalar_points.append(
                    np.array(
                        [
                            mode.position_xyz[0] + radius * np.cos(angle),
                            mode.position_xyz[1] + radius * np.sin(angle),
                            current[2],
                        ],
                        dtype=np.float64,
                    )
                )
    for mode in modes:
        for pose in visited:
            delta = pose[:2] - mode.position_xyz[:2]
            base_angle = float(np.arctan2(delta[1], delta[0]))
            for offset in (0.5 * np.pi, -0.5 * np.pi, np.pi):
                for radius in config.ring_radii_m:
                    angle = base_angle + offset
                    scalar_points.append(
                        np.array(
                            [
                                mode.position_xyz[0] + radius * np.cos(angle),
                                mode.position_xyz[1] + radius * np.sin(angle),
                                current[2],
                            ],
                            dtype=np.float64,
                        )
                    )
    expected = dss_pp._dedupe_points(np.vstack(scalar_points))

    assert actual.shape == expected.shape
    actual_order = np.lexsort((actual[:, 2], actual[:, 1], actual[:, 0]))
    expected_order = np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))
    assert np.allclose(actual[actual_order], expected[expected_order])


def test_dss_grid_augmentation_uses_batched_cell_centers() -> None:
    """Boundary and coverage candidates must avoid scalar cell callbacks."""

    class BatchCellMap:
        """Expose vectorized planning geometry and reject scalar callbacks."""

        origin = (0.0, 0.0)
        cell_size = 1.0
        grid_shape = (3, 2)
        traversable_cells = ((0, 0), (0, 1), (1, 0), (2, 1))

        def __init__(self) -> None:
            """Initialize callback accounting."""
            self.center_batch_calls = 0

        def cell_center(self, _cell: tuple[int, int]) -> tuple[float, float]:
            """Reject accidental scalar center construction."""
            raise AssertionError("scalar cell-center path must not be selected")

        def cell_centers_batch(self, cells: np.ndarray) -> np.ndarray:
            """Return all requested centers in one numerical batch."""
            self.center_batch_calls += 1
            return np.asarray(cells, dtype=np.float64) + 0.5

        @staticmethod
        def is_free_batch(points: np.ndarray) -> np.ndarray:
            """Accept the fixture's candidate batch."""
            return np.ones(np.asarray(points).shape[0], dtype=bool)

    planning_map = BatchCellMap()
    result = dss_pp.augment_candidate_stations(
        np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
        modes_by_isotope={},
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        visited_poses_xyz=None,
        map_api=planning_map,
        bounds_xyz=None,
        config=DSSPPConfig(max_augmented_candidates=32),
        rng=np.random.default_rng(37),
    )

    assert planning_map.center_batch_calls == 2
    assert result.shape[1] == 3
    assert any(np.allclose(row, [2.5, 1.5, 0.5]) for row in result)


def test_batched_bearing_diversity_matches_scalar_test_oracle() -> None:
    """Candidate-by-mode-by-visit bearings must match the scalar oracle."""
    candidates = np.array(
        [[1.0, 2.0, 0.5], [3.0, -1.0, 1.0], [5.0, 4.0, 2.0]],
        dtype=np.float64,
    )
    visited = np.array(
        [[-2.0, 1.0, 0.5], [4.0, 3.0, 1.5]],
        dtype=np.float64,
    )
    modes_by_isotope = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                strength_cps_1m=100.0,
                weight=0.6,
                spread_m=0.2,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([4.0, 0.5, 1.0], dtype=np.float64),
                strength_cps_1m=80.0,
                weight=0.4,
                spread_m=0.3,
            ),
        ]
    }

    actual = dss_pp._bearing_diversity_gains_batch(
        candidates,
        visited,
        modes_by_isotope,
    )
    expected = np.asarray(
        [
            dss_pp._bearing_diversity_gain(
                candidate,
                visited,
                modes_by_isotope,
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_dss_path_filter_prefers_batch_lengths_over_reachability_flags() -> None:
    """Candidate filtering uses finite batch lengths before legacy flags."""

    class BatchLengthMap:
        """Expose both APIs while making legacy use an immediate failure."""

        def __init__(self) -> None:
            """Initialize the batch-call counter."""
            self.batch_calls = 0

        def motion_path_lengths_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Return one unreachable candidate between two reachable ones."""
            del start_xyz
            self.batch_calls += 1
            assert goals_xyz.shape == (3, 3)
            return np.array([1.0, float("inf"), 2.0], dtype=float)

        def is_motion_reachable_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Fail if the compatibility path is selected before lengths."""
            del start_xyz, goals_xyz
            raise AssertionError("legacy reachability API should not be used")

    planning_map = BatchLengthMap()
    candidates = np.array(
        [
            [1.0, 0.0, 0.5],
            [2.0, 0.0, 0.5],
            [3.0, 0.0, 0.5],
        ],
        dtype=float,
    )

    filtered, removed = dss_pp._filter_path_reachable_stations(
        candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        map_api=planning_map,
    )

    assert planning_map.batch_calls == 1
    assert removed == 1
    np.testing.assert_allclose(filtered, candidates[[0, 2]])


def test_dss_batch_path_length_helper_requires_native_map_motion() -> None:
    """DSS path lengths use native motion or explicit obstacle-free geometry."""

    class NativeBatchMap:
        """Return deterministic native batch lengths for dispatch testing."""

        def __init__(self) -> None:
            """Initialize the native batch-call counter."""
            self.batch_calls = 0

        def motion_path_lengths_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Return direct distances with a deterministic offset."""
            self.batch_calls += 1
            return np.linalg.norm(goals_xyz - start_xyz[None, :], axis=1) + 0.25

    start = np.array([0.0, 0.0, 0.5], dtype=float)
    goals = np.array(
        [[1.0, 0.0, 0.5], [0.0, 2.0, 1.5]],
        dtype=float,
    )
    native_map = NativeBatchMap()

    native = dss_pp._node_path_lengths_batch(native_map, start, goals)
    no_map = dss_pp._node_path_lengths_batch(None, start, goals)

    assert native_map.batch_calls == 1
    np.testing.assert_allclose(
        native,
        np.linalg.norm(goals - start[None, :], axis=1) + 0.25,
    )
    np.testing.assert_allclose(
        no_map,
        np.linalg.norm(goals - start[None, :], axis=1),
    )
    with pytest.raises(TypeError, match="must provide motion_path_lengths_batch"):
        dss_pp._node_path_lengths_batch(object(), start, goals)


def test_dss_selection_uses_batch_lengths_for_filter_and_node_build() -> None:
    """End-to-end station selection dispatches both path phases in batches."""

    class TrackingBatchMap:
        """Count runtime-native batch path requests."""

        def __init__(self) -> None:
            """Initialize the batch-call counter."""
            self.batch_calls = 0

        def motion_path_lengths_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Return finite direct lengths for every candidate in one call."""
            self.batch_calls += 1
            return np.linalg.norm(goals_xyz - start_xyz[None, :], axis=1)

        def is_motion_reachable_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Fail if selection bypasses the preferred path-length API."""
            del start_xyz, goals_xyz
            raise AssertionError("legacy reachability API should not be used")

    estimator = _build_simple_estimator()
    planning_map = TrackingBatchMap()
    candidates = np.array(
        [[1.5, 1.5, 0.5], [2.5, 1.5, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=estimator,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        visited_poses_xyz=None,
        map_api=planning_map,
        config=DSSPPConfig(
            max_programs=4,
            program_length=1,
            forced_program_pair_ids=(0,),
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=1.0,
            lambda_rotation=0.0,
            lambda_coverage=0.0,
            eta_revisit=0.0,
            min_station_separation_m=0.0,
            augment_candidates=False,
        ),
    )

    assert planning_map.batch_calls == 2
    assert any(np.allclose(result.next_pose, pose) for pose in candidates)


def test_dss_runtime_motion_times_need_no_local_map_geometry() -> None:
    """Runtime reachability and time costs must replace local map surrogates."""
    estimator = _build_simple_estimator()
    candidates = np.array(
        [[1.5, 1.5, 0.5], [2.5, 1.5, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=estimator,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        candidate_motion_times_s=np.array([4.0, 2.0], dtype=float),
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        map_api=object(),
        config=DSSPPConfig(
            max_programs=4,
            program_length=1,
            forced_program_pair_ids=(0,),
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_time=1.0,
            lambda_rotation=0.0,
            lambda_coverage=0.0,
            augment_candidates=False,
        ),
    )

    np.testing.assert_allclose(result.next_pose, candidates[1])
    assert result.diagnostics["path_filtered_candidates"] == 0
    assert result.diagnostics["runtime_motion_times_applied"] is True
    assert result.diagnostics["planning_eig_shortlist"]["path_length_support"] == (
        "runtime_reachable_candidates_with_time_cost_only"
    )


def test_estimate_lambda_cost_range_scales_motion() -> None:
    """Range-based lambda should match uncertainty and motion-cost ranges."""
    uncertainties = np.array([1.0, 2.0, 4.0], dtype=float)
    distances = np.array([0.5, 1.0, 2.5], dtype=float)
    lam = estimate_lambda_cost(uncertainties, distances, method="range")
    expected = (4.0 - 1.0) / (2.5 - 0.5)
    assert lam == pytest.approx(expected)


def test_dss_pp_program_library_balances_every_shield_pair() -> None:
    """Every canonical pair must be represented without within-station repeats."""
    normals = np.asarray(generate_octant_orientations(), dtype=float)
    programs = build_shield_program_library(
        normals,
        program_length=8,
        max_programs=48,
    )
    occurrences = np.bincount(
        np.asarray(
            [pair_id for program in programs for pair_id in program.pair_ids],
            dtype=np.int64,
        ),
        minlength=64,
    )

    assert len(programs) == 48
    assert all(
        program.kind == "all_pair_balanced_multi_partition" for program in programs
    )
    assert all(len(program.pair_ids) == 8 for program in programs)
    assert all(len(set(program.pair_ids)) == 8 for program in programs)
    assert set(np.flatnonzero(occurrences)) == set(range(64))
    assert np.max(occurrences) == np.min(occurrences) == 6
    companions = {pair_id: set() for pair_id in range(64)}
    for program in programs:
        for pair_id in program.pair_ids:
            companions[pair_id].update(
                other for other in program.pair_ids if other != pair_id
            )
    assert min(len(values) for values in companions.values()) >= 28


@pytest.mark.parametrize(
    ("orientation_count", "program_length", "max_programs"),
    ((8, 8, 48), (5, 3, 9)),
)
def test_shield_program_batch_schedule_matches_scalar_test_oracle(
    orientation_count: int,
    program_length: int,
    max_programs: int,
) -> None:
    """Vectorized pair schedules must equal an independent scalar oracle."""
    programs = build_shield_program_library(
        np.zeros((orientation_count, 3), dtype=np.float64),
        program_length=program_length,
        max_programs=max_programs,
    )
    expected: list[tuple[str, tuple[int, ...]]] = []
    if orientation_count == 8 and program_length == 8:
        partitions: list[tuple[str, list[tuple[int, ...]]]] = []
        for slope in range(orientation_count):
            if math.gcd(slope, orientation_count) != 1:
                continue
            rows = []
            for offset in range(orientation_count):
                rows.append(
                    tuple(
                        fe_index * orientation_count
                        + (slope * fe_index + offset) % orientation_count
                        for fe_index in range(orientation_count)
                    )
                )
            partitions.append((f"latin_slope_{slope}", rows))
        partitions.append(
            (
                "fixed_fe",
                [
                    tuple(
                        fe_index * orientation_count + pb_index
                        for pb_index in range(orientation_count)
                    )
                    for fe_index in range(orientation_count)
                ],
            )
        )
        partitions.append(
            (
                "fixed_pb",
                [
                    tuple(
                        fe_index * orientation_count + pb_index
                        for fe_index in range(orientation_count)
                    )
                    for pb_index in range(orientation_count)
                ],
            )
        )
        expected = [
            (f"{partition_name}_{index:02d}", row)
            for partition_name, rows in partitions
            for index, row in enumerate(rows)
        ]
    else:
        ordered = tuple(
            fe_index * orientation_count + (fe_index + offset) % orientation_count
            for offset in range(orientation_count)
            for fe_index in range(orientation_count)
        )
        required = int(
            np.ceil(orientation_count * orientation_count / float(program_length))
        )
        for index in range(required):
            start = index * program_length
            selected = list(ordered[start : start + program_length])
            selected.extend(ordered[: program_length - len(selected)])
            expected.append((f"all_pair_balanced_{index:02d}", tuple(selected)))

    assert [(program.name, program.pair_ids) for program in programs] == expected


def test_dss_pp_program_library_rejects_insufficient_pair_capacity() -> None:
    """A truncated library must not silently hide valid shield actions."""
    normals = np.asarray(generate_octant_orientations(), dtype=float)

    with pytest.raises(ValueError, match="too small.*multi-partition"):
        build_shield_program_library(
            normals,
            program_length=8,
            max_programs=47,
        )


def test_extract_signature_modes_uses_pf_posterior_weights() -> None:
    """Signature modes should preserve normalized PF posterior mass."""
    states = [
        _encoded_surface_state(
            np.array([[1.0, 0.0, 0.0]], dtype=float),
            np.array([100.0], dtype=float),
        ),
        _encoded_surface_state(
            np.array([[5.0, 0.0, 0.0]], dtype=float),
            np.array([100.0], dtype=float),
        ),
    ]
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        filters={
            "Cs-137": SimpleNamespace(
                continuous_state_positions=_decode_encoded_surface_state,
            )
        },
        planning_particles=lambda **_kwargs: {
            "Cs-137": (states, np.array([0.8, 0.2], dtype=float))
        },
    )

    modes = dss_pp.extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        rng=np.random.default_rng(1),
    )["Cs-137"]

    assert len(modes) == 2
    assert np.allclose(modes[0].position_xyz, np.array([1.0, 0.0, 0.0]))
    assert modes[0].weight == pytest.approx(0.8)
    assert modes[1].weight == pytest.approx(0.2)
    assert modes[0].weight > modes[1].weight


def test_extract_signature_modes_preserves_k_zero_probability() -> None:
    """A tiny K>0 tail must not become a certain planner source."""
    states = [
        IsotopeState(
            num_sources=0,
            strengths=np.zeros(0, dtype=float),
            surface_chart_ids=np.zeros(0, dtype=np.int64),
            surface_uv=np.zeros((0, 2), dtype=float),
        ),
        _encoded_surface_state(
            np.array([[1.0, 0.0, 0.0]], dtype=float),
            np.array([100.0], dtype=float),
        ),
    ]
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        filters={
            "Cs-137": SimpleNamespace(
                continuous_state_positions=_decode_encoded_surface_state,
            )
        },
        planning_particles=lambda **_kwargs: {
            "Cs-137": (states, np.array([0.99, 0.01], dtype=float))
        },
    )

    modes = dss_pp.extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=1,
        rng=np.random.default_rng(1),
    )["Cs-137"]

    assert len(modes) == 1
    assert modes[0].weight == pytest.approx(0.01)
    assert modes[0].isotope_presence_probability == pytest.approx(0.01)


def test_extract_signature_modes_preserves_all_marginal_mode_mass() -> None:
    """Marginal multimodality may exceed Kmax and must never be truncated."""
    states = [
        _encoded_surface_state(
            np.asarray([[position, 0.0, 0.0]], dtype=float),
            np.asarray([100.0], dtype=float),
        )
        for position in (1.0, 5.0, 9.0)
    ]
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        filters={
            "Cs-137": SimpleNamespace(
                continuous_state_positions=_decode_encoded_surface_state,
            )
        },
        planning_particles=lambda **_kwargs: {
            "Cs-137": (
                states,
                np.asarray([0.4, 0.35, 0.25], dtype=float),
            )
        },
    )

    modes = dss_pp.extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=1,
        rng=np.random.default_rng(2),
    )["Cs-137"]

    assert len(modes) == 3
    assert sum(mode.weight for mode in modes) == pytest.approx(1.0)
    assert all(
        mode.isotope_presence_probability == pytest.approx(1.0) for mode in modes
    )


def test_candidate_augmentation_interleaves_every_material_mode() -> None:
    """A finite augmentation budget must represent every material mode once."""
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.asarray([position, 0.0, 0.0], dtype=float),
            strength_cps_1m=300_000.0,
            weight=weight,
            spread_m=0.1,
            isotope_presence_probability=1.0,
        )
        for position, weight in zip(
            (1.0, 5.0, 9.0),
            (0.8, 0.15, 0.05),
            strict=True,
        )
    ]
    config = DSSPPConfig(
        max_modes_per_isotope=1,
        max_augmented_candidates=3,
        ring_radii_m=(1.0,),
        ring_angles=4,
    )

    augmented = dss_pp.augment_candidate_stations(
        np.zeros((0, 3), dtype=float),
        modes_by_isotope={"Cs-137": modes},
        current_pose_xyz=np.asarray([0.0, 0.0, 1.0], dtype=float),
        visited_poses_xyz=None,
        map_api=None,
        bounds_xyz=None,
        config=config,
        rng=np.random.default_rng(3),
    )

    np.testing.assert_allclose(
        augmented,
        np.asarray(
            [
                [2.0, 0.0, 1.0],
                [6.0, 0.0, 1.0],
                [10.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_candidate_augmentation_expands_before_dropping_material_modes() -> None:
    """An undersized ordinary budget must expand to retain every mode."""
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.asarray([float(index), 0.0, 0.0]),
            strength_cps_1m=300_000.0,
            weight=0.25,
            spread_m=0.1,
            isotope_presence_probability=1.0,
        )
        for index in range(4)
    ]

    augmented = dss_pp.augment_candidate_stations(
        np.zeros((0, 3), dtype=float),
        modes_by_isotope={"Cs-137": modes},
        current_pose_xyz=np.asarray([0.0, 0.0, 1.0], dtype=float),
        visited_poses_xyz=None,
        map_api=None,
        bounds_xyz=None,
        config=DSSPPConfig(max_augmented_candidates=3),
        rng=np.random.default_rng(4),
    )

    np.testing.assert_allclose(
        augmented,
        np.asarray(
            [
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [4.0, 0.0, 1.0],
                [5.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_extract_signature_modes_packed_joint_matches_state_oracle() -> None:
    """Packed aligned mode extraction must equal the state-object oracle."""
    states = [
        IsotopeState(
            num_sources=0,
            strengths=np.zeros(0, dtype=float),
            surface_chart_ids=np.zeros(0, dtype=np.int64),
            surface_uv=np.zeros((0, 2), dtype=float),
        ),
        _encoded_surface_state(
            np.array([[1.0, 0.0, 0.0]], dtype=float),
            np.array([100.0], dtype=float),
        ),
        _encoded_surface_state(
            np.array(
                [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                dtype=float,
            ),
            np.array([90.0, 30.0], dtype=float),
        ),
    ]
    weights = np.array([0.1, 0.3, 0.6], dtype=float)
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        filters={
            "Cs-137": SimpleNamespace(
                continuous_state_positions=_decode_encoded_surface_state,
            )
        },
        planning_particles=lambda **_kwargs: {"Cs-137": (states, weights)},
    )
    packed_positions = np.zeros((3, 2, 3), dtype=float)
    packed_positions[1, 0] = np.array([1.0, 0.0, 0.0])
    packed_positions[2] = np.array(
        [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=float,
    )
    packed_strengths = np.array(
        [[0.0, 0.0], [100.0, 0.0], [90.0, 30.0]],
        dtype=float,
    )
    packed_mask = packed_strengths > 0.0
    joint = dss_pp.JointPlanningParticles(
        isotope_order=("Cs-137",),
        weights_n=weights,
        positions_nk3_by_isotope={"Cs-137": packed_positions},
        surface_chart_ids_nk_by_isotope={"Cs-137": np.zeros((3, 2), dtype=np.int64)},
        surface_uv_nk2_by_isotope={"Cs-137": np.zeros((3, 2, 2), dtype=float)},
        strengths_nk_by_isotope={"Cs-137": packed_strengths},
        source_mask_nk_by_isotope={"Cs-137": packed_mask},
        original_particle_indices=np.arange(3, dtype=np.int64),
    )

    oracle = dss_pp.extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        rng=np.random.default_rng(1),
    )["Cs-137"]
    packed = dss_pp.extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        rng=np.random.default_rng(1),
        joint_particles=joint,
    )["Cs-137"]

    assert len(packed) == len(oracle) == 2
    for packed_mode, oracle_mode in zip(packed, oracle):
        assert packed_mode.position_xyz == pytest.approx(oracle_mode.position_xyz)
        assert packed_mode.strength_cps_1m == pytest.approx(oracle_mode.strength_cps_1m)
        assert packed_mode.weight == pytest.approx(oracle_mode.weight)
        assert packed_mode.isotope_presence_probability == pytest.approx(
            oracle_mode.isotope_presence_probability
        )


def test_surface_mode_representative_uses_intrinsic_medoid() -> None:
    """A folded surface must use path distance rather than XYZ averaging."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=float,
    )
    lookup = {
        (0, 1): 100.0,
        (1, 0): 100.0,
        (0, 2): 1.0,
        (2, 0): 1.0,
        (1, 2): 1.0,
        (2, 1): 1.0,
    }

    def surface_distance(
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Return a deterministic folded-surface path metric."""
        left_ids = np.argmin(
            np.linalg.norm(left[:, None, :] - positions[None, :, :], axis=2),
            axis=1,
        )
        right_ids = np.argmin(
            np.linalg.norm(right[:, None, :] - positions[None, :, :], axis=2),
            axis=1,
        )
        return np.asarray(
            [
                0.0 if left_id == right_id else lookup[(left_id, right_id)]
                for left_id, right_id in zip(left_ids, right_ids)
            ],
            dtype=float,
        )

    medoid = dss_pp._weighted_surface_medoid_index(
        positions,
        np.full(3, 1.0 / 3.0, dtype=float),
        surface_path_distance=surface_distance,
    )

    assert medoid == 2


def test_surface_mode_clustering_uses_authoritative_chart_coordinates() -> None:
    """Coincident XYZ points on disconnected charts must remain distinct."""
    positions = np.zeros((2, 3), dtype=float)
    chart_ids = np.asarray([4, 9], dtype=np.int64)
    surface_uv = np.asarray(
        [[0.25, 0.75], [0.25, 0.75]],
        dtype=float,
    )

    def coordinate_distance(
        left_ids: np.ndarray,
        left_uv: np.ndarray,
        right_ids: np.ndarray,
        right_uv: np.ndarray,
    ) -> np.ndarray:
        """Return zero within a chart and infinity across components."""
        del left_uv, right_uv
        left, right = np.broadcast_arrays(left_ids, right_ids)
        return np.where(left == right, 0.0, np.inf)

    modes = dss_pp._cluster_source_samples(
        "Cs-137",
        positions,
        np.asarray([300_000.0, 500_000.0]),
        np.asarray([0.4, 0.6]),
        radius_m=1.0,
        max_modes=2,
        particle_ids=np.asarray([0, 1], dtype=np.int64),
        isotope_presence_probability=1.0,
        surface_chart_ids=chart_ids,
        surface_uv=surface_uv,
        surface_coordinate_path_distance=coordinate_distance,
    )

    assert len(modes) == 2
    assert {mode.surface_chart_id for mode in modes} == {4, 9}
    assert all(mode.surface_uv == pytest.approx((0.25, 0.75)) for mode in modes)


def test_information_gain_cannot_resurrect_zero_mass_particles() -> None:
    """A zero-prior particle must remain absent even at enormous likelihood."""
    likelihood = np.asarray([[[0.0, 1_000.0]]], dtype=float)

    gain = dss_pp._information_gain_from_log_likelihood(
        likelihood,
        np.asarray([1.0, 0.0], dtype=float),
    )

    assert gain == pytest.approx(np.asarray([0.0]))


def test_dss_probability_consumers_reject_numeric_strings() -> None:
    """Planner summaries must not cast textual scores into posterior mass."""
    with pytest.raises(ValueError, match="real number"):
        dss_pp._posterior_mode_weights(["0.5"])
    hostile_mode = dss_pp.SignatureMode(
        isotope="Cs-137",
        position_xyz=np.zeros(3, dtype=np.float64),
        strength_cps_1m=300_000.0,
        weight=0.5,
        spread_m=0.1,
        isotope_presence_probability="0.5",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="real number"):
        dss_pp._isotope_presence_probability([hostile_mode])


def test_information_gain_fails_if_positive_mass_has_no_support() -> None:
    """An impossible predictive observation must not become a finite EIG."""
    with pytest.raises(RuntimeError, match="outside every positive-mass"):
        dss_pp._information_gain_from_log_likelihood(
            np.full((1, 1, 2), -np.inf, dtype=float),
            np.asarray([0.5, 0.5], dtype=float),
        )


def test_finite_sample_eig_bound_is_not_replaced_by_prior_entropy() -> None:
    """A rare diagnostic draw may exceed entropy but not its support KL bound."""
    weights = np.asarray([0.99, 0.01], dtype=np.float64)
    likelihood = np.asarray([[[-1_000.0, 0.0]]], dtype=np.float64)

    sampled_gain = float(
        dss_pp._information_gain_from_log_likelihood(
            likelihood,
            weights,
        )[0]
    )
    entropy = float(-np.sum(weights * np.log(weights)))
    finite_sample_bound = dss_pp._finite_sample_information_gain_upper_bound(weights)

    assert sampled_gain > entropy
    assert sampled_gain <= finite_sample_bound + 1.0e-12
    assert finite_sample_bound == pytest.approx(-np.log(0.01))


def test_planner_configuration_does_not_silently_expand_ring_angles() -> None:
    """Invalid ring diversity must fail instead of being clamped at runtime."""
    with pytest.raises(ValueError, match="ring_angles"):
        DSSPPConfig(ring_angles=3)


def test_forced_pair_support_comes_from_estimator_orientation_count() -> None:
    """A globally plausible pair ID must fail for a smaller shield state space."""
    estimator = _build_simple_estimator()

    with pytest.raises(ValueError, match="shield-pair support"):
        select_dss_pp_next_station(
            estimator=estimator,
            rng=np.random.default_rng(7),
            candidate_poses_xyz=np.asarray(
                [[1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            current_pose_xyz=np.zeros(3, dtype=np.float64),
            config=DSSPPConfig(
                lambda_eig=0.0,
                lambda_distance=0.0,
                augment_candidates=False,
                min_station_separation_m=0.0,
                forced_program_pair_ids=(63,),
            ),
        )


def test_forced_program_must_contain_a_physical_view() -> None:
    """An empty forced program must fail before it can skip all measurements."""
    with pytest.raises(ValueError, match="at least one pair"):
        DSSPPConfig(forced_program_pair_ids=())


@pytest.mark.parametrize("current_pair_id", (-1, 4, True, 1.5))
def test_current_pair_must_match_estimator_shield_support(
    current_pair_id: object,
) -> None:
    """Rotation cost must not decode an invalid previous shield state."""
    estimator = _build_simple_estimator()

    with pytest.raises(ValueError, match="current_pair_id"):
        select_dss_pp_next_station(
            estimator=estimator,
            rng=np.random.default_rng(7),
            candidate_poses_xyz=np.asarray(
                [[1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            current_pose_xyz=np.zeros(3, dtype=np.float64),
            current_pair_id=current_pair_id,  # type: ignore[arg-type]
            config=DSSPPConfig(
                lambda_eig=0.0,
                lambda_distance=0.0,
                augment_candidates=False,
                min_station_separation_m=0.0,
                forced_program_pair_ids=(0,),
            ),
        )


def test_planner_geometry_modes_match_official_joint_map_projection() -> None:
    """Planner cardinality and medoids must equal the official joint report."""
    estimator = SimpleNamespace(
        isotopes=("Cs-137", "Co-60"),
        joint_isotope_order=lambda: ("Cs-137", "Co-60"),
        posterior_point_estimate=lambda: {
            "Cs-137": SimpleNamespace(
                map_cardinality=1,
                cardinality_distribution={0: 0.3, 1: 0.7},
                selected_stratum_mass=0.7,
                modes=(
                    SimpleNamespace(
                        position_medoid_xyz=(2.0, 3.0, 4.0),
                        strength_representative_cps_1m=400_000.0,
                        posterior_mass=0.7,
                        credible_surface_path_radius_95_m=0.6,
                        credible_radius_95_m=0.4,
                        surface_chart_id=3,
                        surface_uv=(0.25, 0.75),
                    ),
                ),
            ),
            "Co-60": SimpleNamespace(
                map_cardinality=0,
                cardinality_distribution={0: 0.7, 1: 0.3},
                selected_stratum_mass=0.7,
                modes=(),
            ),
        },
        posterior_joint_cardinality_distribution=lambda: {
            (0, 1): 0.3,
            (1, 0): 0.7,
        },
    )

    modes, diagnostics = dss_pp._official_signature_modes(
        estimator,
        max_modes_per_isotope=5,
    )

    assert diagnostics["joint_map_cardinality_vector"] == [1, 0]
    assert diagnostics["joint_map_stratum_mass"] == pytest.approx(0.7)
    assert diagnostics["verified_against_joint_cardinality_distribution"] is True
    assert diagnostics["medoids_by_isotope"]["Cs-137"] == [[2.0, 3.0, 4.0]]
    assert modes["Co-60"] == []
    assert len(modes["Cs-137"]) == 1
    np.testing.assert_allclose(
        modes["Cs-137"][0].position_xyz,
        np.asarray([2.0, 3.0, 4.0]),
    )
    assert modes["Cs-137"][0].weight == pytest.approx(0.7)
    assert modes["Cs-137"][0].isotope_presence_probability == pytest.approx(0.7)


def test_planner_rejects_geometry_modes_from_nonofficial_joint_map() -> None:
    """A planner snapshot may not diverge from the official joint MAP tuple."""
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        posterior_point_estimate=lambda: {
            "Cs-137": SimpleNamespace(
                map_cardinality=1,
                cardinality_distribution={0: 0.4, 1: 0.6},
                selected_stratum_mass=0.6,
                modes=(
                    SimpleNamespace(
                        position_medoid_xyz=(1.0, 0.0, 0.0),
                        strength_representative_cps_1m=300_000.0,
                        posterior_mass=0.6,
                        credible_surface_path_radius_95_m=0.2,
                        credible_radius_95_m=0.2,
                        surface_chart_id=2,
                        surface_uv=(0.4, 0.6),
                    ),
                ),
            )
        },
        posterior_joint_cardinality_distribution=lambda: {
            (0,): 0.7,
            (1,): 0.3,
        },
    )

    with pytest.raises(RuntimeError, match="official joint MAP"):
        dss_pp._official_signature_modes(
            estimator,
            max_modes_per_isotope=5,
        )


def test_dss_rejects_mode_capacity_below_pf_cardinality() -> None:
    """Planning cannot silently omit one state-supported source mode."""
    estimator = _build_simple_estimator()
    estimator.pf_config.max_sources = 2
    estimator.pf_config.hard_max_sources = 4
    with pytest.raises(ValueError, match="at least the PF cardinality capacity"):
        dss_pp._validate_mode_capacity(
            estimator,
            DSSPPConfig(max_modes_per_isotope=3),
        )


def test_dss_accepts_mode_capacity_equal_to_pf_hard_capacity() -> None:
    """Planning must accept every state slot, including the thin K tail."""
    estimator = _build_simple_estimator()
    estimator.pf_config.max_sources = 5
    estimator.pf_config.hard_max_sources = 8

    capacity = dss_pp._validate_mode_capacity(
        estimator,
        DSSPPConfig(max_modes_per_isotope=8),
    )

    assert capacity == 8


def test_extract_signature_modes_keeps_distinct_pf_posterior_sources() -> None:
    """Every distinct active source in the PF posterior should remain visible."""
    states = [
        _encoded_surface_state(
            np.array(
                [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                dtype=float,
            ),
            np.array([90.0, 10.0], dtype=float),
        ),
    ]
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        filters={
            "Cs-137": SimpleNamespace(
                continuous_state_positions=_decode_encoded_surface_state,
            )
        },
        planning_particles=lambda **_kwargs: {
            "Cs-137": (states, np.array([1.0], dtype=float))
        },
    )

    modes = dss_pp.extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        rng=np.random.default_rng(1),
    )["Cs-137"]

    assert len(modes) == 2
    assert {tuple(np.asarray(mode.position_xyz, dtype=float)) for mode in modes} == {
        (1.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
    }
    assert [mode.weight for mode in modes] == pytest.approx([1.0, 1.0])


def test_dss_pp_has_no_external_or_tentative_mode_boundary() -> None:
    """DSS-PP should accept only the ordinary PF posterior mode source."""
    extract_parameters = inspect.signature(dss_pp.extract_signature_modes).parameters
    select_parameters = inspect.signature(dss_pp.select_dss_pp_next_station).parameters

    assert "tentative_weight_multiplier" not in extract_parameters
    assert "_planner_only_external_mode_token" not in extract_parameters
    assert "_planner_only_external_mode_token" not in select_parameters
    assert not hasattr(dss_pp, "_preserve_external_modes")


def test_dss_pp_requires_one_persistent_rng() -> None:
    """Planning must not recreate an identical fixed-seed stream per call."""
    estimator = _build_simple_estimator()
    with pytest.raises(TypeError, match="persistent explicit rng"):
        select_dss_pp_next_station(
            estimator=estimator,
            candidate_poses_xyz=np.array([[1.0, 0.0, 0.0]], dtype=float),
            current_pose_xyz=np.zeros(3, dtype=float),
            config=DSSPPConfig(
                lambda_eig=0.0,
                augment_candidates=False,
                min_station_separation_m=0.0,
            ),
        )


def test_dss_evaluates_every_action_below_shortlist_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small action set must remain an exhaustive exact-EIG evaluation."""
    estimator = _build_simple_estimator()
    candidates = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.5], [3.0, 0.0, 1.0]],
        dtype=float,
    )
    evaluated: list[np.ndarray] = []

    def fake_validate(*_args: object, **_kwargs: object) -> None:
        """Bypass the generative-model fixture in this routing-only test."""

    def fake_eig(
        _estimator: object,
        detector_positions: np.ndarray,
        programs_by_pose: list[list[object]],
        **_kwargs: object,
    ) -> list[np.ndarray]:
        """Record all actions and return one finite value per program."""
        evaluated.append(np.asarray(detector_positions, dtype=float).copy())
        return [np.zeros(len(programs), dtype=float) for programs in programs_by_pose]

    monkeypatch.setattr(dss_pp, "_validate_eig_likelihood_contract", fake_validate)
    monkeypatch.setattr(dss_pp, "_program_information_gains_for_poses", fake_eig)
    result = select_dss_pp_next_station(
        estimator=estimator,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.zeros(3, dtype=float),
        config=DSSPPConfig(
            lambda_eig=1.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            max_programs=4,
            program_length=1,
            forced_program_pair_ids=(0,),
            augment_candidates=False,
            min_station_separation_m=0.0,
        ),
    )

    assert len(evaluated) == 1
    np.testing.assert_allclose(evaluated[0], candidates)
    diagnostics = result.diagnostics["planning_eig_shortlist"]
    assert diagnostics["total_action_count"] == 3
    assert diagnostics["proxy_action_count"] == 0
    assert diagnostics["exact_action_count"] == 3
    assert diagnostics["proxy_wall_s"] == 0.0
    assert diagnostics["exact_eig_wall_s"] >= 0.0


def test_dss_exact_eig_respects_predeclared_action_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact stage must not expand past its real-time action budget."""
    estimator = _build_simple_estimator()
    candidates = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    exact_by_x = {
        1.0: 0.10,
        2.0: float(np.log(2.0)),
        3.0: 0.20,
        4.0: 0.30,
    }

    def fake_exact_eig(
        _estimator: object,
        detector_positions: np.ndarray,
        programs_by_pose: list[list[object]],
        **_kwargs: object,
    ) -> list[np.ndarray]:
        """Return deterministic exact values for requested actions only."""
        return [
            np.full(
                len(programs),
                exact_by_x[float(position[0])],
                dtype=np.float64,
            )
            for position, programs in zip(
                np.asarray(detector_positions, dtype=np.float64),
                programs_by_pose,
                strict=True,
            )
        ]

    def fake_proxy(
        _estimator: object,
        detector_positions: np.ndarray,
        programs: list[object],
        **_kwargs: object,
    ) -> np.ndarray:
        """Deliberately rank the exact optimum last."""
        proxy_by_x = {
            1.0: 0.40,
            2.0: 0.01,
            3.0: 0.30,
            4.0: 0.20,
        }
        return np.asarray(
            [
                [proxy_by_x[float(position[0])] for _program in programs]
                for position in np.asarray(
                    detector_positions,
                    dtype=np.float64,
                )
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(
        dss_pp,
        "_program_information_gains_for_poses",
        fake_exact_eig,
    )
    monkeypatch.setattr(
        dss_pp,
        "_program_information_proxy_for_poses",
        fake_proxy,
    )
    common_config = {
        "lambda_eig": 1.0,
        "lambda_distance": 0.0,
        "lambda_time": 0.0,
        "lambda_rotation": 0.0,
        "max_programs": 1,
        "program_length": 1,
        "forced_program_pair_ids": (0,),
        "augment_candidates": False,
        "min_station_separation_m": 0.0,
        "exact_eig_coverage_reserve": 0,
        "exact_eig_program_diversity_reserve": 0,
    }
    exhaustive = select_dss_pp_next_station(
        estimator=estimator,
        rng=np.random.default_rng(17),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.zeros(3, dtype=np.float64),
        config=DSSPPConfig(
            **common_config,
            exact_eig_action_limit=4,
        ),
    )
    shortlisted = select_dss_pp_next_station(
        estimator=estimator,
        rng=np.random.default_rng(17),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.zeros(3, dtype=np.float64),
        config=DSSPPConfig(
            **common_config,
            exact_eig_action_limit=2,
        ),
    )

    assert exhaustive.next_pose[0] == pytest.approx(2.0)
    assert shortlisted.next_pose[0] == pytest.approx(3.0)
    diagnostics = shortlisted.diagnostics["planning_eig_shortlist"]
    assert diagnostics["total_action_count"] == 4
    assert diagnostics["proxy_action_count"] == 4
    assert diagnostics["exact_action_count"] == 2
    assert diagnostics["adaptive_exact_eig_exhausted_all_actions"] is False
    assert diagnostics["adaptive_exact_eig_round_count"] == 1
    assert diagnostics["proxy_wall_s"] >= 0.0
    assert diagnostics["exact_eig_wall_s"] >= 0.0
    assert (
        diagnostics["shortlisted_exact_bin_state_operations"]
        < diagnostics["legacy_all_exact_bin_state_operations"]
    )
    assert diagnostics["shortlist_selected_proxy_rank"] == 2
    assert diagnostics["shortlist_mc_winner_exceeds_universal_excluded_bound"] is False
    assert diagnostics["shortlist_formal_recall_certificate_available"] is False
    assert "joint_full_spectrum" in diagnostics["proxy_contract"]


def test_dss_canonical_program_diagnostics_cover_all_pairs_at_horizon_one() -> None:
    """The runtime library must expose all 64 pairs in a one-step policy."""
    estimator = _build_simple_estimator(canonical_octants=True)
    result = select_dss_pp_next_station(
        estimator=estimator,
        rng=np.random.default_rng(31),
        candidate_poses_xyz=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        current_pose_xyz=np.zeros(3, dtype=np.float64),
        config=DSSPPConfig(
            max_programs=48,
            program_length=8,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_time=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
            min_station_separation_m=0.0,
        ),
    )

    assert result.diagnostics["planning_policy"] == "one_step_joint_eig"
    assert len(result.sequence) == 1
    assert result.diagnostics["program_count"] == 48
    assert result.diagnostics["program_library_unique_pair_count"] == 64
    assert result.diagnostics["program_library_pair_occurrence_min"] == 6
    assert result.diagnostics["program_library_pair_occurrence_max"] == 6


def test_dss_pp_selects_station_and_shield_program() -> None:
    """DSS-PP should jointly return a pose and executable shield program."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    shield_normals = np.asarray(generate_octant_orientations(), dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        variable_cardinality=False,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=_state_on_filter(
                filt,
                np.array([[0.0, 0.0, 0.5]], dtype=float),
                np.array([2000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=filt.continuous_particles[0].joint_row_identity,
        ),
        IsotopeParticle(
            state=_state_on_filter(
                filt,
                np.array([[4.0, 0.0, 0.5]], dtype=float),
                np.array([2000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=filt.continuous_particles[1].joint_row_identity,
        ),
    ]
    candidates = np.array(
        [[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]],
        dtype=float,
    )
    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([2.0, 2.0, 0.5], dtype=float),
        config=DSSPPConfig(
            max_programs=32,
            program_length=2,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
        ),
    )

    assert result.next_pose.shape == (3,)
    assert result.shield_program.pair_ids
    assert result.diagnostics["node_count"] > 0
    assert result.diagnostics["planner_belief_sources"] == ["pf_posterior"]
    assert result.diagnostics["ranked_nodes"]
    assert (
        result.diagnostics["ranked_nodes"][0]["score"]
        >= result.diagnostics["ranked_nodes"][-1]["score"]
    )
    assert np.allclose(result.next_pose, candidates[result.next_pose_index])
    assert "component_leaders" in result.diagnostics
    assert "score" in result.diagnostics["component_leaders"]


def test_dss_pp_forced_program_scores_only_baseline_pairs() -> None:
    """Forced DSS-PP programs should match baseline shield-policy execution."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    shield_normals = np.asarray(generate_octant_orientations(), dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        variable_cardinality=False,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=_state_on_filter(
                filt,
                np.array([[0.0, 0.0, 0.5]], dtype=float),
                np.array([2000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=filt.continuous_particles[0].joint_row_identity,
        ),
        IsotopeParticle(
            state=_state_on_filter(
                filt,
                np.array([[4.0, 0.0, 0.5]], dtype=float),
                np.array([2000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=filt.continuous_particles[1].joint_row_identity,
        ),
    ]
    forced_pairs = (7, 8, 9, 10)
    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=np.array([[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]]),
        current_pose_xyz=np.array([2.0, 2.0, 0.5], dtype=float),
        config=DSSPPConfig(
            max_programs=32,
            program_length=4,
            forced_program_pair_ids=forced_pairs,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
        ),
    )

    assert result.shield_program.pair_ids == forced_pairs
    assert result.shield_program.kind == "forced_baseline"
    assert result.diagnostics["program_count"] == 1
    assert {tuple(node["pair_ids"]) for node in result.diagnostics["ranked_nodes"]} == {
        forced_pairs
    }


def test_dss_pp_ranked_node_limit_zero_disables_ranked_payload() -> None:
    """A zero DSS-PP ranked-node limit should skip diagnostic node payloads."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    shield_normals = np.asarray(generate_octant_orientations(), dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        variable_cardinality=False,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": 0.08},
        pf_config=config,
        shield_params=ShieldParams(),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=_state_on_filter(
                filt,
                np.array([[0.0, 0.0, 0.5]], dtype=float),
                np.array([2000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=filt.continuous_particles[0].joint_row_identity,
        ),
        IsotopeParticle(
            state=_state_on_filter(
                filt,
                np.array([[4.0, 0.0, 0.5]], dtype=float),
                np.array([2000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=filt.continuous_particles[1].joint_row_identity,
        ),
    ]
    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=np.array([[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]]),
        current_pose_xyz=np.array([2.0, 2.0, 0.5], dtype=float),
        config=DSSPPConfig(
            max_programs=32,
            program_length=2,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
            diagnostic_ranked_node_limit=0,
        ),
    )

    assert result.next_pose.shape == (3,)
    assert result.diagnostics["node_count"] > 0
    assert result.diagnostics["diagnostic_ranked_node_limit"] == 0
    assert result.diagnostics["ranked_nodes"] == []
    assert result.sequence


def test_dss_pp_coverage_term_prefers_unvisited_free_space() -> None:
    """DSS-PP should move toward uncovered traversable cells when weighted."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5]], dtype=float)
    shield_normals = np.asarray(generate_octant_orientations(), dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=1,
        variable_cardinality=False,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    est.add_measurement_pose(np.array([1.0, 1.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    est.filters["Cs-137"].continuous_particles = [
        IsotopeParticle(
            state=_state_on_filter(
                est.filters["Cs-137"],
                np.array([[0.0, 0.0, 0.5]], dtype=float),
                np.array([100.0], dtype=float),
            ),
            log_weight=0.0,
            joint_row_identity=est.filters["Cs-137"]
            .continuous_particles[0]
            .joint_row_identity,
        )
    ]
    candidates = np.array(
        [[1.5, 1.5, 0.5], [8.5, 8.5, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([1.0, 1.0, 0.5], dtype=float),
        visited_poses_xyz=np.array([[1.0, 1.0, 0.5]], dtype=float),
        map_api=None,
        config=DSSPPConfig(
            max_programs=4,
            program_length=1,
            forced_program_pair_ids=(0,),
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            lambda_coverage=1.0,
            eta_revisit=1.0,
            min_station_separation_m=3.0,
            coverage_radius_m=2.0,
            augment_candidates=False,
        ),
    )

    assert result.diagnostics["first_coverage_gain"] > 0.0
    assert np.allclose(result.next_pose, candidates[1])


def test_dss_pp_coverage_floor_rejects_low_coverage_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage-floor scoring should keep surface exploration from collapsing."""
    est = _build_simple_estimator()
    quadrature = SimpleNamespace(
        positions_s3=np.array(
            [[8.5, 8.5, 0.5]],
            dtype=float,
        ),
        area_weights_m2_s=np.ones(1, dtype=float),
        diagnostics=lambda: {
            "complete_chart_coverage": True,
            "chart_count": 1,
        },
    )
    monkeypatch.setattr(
        est,
        "surface_atlas_area_quadrature",
        lambda **_kwargs: quadrature,
    )
    candidates = np.array(
        [[1.5, 1.5, 0.5], [8.5, 8.5, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([1.0, 1.0, 0.5], dtype=float),
        visited_poses_xyz=np.array([[1.0, 1.0, 0.5]], dtype=float),
        map_api=None,
        config=DSSPPConfig(
            max_programs=4,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            lambda_coverage=0.0,
            coverage_floor_quantile=1.0,
            coverage_floor_weight=100.0,
            eta_revisit=0.0,
            min_station_separation_m=0.0,
            coverage_radius_m=2.0,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])


def test_dss_pp_fails_when_generic_separation_removes_every_candidate() -> None:
    """DSS must not bypass the one generic 3-D station-separation rule."""
    est = _build_simple_estimator()
    current = np.array([0.5, 0.5, 0.0], dtype=float)
    candidates = np.array([[1.5, 0.5, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="generic 3-D station-separation"):
        select_dss_pp_next_station(
            estimator=est,
            rng=np.random.default_rng(123),
            candidate_poses_xyz=candidates,
            current_pose_xyz=current,
            visited_poses_xyz=np.array([current], dtype=float),
            map_api=None,
            config=DSSPPConfig(
                max_programs=4,
                program_length=1,
                lambda_eig=0.0,
                lambda_distance=0.0,
                lambda_rotation=0.0,
                min_station_separation_m=2.0,
                augment_candidates=False,
            ),
        )


def test_dss_pp_production_estimator_requires_continuous_surface_atlas() -> None:
    """Production coverage must never fall back to an XY/free-cell grid."""
    est = _build_simple_estimator()
    est.surface_atlas_area_quadrature = None  # type: ignore[method-assign]
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [2.0, 1.0, 0.5],
            [8.0, 18.0, 0.5],
        ],
        dtype=float,
    )

    with pytest.raises(RuntimeError, match="continuous physical surface atlas"):
        select_dss_pp_next_station(
            estimator=est,
            rng=np.random.default_rng(123),
            candidate_poses_xyz=candidates,
            current_pose_xyz=current,
            visited_poses_xyz=visited,
            bounds_xyz=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([10.0, 20.0, 1.0], dtype=float),
            ),
            config=DSSPPConfig(
                augment_candidates=False,
                max_programs=2,
                forced_program_pair_ids=(0, 1),
                lambda_eig=0.0,
                lambda_distance=0.0,
                lambda_coverage=10.0,
                coverage_radius_m=3.0,
                min_station_separation_m=0.0,
            ),
        )


def test_dss_coverage_uses_full_surface_atlas_without_pf_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High walls and ceilings must remain coverage targets before detection."""
    estimator = _build_simple_estimator()
    atlas_points = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 3.0, 5.0],
            [4.0, 3.0, 10.0],
        ],
        dtype=float,
    )
    observed: list[np.ndarray] = []

    def fake_surface_quadrature(
        *,
        max_points: int,
        maximum_hausdorff_bound_m: float,
    ) -> object:
        """Return a complete weighted surface fixture."""
        assert max_points == 3
        assert maximum_hausdorff_bound_m == pytest.approx(1.0)
        return SimpleNamespace(
            positions_s3=atlas_points,
            area_weights_m2_s=np.asarray([1.0, 2.0, 3.0]),
            diagnostics=lambda: {
                "sample_count": 3,
                "every_chart_represented": True,
                "area_weighted": True,
            },
        )

    estimator.surface_atlas_area_quadrature = (  # type: ignore[method-assign]
        fake_surface_quadrature
    )

    def fake_coverage(**kwargs: object) -> np.ndarray:
        """Record the atlas support passed to the response coverage kernel."""
        observed.append(np.asarray(kwargs["surface_points_xyz"], dtype=float).copy())
        candidates = np.asarray(kwargs["candidate_poses_xyz"], dtype=float)
        return np.zeros(candidates.shape[0], dtype=float)

    monkeypatch.setattr(
        dss_pp,
        "_response_equivalent_surface_coverage_gains",
        fake_coverage,
    )
    select_dss_pp_next_station(
        estimator=estimator,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=np.array(
            [[1.0, 0.0, 0.5], [2.0, 0.0, 1.5]],
            dtype=float,
        ),
        current_pose_xyz=np.zeros(3, dtype=float),
        config=DSSPPConfig(
            lambda_eig=0.0,
            augment_candidates=False,
            min_station_separation_m=0.0,
            coverage_surface_quadrature_max_points=3,
            coverage_surface_max_hausdorff_m=1.0,
        ),
    )

    assert len(observed) == 1
    np.testing.assert_allclose(observed[0], atlas_points)
    assert float(np.max(observed[0][:, 2])) == pytest.approx(10.0)


def test_complete_surface_quadrature_includes_every_physical_face() -> None:
    """Tiny, high-wall, ceiling, and obstacle charts must all be represented."""
    vertices = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.1, 0.0],
                [0.0, 0.1, 0.0],
            ],
            [
                [0.0, 0.0, 8.0],
                [0.0, 0.5, 8.0],
                [0.0, 0.5, 9.0],
                [0.0, 0.0, 9.0],
            ],
            [
                [1.0, 1.0, 10.0],
                [2.0, 1.0, 10.0],
                [2.0, 2.0, 10.0],
                [1.0, 2.0, 10.0],
            ],
            [
                [2.0, 2.0, 0.0],
                [2.0, 2.2, 0.0],
                [2.0, 2.2, 0.2],
                [2.0, 2.0, 0.2],
            ],
        ],
        dtype=np.float64,
    )
    areas = np.linalg.norm(
        np.cross(
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=1,
    )
    geometry = SurfaceChartGeometry(
        centers_xyz=np.mean(vertices, axis=1),
        areas_m2=areas,
        kinds=("floor", "wall", "ceiling", "obstacle_side"),
        face_ids=(
            "tiny_floor",
            "high_wall",
            "ceiling",
            "obstacle_boundary",
        ),
        normals_xyz=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        local_uv_m=np.zeros((4, 2), dtype=np.float64),
        vertices_xyz=vertices,
        adjacency_edges=np.zeros((0, 2), dtype=np.int64),
        shared_edge_lengths_m=np.zeros(0, dtype=np.float64),
        obstacle_geometry_source="test",
        obstacle_surfaces_available=True,
        obstacle_component_count=1,
    )
    quadrature = build_complete_surface_atlas_quadrature(
        ContinuousSurfaceAtlas(geometry),
        max_points=4,
        maximum_hausdorff_bound_m=0.75,
    )

    np.testing.assert_allclose(quadrature.positions_s3, geometry.centers_xyz)
    np.testing.assert_allclose(quadrature.area_weights_m2_s, areas)
    assert set(quadrature.face_ids) == {
        "tiny_floor",
        "high_wall",
        "ceiling",
        "obstacle_boundary",
    }
    assert quadrature.area_weights_m2_s[0] == pytest.approx(0.01)
    assert float(np.max(quadrature.positions_s3[:, 2])) == pytest.approx(10.0)
    diagnostics = quadrature.diagnostics()
    assert diagnostics["every_chart_represented"] is True
    assert diagnostics["area_weighted"] is True
    assert diagnostics["physical_face_count"] == 4

    with pytest.raises(RuntimeError, match="cannot represent every chart"):
        build_complete_surface_atlas_quadrature(
            ContinuousSurfaceAtlas(geometry),
            max_points=3,
            maximum_hausdorff_bound_m=0.75,
        )
    with pytest.raises(RuntimeError, match="Hausdorff bound"):
        build_complete_surface_atlas_quadrature(
            ContinuousSurfaceAtlas(geometry),
            max_points=4,
            maximum_hausdorff_bound_m=0.1,
        )


def test_surface_coverage_gain_uses_physical_area_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage gain must not give tiny and large charts equal mass."""
    candidate_masks = np.asarray(
        [[True, False], [False, True]],
        dtype=bool,
    )

    def fake_masks(**_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic candidate and acquired coverage masks."""
        return candidate_masks, np.zeros(2, dtype=bool)

    monkeypatch.setattr(
        dss_pp,
        "_response_equivalent_surface_coverage_masks",
        fake_masks,
    )
    gains = dss_pp._response_equivalent_surface_coverage_gains(
        kernel=object(),
        estimator=object(),
        surface_points_xyz=np.zeros((2, 3), dtype=np.float64),
        surface_area_weights_m2=np.asarray([1.0, 9.0], dtype=np.float64),
        candidate_poses_xyz=np.zeros((2, 3), dtype=np.float64),
        reference_radius_m=3.0,
    )

    np.testing.assert_allclose(gains, np.asarray([0.1, 0.9]))


def test_sparse_surface_observability_matches_dense_physics_oracle() -> None:
    """Sparse station coverage must preserve its dense unshielded response test."""
    estimator = _build_simple_estimator()
    estimator.measurements.append(
        SimpleNamespace(
            detector_position_xyz_m=(0.0, 0.0, 1.0),
            fe_index=0,
            pb_index=1,
        )
    )
    kernel = dss_pp._continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=1,
    )
    surfaces = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    candidates = np.asarray(
        [[0.0, 0.0, 1.0], [4.0, 0.0, 1.0], [8.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    radius = 2.0

    sparse_candidate, sparse_acquired = (
        dss_pp._response_equivalent_surface_coverage_masks(
            kernel=kernel,
            estimator=estimator,
            surface_points_xyz=surfaces,
            candidate_poses_xyz=candidates,
            reference_radius_m=radius,
        )
    )
    reference = float(
        dss_pp._finite_sphere_geometric_terms_batched(
            np.zeros((1, 3), dtype=float),
            np.asarray([[radius, 0.0, 0.0]], dtype=float),
            detector_radius_m=float(kernel.detector_radius_m),
        )[0, 0]
    )
    dense_candidate = (
        kernel.kernel_values_unshielded_for_detectors(
            isotope="Cs-137",
            detector_positions=candidates,
            sources=surfaces,
        )
        >= reference
    )
    dense_acquired_response = kernel.kernel_values_unshielded_for_detectors(
        isotope="Cs-137",
        detector_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=float),
        sources=surfaces,
    )
    dense_acquired = np.max(dense_acquired_response, axis=0) >= reference

    np.testing.assert_array_equal(sparse_candidate, dense_candidate)
    np.testing.assert_array_equal(sparse_acquired, dense_acquired)


def test_surface_observability_resolves_horizontal_and_vertical_actions() -> None:
    """Atlas coverage must score horizontal and vertical geometry in full 3-D."""
    estimator = _build_simple_estimator()
    kernel = dss_pp._continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=1,
    )
    surfaces = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 8.0], [8.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    candidates = np.asarray(
        [[0.0, 0.5, 0.0], [0.0, 0.5, 8.0], [8.0, 0.5, 0.0]],
        dtype=np.float64,
    )

    candidate_masks, acquired = dss_pp._response_equivalent_surface_coverage_masks(
        kernel=kernel,
        estimator=estimator,
        surface_points_xyz=surfaces,
        candidate_poses_xyz=candidates,
        reference_radius_m=1.0,
    )

    np.testing.assert_array_equal(candidate_masks, np.eye(3, dtype=bool))
    np.testing.assert_array_equal(acquired, np.zeros(3, dtype=bool))


def test_dss_pp_filters_near_revisit_when_alternatives_exist() -> None:
    """Station separation should remove near revisits after augmentation."""
    est = _build_simple_estimator()
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [1.5, 1.0, 0.5],
            [5.5, 1.0, 0.5],
        ],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        bounds_xyz=(
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([10.0, 10.0, 1.0], dtype=float),
        ),
        config=DSSPPConfig(
            augment_candidates=False,
            max_programs=2,
            forced_program_pair_ids=(0, 1),
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_coverage=0.0,
            min_station_separation_m=3.0,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])
    assert int(result.diagnostics["separation_filtered_candidates"]) == 1


def test_dss_pp_augments_with_global_unvisited_coverage_candidates() -> None:
    """DSS-PP should add global coverage candidates when base candidates revisit."""
    est = _build_simple_estimator()
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array([[1.2, 1.0, 0.5]], dtype=float)

    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 0.5], dtype=float),
        ),
        config=DSSPPConfig(
            augment_candidates=True,
            max_augmented_candidates=32,
            max_programs=2,
            forced_program_pair_ids=(0, 1),
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_coverage=10.0,
            lambda_rotation=0.0,
            min_station_separation_m=3.0,
            coverage_radius_m=2.0,
        ),
    )

    assert not np.allclose(result.next_pose, candidates[0])
    assert result.diagnostics["candidate_count"] > 1


def test_dss_pp_local_orbit_prefers_informative_annulus() -> None:
    """Local-orbit scoring should choose an offset station over source chasing."""
    est = _build_simple_estimator()
    candidates = np.array(
        [
            [0.1, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([5.0, 5.0, 0.0], dtype=float),
        config=DSSPPConfig(
            max_programs=4,
            program_length=1,
            forced_program_pair_ids=(0,),
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            lambda_local_orbit=10.0,
            ring_radii_m=(3.0,),
            local_orbit_sigma_m=0.5,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])
    assert result.diagnostics["first_local_orbit_gain"] > 0.0


def test_dss_pp_bearing_diversity_is_isotope_agnostic() -> None:
    """Bearing diversity should favor angularly separating any same-isotope modes."""
    isotopes = ["Co-60"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    shield_normals = np.asarray(generate_octant_orientations(), dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        variable_cardinality=False,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Co-60": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    est.add_measurement_pose(np.array([2.0, 3.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    est.filters["Co-60"].continuous_particles = [
        IsotopeParticle(
            state=_state_on_filter(
                est.filters["Co-60"],
                np.array([[0.0, 0.0, 0.5]], dtype=float),
                np.array([1000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=est.filters["Co-60"]
            .continuous_particles[0]
            .joint_row_identity,
        ),
        IsotopeParticle(
            state=_state_on_filter(
                est.filters["Co-60"],
                np.array([[4.0, 0.0, 0.5]], dtype=float),
                np.array([1000.0], dtype=float),
            ),
            log_weight=np.log(0.5),
            joint_row_identity=est.filters["Co-60"]
            .continuous_particles[1]
            .joint_row_identity,
        ),
    ]
    candidates = np.array(
        [[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([2.0, 3.0, 0.5], dtype=float),
        config=DSSPPConfig(
            max_programs=32,
            forced_program_pair_ids=(0, 1),
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_coverage=0.0,
            lambda_bearing_diversity=10.0,
            lambda_rotation=0.0,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[0])
    assert result.diagnostics["first_bearing_diversity_gain"] > 0.0


def test_dss_pp_turn_smoothness_discourages_backtracking() -> None:
    """Turn smoothness should prefer continuing outward over reversing course."""
    est = _build_simple_estimator()
    current = np.array([1.0, 0.0, 0.0], dtype=float)
    visited = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )
    candidates = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        rng=np.random.default_rng(123),
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        config=DSSPPConfig(
            max_programs=2,
            forced_program_pair_ids=(0, 1),
            lambda_eig=0.0,
            lambda_distance=0.0,
            lambda_coverage=0.0,
            lambda_turn_smoothness=5.0,
            lambda_rotation=0.0,
            min_station_separation_m=0.0,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])
    assert result.diagnostics["first_turn_penalty"] == pytest.approx(0.0)


def test_route_turn_batch_handles_second_three_dimensional_station() -> None:
    """The second planning call must evaluate full 3-D turn vectors."""
    current = np.array([1.0, 2.0, 1.5], dtype=float)
    visited = np.array(
        [[0.0, 0.0, 0.5], current],
        dtype=float,
    )
    candidates = np.array(
        [
            [2.0, 4.0, 2.5],
            [0.0, 0.0, 0.5],
            [1.0, 2.0, 3.5],
        ],
        dtype=float,
    )

    batched = dss_pp._route_turn_penalties_batch(
        candidates,
        current,
        visited,
    )
    scalar = np.asarray(
        [
            dss_pp._route_turn_penalty(candidate, current, visited)
            for candidate in candidates
        ],
        dtype=float,
    )

    np.testing.assert_allclose(batched, scalar, rtol=0.0, atol=1.0e-12)

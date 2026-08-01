"""CPU/Torch equivalence tests for pure-PF exact-RJ candidate targets."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from pf.estimator import (
    JointStationObservation,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.particle_filter import IsotopeParticle
from pf.state import IsotopeState
from tests.pure_pf_test_support import (
    TEST_ISOTOPES,
    approved_full_spectrum_model,
)


def _positive_line_transport_table() -> dict[str, tuple[dict[str, float], ...]]:
    """Return the full-spectrum model's authoritative positive-line table."""
    model = approved_full_spectrum_model()
    return {
        isotope: tuple(
            {
                "weight": float(line["branching_weight"]),
                "fe": float(line["mu_fe_cm_inv"]),
                "pb": float(line["mu_pb_cm_inv"]),
                "energy_keV": float(line["energy_keV"]),
            }
            for line in model.line_identity
            if str(line["isotope"]) == isotope
        )
        for isotope in TEST_ISOTOPES
    }


def _estimator(
    *,
    use_gpu: bool,
    gpu_device: str = "cpu",
) -> RotatingShieldPFEstimator:
    """Build the same small joint PF on NumPy or Torch-CPU transport."""
    model = approved_full_spectrum_model()
    estimator = RotatingShieldPFEstimator(
        isotopes=TEST_ISOTOPES,
        surface_diagnostic_points=np.asarray(
            [[0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in TEST_ISOTOPES},
        line_mu_by_isotope=_positive_line_transport_table(),
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=model,
        random_seed=17,
    )
    estimator.add_measurement_pose(
        np.asarray([1.5, 1.5, 1.5], dtype=np.float64)
    )
    estimator._ensure_kernel_cache()
    return estimator


def _station() -> JointStationObservation:
    """Return one exact-count two-view station for backend comparison."""
    model = approved_full_spectrum_model()
    spectrum = np.zeros(
        (2, int(model.energy_axis_keV.size)),
        dtype=np.int64,
    )
    spectrum[0, [0, 331, 586, 798]] = [3, 17, 11, 5]
    spectrum[1, [0, 331, 586, 798]] = [2, 7, 13, 19]
    return JointStationObservation(
        spectrum_vb=spectrum,
        energy_axis_keV=model.energy_axis_keV,
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=(1.5, 1.5, 1.5),
        fe_indices=np.asarray([0, 3], dtype=np.int64),
        pb_indices=np.asarray([2, 5], dtype=np.int64),
        live_times_s=np.asarray([1.0, 2.0], dtype=np.float64),
        station_sequence_id=0,
    )


def _conditional_target(
    estimator: RotatingShieldPFEstimator,
    station: JointStationObservation,
) -> np.ndarray:
    """Evaluate two continuous Cs-137 candidates under the joint target."""
    isotope = "Cs-137"
    particle_filter = estimator.filters[isotope]
    atlas = particle_filter._structural_rj_surface_atlas
    assert atlas is not None
    chart_ids = np.asarray([[0], [1]], dtype=np.int64)
    surface_uv = np.asarray(
        [[[0.2, 0.3]], [[0.7, 0.8]]],
        dtype=np.float64,
    )
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    strengths = np.asarray([[1_000.0], [900_000.0]], dtype=np.float64)
    stations = (station,)
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    evidence = estimator._joint_history_structural_geometry(
        isotope,
        stations,
    )
    try:
        return estimator._joint_structural_target_evaluator(
            filt=particle_filter,
            data=evidence,
            positions_pks=positions,
            chart_ids_pk=chart_ids,
            strengths_pk=strengths,
            particle_indices=np.arange(2, dtype=np.int64),
            target_beta=0.7,
            tempering_start_row=0,
        )
    finally:
        estimator._joint_structural_transport_cache = None
        estimator._active_joint_station_history = None


def test_exact_rj_candidate_target_matches_numpy_and_torch_cpu() -> None:
    """PF packing, physics, line layout, and target math must be equivalent."""
    pytest.importorskip("torch")
    station = _station()
    numpy_estimator = _estimator(use_gpu=False)
    torch_estimator = _estimator(use_gpu=True)

    numpy_components = (
        numpy_estimator._joint_station_transport_components_torch(station)
    )
    torch_components = (
        torch_estimator._joint_station_transport_components_torch(station)
    )
    for numpy_value, torch_value in zip(
        numpy_components,
        torch_components,
        strict=True,
    ):
        np.testing.assert_allclose(
            numpy_value.detach().cpu().numpy(),
            torch_value.detach().cpu().numpy(),
            rtol=2.0e-10,
            atol=3.0e-11,
        )

    numpy_target = _conditional_target(numpy_estimator, station)
    torch_target = _conditional_target(torch_estimator, station)
    np.testing.assert_allclose(
        numpy_target,
        torch_target,
        rtol=2.0e-12,
        atol=1.0e-8,
    )


def test_cached_fixed_state_transport_matches_direct_batch_kernel() -> None:
    """Station caching must preserve the former direct transport arithmetic."""
    estimator = _estimator(use_gpu=True)
    station = _station()
    layout = estimator._joint_line_layout()
    for isotope in estimator.joint_isotope_order():
        global_columns, local_indices, branching_weights = layout[isotope]
        filt = estimator.filters[isotope]
        direct = (
            filt._continuous_expected_line_transport_components_pair_sequence_torch(
                pose_idx=int(station.pose_idx),
                fe_indices=station.fe_indices,
                pb_indices=station.pb_indices,
                live_times_s=station.live_times_s,
                positive_line_indices=local_indices,
            )
        )
        cached_total, cached_uncollided, cached_features = (
            estimator._joint_isotope_station_transport_components_torch(
                station,
                isotope,
            )
        )
        branch = branching_weights.reshape(1, 1, 1, -1)
        np.testing.assert_allclose(
            cached_total[..., global_columns].detach().cpu().numpy(),
            direct.total_kernel.detach().cpu().numpy() * branch,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            cached_uncollided[..., global_columns].detach().cpu().numpy(),
            direct.uncollided_kernel.detach().cpu().numpy() * branch,
            rtol=0.0,
            atol=0.0,
        )
        direct_features = np.stack(
            [
                direct.tau_fe.detach().cpu().numpy(),
                direct.tau_pb.detach().cpu().numpy(),
                direct.tau_obstacle.detach().cpu().numpy(),
                direct.distance_m.detach().cpu().numpy(),
            ],
            axis=-1,
        )
        np.testing.assert_allclose(
            cached_features[..., global_columns, :]
            .detach()
            .cpu()
            .numpy(),
            direct_features,
            rtol=0.0,
            atol=0.0,
        )


def test_exact_rj_candidate_target_matches_numpy_and_cuda() -> None:
    """The production CUDA target must preserve the NumPy PF target."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    station = _station()
    numpy_estimator = _estimator(use_gpu=False)
    cuda_estimator = _estimator(use_gpu=True, gpu_device="cuda")

    numpy_target = _conditional_target(numpy_estimator, station)
    cuda_target = _conditional_target(cuda_estimator, station)
    np.testing.assert_allclose(
        numpy_target,
        cuda_target,
        rtol=2.0e-12,
        atol=1.0e-8,
    )


def test_torch_rj_target_keeps_history_resident_and_reuses_unit_transport() -> None:
    """Torch RJ must retain history tensors and cache exact surface responses."""
    torch = pytest.importorskip("torch")
    station = _station()
    estimator = _estimator(use_gpu=True)
    estimator._active_joint_station_history = (station,)
    estimator._refresh_joint_structural_transport_cache((station,))
    cache = estimator._joint_structural_transport_cache
    assert cache is not None
    assert all(torch.is_tensor(value) for value in cache)
    assert all(str(value.device) == "cpu" for value in cache)
    estimator._joint_structural_transport_cache = None
    estimator._active_joint_station_history = None

    first = _conditional_target(estimator, station)
    first_hits = estimator.last_joint_structural_unit_cache_hits
    first_misses = estimator.last_joint_structural_unit_cache_misses
    second = _conditional_target(estimator, station)

    assert first_misses > 0
    assert estimator.last_joint_structural_unit_cache_hits > first_hits
    assert estimator.last_joint_structural_unit_cache_misses == first_misses
    np.testing.assert_array_equal(second, first)


def test_unit_transport_cache_preserves_completed_station_shards() -> None:
    """Appending a station must reuse exact responses from completed stations."""
    estimator = _estimator(use_gpu=True)
    filt = estimator.filters["Cs-137"]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    chart_ids = np.asarray([0, 1], dtype=np.int64)
    surface_uv = np.asarray(
        [[0.2, 0.3], [0.7, 0.8]],
        dtype=np.float64,
    )
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    local_indices = estimator._joint_line_layout()["Cs-137"][1]
    first_station = _station()
    second_station = replace(
        first_station,
        fe_indices=np.asarray([1, 4], dtype=np.int64),
        pb_indices=np.asarray([6, 0], dtype=np.int64),
        station_sequence_id=1,
    )
    first_geometry = estimator._joint_history_structural_geometry(
        "Cs-137",
        (first_station,),
    )
    combined_geometry = estimator._joint_history_structural_geometry(
        "Cs-137",
        (first_station, second_station),
    )

    first = estimator._joint_cached_continuous_unit_components(
        filt=filt,
        data=first_geometry,
        positions_s3=positions,
        chart_ids_s=chart_ids,
        positive_line_indices=local_indices,
    )
    first_misses = estimator.last_joint_structural_unit_cache_misses
    combined = estimator._joint_cached_continuous_unit_components(
        filt=filt,
        data=combined_geometry,
        positions_s3=positions,
        chart_ids_s=chart_ids,
        positive_line_indices=local_indices,
    )

    assert estimator.last_joint_structural_unit_cache_hits >= chart_ids.size
    assert (
        estimator.last_joint_structural_unit_cache_misses
        == first_misses + chart_ids.size
    )
    for first_values, combined_values in zip(
        first,
        combined,
        strict=True,
    ):
        np.testing.assert_array_equal(
            combined_values[: first_geometry.row_count],
            first_values,
        )

    misses_before_repeat = estimator.last_joint_structural_unit_cache_misses
    repeated = estimator._joint_cached_continuous_unit_components(
        filt=filt,
        data=combined_geometry,
        positions_s3=positions,
        chart_ids_s=chart_ids,
        positive_line_indices=local_indices,
    )
    assert (
        estimator.last_joint_structural_unit_cache_misses
        == misses_before_repeat
    )
    for repeated_values, combined_values in zip(
        repeated,
        combined,
        strict=True,
    ):
        np.testing.assert_array_equal(repeated_values, combined_values)


def test_raw_spectrum_joint_smc_concentrates_on_physical_truth() -> None:
    """An exact raw-spectrum update must retain truth support and target ESS."""
    model = approved_full_spectrum_model()
    particle_count = 48
    estimator = RotatingShieldPFEstimator(
        isotopes=TEST_ISOTOPES,
        surface_diagnostic_points=np.asarray(
            [[0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in TEST_ISOTOPES},
        line_mu_by_isotope=_positive_line_transport_table(),
        pf_config=RotatingShieldPFConfig(
            num_particles=particle_count,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            structural_rj_surface_chart_max_edge_m=2.0,
            strength_prior_min_cps_1m=1_000.0,
            strength_prior_max_cps_1m=10_000.0,
            structural_cardinality_prior_mean=1.0,
            target_ess_ratio=0.25,
            max_temper_steps=64,
            min_delta_beta=1.0e-8,
        ),
        full_spectrum_generative_model=model,
        random_seed=1_827,
    )
    detector_position = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    truth_position = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
    far_position = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    estimator.add_measurement_pose(detector_position)
    estimator._ensure_kernel_cache()
    cs_filter = estimator.filters["Cs-137"]
    atlas = cs_filter._structural_rj_surface_atlas
    assert atlas is not None

    def _empty_state() -> IsotopeState:
        """Return one zero-source continuous-surface state."""
        return IsotopeState(
            num_sources=0,
            strengths=np.empty(0, dtype=np.float64),
            surface_chart_ids=np.empty(0, dtype=np.int64),
            surface_uv=np.empty((0, 2), dtype=np.float64),
        )

    def _source_state(position: np.ndarray) -> IsotopeState:
        """Return one 5 kcps source at an exact continuous surface point."""
        chart_ids, surface_uv = atlas.locate_positions(
            np.asarray(position, dtype=np.float64).reshape(1, 3)
        )
        return IsotopeState(
            num_sources=1,
            strengths=np.asarray([5_000.0], dtype=np.float64),
            surface_chart_ids=chart_ids,
            surface_uv=surface_uv,
        )

    common_log_weight = -math.log(particle_count)
    for isotope in TEST_ISOTOPES:
        filt = estimator.filters[isotope]
        row_identities = [
            particle.joint_row_identity
            for particle in filt.continuous_particles
        ]
        if isotope == "Cs-137":
            states = (
                [_empty_state() for _ in range(16)]
                + [_source_state(truth_position) for _ in range(16)]
                + [_source_state(far_position) for _ in range(16)]
            )
        else:
            states = [_empty_state() for _ in range(particle_count)]
        filt.continuous_particles = [
                IsotopeParticle(
                    state=state,
                    log_weight=common_log_weight,
                    joint_row_identity=row_identities[row],
                )
                for row, state in enumerate(states)
            ]

    fe_indices = np.arange(8, dtype=np.int64)
    pb_indices = np.arange(7, -1, -1, dtype=np.int64)
    live_times = np.full(8, 2.0, dtype=np.float64)
    empty_observation = JointStationObservation(
        spectrum_vb=np.zeros(
            (8, int(model.energy_axis_keV.size)),
            dtype=np.int64,
        ),
        energy_axis_keV=model.energy_axis_keV,
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=tuple(detector_position.tolist()),
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_times_s=live_times,
        station_sequence_id=0,
    )
    total, uncollided, features = (
        estimator._joint_station_transport_components_torch(
            empty_observation
        )
    )
    observed = model.sample_predictive_numpy(
        total[16:17].detach().cpu().numpy(),
        uncollided[16:17].detach().cpu().numpy(),
        features[16:17].detach().cpu().numpy(),
        live_times,
        sample_count=1,
        rng=np.random.default_rng(444),
    )[0, 0]
    assert np.issubdtype(observed.dtype, np.integer)
    station = JointStationObservation(
        spectrum_vb=observed,
        energy_axis_keV=model.energy_axis_keV,
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=tuple(detector_position.tolist()),
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_times_s=live_times,
        station_sequence_id=0,
    )
    likelihood = (
        estimator._joint_station_log_likelihood_torch(station)
        .detach()
        .cpu()
        .numpy()
    )
    assert np.all(np.isfinite(likelihood))
    assert likelihood[16] > likelihood[32]

    records = tuple(
        (
            observed[view_index],
            int(fe_indices[view_index]),
            int(pb_indices[view_index]),
            float(live_times[view_index]),
        )
        for view_index in range(8)
    )
    estimator.update_spectrum_station(
        records,
        pose_idx=0,
        generative_contract_hash_sha256=model.contract_hash_sha256,
    )

    weights = np.exp(
        np.asarray(
            [
                particle.log_weight
                for particle in cs_filter.continuous_particles
            ],
            dtype=np.float64,
        )
    )
    weights /= float(np.sum(weights))
    cardinality_one_mass = 0.0
    truth_neighborhood_mass = 0.0
    for particle, weight in zip(
        cs_filter.continuous_particles,
        weights,
        strict=True,
    ):
        if particle.state.num_sources != 1:
            continue
        cardinality_one_mass += float(weight)
        inferred = cs_filter.continuous_state_positions(
            particle.state
        )[0]
        if float(np.linalg.norm(inferred - truth_position)) < 0.5:
            truth_neighborhood_mass += float(weight)

    assert cardinality_one_mass > 0.99
    assert truth_neighborhood_mass > 0.99
    assert cs_filter.last_ess >= 0.25 * particle_count - 1.0e-9
    assert estimator.last_joint_temper_steps[-1]["beta_total"] == 1.0
    assert (
        1
        <= estimator.last_joint_station_unique_ancestor_count
        <= particle_count
    )

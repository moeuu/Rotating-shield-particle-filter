"""CPU/Torch equivalence tests for pure-PF exact-RJ candidate targets."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import pf.estimator as estimator_module
import pf.estimator_likelihood as estimator_likelihood_module
from pf.estimator import (
    JointStationObservation,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.particle_filter import IsotopeParticle, StructuralGeometryBatch
from pf.joint_transport_cache import JointTransportCache
from pf.state import IsotopeState
from pf.strength_prior import BoundedUniformStrengthPriorTestConfig
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
    num_particles: int = 2,
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
            num_particles=num_particles,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            position_max=(3.0, 3.0, 3.0),
        ),
        detector_radius_m=0.025,
        detector_aperture_radius_m=0.0395,
        detector_aperture_samples=33,
        full_spectrum_generative_model=model,
        random_seed=17,
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5], dtype=np.float64))
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


def _set_single_source_states(
    estimator: RotatingShieldPFEstimator,
    isotope: str,
    *,
    strength_cps_1m: float,
    surface_u: float = 0.25,
) -> None:
    """Set deterministic aligned one-source rows without changing identities."""
    filt = estimator.filters[isotope]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    for row, particle in enumerate(filt.continuous_particles):
        particle.state = IsotopeState(
            num_sources=1,
            strengths=np.asarray([strength_cps_1m], dtype=np.float64),
            surface_chart_ids=np.asarray(
                [row % int(atlas.chart_count)],
                dtype=np.int64,
            ),
            surface_uv=np.asarray([[surface_u, 0.5]], dtype=np.float64),
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

    numpy_components = numpy_estimator._joint_station_transport_components_torch(
        station
    )
    torch_components = torch_estimator._joint_station_transport_components_torch(
        station
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
    assert isinstance(
        torch_estimator._joint_persistent_structural_transport_cache,
        JointTransportCache,
    )
    assert torch_estimator.last_joint_slot_overlay_likelihood_calls > 0
    assert torch_estimator.last_joint_full_history_clone_count == 0
    diagnostics = (
        torch_estimator._full_spectrum_model().last_torch_slot_overlay_diagnostics
    )
    assert diagnostics["mode"] == "bounded_exact_slot_overlay"
    assert diagnostics["full_history_clone_count"] == 0


def test_station_cache_signature_binds_observed_spectrum() -> None:
    """Transport and station-likelihood identity must bind MeasurementLog data."""
    station = _station()
    changed_spectrum = np.asarray(station.spectrum_vb).copy()
    changed_spectrum[0, 0] += 1
    changed = replace(station, spectrum_vb=changed_spectrum)

    assert (
        RotatingShieldPFEstimator._joint_station_cache_signature(station)
        != RotatingShieldPFEstimator._joint_station_cache_signature(changed)
    )


def test_standard_joint_estimator_installs_exact_target_for_every_isotope() -> None:
    """Production filters must use the estimator-owned joint exact target."""
    estimator = _estimator(use_gpu=True, gpu_device="cpu")
    expected = estimator._joint_structural_target_evaluator.__func__
    for filt in estimator.filters.values():
        evaluator = filt._joint_target_evaluator
        assert evaluator is not None
        assert evaluator.__self__ is estimator
        assert evaluator.__func__ is expected


def test_exact_unit_cache_key_binds_geometry_content_not_object_identity() -> None:
    """Exact unit caches must bind geometry content, not transient object ID."""
    pytest.importorskip("torch")
    estimator = _estimator(use_gpu=True, gpu_device="cpu")
    station = _station()
    stations = (station,)
    isotope = "Cs-137"
    filt = estimator.filters[isotope]
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    cache = estimator._joint_structural_transport_cache
    assert isinstance(cache, JointTransportCache)
    geometry = estimator._joint_history_structural_geometry(isotope, stations)
    geometry_copy = StructuralGeometryBatch(
        detector_positions=geometry.detector_positions.copy(),
        fe_indices=geometry.fe_indices.copy(),
        pb_indices=geometry.pb_indices.copy(),
        live_times=geometry.live_times.copy(),
        station_sequence_ids=geometry.station_sequence_ids.copy(),
    )
    changed_fe = geometry.fe_indices.copy()
    changed_fe[0] = (int(changed_fe[0]) + 1) % 8
    changed = StructuralGeometryBatch(
        detector_positions=geometry.detector_positions.copy(),
        fe_indices=changed_fe,
        pb_indices=geometry.pb_indices.copy(),
        live_times=geometry.live_times.copy(),
        station_sequence_ids=geometry.station_sequence_ids.copy(),
    )
    local_lines = estimator._joint_line_layout()[isotope][1]
    first = estimator._joint_cuda_accepted_unit_cache_entry(
        filt=filt,
        data=geometry,
        positive_line_indices=local_lines,
        reference=cache[0],
    )
    equivalent = estimator._joint_cuda_accepted_unit_cache_entry(
        filt=filt,
        data=geometry_copy,
        positive_line_indices=local_lines,
        reference=cache[0],
    )
    distinct = estimator._joint_cuda_accepted_unit_cache_entry(
        filt=filt,
        data=changed,
        positive_line_indices=local_lines,
        reference=cache[0],
    )
    assert equivalent is first
    assert distinct is not first


def test_staged_accepted_transport_commits_without_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted exact proposal columns must become the global cache directly."""
    pytest.importorskip("torch")
    estimator = _estimator(use_gpu=True, gpu_device="cpu")
    isotope = "Cs-137"
    filt = estimator.filters[isotope]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
        surface_u=0.2,
    )
    station = _station()
    stations = (station,)
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    evidence = estimator._joint_history_structural_geometry(isotope, stations)
    sweep_entry_state = estimator._joint_isotope_cache_state(filt)
    chart_ids = np.asarray([[2], [3]], dtype=np.int64)
    surface_uv = np.asarray(
        [[[0.65, 0.35]], [[0.75, 0.45]]],
        dtype=np.float64,
    )
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    strengths = np.asarray([[7_500.0], [9_500.0]], dtype=np.float64)
    estimator._joint_structural_target_evaluator(
        filt=filt,
        data=evidence,
        positions_pks=positions,
        chart_ids_pk=chart_ids,
        strengths_pk=strengths,
        particle_indices=np.arange(2, dtype=np.int64),
        target_beta=1.0,
        tempering_start_row=0,
        stage_unit_transport=True,
    )
    for row, particle in enumerate(filt.continuous_particles):
        particle.state = IsotopeState(
            num_sources=1,
            strengths=strengths[row].copy(),
            surface_chart_ids=chart_ids[row].copy(),
            surface_uv=surface_uv[row].copy(),
        )

    expected = estimator._joint_isotope_station_transport_components_torch(
        station,
        isotope,
    )

    def _forbid_recomputation(*args: object, **kwargs: object) -> object:
        """Fail if the post-sweep path launches accepted-state transport."""
        del args, kwargs
        raise AssertionError("accepted transport was recomputed")

    monkeypatch.setattr(
        estimator,
        "_joint_isotope_station_transport_components_torch",
        _forbid_recomputation,
    )
    estimator._joint_commit_staged_cuda_transport_cache_isotope(
        filt=filt,
        data=evidence,
        stations=stations,
        particle_indices=np.arange(2, dtype=np.int64),
        sweep_entry_state=sweep_entry_state,
    )

    committed = estimator._joint_structural_transport_cache
    assert committed is not None
    order = estimator.joint_isotope_order()
    slot_start = order.index(isotope) * estimator.pf_config.cardinality_capacity
    slot_stop = slot_start + estimator.pf_config.cardinality_capacity
    for committed_values, expected_values in zip(
        committed,
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(
            committed_values[:, :, slot_start:slot_stop]
            .detach()
            .cpu()
            .numpy(),
            expected_values.detach().cpu().numpy(),
            rtol=2.0e-12,
            atol=1.0e-12,
        )
    assert estimator.last_joint_staged_transport_commit_rows == 2
    assert isinstance(committed, JointTransportCache)
    assert committed.state_sha256 == estimator._joint_structural_state_sha256()


@pytest.mark.parametrize("gpu_device", ["cpu", "cuda"])
def test_strength_only_commit_reuses_exact_sweep_entry_transport(
    gpu_device: str,
) -> None:
    """A strength-only change must reuse exact accepted geometry transport."""
    torch = pytest.importorskip("torch")
    if gpu_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    estimator = _estimator(use_gpu=True, gpu_device=gpu_device)
    isotope = "Cs-137"
    filt = estimator.filters[isotope]
    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
    )
    station = _station()
    stations = (station,)
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    evidence = estimator._joint_history_structural_geometry(isotope, stations)
    cache = estimator._joint_structural_transport_cache
    assert isinstance(cache, JointTransportCache)
    sweep_entry_state = estimator._joint_isotope_cache_state(filt)
    if gpu_device == "cuda":
        assert filt._begin_continuous_rj_station_device_state(cache[0])
    else:
        filt._initialize_continuous_rj_device_state(cache[0])
    state = filt._structural_rj_device_state
    assert state is not None
    before = tuple(
        value[0].clone()
        for value in cache
    )
    if gpu_device == "cuda":
        state["strengths"][0, 0] *= 1.75
    else:
        original_state = filt.continuous_particles[0].state
        filt.continuous_particles[0].state = IsotopeState(
            num_sources=1,
            strengths=original_state.strengths * 1.75,
            surface_chart_ids=original_state.surface_chart_ids.copy(),
            surface_uv=original_state.surface_uv.copy(),
        )

    estimator._joint_commit_staged_cuda_transport_cache_isotope(
        filt=filt,
        data=evidence,
        stations=stations,
        particle_indices=np.asarray([0], dtype=np.int64),
        sweep_entry_state=sweep_entry_state,
    )

    line_columns = estimator._joint_line_layout()[isotope][0]
    slot_start = estimator.joint_isotope_order().index(isotope) * int(
        filt.config.hard_max_sources
    )
    np.testing.assert_allclose(
        cache[0][0, :, slot_start, line_columns].detach().cpu().numpy(),
        before[0][:, slot_start, line_columns].detach().cpu().numpy() * 1.75,
        rtol=2.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        cache[1][0, :, slot_start, line_columns].detach().cpu().numpy(),
        before[1][:, slot_start, line_columns].detach().cpu().numpy() * 1.75,
        rtol=2.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        cache[2][0, :, slot_start, line_columns].detach().cpu().numpy(),
        before[2][:, slot_start, line_columns].detach().cpu().numpy(),
    )


def test_staged_transport_commit_fails_closed_for_unknown_geometry() -> None:
    """A geometry absent from both exact caches must fail without recomputing."""
    pytest.importorskip("torch")
    estimator = _estimator(use_gpu=True, gpu_device="cpu")
    isotope = "Cs-137"
    filt = estimator.filters[isotope]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
    )
    station = _station()
    stations = (station,)
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    evidence = estimator._joint_history_structural_geometry(isotope, stations)
    cache = estimator._joint_structural_transport_cache
    assert isinstance(cache, JointTransportCache)
    sweep_entry_state = estimator._joint_isotope_cache_state(filt)
    filt._initialize_continuous_rj_device_state(cache[0])
    state = filt._structural_rj_device_state
    assert state is not None
    original_state = filt.continuous_particles[0].state
    changed_uv = original_state.surface_uv[0].copy()
    changed_uv[0] = min(float(changed_uv[0]) + 0.125, 0.95)
    filt.continuous_particles[0].state = IsotopeState(
        num_sources=1,
        strengths=original_state.strengths.copy(),
        surface_chart_ids=original_state.surface_chart_ids.copy(),
        surface_uv=changed_uv.reshape(1, 2),
    )

    with pytest.raises(
        RuntimeError,
        match="lacks exact cached unit transport",
    ):
        estimator._joint_commit_staged_cuda_transport_cache_isotope(
            filt=filt,
            data=evidence,
            stations=stations,
            particle_indices=np.asarray([0], dtype=np.int64),
            sweep_entry_state=sweep_entry_state,
        )


@pytest.mark.parametrize(
    ("use_gpu", "gpu_device"),
    [(False, "cpu"), (True, "cpu"), (True, "cuda")],
)
def test_sparse_strength_grid_matches_expanded_exact_target(
    use_gpu: bool,
    gpu_device: str,
) -> None:
    """Differential source-axis assembly must equal the expanded target."""
    if use_gpu:
        torch = pytest.importorskip("torch")
        if gpu_device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available.")
    estimator = _estimator(use_gpu=use_gpu, gpu_device=gpu_device)
    station = _station()
    stations = (station,)
    isotope = "Cs-137"
    filt = estimator.filters[isotope]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    chart_ids = np.asarray([[0], [1]], dtype=np.int64)
    surface_uv = np.asarray(
        [[[0.2, 0.3]], [[0.7, 0.8]]],
        dtype=np.float64,
    )
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    strengths = np.asarray(
        [
            [[300_000.0], [600_000.0], [900_000.0]],
            [[400_000.0], [800_000.0], [1_200_000.0]],
        ],
        dtype=np.float64,
    )
    indices = np.arange(2, dtype=np.int64)
    geometry = estimator._joint_history_structural_geometry(isotope, stations)
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    try:
        sparse = estimator._joint_structural_strength_grid_target_batch(
            filt=filt,
            data=geometry,
            stations=stations,
            positions_bks=positions,
            chart_ids_bk=chart_ids,
            strengths_bgk=strengths,
            particle_indices=indices,
            target_beta=0.7,
        )
        grid_count = strengths.shape[1]
        expanded = estimator._joint_structural_target_evaluator(
            filt=filt,
            data=geometry,
            positions_pks=np.broadcast_to(
                positions[:, None],
                (2, grid_count, 1, 3),
            ).reshape(2 * grid_count, 1, 3),
            chart_ids_pk=np.broadcast_to(
                chart_ids[:, None],
                (2, grid_count, 1),
            ).reshape(2 * grid_count, 1),
            strengths_pk=strengths.reshape(2 * grid_count, 1),
            particle_indices=np.repeat(indices, grid_count),
            target_beta=0.7,
            tempering_start_row=0,
        ).reshape(2, grid_count)
    finally:
        estimator._joint_structural_transport_cache = None
        estimator._active_joint_station_history = None

    np.testing.assert_allclose(sparse, expanded, rtol=2.0e-12, atol=1.0e-8)
    assert estimator.last_joint_strength_grid_source_slots_after <= (
        estimator.last_joint_strength_grid_source_slots_before
    )


def test_cuda_strength_grid_autotunes_real_batches_above_128() -> None:
    """CUDA must time real 128/256/512 row slabs before caching a width."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    particle_count = 896
    estimator = _estimator(
        use_gpu=True,
        gpu_device="cuda",
        num_particles=particle_count,
    )
    station = _station()
    stations = (station,)
    isotope = "Cs-137"
    filt = estimator.filters[isotope]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    chart_ids = (np.arange(particle_count, dtype=np.int64) % 2)[:, None]
    phase = np.arange(particle_count, dtype=np.float64)
    surface_uv = np.stack(
        (
            0.1 + 0.8 * ((phase % 97.0) / 97.0),
            0.1 + 0.8 * ((phase % 89.0) / 89.0),
        ),
        axis=1,
    )[:, None, :]
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    strength_scale = np.linspace(
        0.75,
        1.25,
        particle_count,
        dtype=np.float64,
    )
    strengths = (
        strength_scale[:, None, None]
        * np.asarray([250_000.0, 500_000.0, 750_000.0, 1_000_000.0, 1_250_000.0])[
            None, :, None
        ]
    )
    indices = np.arange(particle_count, dtype=np.int64)
    geometry = estimator._joint_history_structural_geometry(isotope, stations)
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    try:
        target = estimator._joint_structural_strength_grid_target_evaluator(
            filt=filt,
            data=geometry,
            positions_pks=positions,
            chart_ids_pk=chart_ids,
            strengths_pgk=strengths,
            particle_indices=indices,
            target_beta=0.7,
            tempering_start_row=0,
        )
        first_diagnostics = dict(
            estimator.last_joint_strength_grid_batch_diagnostics
        )
        repeated = estimator._joint_structural_strength_grid_target_evaluator(
            filt=filt,
            data=geometry,
            positions_pks=positions[:10],
            chart_ids_pk=chart_ids[:10],
            strengths_pgk=strengths[:10],
            particle_indices=indices[:10],
            target_beta=0.7,
            tempering_start_row=0,
        )
        cached_diagnostics = dict(
            estimator.last_joint_strength_grid_batch_diagnostics
        )
    finally:
        filt._clear_continuous_rj_device_state()
        estimator._joint_structural_transport_cache = None
        estimator._active_joint_station_history = None
    assert target.shape == (particle_count, 5)
    assert np.all(np.isfinite(target))
    np.testing.assert_allclose(
        repeated,
        target[:10],
        rtol=5.0e-12,
        atol=1.0e-10,
    )
    assert first_diagnostics["mode"] == "empirical_cuda_autotune"
    trials = first_diagnostics["trials"]
    assert isinstance(trials, list)
    assert [trial["batch_size"] for trial in trials] == [128, 256, 512]
    selected = first_diagnostics["selected_batch_size"]
    assert selected in {128, 256, 512}
    assert cached_diagnostics["mode"] == "cached_cuda_autotune"
    assert cached_diagnostics["selected_batch_size"] == selected


def test_device_delta_reuses_unchanged_positions_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted GPU columns may replace recomputation for unchanged sources."""
    pytest.importorskip("torch")
    station = _station()
    numpy_estimator = _estimator(use_gpu=False)
    torch_estimator = _estimator(use_gpu=True)
    chart_ids = np.asarray([[0], [1]], dtype=np.int64)
    surface_uv = np.asarray(
        [[[0.2, 0.3]], [[0.7, 0.8]]],
        dtype=np.float64,
    )
    accepted_strengths = np.asarray([700_000.0, 900_000.0])
    for estimator in (numpy_estimator, torch_estimator):
        filt = estimator.filters["Cs-137"]
        for row in range(2):
            filt.continuous_particles[row].state = IsotopeState(
                num_sources=1,
                strengths=np.asarray([accepted_strengths[row]]),
                surface_chart_ids=chart_ids[row].copy(),
                surface_uv=surface_uv[row].copy(),
            )

    def _evaluate(estimator: RotatingShieldPFEstimator) -> np.ndarray:
        """Evaluate a strength change at unchanged accepted positions."""
        filt = estimator.filters["Cs-137"]
        atlas = filt._structural_rj_surface_atlas
        assert atlas is not None
        positions = atlas.positions_xyz(chart_ids, surface_uv)
        stations = (station,)
        estimator._active_joint_station_history = stations
        estimator._refresh_joint_structural_transport_cache(stations)
        evidence = estimator._joint_history_structural_geometry(
            "Cs-137",
            stations,
        )
        try:
            return estimator._joint_structural_target_evaluator(
                filt=filt,
                data=evidence,
                positions_pks=positions,
                chart_ids_pk=chart_ids,
                strengths_pk=(accepted_strengths * 1.2).reshape(2, 1),
                particle_indices=np.arange(2, dtype=np.int64),
                target_beta=0.7,
                tempering_start_row=0,
            )
        finally:
            estimator._joint_structural_transport_cache = None
            estimator._active_joint_station_history = None

    expected = _evaluate(numpy_estimator)
    torch_filter = torch_estimator.filters["Cs-137"]
    original = torch_filter._continuous_rj_line_transport_component_columns
    device_calls = 0

    def _capture(*args: object, **kwargs: object) -> object:
        """Count proposal transport calls after accepted-cache creation."""
        nonlocal device_calls
        if bool(kwargs.get("device_resident", False)):
            device_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        torch_filter,
        "_continuous_rj_line_transport_component_columns",
        _capture,
    )
    actual = _evaluate(torch_estimator)

    assert device_calls == 0
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=1.0e-8)


def test_sweep_local_unit_cache_commits_only_accepted_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact accepted-proposal columns may enter the sweep-local cache."""
    pytest.importorskip("torch")
    estimator = _estimator(use_gpu=True)
    station = _station()
    filt = estimator.filters["Cs-137"]
    empty = IsotopeState(
        num_sources=0,
        strengths=np.empty(0, dtype=np.float64),
        surface_chart_ids=np.empty(0, dtype=np.int64),
        surface_uv=np.empty((0, 2), dtype=np.float64),
    )
    for particle in filt.continuous_particles:
        particle.state = empty.copy()
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    chart_ids = np.asarray([[0], [1]], dtype=np.int64)
    surface_uv = np.asarray(
        [[[0.2, 0.3]], [[0.7, 0.8]]],
        dtype=np.float64,
    )
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    strengths = np.asarray([[700_000.0], [900_000.0]], dtype=np.float64)
    indices = np.arange(2, dtype=np.int64)
    stations = (station,)
    estimator._active_joint_station_history = stations
    estimator._refresh_joint_structural_transport_cache(stations)
    evidence = estimator._joint_history_structural_geometry("Cs-137", stations)
    original = filt._continuous_rj_line_transport_component_columns
    call_rows: list[int] = []

    def _capture(*args: object, **kwargs: object) -> object:
        """Record the number of source geometries sent to transport."""
        positions_arg = np.asarray(args[1], dtype=np.float64).reshape(-1, 3)
        if bool(kwargs.get("device_resident", False)):
            call_rows.append(int(positions_arg.shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        filt,
        "_continuous_rj_line_transport_component_columns",
        _capture,
    )
    try:
        first = estimator._joint_structural_target_evaluator(
            filt=filt,
            data=evidence,
            positions_pks=positions,
            chart_ids_pk=chart_ids,
            strengths_pk=strengths,
            particle_indices=indices,
            target_beta=0.7,
            tempering_start_row=0,
        )
        estimator._joint_structural_target_evaluator(
            filt=filt,
            data=evidence,
            positions_pks=positions[:1],
            chart_ids_pk=chart_ids[:1],
            strengths_pk=strengths[:1],
            particle_indices=indices[:1],
            target_beta=0.7,
            tempering_start_row=0,
            stage_unit_transport=True,
        )
        filt._commit_continuous_rj_states(
            indices,
            np.asarray([True, False]),
            chart_ids,
            surface_uv,
            positions,
            strengths,
        )
        second = estimator._joint_structural_target_evaluator(
            filt=filt,
            data=evidence,
            positions_pks=positions,
            chart_ids_pk=chart_ids,
            strengths_pk=strengths,
            particle_indices=indices,
            target_beta=0.7,
            tempering_start_row=0,
        )
    finally:
        estimator._joint_structural_transport_cache = None
        estimator._active_joint_station_history = None
        filt._clear_continuous_rj_device_state()

    assert call_rows == [2, 1, 1]
    np.testing.assert_allclose(second, first, rtol=2.0e-12, atol=1.0e-8)


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
        direct_features = np.concatenate(
            (
                np.stack(
                    [
                        direct.tau_fe.detach().cpu().numpy(),
                            direct.tau_pb.detach().cpu().numpy(),
                            direct.tau_obstacle.detach().cpu().numpy(),
                            direct.tau_obstacle_compton.detach().cpu().numpy(),
                            direct.distance_m.detach().cpu().numpy(),
                    ],
                    axis=-1,
                ),
                direct.uncollided_impact_fractions.detach().cpu().numpy(),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(
            cached_features[..., global_columns, :].detach().cpu().numpy(),
            direct_features,
            rtol=0.0,
            atol=0.0,
        )


def test_cuda_device_unit_cache_reuses_geometry_across_strengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm CUDA mirror must skip host assembly and rescale exactly."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    estimator = _estimator(
        use_gpu=True,
        gpu_device="cuda",
        num_particles=8,
    )
    isotope = "Cs-137"
    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
    )
    original = estimator._joint_cached_continuous_unit_components
    host_assembly_calls = 0

    def _counted_host_assembly(*args: object, **kwargs: object) -> object:
        """Count host-cache assembly without changing transport values."""
        nonlocal host_assembly_calls
        host_assembly_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        estimator,
        "_joint_cached_continuous_unit_components",
        _counted_host_assembly,
    )
    station = _station()
    first = estimator._joint_isotope_station_transport_components_torch(
        station,
        isotope,
    )
    second = estimator._joint_isotope_station_transport_components_torch(
        station,
        isotope,
    )
    assert host_assembly_calls == 1
    assert estimator.last_joint_device_unit_cache_misses == 1
    assert estimator.last_joint_device_unit_cache_hits == 1
    assert len(estimator._joint_device_unit_transport_cache) == 1
    cache_key = next(iter(estimator._joint_device_unit_transport_cache))
    assert cache_key[-2:] == (
        f"cuda:{torch.cuda.current_device()}",
        "torch.float64",
    )
    for first_values, second_values in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            second_values.detach().cpu().numpy(),
            first_values.detach().cpu().numpy(),
        )

    for particle in estimator.filters[isotope].continuous_particles:
        particle.state.strengths[:] = 6_000.0
    rescaled = estimator._joint_isotope_station_transport_components_torch(
        station,
        isotope,
    )
    assert host_assembly_calls == 1
    assert estimator.last_joint_device_unit_cache_hits == 2
    np.testing.assert_allclose(
        rescaled[0].detach().cpu().numpy(),
        1.2 * first[0].detach().cpu().numpy(),
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rescaled[1].detach().cpu().numpy(),
        1.2 * first[1].detach().cpu().numpy(),
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        rescaled[2].detach().cpu().numpy(),
        first[2].detach().cpu().numpy(),
    )

    changed_station = replace(
        station,
        fe_indices=np.asarray([1, 4], dtype=np.int64),
        pb_indices=np.asarray([6, 0], dtype=np.int64),
    )
    estimator._joint_isotope_station_transport_components_torch(
        changed_station,
        isotope,
    )
    assert host_assembly_calls == 2
    assert estimator.last_joint_device_unit_cache_misses == 2

    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=6_000.0,
        surface_u=0.75,
    )
    estimator._joint_isotope_station_transport_components_torch(station, isotope)
    assert host_assembly_calls == 3
    assert estimator.last_joint_device_unit_cache_misses == 3


def test_torch_cpu_transport_does_not_retain_device_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch-CPU must keep the existing host LRU without a dense duplicate."""
    pytest.importorskip("torch")
    estimator = _estimator(use_gpu=True, gpu_device="cpu", num_particles=8)
    isotope = "Cs-137"
    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
    )
    original = estimator._joint_cached_continuous_unit_components
    host_assembly_calls = 0

    def _counted_host_assembly(*args: object, **kwargs: object) -> object:
        """Count host assembly without changing CPU transport values."""
        nonlocal host_assembly_calls
        host_assembly_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        estimator,
        "_joint_cached_continuous_unit_components",
        _counted_host_assembly,
    )
    first = estimator._joint_isotope_station_transport_components_torch(
        _station(),
        isotope,
    )
    second = estimator._joint_isotope_station_transport_components_torch(
        _station(),
        isotope,
    )
    assert host_assembly_calls == 2
    assert estimator._joint_device_unit_transport_cache == {}
    assert estimator.last_joint_device_unit_cache_hits == 0
    assert estimator.last_joint_device_unit_cache_misses == 0
    for first_values, second_values in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            second_values.detach().numpy(),
            first_values.detach().numpy(),
        )


def test_device_unit_transport_cache_matches_torch_cpu_and_cuda() -> None:
    """The CUDA mirror must preserve the exact Torch-CPU transport model."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    station = _station()
    results = []
    for device_name in ("cpu", "cuda"):
        estimator = _estimator(
            use_gpu=True,
            gpu_device=device_name,
            num_particles=8,
        )
        _set_single_source_states(
            estimator,
            "Cs-137",
            strength_cps_1m=5_000.0,
        )
        estimator._joint_isotope_station_transport_components_torch(
            station,
            "Cs-137",
        )
        warm = estimator._joint_isotope_station_transport_components_torch(
            station,
            "Cs-137",
        )
        expected_hits = 0 if device_name == "cpu" else 1
        assert estimator.last_joint_device_unit_cache_hits == expected_hits
        assert bool(estimator._joint_device_unit_transport_cache) == (
            device_name == "cuda"
        )
        results.append(
            tuple(value.detach().cpu().numpy() for value in warm)
        )
    for cpu_values, cuda_values in zip(results[0], results[1], strict=True):
        np.testing.assert_allclose(
            cuda_values,
            cpu_values,
            rtol=2.0e-12,
            atol=1.0e-12,
        )


def test_device_unit_transport_cache_is_content_invalidated_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed geometry must miss and LRU eviction must enforce its byte cap."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    estimator = _estimator(use_gpu=True, gpu_device="cuda", num_particles=8)
    isotope = "Cs-137"
    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
        surface_u=0.25,
    )
    station = _station()
    original = estimator._joint_cached_continuous_unit_components
    host_assembly_calls = 0

    def _counted_host_assembly(*args: object, **kwargs: object) -> object:
        """Count exact host assembly for each device-cache miss."""
        nonlocal host_assembly_calls
        host_assembly_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        estimator,
        "_joint_cached_continuous_unit_components",
        _counted_host_assembly,
    )
    estimator._joint_isotope_station_transport_components_torch(station, isotope)
    entry_bytes = estimator.joint_device_unit_transport_cache_bytes
    assert entry_bytes > 0
    monkeypatch.setattr(
        estimator_likelihood_module,
        "JOINT_DEVICE_UNIT_TRANSPORT_CACHE_MAX_BYTES",
        entry_bytes,
    )

    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
        surface_u=0.75,
    )
    estimator._joint_isotope_station_transport_components_torch(station, isotope)
    assert host_assembly_calls == 2
    assert len(estimator._joint_device_unit_transport_cache) == 1
    assert estimator.joint_device_unit_transport_cache_bytes <= entry_bytes

    _set_single_source_states(
        estimator,
        isotope,
        strength_cps_1m=5_000.0,
        surface_u=0.25,
    )
    estimator._joint_isotope_station_transport_components_torch(station, isotope)
    assert host_assembly_calls == 3
    assert estimator.last_joint_device_unit_cache_misses == 3
    assert len(estimator._joint_device_unit_transport_cache) == 1
    assert estimator.joint_device_unit_transport_cache_bytes <= entry_bytes


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


def test_torch_rj_target_keeps_history_and_proposals_device_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch RJ must bypass the host LRU without changing exact targets."""
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
    filt = estimator.filters["Cs-137"]
    original = filt._continuous_rj_line_transport_component_columns
    device_flags: list[bool] = []

    def _capture_device_path(*args: object, **kwargs: object) -> object:
        """Record selection of the device-resident transport API."""
        device_flags.append(bool(kwargs.get("device_resident", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        filt,
        "_continuous_rj_line_transport_component_columns",
        _capture_device_path,
    )

    misses_before = estimator.last_joint_structural_unit_cache_misses
    first = _conditional_target(estimator, station)
    first_misses = estimator.last_joint_structural_unit_cache_misses
    second = _conditional_target(estimator, station)

    assert first_misses > 0
    assert estimator.last_joint_structural_unit_cache_hits == 0
    assert (
        estimator.last_joint_structural_unit_cache_misses - first_misses
        == first_misses - misses_before
    )
    assert device_flags == [True, True]
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
    assert estimator.last_joint_structural_unit_cache_misses == misses_before_repeat
    for repeated_values, combined_values in zip(
        repeated,
        combined,
        strict=True,
    ):
        np.testing.assert_array_equal(repeated_values, combined_values)


def test_unit_transport_cache_fuses_multi_station_miss_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fused misses must equal serial shards with one exact kernel call."""
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
    line_indices = estimator._joint_line_layout()["Cs-137"][1]
    first_station = _station()
    second_station = replace(
        first_station,
        detector_position_xyz_m=(2.0, 1.0, 1.5),
        fe_indices=np.asarray([1, 4], dtype=np.int64),
        pb_indices=np.asarray([6, 0], dtype=np.int64),
        station_sequence_id=1,
    )
    combined_geometry = estimator._joint_history_structural_geometry(
        "Cs-137",
        (first_station, second_station),
    )
    shards = estimator._joint_structural_station_geometry_shards(combined_geometry)
    original = filt._continuous_rj_line_transport_component_columns
    call_rows: list[int] = []

    def _counted(*args: object, **kwargs: object) -> object:
        """Record each exact transport launch without changing its result."""
        data = args[0]
        assert hasattr(data, "row_count")
        call_rows.append(int(getattr(data, "row_count")))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        filt,
        "_continuous_rj_line_transport_component_columns",
        _counted,
    )
    serial_shards = [
        estimator._joint_cached_continuous_unit_components_shard(
            filt=filt,
            data=shard,
            positions_s3=positions,
            chart_ids_s=chart_ids,
            positive_line_indices=line_indices,
        )
        for shard in shards
    ]
    serial = tuple(
        np.concatenate(
            [values[index] for values in serial_shards],
            axis=0,
        )
        for index in range(len(serial_shards[0]))
    )
    assert call_rows == [2, 2]

    estimator._joint_structural_unit_transport_cache.clear()
    call_rows.clear()
    fused = estimator._joint_cached_continuous_unit_components(
        filt=filt,
        data=combined_geometry,
        positions_s3=positions,
        chart_ids_s=chart_ids,
        positive_line_indices=line_indices,
    )

    assert call_rows == [4]
    for fused_values, serial_values in zip(fused, serial, strict=True):
        np.testing.assert_allclose(
            fused_values,
            serial_values,
            rtol=2.0e-12,
            atol=1.0e-12,
        )


def test_unit_transport_cache_retains_reused_state_during_proposal_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-shot proposals must not evict repeatedly used accepted positions."""
    estimator = _estimator(use_gpu=True)
    filt = estimator.filters["Cs-137"]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    station = _station()
    geometry = estimator._joint_history_structural_geometry(
        "Cs-137",
        (station,),
    )
    line_indices = estimator._joint_line_layout()["Cs-137"][1]
    chart_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
    surface_uv = np.asarray(
        [[0.2, 0.3], [0.7, 0.8], [0.1, 0.9], [0.9, 0.1]],
        dtype=np.float64,
    )
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    bytes_per_entry = (
        estimator_module.JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE.itemsize
        + np.dtype(np.int64).itemsize
        + len(estimator_module.JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES)
        * geometry.row_count
        * line_indices.size
        * np.dtype(np.float64).itemsize
    )
    monkeypatch.setattr(
        estimator_module,
        "JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES",
        3 * bytes_per_entry,
    )

    estimator._joint_cached_continuous_unit_components(
        filt=filt,
        data=geometry,
        positions_s3=positions[:2],
        chart_ids_s=chart_ids[:2],
        positive_line_indices=line_indices,
    )
    estimator._joint_cached_continuous_unit_components(
        filt=filt,
        data=geometry,
        positions_s3=positions,
        chart_ids_s=chart_ids,
        positive_line_indices=line_indices,
    )
    misses_before_reuse = estimator.last_joint_structural_unit_cache_misses
    estimator._joint_cached_continuous_unit_components(
        filt=filt,
        data=geometry,
        positions_s3=positions[:2],
        chart_ids_s=chart_ids[:2],
        positive_line_indices=line_indices,
    )

    assert estimator.last_joint_structural_unit_cache_misses == misses_before_reuse


@pytest.mark.parametrize(
    ("use_gpu", "gpu_device"),
    [(False, "cpu"), (True, "cpu"), (True, "cuda")],
)
def test_raw_spectrum_joint_smc_concentrates_on_physical_truth(
    use_gpu: bool,
    gpu_device: str,
) -> None:
    """An exact raw-spectrum update must retain truth support and target ESS."""
    if gpu_device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available.")
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
            hard_max_sources=2,
            variable_cardinality=True,
                init_num_sources=(0, 1),
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            position_max=(3.0, 3.0, 3.0),
            structural_rj_surface_chart_max_edge_m=2.0,
            strength_prior=BoundedUniformStrengthPriorTestConfig(
                minimum_cps_1m=1_000.0,
                maximum_cps_1m=10_000.0,
            ),
            structural_cardinality_prior_mean=1.0,
            target_ess_ratio=0.25,
            max_temper_steps=64,
            min_delta_beta=1.0e-8,
            joint_rejuvenation_min_state_change_weight_mass=0.0,
            joint_rejuvenation_min_surface_esjd_m2=0.0,
            joint_rejuvenation_min_log_strength_esjd=0.0,
        ),
        detector_radius_m=0.025,
        detector_aperture_radius_m=0.0395,
        detector_aperture_samples=33,
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
            particle.joint_row_identity for particle in filt.continuous_particles
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
    total, uncollided, features = estimator._joint_station_transport_components_torch(
        empty_observation
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
        estimator._joint_station_log_likelihood_torch(station).detach().cpu().numpy()
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
            [particle.log_weight for particle in cs_filter.continuous_particles],
            dtype=np.float64,
        )
    )
    weights /= float(np.sum(weights))
    cardinality_one_mass = 0.0
    truth_neighborhood_mass = 0.0
    far_neighborhood_mass = 0.0
    for particle, weight in zip(
        cs_filter.continuous_particles,
        weights,
        strict=True,
    ):
        if particle.state.num_sources != 1:
            continue
        cardinality_one_mass += float(weight)
        inferred = cs_filter.continuous_state_positions(particle.state)[0]
        if float(np.linalg.norm(inferred - truth_position)) < 0.5:
            truth_neighborhood_mass += float(weight)
        if float(np.linalg.norm(inferred - far_position)) < 0.5:
            far_neighborhood_mass += float(weight)

    assert cardinality_one_mass > 0.99
    assert truth_neighborhood_mass > 0.95
    assert truth_neighborhood_mass > 20.0 * far_neighborhood_mass
    assert cs_filter.last_ess >= 0.25 * particle_count - 1.0e-9
    assert estimator.last_joint_temper_steps[-1]["beta_total"] == 1.0
    assert 1 <= estimator.last_joint_station_unique_ancestor_count <= particle_count
    if gpu_device == "cuda":
        assert estimator.last_joint_staged_transport_commit_rows > 0
    elif use_gpu:
        assert estimator.last_joint_staged_transport_commit_rows == 0


def test_cuda_joint_moves_keep_state_and_mh_on_device() -> None:
    """Joint strength/state proposals must not materialize station PF rows."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    estimator = _estimator(
        use_gpu=True,
        gpu_device="cuda",
        num_particles=8,
    )
    estimator.pf_config.joint_strength_block_probability = 1.0
    estimator.pf_config.joint_cross_isotope_state_block_probability = 1.0
    estimator.pf_config.structural_rj_move_probability = 0.0
    estimator.pf_config.structural_rj_split_merge_probability = 0.0
    estimator.pf_config.structural_rj_multi_component_probability = 0.0
    estimator.pf_config.structural_rj_block_independence_probability = 0.0
    estimator.pf_config.structural_rj_position_move_probability = 0.0
    estimator.pf_config.structural_rj_local_position_move_probability = 0.0
    estimator.pf_config.structural_rj_strength_move_probability = 0.0
    for isotope in TEST_ISOTOPES:
        _set_single_source_states(
            estimator,
            isotope,
            strength_cps_1m=5_000.0,
        )
        filt_config = estimator.filters[isotope].config
        filt_config.structural_rj_move_probability = 0.0
        filt_config.structural_rj_split_merge_probability = 0.0
        filt_config.structural_rj_multi_component_probability = 0.0
        filt_config.structural_rj_block_independence_probability = 0.0
        filt_config.structural_rj_position_move_probability = 0.0
        filt_config.structural_rj_local_position_move_probability = 0.0
        filt_config.structural_rj_strength_move_probability = 0.0
    station = _station()
    stations = (station,)
    estimator._refresh_joint_structural_transport_cache(stations)
    reference = estimator._joint_station_log_likelihood_torch(station)
    for filt in estimator.filters.values():
        assert filt._begin_continuous_rj_station_device_state(reference)

    diagnostics = estimator._joint_rejuvenate(
        stations,
        target_beta=0.5,
    )

    assert diagnostics["joint_strength_attempted_weight_mass"] == pytest.approx(
        1.0
    )
    assert diagnostics[
        "cross_isotope_state_attempted_weight_mass"
    ] == pytest.approx(1.0)
    assert estimator.last_joint_device_mh_acceptance_calls >= 2
    for filt in estimator.filters.values():
        assert filt.last_structural_device_diagnostics["proposal_backend"] == "torch"
        assert filt.last_structural_device_diagnostics["materialization_calls"] == 0
        assert filt._structural_rj_device_state_authoritative

    for filt in estimator.filters.values():
        filt._end_continuous_rj_station_device_state()
        assert filt.last_structural_device_diagnostics["materialization_calls"] == 1

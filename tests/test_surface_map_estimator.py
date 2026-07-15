"""Estimator integration tests for PF-independent spectral surface maps."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from measurement.kernels import ShieldParams
from measurement.surface_patches import SurfacePatchDictionary
from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.likelihood import expected_counts_per_source
from pf.particle_filter import MeasurementData
from pf.state import IsotopeState
from pf.surface_map import (
    SurfaceMapConfig,
    aggregate_contiguous_poisson_bins,
)


def _test_patch_dictionary() -> SurfacePatchDictionary:
    """Return two connected floor patches with unequal physical areas."""
    return SurfacePatchDictionary(
        centers_xyz=np.asarray(
            [[0.2, 0.2, 0.0], [1.8, 0.2, 0.0]],
            dtype=float,
        ),
        areas_m2=np.asarray([1.0, 0.5], dtype=float),
        kinds=("floor", "floor"),
        face_ids=("room_floor", "room_floor"),
        normals_xyz=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=float,
        ),
        local_uv_m=np.asarray([[0.2, 0.2], [1.8, 0.2]], dtype=float),
        adjacency_edges=np.asarray([[0, 1]], dtype=np.int64),
        shared_edge_lengths_m=np.asarray([0.5], dtype=float),
    )


def _estimator_with_spectral_history(
    *,
    nuisance_enable: bool = False,
) -> tuple[RotatingShieldPFEstimator, SurfacePatchDictionary]:
    """Return an estimator with deterministic aligned full-spectrum history."""
    isotope = "Cs-137"
    candidate_sources = np.asarray(
        [[0.0, 0.0, 0.1], [2.0, 0.0, 0.1]],
        dtype=float,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=candidate_sources,
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            min_particles=4,
            max_particles=4,
            max_sources=1,
            birth_enable=False,
            sparse_poisson_spectral_nuisance_enable=bool(nuisance_enable),
            use_gpu=False,
        ),
        shield_params=ShieldParams(
            thickness_fe_cm=0.0,
            thickness_pb_cm=0.0,
        ),
    )
    detector_positions = np.asarray(
        [
            [0.3, 1.0, 0.6],
            [1.7, 1.1, 0.8],
            [0.8, 1.8, 0.4],
            [2.4, 1.5, 1.0],
        ],
        dtype=float,
    )
    for detector_position in detector_positions:
        estimator.add_measurement_pose(detector_position)
    estimator._ensure_kernel_cache()

    patches = _test_patch_dictionary()
    live_times = np.ones(detector_positions.shape[0], dtype=float)
    pair_indices = np.zeros(detector_positions.shape[0], dtype=int)
    unit_response = expected_counts_per_source(
        kernel=estimator.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=detector_positions,
        sources=np.asarray(patches.centers_xyz, dtype=float),
        strengths=np.ones(patches.patch_count, dtype=float),
        live_times=live_times,
        fe_indices=pair_indices,
        pb_indices=pair_indices,
        source_scale=1.0,
    )
    template = np.asarray([0.7, 0.2, 0.1], dtype=float)
    integrated_strengths = np.asarray([40.0, 18.0], dtype=float)
    background = np.full(
        (detector_positions.shape[0], template.size),
        0.25,
        dtype=float,
    )
    spectra = background + np.einsum(
        "mc,b,c->mb",
        np.asarray(unit_response, dtype=float),
        template,
        integrated_strengths,
    )
    scalar_counts = np.asarray(unit_response, dtype=float) @ integrated_strengths
    for pose_index in range(detector_positions.shape[0]):
        estimator.measurements.append(
            MeasurementRecord(
                z_k={isotope: float(scalar_counts[pose_index])},
                pose_idx=int(pose_index),
                orient_idx=0,
                live_time_s=1.0,
                fe_index=0,
                pb_index=0,
                spectrum_counts=tuple(float(value) for value in spectra[pose_index]),
                spectrum_background=tuple(
                    float(value) for value in background[pose_index]
                ),
                spectrum_background_source="independent_test_calibration",
                spectrum_background_observation_independent=True,
                spectrum_response_templates_by_isotope={
                    isotope: tuple(float(value) for value in template)
                },
            )
        )
    return estimator, patches


def test_estimator_surface_map_is_invariant_to_pf_particle_mutation() -> None:
    """Surface fitting must not select or initialize candidates from PF particles."""
    estimator, patches = _estimator_with_spectral_history()
    config = SurfaceMapConfig(
        l1_weight=1.0e-3,
        tv_weight=1.0e-3,
        max_iterations=800,
        tolerance=1.0e-7,
        objective_tolerance=1.0e-8,
    )

    before = estimator.fit_surface_map(patches, config)
    particles = estimator.filters["Cs-137"].continuous_particles
    for particle_index, particle in enumerate(particles):
        particle.state = IsotopeState(
            num_sources=1,
            positions=np.asarray(
                [[100.0 + particle_index, -50.0, 25.0]],
                dtype=float,
            ),
            strengths=np.asarray([1.0e6 + particle_index], dtype=float),
            background=1.0e5 + particle_index,
        )
        particle.log_weight = float(1000.0 - 100.0 * particle_index)
    after = estimator.fit_surface_map(patches, config)

    assert before == after
    assert before["available"] is True
    assert before["isotope_order"] == ["Cs-137"]
    assert before["patch_metadata"]["obstacle_geometry_source"] == "none"
    assert before["patch_metadata"]["obstacle_surfaces_available"] is True
    assert np.asarray(before["densities_cps_1m_m2"]).shape == (2, 1)
    assert np.asarray(before["integrated_strengths_cps_1m"]).shape == (2, 1)
    json.dumps(before, allow_nan=False)


def test_factored_spectral_builder_preserves_candidate_grid_response() -> None:
    """The legacy candidate-grid tensor must retain its exact response formula."""
    estimator, _patches = _estimator_with_spectral_history()
    isotope = "Cs-137"
    history = estimator._spectrum_history_arrays([isotope])
    assert history is not None

    legacy_tensor, data_by_isotope = estimator._spectral_response_tensor_for_isotopes(
        history, [isotope]
    )
    generic_tensor, generic_data = estimator._spectral_response_tensor_at_positions(
        history,
        [isotope],
        np.asarray(estimator.candidate_sources, dtype=float),
    )
    design = estimator._cached_candidate_grid_counts(
        filt=estimator.filters[isotope],
        isotope=isotope,
        data=data_by_isotope[isotope],
    )
    template = np.asarray(
        history["templates_by_isotope"][isotope],
        dtype=float,
    )
    formula_oracle = (
        np.asarray(design, dtype=float)[:, None, :]
        * np.maximum(template, 0.0)[:, :, None]
    )

    np.testing.assert_allclose(legacy_tensor[:, :, :, 0], formula_oracle)
    np.testing.assert_allclose(generic_tensor, legacy_tensor)
    assert generic_data.keys() == data_by_isotope.keys()


def test_estimator_surface_map_aggregates_all_spectrum_fields_identically() -> None:
    """Estimator bin capping must preserve grouped Poisson means exactly."""
    estimator, patches = _estimator_with_spectral_history()
    isotope = "Cs-137"
    raw_history = estimator._spectrum_history_arrays([isotope])
    assert raw_history is not None
    raw_response, _ = estimator._spectral_response_tensor_at_positions(
        raw_history,
        [isotope],
        np.asarray(patches.centers_xyz, dtype=float),
    )
    aggregated_history, aggregation = estimator._aggregate_surface_map_spectrum_history(
        raw_history,
        max_spectrum_bins=2,
    )
    aggregated_response, _ = estimator._spectral_response_tensor_at_positions(
        aggregated_history,
        [isotope],
        np.asarray(patches.centers_xyz, dtype=float),
    )

    np.testing.assert_allclose(
        aggregated_history["spectrum_counts"],
        aggregate_contiguous_poisson_bins(
            raw_history["spectrum_counts"],
            aggregation,
            axis=1,
        ),
    )
    np.testing.assert_allclose(
        aggregated_history["background_spectrum"],
        aggregate_contiguous_poisson_bins(
            raw_history["background_spectrum"],
            aggregation,
            axis=1,
        ),
    )
    np.testing.assert_allclose(
        aggregated_response,
        aggregate_contiguous_poisson_bins(raw_response, aggregation, axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    payload = estimator.fit_surface_map(
        patches,
        SurfaceMapConfig(max_spectrum_bins=2, max_iterations=50),
    )
    assert payload["spectrum_original_bin_count"] == 3
    assert payload["spectrum_bin_count"] == 2
    assert payload["spectrum_aggregation"] == {
        "method": "contiguous_full_spectrum_poisson_sum",
        "max_spectrum_bins": 2,
        "original_bin_count": 3,
        "aggregated_bin_count": 2,
        "group_start_indices": [0, 1],
        "group_end_indices_exclusive": [1, 3],
        "group_widths": [1, 2],
        "covers_all_original_bins_once": True,
        "poisson_sum_preserving": True,
        "bin_selection_applied": False,
    }
    json.dumps(payload, allow_nan=False)


def test_estimator_surface_map_reports_unavailable_without_aligned_history() -> None:
    """Missing spectrum-bin history should produce an explicit JSON-safe payload."""
    estimator, patches = _estimator_with_spectral_history()
    estimator.measurements.clear()

    payload = estimator.fit_surface_map(patches)

    assert payload["available"] is False
    assert payload["reason"] == "no_aligned_spectral_history"
    assert payload["isotope_order"] == ["Cs-137"]
    assert payload["patch_count"] == patches.patch_count
    json.dumps(payload, allow_nan=False)


def test_estimator_surface_map_reports_nuisance_basis_units_and_normalization() -> None:
    """Nuisance coefficients should carry aligned physical basis semantics."""
    estimator, patches = _estimator_with_spectral_history(nuisance_enable=True)

    payload = estimator.fit_surface_map(
        patches,
        SurfaceMapConfig(max_iterations=20),
    )

    nuisance = payload["nuisance"]
    assert nuisance["parameter_count"] == 4
    assert len(nuisance["coefficients"]) == 4
    assert [row["station_visit_id"] for row in nuisance["basis"]] == [0, 1, 2, 3]
    assert all(
        row["kind"] == "station_configured_background_spectrum"
        for row in nuisance["basis"]
    )
    assert all(row["coefficient_unit"] == "dimensionless" for row in nuisance["basis"])
    assert all(row["column_unit"] == "counts" for row in nuisance["basis"])
    assert all(
        row["normalization"] == "independent_configured_background_counts"
        for row in nuisance["basis"]
    )
    assert all(row["baseline_included_separately"] is True for row in nuisance["basis"])
    assert payload["background_model"] == {
        "fixed_background_source_counts": {"independent_test_calibration": 4},
        "fixed_background_observation_independent": True,
        "rejected_observation_fitted_background_count": 0,
        "unknown_or_excess_background_fit": "nonnegative_station_visit_nuisance",
    }
    json.dumps(payload, allow_nan=False)


def test_flat_nuisance_basis_reports_count_rate_normalization() -> None:
    """The zero-background fallback should disclose its unit-integral flat basis."""
    estimator, _patches = _estimator_with_spectral_history(nuisance_enable=True)
    history = estimator._spectrum_history_arrays(["Cs-137"])
    assert history is not None
    history = {
        **history,
        "background_spectrum": np.zeros_like(history["background_spectrum"]),
    }
    history, _aggregation = estimator._aggregate_surface_map_spectrum_history(
        history,
        max_spectrum_bins=2,
    )

    matrix, basis = estimator._spectral_nuisance_basis(
        history,
        target_isotope=None,
    )

    assert matrix is not None
    assert matrix.shape[1] == 4
    assert all(row["kind"] == "station_flat_spectrum_background" for row in basis)
    assert all(row["coefficient_unit"] == "counts_per_second" for row in basis)
    assert all(row["column_unit"] == "seconds_per_bin" for row in basis)
    assert all(
        row["normalization"] == "unit_integral_across_spectrum_bins"
        for row in basis
    )
    expected_row = np.asarray([1.0 / 3.0, 2.0 / 3.0], dtype=float)
    np.testing.assert_allclose(
        np.sum(matrix, axis=1).reshape(-1, 2),
        np.broadcast_to(expected_row, (4, 2)),
    )


def test_surface_history_rejects_observation_fitted_fixed_background() -> None:
    """Unlabelled same-observation background fits must remain joint nuisances."""
    estimator, _patches = _estimator_with_spectral_history(nuisance_enable=True)
    estimator.measurements = [
        replace(
            record,
            spectrum_background_source="same_observation_response_fit",
            spectrum_background_observation_independent=False,
        )
        for record in estimator.measurements
    ]

    history = estimator._spectrum_history_arrays(["Cs-137"])

    assert history is not None
    np.testing.assert_allclose(history["background_spectrum"], 0.0)
    assert history["background_source_counts"] == {}
    assert history["rejected_observation_fitted_background_count"] == 4


def test_configured_response_kernel_survives_empty_active_filter_set() -> None:
    """Configured response evaluation must not require active PF particles."""
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137", "Co-60"],
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            background_level={"Cs-137": 0.5, "Co-60": 2.0},
            use_gpu=False,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    detector_positions = np.asarray(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.zeros(2, dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    sources = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    shared_kernel = estimator.configured_isotope_response_kernel("Co-60")
    oracle = expected_counts_per_source(
        kernel=shared_kernel,
        isotope="Co-60",
        detector_positions=detector_positions,
        sources=sources,
        strengths=np.ones(2, dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )

    estimator.restrict_isotopes([], allow_empty=True)
    actual = estimator.configured_isotope_response_counts(
        "Co-60",
        data,
        sources,
    )

    assert estimator.filters == {}
    assert estimator.configured_isotope_order() == ("Cs-137", "Co-60")
    assert estimator.configured_isotope_response_kernel("Cs-137") is shared_kernel
    np.testing.assert_allclose(actual, oracle, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        estimator._background_counts_for_report_refit(
            "Co-60",
            np.asarray([1.0, 2.0], dtype=float),
        ),
        [2.0, 4.0],
    )


def test_measurement_ingestion_retains_registered_inactive_spectrum_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measurement history must carry configured responses absent from PF payloads."""
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137", "Co-60"],
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            min_particles=2,
            max_particles=2,
            birth_enable=False,
            adaptive_strength_prior_steps=0,
            conditional_strength_refit=False,
            sparse_poisson_spectral_evidence_enable=False,
            use_gpu=False,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    estimator.add_measurement_pose(np.asarray([1.0, 1.0, 0.5], dtype=float))
    estimator.restrict_isotopes(["Cs-137"])
    estimator._ensure_kernel_cache()
    monkeypatch.setattr(
        estimator.filters["Cs-137"],
        "update_continuous_pair",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(estimator, "_apply_birth_death", lambda **_kwargs: None)
    estimator.register_configured_isotope_spectrum_responses(
        {"Co-60": np.asarray([0.0, 1.0], dtype=float)}
    )

    estimator.update_pair(
        {"Cs-137": 0.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
        spectrum_payload={
            "spectrum_counts": np.asarray([0.0, 3.0], dtype=float),
            "spectrum_background": np.zeros(2, dtype=float),
            "spectrum_response_templates_by_isotope": {
                "Cs-137": np.asarray([1.0, 0.0], dtype=float)
            },
        },
    )

    stored = estimator.measurements[-1].spectrum_response_templates_by_isotope
    assert stored is not None
    assert set(stored) == {"Cs-137", "Co-60"}
    np.testing.assert_allclose(stored["Co-60"], [0.0, 1.0])


def test_surface_map_recovers_inactive_configured_isotope_from_registry() -> None:
    """Final L1 fitting must retain and recover an isotope pruned from the PF."""
    isotopes = ("Cs-137", "Co-60")
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            min_particles=2,
            max_particles=2,
            max_sources=1,
            birth_enable=False,
            use_gpu=False,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    detector_positions = np.asarray(
        [
            [0.3, 1.0, 0.6],
            [1.7, 1.1, 0.8],
            [0.8, 1.8, 0.4],
            [2.4, 1.5, 1.0],
        ],
        dtype=float,
    )
    for detector_position in detector_positions:
        estimator.add_measurement_pose(detector_position)
    estimator._ensure_kernel_cache()
    patches = _test_patch_dictionary()
    data = MeasurementData(
        z_k=np.zeros(detector_positions.shape[0], dtype=float),
        observation_variances=np.ones(detector_positions.shape[0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
    )
    unit_response = estimator.configured_isotope_response_counts(
        "Co-60",
        data,
        np.asarray(patches.centers_xyz, dtype=float),
    )
    true_co_strengths = np.asarray([0.0, 55.0], dtype=float)
    templates = {
        "Cs-137": np.asarray([1.0, 0.0], dtype=float),
        "Co-60": np.asarray([0.0, 1.0], dtype=float),
    }
    estimator.register_configured_isotope_spectrum_responses(templates)
    background = np.full((detector_positions.shape[0], 2), 0.2, dtype=float)
    spectra = background.copy()
    spectra[:, 1] += np.asarray(unit_response, dtype=float) @ true_co_strengths
    co_counts = np.asarray(unit_response, dtype=float) @ true_co_strengths
    for measurement_index in range(detector_positions.shape[0]):
        record_templates = (
            {isotope: tuple(template) for isotope, template in templates.items()}
            if measurement_index == 0
            else {"Cs-137": tuple(templates["Cs-137"])}
        )
        estimator.measurements.append(
            MeasurementRecord(
                z_k={"Cs-137": 0.0, "Co-60": float(co_counts[measurement_index])},
                pose_idx=int(measurement_index),
                orient_idx=0,
                live_time_s=1.0,
                fe_index=0,
                pb_index=0,
                spectrum_counts=tuple(
                    float(value) for value in spectra[measurement_index]
                ),
                spectrum_background=tuple(
                    float(value) for value in background[measurement_index]
                ),
                spectrum_background_source="independent_test_calibration",
                spectrum_background_observation_independent=True,
                spectrum_response_templates_by_isotope=record_templates,
            )
        )

    estimator.restrict_isotopes(["Cs-137"])
    assert "Co-60" not in estimator.filters
    inactive_history = estimator.configured_isotope_measurement_history("Co-60")
    assert inactive_history is not None
    np.testing.assert_allclose(inactive_history.z_k, co_counts)
    payload = estimator.fit_surface_map(
        patches,
        SurfaceMapConfig(
            l1_weight=1.0e-5,
            tv_weight=0.0,
            max_iterations=2000,
            tolerance=1.0e-8,
            objective_tolerance=1.0e-9,
        ),
    )

    reconstructed = np.asarray(payload["integrated_strengths_cps_1m"], dtype=float)
    assert payload["available"] is True
    assert payload["isotope_order"] == ["Cs-137", "Co-60"]
    assert payload["active_isotopes_at_fit"] == ["Cs-137"]
    assert payload["inactive_isotopes_evaluated"] == ["Co-60"]
    assert payload["response_source"] == ("shared_configured_isotope_kernel_registry")
    assert payload["pf_particle_state_independent"] is True
    assert reconstructed.shape == (patches.patch_count, len(isotopes))
    np.testing.assert_allclose(reconstructed[:, 0], 0.0, atol=1.0e-5)
    np.testing.assert_allclose(
        reconstructed[:, 1],
        true_co_strengths,
        rtol=2.0e-3,
        atol=2.0e-2,
    )
    assert payload["template_source_counts_by_isotope"]["Co-60"] == {
        "record": 1,
        "configured_registry": 3,
    }
    json.dumps(payload, allow_nan=False)

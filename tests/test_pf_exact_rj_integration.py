"""Integration tests for target-preserving finite-surface PF rejuvenation."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.shielding import generate_octant_orientations
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import (
    IsotopeParticleFilter,
    MeasurementData,
    PFConfig,
)


def _exact_config(**overrides: object) -> PFConfig:
    """Return a small exact RJ-MH configuration for deterministic tests."""
    values: dict[str, object] = {
        "num_particles": 18,
        "min_particles": 18,
        "max_particles": 18,
        "max_sources": 2,
        "use_gpu": False,
        "position_min": (0.0, 0.0, 0.0),
        "position_max": (2.0, 2.0, 2.0),
        "source_position_prior": "surface",
        "init_num_sources": (0, 2),
        "init_grid_repeats": 1,
        "init_strength_prior": "uniform",
        "init_strength_min": 1.0,
        "init_strength_max": 3.0,
        "structural_kernel_mode": "rj_mh",
        "structural_rj_patch_spacing_m": 1.0,
        "structural_rj_move_probability": 1.0,
        "structural_rj_birth_probability": 0.5,
        "structural_rj_death_probability": 0.5,
        "structural_rj_position_move_probability": 1.0,
        "structural_rj_local_position_move_probability": 1.0,
        "structural_rj_strength_move_probability": 1.0,
        "cardinality_preserving_resample": False,
        "mode_preserving_resample": False,
        "surface_rejuvenation_enable": False,
        "pseudo_source_verification_enable": False,
        "split_prob": 0.0,
        "merge_prob": 0.0,
        "source_detector_exclusion_m": 0.0,
        "init_source_min_separation_m": 0.0,
    }
    values.update(overrides)
    return PFConfig(**values)


def _measurement_data(*, live_time: float) -> MeasurementData:
    """Return a two-row count-likelihood block for exact-move tests."""
    return MeasurementData(
        z_k=np.zeros(2, dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=np.asarray(
            [[0.4, 0.6, 0.5], [1.4, 1.2, 0.5]],
            dtype=float,
        ),
        fe_indices=np.zeros(2, dtype=np.int64),
        pb_indices=np.zeros(2, dtype=np.int64),
        live_times=np.full(2, live_time, dtype=float),
        station_sequence_ids=np.asarray([0, 1], dtype=np.int64),
        runtime_likelihood_routes=np.asarray(["count", "count"], dtype="<U16"),
    )


def _extended_measurement_data(*, live_time: float) -> MeasurementData:
    """Return a three-row history whose first two response rows are unchanged."""
    return MeasurementData(
        z_k=np.zeros(3, dtype=float),
        observation_variances=np.ones(3, dtype=float),
        detector_positions=np.asarray(
            [
                [0.4, 0.6, 0.5],
                [1.4, 1.2, 0.5],
                [1.0, 0.8, 1.5],
            ],
            dtype=float,
        ),
        fe_indices=np.zeros(3, dtype=np.int64),
        pb_indices=np.zeros(3, dtype=np.int64),
        live_times=np.full(3, live_time, dtype=float),
        station_sequence_ids=np.asarray([0, 1, 2], dtype=np.int64),
        runtime_likelihood_routes=np.asarray(
            ["count", "count", "count"],
            dtype="<U16",
        ),
    )


def _build_filter(**config_overrides: object) -> IsotopeParticleFilter:
    """Construct a reproducible exact filter without obstacle geometry."""
    np.random.seed(20260727)
    return IsotopeParticleFilter(
        "Cs-137",
        kernel=None,
        config=_exact_config(**config_overrides),
    )


def _discrete_kernel(
    *,
    poses: np.ndarray | None = None,
    orientations: np.ndarray | None = None,
    shield_params: ShieldParams | None = None,
    mu_by_isotope: dict[str, object] | None = None,
) -> KernelPrecomputer:
    """Build a CPU discrete kernel for cache-replacement tests."""
    return KernelPrecomputer(
        candidate_sources=np.asarray([[0.5, 0.5, 0.5]], dtype=float),
        poses=(
            np.asarray([[0.4, 0.6, 0.5]], dtype=float)
            if poses is None
            else np.asarray(poses, dtype=float)
        ),
        orientations=(
            generate_octant_orientations()
            if orientations is None
            else np.asarray(orientations, dtype=float)
        ),
        shield_params=ShieldParams() if shield_params is None else shield_params,
        mu_by_isotope={} if mu_by_isotope is None else mu_by_isotope,
        use_gpu=False,
    )


def _build_exact_estimator() -> RotatingShieldPFEstimator:
    """Build a small exact estimator for appended-pose cache tests."""
    config = RotatingShieldPFConfig(
        num_particles=18,
        min_particles=18,
        max_particles=18,
        max_sources=2,
        use_gpu=False,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 2.0, 2.0),
        source_position_prior="surface",
        init_num_sources=(0, 2),
        init_strength_prior="uniform",
        init_strength_min=1.0,
        init_strength_max=3.0,
        structural_kernel_mode="rj_mh",
        structural_rj_patch_spacing_m=1.0,
        cardinality_preserving_resample=False,
        mode_preserving_resample=False,
        surface_rejuvenation_enable=False,
        pseudo_source_verification_enable=False,
        split_prob=0.0,
        merge_prob=0.0,
        source_detector_exclusion_m=0.0,
        init_source_min_separation_m=0.0,
    )
    shield = ShieldParams()
    return RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.asarray([[0.5, 0.5, 0.5]], dtype=float),
        shield_normals=generate_octant_orientations(),
        mu_by_isotope={
            "Cs-137": {
                "fe": shield.mu_fe,
                "pb": shield.mu_pb,
            }
        },
        pf_config=config,
        shield_params=shield,
    )


def test_exact_initialization_uses_complete_canonical_surface_prior() -> None:
    """Every initial state must use unique canonical patches and prior strengths."""
    particle_filter = _build_filter()
    patches = particle_filter._structural_rj_surface_patches

    assert patches is not None
    assert particle_filter.N == particle_filter.config.num_particles
    cardinalities = {
        particle.state.num_sources
        for particle in particle_filter.continuous_particles
    }
    assert cardinalities == {0, 1, 2}
    assert np.sum(particle_filter.continuous_weights) == 1.0
    for particle in particle_filter.continuous_particles:
        patch_indices = particle_filter._canonicalize_structural_rj_state(
            particle.state
        )
        if patch_indices.size > 1:
            assert np.all(np.diff(patch_indices) > 0)
        assert np.all(
            particle_filter._strength_prior.in_support(
                particle.state.strengths
            )
        )


def test_exact_initialization_ignores_legacy_grid_repeats() -> None:
    """Legacy grid repeats must not expand an exact prior-sampling population."""
    config = _exact_config()
    config.init_grid_repeats = 100
    particle_filter = IsotopeParticleFilter(
        "Cs-137",
        kernel=None,
        config=config,
    )

    assert particle_filter.N == config.num_particles


def test_exact_initialization_reproduces_explicit_cardinality_mass() -> None:
    """Stratified initial particles must carry the configured total mass per K."""
    expected = np.asarray([0.10, 0.25, 0.65], dtype=float)
    particle_filter = _build_filter(
        structural_cardinality_prior_probs=tuple(expected),
    )
    cardinalities = np.asarray(
        [
            particle.state.num_sources
            for particle in particle_filter.continuous_particles
        ],
        dtype=np.int64,
    )
    weights = np.asarray(particle_filter.continuous_weights, dtype=float)
    observed = np.bincount(
        cardinalities,
        weights=weights,
        minlength=expected.size,
    )

    np.testing.assert_allclose(observed, expected, atol=1.0e-14, rtol=0.0)


def test_exact_batched_response_matches_scalar_particle_oracle() -> None:
    """Batched equal-K expected counts must match the scalar physics path."""
    particle_filter = _build_filter()
    data = _measurement_data(live_time=1.0)
    response_dictionary = (
        particle_filter._structural_rj_response_dictionary(data)
    )
    particle_indices = np.asarray(
        [
            index
            for index, particle in enumerate(
                particle_filter.continuous_particles
            )
            if particle.state.num_sources == 2
        ][:4],
        dtype=np.int64,
    )
    patch_sets, strengths, backgrounds = (
        particle_filter._structural_rj_group_arrays(
            particle_indices,
            cardinality=2,
        )
    )
    batched = particle_filter._structural_rj_lambda_from_arrays(
        response_dictionary,
        patch_sets,
        strengths,
        backgrounds,
        data.live_times,
    )
    scalar = np.column_stack(
        [
            particle_filter._lambda_components(
                particle_filter.continuous_particles[int(index)].state,
                data,
            )[1]
            for index in particle_indices
        ]
    )

    np.testing.assert_allclose(batched, scalar, rtol=1.0e-12, atol=1.0e-12)


def test_exact_response_cache_is_lazy_and_extends_only_missing_suffixes() -> None:
    """Only referenced patches and newly appended response rows should be built."""
    particle_filter = _build_filter()
    data = _measurement_data(live_time=1.0)
    response_dictionary = particle_filter._structural_rj_response_dictionary(
        data,
        patch_indices=np.zeros(0, dtype=np.int64),
    )
    assert np.all(np.isnan(response_dictionary))
    particle_indices = np.asarray(
        [
            index
            for index, particle in enumerate(
                particle_filter.continuous_particles
            )
            if particle.state.num_sources == 2
        ][:3],
        dtype=np.int64,
    )
    patch_sets, strengths, backgrounds = (
        particle_filter._structural_rj_group_arrays(
            particle_indices,
            cardinality=2,
        )
    )
    required = np.unique(patch_sets)
    particle_filter._structural_rj_group_log_likelihood(
        data,
        response_dictionary,
        patch_sets,
        strengths,
        backgrounds,
    )

    assert np.all(np.isfinite(response_dictionary[:, required]))
    unreferenced = np.setdiff1d(
        np.arange(response_dictionary.shape[1], dtype=np.int64),
        required,
    )
    assert np.all(np.isnan(response_dictionary[:, unreferenced]))
    assert (
        particle_filter._structural_rj_response_evaluated_cells
        == data.z_k.size * required.size
    )

    extended = _extended_measurement_data(live_time=1.0)
    extended_dictionary = (
        particle_filter._structural_rj_response_dictionary(
            extended,
            patch_indices=np.zeros(0, dtype=np.int64),
        )
    )
    np.testing.assert_array_equal(
        extended_dictionary[: data.z_k.size, required],
        response_dictionary[:, required],
    )
    assert np.all(
        np.isnan(extended_dictionary[data.z_k.size :, required])
    )
    evaluated_before = (
        particle_filter._structural_rj_response_evaluated_cells
    )
    particle_filter._structural_rj_group_log_likelihood(
        extended,
        extended_dictionary,
        patch_sets,
        strengths,
        backgrounds,
    )
    assert (
        particle_filter._structural_rj_response_evaluated_cells
        - evaluated_before
        == required.size
    )

    unseen_patch = int(unreferenced[0])
    evaluated_before = (
        particle_filter._structural_rj_response_evaluated_cells
    )
    particle_filter._structural_rj_response_dictionary(
        extended,
        patch_indices=np.asarray([unseen_patch], dtype=np.int64),
    )
    assert (
        particle_filter._structural_rj_response_evaluated_cells
        - evaluated_before
        == extended.z_k.size
    )
    assert np.all(np.isfinite(extended_dictionary[:, unseen_patch]))

    changed_history = _measurement_data(live_time=2.0)
    reset_dictionary = (
        particle_filter._structural_rj_response_dictionary(
            changed_history,
            patch_indices=np.zeros(0, dtype=np.int64),
        )
    )
    assert np.all(np.isnan(reset_dictionary))


def test_appended_pose_reuses_exact_prefix_across_equivalent_kernel_rebuild() -> None:
    """A pose-only discrete rebuild must evaluate only the new response suffix."""
    estimator = _build_exact_estimator()
    data = _measurement_data(live_time=1.0)
    for pose in data.detector_positions:
        estimator.add_measurement_pose(pose, reset_filters=False)
    estimator._ensure_kernel_cache()
    particle_filter = estimator.filters["Cs-137"]
    selected_patch = np.asarray([0], dtype=np.int64)
    response = particle_filter._structural_rj_response_dictionary(
        data,
        patch_indices=selected_patch,
    )
    prefix = response[:, selected_patch].copy()
    old_discrete_kernel = estimator.kernel_cache
    old_continuous_kernel = particle_filter.continuous_kernel
    old_response_cache = particle_filter._structural_rj_response_cache
    evaluated_before = (
        particle_filter._structural_rj_response_evaluated_cells
    )

    estimator.normals = np.asarray(estimator.normals, dtype=float).copy()
    estimator.shield_params = ShieldParams()
    estimator.mu_by_isotope = {
        "Cs-137": (
            float(ShieldParams().mu_fe),
            float(ShieldParams().mu_pb),
        )
    }
    extended = _extended_measurement_data(live_time=1.0)
    estimator.add_measurement_pose(
        extended.detector_positions[-1],
        reset_filters=False,
    )
    estimator._ensure_kernel_cache()

    assert estimator.filters["Cs-137"] is particle_filter
    assert estimator.kernel_cache is not old_discrete_kernel
    assert particle_filter.kernel is estimator.kernel_cache
    assert particle_filter.continuous_kernel is old_continuous_kernel
    assert particle_filter._structural_rj_response_cache is old_response_cache
    extended_response = (
        particle_filter._structural_rj_response_dictionary(
            extended,
            patch_indices=np.zeros(0, dtype=np.int64),
        )
    )
    np.testing.assert_array_equal(
        extended_response[: data.z_k.size, selected_patch],
        prefix,
    )
    assert np.all(
        np.isnan(extended_response[data.z_k.size :, selected_patch])
    )

    particle_filter._structural_rj_response_dictionary(
        extended,
        patch_indices=selected_patch,
    )
    assert (
        particle_filter._structural_rj_response_evaluated_cells
        - evaluated_before
        == 1
    )
    eager_filter = _build_filter()
    eager_response = eager_filter._structural_rj_response_dictionary(
        extended,
        patch_indices=selected_patch,
    )
    np.testing.assert_allclose(
        extended_response[:, selected_patch],
        eager_response[:, selected_patch],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "physics_change",
    ["effective_mu", "shield_geometry", "orientation_order"],
)
def test_exact_response_cache_invalidates_on_kernel_physics_change(
    physics_change: str,
) -> None:
    """Changed continuous physics must discard every cached response row."""
    particle_filter = _build_filter()
    data = _measurement_data(live_time=1.0)
    selected_patch = np.asarray([0], dtype=np.int64)
    particle_filter._structural_rj_response_dictionary(
        data,
        patch_indices=selected_patch,
    )
    previous_continuous_kernel = particle_filter.continuous_kernel
    default_shield = ShieldParams()
    orientations = generate_octant_orientations()
    shield_params = default_shield
    mu_by_isotope: dict[str, object] = {}
    if physics_change == "effective_mu":
        mu_by_isotope = {
            "Cs-137": {
                "fe": 1.01 * default_shield.mu_fe,
                "pb": default_shield.mu_pb,
            }
        }
    elif physics_change == "shield_geometry":
        shield_params = ShieldParams(
            thickness_pb_cm=default_shield.thickness_pb_cm + 0.1
        )
    else:
        orientations = orientations.copy()
        orientations[[0, 1]] = orientations[[1, 0]]
    incoming = _discrete_kernel(
        orientations=orientations,
        shield_params=shield_params,
        mu_by_isotope=mu_by_isotope,
    )

    particle_filter.set_kernel(incoming)

    assert particle_filter.kernel is incoming
    assert particle_filter.continuous_kernel is not previous_continuous_kernel
    assert particle_filter._structural_rj_response_cache is None
    assert particle_filter._structural_rj_response_cache_signatures is None
    reset_response = particle_filter._structural_rj_response_dictionary(
        data,
        patch_indices=np.zeros(0, dtype=np.int64),
    )
    assert np.all(np.isnan(reset_response))
    evaluated_before = (
        particle_filter._structural_rj_response_evaluated_cells
    )
    particle_filter._structural_rj_response_dictionary(
        data,
        patch_indices=selected_patch,
    )
    assert (
        particle_filter._structural_rj_response_evaluated_cells
        - evaluated_before
        == data.z_k.size
    )


def test_equivalent_kernel_replacement_preserves_nonexact_runtime_output() -> None:
    """Legacy mode must update its discrete kernel without changing responses."""
    original = _discrete_kernel()
    particle_filter = IsotopeParticleFilter(
        "Cs-137",
        kernel=original,
        config=PFConfig(
            num_particles=4,
            min_particles=4,
            max_particles=4,
            max_sources=1,
            use_gpu=False,
        ),
    )
    continuous_kernel = particle_filter.continuous_kernel
    detector_positions = np.asarray([[0.4, 0.6, 0.5]], dtype=float)
    sources = np.asarray([[1.2, 1.0, 0.8]], dtype=float)
    before = continuous_kernel.kernel_values_selected_pairs_for_detectors(
        isotope="Cs-137",
        detector_positions=detector_positions,
        sources=sources,
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([1], dtype=np.int64),
    )
    replacement = _discrete_kernel(
        poses=np.asarray(
            [[0.4, 0.6, 0.5], [1.4, 1.2, 0.5]],
            dtype=float,
        ),
        mu_by_isotope={
            "Cs-137": {
                "fe": ShieldParams().mu_fe,
                "pb": ShieldParams().mu_pb,
            }
        },
    )

    particle_filter.set_kernel(replacement)
    after = particle_filter.continuous_kernel.kernel_values_selected_pairs_for_detectors(
        isotope="Cs-137",
        detector_positions=detector_positions,
        sources=sources,
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([1], dtype=np.int64),
    )

    assert particle_filter.kernel is replacement
    assert particle_filter.continuous_kernel is continuous_kernel
    np.testing.assert_array_equal(after, before)


def test_lazy_response_cache_matches_eager_exact_move_results() -> None:
    """Lazy response evaluation must preserve RNG use and accepted PF states."""
    eager_filter = _build_filter()
    lazy_filter = _build_filter()
    data = _measurement_data(live_time=1.0)
    eager_filter._structural_rj_response_dictionary(data)

    eager_filter.apply_structural_moves(data)
    lazy_filter.apply_structural_moves(data)

    assert eager_filter.last_birth_count == lazy_filter.last_birth_count
    assert eager_filter.last_kill_count == lazy_filter.last_kill_count
    assert (
        eager_filter.last_structural_timing_s[
            "rj_local_position_accepted"
        ]
        == lazy_filter.last_structural_timing_s[
            "rj_local_position_accepted"
        ]
    )
    assert (
        eager_filter.last_structural_timing_s["rj_response_evaluated_cells"]
        == 0.0
    )
    assert (
        lazy_filter.last_structural_timing_s["rj_response_evaluated_cells"]
        > 0.0
    )
    for eager_particle, lazy_particle in zip(
        eager_filter.continuous_particles,
        lazy_filter.continuous_particles,
    ):
        assert eager_particle.state.num_sources == lazy_particle.state.num_sources
        np.testing.assert_array_equal(
            eager_particle.state.positions,
            lazy_particle.state.positions,
        )
        np.testing.assert_array_equal(
            eager_particle.state.strengths,
            lazy_particle.state.strengths,
        )
        assert eager_particle.state.background == lazy_particle.state.background
        assert eager_particle.log_weight == lazy_particle.log_weight


def test_exact_standard_path_runs_global_and_local_position_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard exact path must retain global moves and invoke local moves."""
    particle_filter = _build_filter(
        structural_rj_move_probability=0.0,
        structural_rj_position_move_probability=0.0,
        structural_rj_local_position_move_probability=0.0,
        structural_rj_strength_move_probability=0.0,
    )
    data = _measurement_data(live_time=0.0)
    calls: list[str] = []
    global_move = particle_filter._apply_structural_rj_position_moves
    local_move = particle_filter._apply_structural_rj_local_position_moves

    def record_global(*args: object, **kwargs: object) -> int:
        """Record and invoke the global exact position kernel."""
        calls.append("global")
        return global_move(*args, **kwargs)

    def record_local(*args: object, **kwargs: object) -> int:
        """Record and invoke the local exact position kernel."""
        calls.append("local")
        return local_move(*args, **kwargs)

    monkeypatch.setattr(
        particle_filter,
        "_apply_structural_rj_position_moves",
        record_global,
    )
    monkeypatch.setattr(
        particle_filter,
        "_apply_structural_rj_local_position_moves",
        record_local,
    )

    particle_filter.apply_structural_moves(
        data,
        allow_structural_birth_proposals=False,
    )

    assert calls == ["global", "local"]
    assert particle_filter._structural_rj_surface_adjacency is not None
    diagnostics = particle_filter.last_structural_timing_s
    for move in (
        "birth",
        "death",
        "global_position",
        "local_position",
        "strength",
    ):
        assert diagnostics[f"rj_{move}_attempted"] == 0.0
        assert diagnostics[f"rj_{move}_accepted"] == 0.0
    assert diagnostics["rj_local_position_movable"] == 0.0
    assert diagnostics["outer_log_weight_max_abs_diff"] == 0.0
    assert diagnostics["outer_log_weight_array_equal"] == 1.0
    assert diagnostics["weights_preserved"] == 1.0


def test_exact_runtime_records_true_move_attempt_and_accept_counts() -> None:
    """Exact diagnostics must expose valid denominators for every MH kernel."""
    particle_filter = _build_filter()
    data = _measurement_data(live_time=0.0)

    particle_filter.apply_structural_moves(data)

    diagnostics = particle_filter.last_structural_timing_s
    particle_count = len(particle_filter.continuous_particles)
    assert (
        diagnostics["rj_birth_attempted"]
        + diagnostics["rj_death_attempted"]
        == particle_count
    )
    for move in (
        "birth",
        "death",
        "global_position",
        "local_position",
        "strength",
    ):
        attempted = diagnostics[f"rj_{move}_attempted"]
        accepted = diagnostics[f"rj_{move}_accepted"]
        assert 0.0 <= accepted <= attempted <= particle_count
    assert (
        diagnostics["rj_local_position_accepted"]
        <= diagnostics["rj_local_position_movable"]
        <= diagnostics["rj_local_position_attempted"]
    )
    assert (
        diagnostics["rj_position_accepted"]
        == diagnostics["rj_global_position_accepted"]
    )
    assert (
        diagnostics["rj_position_attempted"]
        == diagnostics["rj_global_position_attempted"]
    )
    assert diagnostics["outer_log_weight_max_abs_diff"] == 0.0
    assert diagnostics["outer_log_weight_array_equal"] == 1.0
    assert diagnostics["weights_preserved"] == 1.0


def test_exact_runtime_records_outer_weight_invariant_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A violated outer-weight invariant must retain its exact diagnostics."""
    particle_filter = _build_filter(
        structural_rj_move_probability=0.0,
        structural_rj_position_move_probability=0.0,
        structural_rj_local_position_move_probability=0.0,
        structural_rj_strength_move_probability=0.0,
    )
    data = _measurement_data(live_time=0.0)
    global_move = particle_filter._apply_structural_rj_position_moves

    def corrupt_outer_weight(*args: object, **kwargs: object) -> int:
        """Inject one test-only outer-weight mutation after the global kernel."""
        accepted = global_move(*args, **kwargs)
        particle_filter.continuous_particles[0].log_weight += 0.25
        return accepted

    monkeypatch.setattr(
        particle_filter,
        "_apply_structural_rj_position_moves",
        corrupt_outer_weight,
    )

    with pytest.raises(RuntimeError, match="must not alter PF weights"):
        particle_filter.apply_structural_moves(
            data,
            allow_structural_birth_proposals=False,
        )

    diagnostics = particle_filter.last_structural_timing_s
    assert diagnostics["outer_log_weight_max_abs_diff"] == pytest.approx(
        0.25
    )
    assert diagnostics["outer_log_weight_array_equal"] == 0.0
    assert diagnostics["weights_preserved"] == 0.0


def test_exact_local_position_runtime_preserves_weights_and_backgrounds() -> None:
    """Batched local acceptance must alter only source surface positions."""
    particle_filter = _build_filter(
        structural_rj_move_probability=0.0,
        structural_rj_position_move_probability=0.0,
        structural_rj_strength_move_probability=0.0,
    )
    data = _measurement_data(live_time=0.0)
    original_weights = np.asarray(
        [
            particle.log_weight
            for particle in particle_filter.continuous_particles
        ],
        dtype=float,
    )
    original_backgrounds = np.asarray(
        [
            particle.state.background
            for particle in particle_filter.continuous_particles
        ],
        dtype=float,
    )
    original_positions = [
        particle.state.positions.copy()
        for particle in particle_filter.continuous_particles
    ]

    particle_filter.apply_structural_moves(
        data,
        allow_structural_birth_proposals=False,
    )

    np.testing.assert_array_equal(
        [
            particle.log_weight
            for particle in particle_filter.continuous_particles
        ],
        original_weights,
    )
    np.testing.assert_array_equal(
        [
            particle.state.background
            for particle in particle_filter.continuous_particles
        ],
        original_backgrounds,
    )
    assert (
        particle_filter.last_structural_timing_s[
            "rj_local_position_accepted"
        ]
        > 0.0
    )
    assert any(
        not np.array_equal(before, particle.state.positions)
        for before, particle in zip(
            original_positions,
            particle_filter.continuous_particles,
        )
    )


def test_exact_commit_preserves_unproposed_background_and_weight() -> None:
    """Accepted structural proposals must not mutate an unproposed background."""
    particle_filter = _build_filter()
    particle_index = next(
        index
        for index, particle in enumerate(particle_filter.continuous_particles)
        if particle.state.num_sources == 1
    )
    particle = particle_filter.continuous_particles[particle_index]
    particle.state.background = 7.25
    original_log_weight = float(particle.log_weight)
    patch_sets, strengths, _ = particle_filter._structural_rj_group_arrays(
        np.asarray([particle_index], dtype=np.int64),
        cardinality=1,
    )

    accepted_count = particle_filter._commit_structural_rj_states(
        np.asarray([particle_index], dtype=np.int64),
        np.asarray([True], dtype=bool),
        patch_sets,
        strengths,
    )

    assert accepted_count == 1
    assert particle.state.background == 7.25
    assert particle.log_weight == original_log_weight


def test_exact_birth_death_runtime_respects_cardinality_boundaries() -> None:
    """One runtime attempt may only move each boundary state inward or stay."""
    particle_filter = _build_filter(
        structural_rj_position_move_probability=0.0,
        structural_rj_local_position_move_probability=0.0,
        structural_rj_strength_move_probability=0.0,
    )
    data = _measurement_data(live_time=0.0)
    response_dictionary = (
        particle_filter._structural_rj_response_dictionary(data)
    )
    before = np.asarray(
        [
            particle.state.num_sources
            for particle in particle_filter.continuous_particles
        ],
        dtype=np.int64,
    )

    particle_filter._apply_structural_rj_birth_death(
        data,
        response_dictionary,
    )
    after = np.asarray(
        [
            particle.state.num_sources
            for particle in particle_filter.continuous_particles
        ],
        dtype=np.int64,
    )
    maximum = int(particle_filter.config.max_sources or 0)

    assert np.all((after >= 0) & (after <= maximum))
    assert np.all(np.isin(after[before == 0], [0, 1]))
    assert np.all(np.isin(after[before == maximum], [maximum - 1, maximum]))
    assert np.all(np.abs(after - before) <= 1)


def test_exact_rejuvenation_preserves_weights_support_and_disables_roughening() -> None:
    """RJ/MH moves may change states but must not change outer PF weights."""
    particle_filter = _build_filter()
    zero_information_data = _measurement_data(live_time=0.0)
    initial_cardinalities = np.asarray(
        [
            particle.state.num_sources
            for particle in particle_filter.continuous_particles
        ],
        dtype=np.int64,
    )

    for _ in range(5):
        log_weights_before = np.asarray(
            [
                particle.log_weight
                for particle in particle_filter.continuous_particles
            ],
            dtype=float,
        )
        particle_filter.apply_structural_moves(zero_information_data)
        log_weights_after = np.asarray(
            [
                particle.log_weight
                for particle in particle_filter.continuous_particles
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(log_weights_after, log_weights_before)

    final_cardinalities = np.asarray(
        [
            particle.state.num_sources
            for particle in particle_filter.continuous_particles
        ],
        dtype=np.int64,
    )
    assert np.any(final_cardinalities != initial_cardinalities)
    assert particle_filter.last_structural_timing_s["weights_preserved"] == 1.0
    for particle in particle_filter.continuous_particles:
        assert np.all(
            particle_filter._strength_prior.in_support(
                particle.state.strengths
            )
        )
        particle_filter._canonicalize_structural_rj_state(particle.state)

    positions_before = [
        particle.state.positions.copy()
        for particle in particle_filter.continuous_particles
    ]
    strengths_before = [
        particle.state.strengths.copy()
        for particle in particle_filter.continuous_particles
    ]
    particle_filter.regularize_continuous(
        sigma_pos=10.0,
        strength_log_sigma=10.0,
    )
    for particle, positions, strengths in zip(
        particle_filter.continuous_particles,
        positions_before,
        strengths_before,
    ):
        np.testing.assert_array_equal(particle.state.positions, positions)
        np.testing.assert_array_equal(particle.state.strengths, strengths)

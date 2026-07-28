"""Tests for the real-Geant4 acceptance backend contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from measurement.obstacles import ObstacleGrid
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    NATIVE_ACCEPTANCE_FIDELITY,
)
from spectrum.geant4_acceptance_backend import (
    ExternalGeant4AcceptanceBackend,
    _geometry_batch,
    _mutated_surface_scene,
    _native_fidelity,
)
from spectrum.response_matrix import NATIVE_GEANT4_BIN_COUNT
from spectrum.transport_spectral import GeometryConditionedSpectralModel


class _FakeKernel:
    """Record isotope-batched geometry calls and return physical tensors."""

    def __init__(self) -> None:
        """Initialize an empty call trace."""
        self.calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def line_transport_components_selected_pairs_for_detectors(
        self,
        isotope: str,
        detectors: np.ndarray,
        sources: np.ndarray,
        fe_indices: np.ndarray,
        pb_indices: np.ndarray,
        local_indices: np.ndarray,
    ) -> SimpleNamespace:
        """Return one finite attenuation batch for all requested pairs."""
        del fe_indices, pb_indices
        self.calls.append((isotope, sources.shape, local_indices.shape))
        shape = (detectors.shape[0], sources.shape[0], local_indices.size)
        unattenuated = np.full(shape, 2.0, dtype=np.float64)
        return SimpleNamespace(
            unattenuated_kernel=unattenuated,
            uncollided_kernel=0.75 * unattenuated,
            tau_fe=np.full(shape, 0.1, dtype=np.float64),
            tau_pb=np.full(shape, 0.2, dtype=np.float64),
            tau_obstacle=np.full(shape, 0.3, dtype=np.float64),
            tau_obstacle_compton=np.full(shape, 0.15, dtype=np.float64),
            distance_m=np.full(shape, 3.0, dtype=np.float64),
        )

    def line_branching_weights(
        self,
        isotope: str,
        local_indices: np.ndarray,
    ) -> np.ndarray:
        """Return normalized positive weights for one isotope subset."""
        del isotope
        return np.full(
            local_indices.shape,
            1.0 / max(local_indices.size, 1),
            dtype=np.float64,
        )


class _Source:
    """Minimal source object consumed by the batched geometry constructor."""

    def __init__(self, isotope: str, position: tuple[float, float, float]) -> None:
        """Store one isotope, transport position, and physical strength."""
        self.isotope = isotope
        self._position = np.asarray(position, dtype=np.float64)
        self.intensity_cps_1m = 500_000.0

    def transport_position_array(self) -> np.ndarray:
        """Return the continuous source transport coordinate."""
        return self._position.copy()


@pytest.mark.parametrize("use_gpu", (0, 1, "false", "true", None))
def test_acceptance_kernel_rejects_coerced_gpu_switch(
    use_gpu: object,
) -> None:
    """Acceptance must not reinterpret an invalid backend switch."""
    backend = object.__new__(ExternalGeant4AcceptanceBackend)
    backend.runtime_config = {"use_gpu": use_gpu}
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
    )

    with pytest.raises(TypeError, match="Acceptance use_gpu"):
        backend._kernel(grid)


def _config() -> SimpleNamespace:
    """Return a minimal exact native-fidelity application config."""
    return SimpleNamespace(
        engine_mode="external",
        physics_profile="balanced",
        thread_count=32,
        source_rate_model="detector_cps_1m",
        detector_scoring_mode="incident_gamma_energy",
        secondary_transport_mode="full_transport",
        primary_sampling_fraction=1.0,
        target_sampled_primaries=None,
        accelerated_weighted_transport_enable=False,
        sample_detector_response=True,
        validation_entry_class_spectra=True,
    )


def _metadata(*, source_count: int) -> dict[str, object]:
    """Return exact postconditions emitted by a native unit-history run."""
    payload: dict[str, object] = {
        "backend": "geant4",
        "engine_mode": "external",
        "physics_profile": "balanced",
        "source_rate_model": "detector_cps_1m",
        "detector_scoring_mode": "incident_gamma_energy",
        "secondary_transport_mode": "full_transport",
        "transport_history_mode": "full_unit_weight",
        "validation_entry_spectrum_space": (
            "pre_dead_time_raw_incident_gamma"
        ),
        "validation_entry_spectrum_grouping": (
            "source_token_initial_gamma_line_entry_class"
        ),
        "requested_threads": 32,
        "primary_sampling_fraction": 1.0,
        "primary_history_weight": 1.0,
        "target_sampled_primaries": 0,
        "spectrum_bin_count": NATIVE_GEANT4_BIN_COUNT,
        "multithreaded_run_manager": True,
        "primary_sampling_budget_enabled": False,
        "history_thinning_enabled": False,
        "transport_tally_weighted": False,
        "weighted_transport": False,
        "theory_tvl_attenuation": False,
        "detector_response_applied_in_native": True,
        "validation_entry_class_spectra": True,
        "source_bias_weighted_transport": False,
        "detector_response_sampling_contract_sha256": (
            NATIVE_ACCEPTANCE_FIDELITY[
                "detector_response_sampling_contract_sha256"
            ]
        ),
        "detector_response_sampling_mode": (
            "multinomial_marking_with_nonparalyzable_event_time"
        ),
        "all_sources_surface_bound": source_count > 0,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256() if source_count else ""
        ),
        "surface_emission_epsilon_m": (
            SURFACE_EMISSION_EPSILON_M if source_count else 0.0
        ),
    }
    return payload


def test_geometry_batch_uses_one_batched_call_per_present_isotope() -> None:
    """All 64 pairs must not be evaluated through scalar pair/source loops."""
    model = GeometryConditionedSpectralModel.standard_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
    )
    sources = (
        _Source("Cs-137", (1.0, 2.0, 3.0)),
        _Source("Co-60", (2.0, 3.0, 4.0)),
        _Source("Cs-137", (3.0, 4.0, 5.0)),
    )
    kernel = _FakeKernel()

    batch = _geometry_batch(
        kernel=kernel,
        model=model,
        detector_pose_xyz=(1.0, 1.0, 0.5),
        sources=sources,
    )

    assert len(kernel.calls) == 2
    assert {call[0] for call in kernel.calls} == {"Cs-137", "Co-60"}
    assert batch.unattenuated_vsl.shape == (64, 3, len(model.line_identity))
    assert batch.features_vslf.shape == (
        64,
        3,
        len(model.line_identity),
        4,
    )
    for source_index, source in enumerate(sources):
        wrong_lines = np.asarray(
            [
                line["isotope"] != source.isotope
                for line in model.line_identity
            ]
        )
        assert np.all(batch.unattenuated_vsl[:, source_index, wrong_lines] == 0)


@pytest.mark.parametrize("source_count", (0, 1))
def test_native_fidelity_accepts_exact_background_and_surface_runs(
    source_count: int,
) -> None:
    """Background and source scenes have distinct strict surface provenance."""
    assert _native_fidelity(
        _metadata(source_count=source_count),
        config=_config(),
        source_count=source_count,
    ) == NATIVE_ACCEPTANCE_FIDELITY


def test_native_fidelity_rejects_truthy_integer_boolean() -> None:
    """Native postconditions must not accept integers as booleans."""
    metadata = _metadata(source_count=1)
    metadata["weighted_transport"] = 0

    with pytest.raises(RuntimeError, match="must be boolean"):
        _native_fidelity(metadata, config=_config(), source_count=1)


def test_signed_surface_scene_variants_use_exact_normal_offsets() -> None:
    """The native gate must probe anchor, air+epsilon, and solid-epsilon."""
    scene = (
        "SOURCE isotope=Cs-137 x=1 y=2 z=3 "
        "anchor_x=1 anchor_y=2 anchor_z=3 "
        "surface_normal_x=0 surface_normal_y=0 surface_normal_z=1\n"
    )

    exact = _mutated_surface_scene(scene, variant="exact_surface_anchor")
    air = _mutated_surface_scene(scene, variant="air_plus_epsilon")
    solid = _mutated_surface_scene(scene, variant="solid_minus_epsilon")

    assert "z=3 " in exact
    assert f"z={format(3.0 + SURFACE_EMISSION_EPSILON_M, '.17g')} " in air
    assert f"z={format(3.0 - SURFACE_EMISSION_EPSILON_M, '.17g')} " in solid

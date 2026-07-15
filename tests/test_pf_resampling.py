"""Tests for PF resampling helper utilities."""

from __future__ import annotations

import numpy as np

from pf.particle_filter import IsotopeParticle, IsotopeParticleFilter, PFConfig
from pf.resampling import systematic_resample_count
from pf.state import IsotopeState


def test_systematic_resample_count_handles_empty_draw() -> None:
    """Zero requested draws should return an empty index array."""
    idx = systematic_resample_count(np.array([0.2, 0.8]), count=0)

    assert idx.dtype == np.int64
    assert idx.size == 0


def test_systematic_resample_count_normalizes_weights() -> None:
    """Positive non-normalized weights should be sampled as probabilities."""
    np.random.seed(3)

    idx = systematic_resample_count(np.array([0.0, 2.0, 8.0]), count=10)

    assert idx.shape == (10,)
    assert np.all(idx >= 0)
    assert np.all(idx < 3)
    assert np.count_nonzero(idx == 2) > np.count_nonzero(idx == 1)


def test_systematic_resample_count_falls_back_to_uniform() -> None:
    """Invalid total mass should fall back to uniform weights."""
    np.random.seed(1)

    idx = systematic_resample_count(np.array([0.0, 0.0, 0.0]), count=6)

    assert idx.shape == (6,)
    assert set(idx.tolist()).issubset({0, 1, 2})


def test_report_cardinality_hint_adds_resampling_quota() -> None:
    """Report-selected cardinality should receive extra mode-preserving quota."""
    config = PFConfig(
        num_particles=6,
        mode_preserving_resample=True,
        mode_preserving_max_modes=4,
        mode_preserving_particles_per_mode=1,
        mode_preserving_cardinality_strata=True,
        mode_preserving_min_particles_per_cardinality=1,
        mode_preserving_report_cardinality_strata=True,
        mode_preserving_report_cardinality_extra_particles=2,
    )
    filt = IsotopeParticleFilter("Cs-137", kernel=None, config=config)
    particles = []
    for idx, count in enumerate((1, 1, 1, 2, 2, 2)):
        positions = np.column_stack(
            [
                np.linspace(float(idx), float(idx) + count - 1, count),
                np.zeros(count),
                np.zeros(count),
            ]
        )
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=count,
                    positions=positions,
                    strengths=np.full(count, 100.0, dtype=float),
                    background=0.0,
                ),
                log_weight=float(-np.log(6.0)),
            )
        )
    filt.continuous_particles = particles
    filt.set_external_protected_cardinalities({2})

    protected = filt._source_mode_preserving_indices(  # noqa: SLF001
        np.full(6, 1.0 / 6.0, dtype=float)
    )

    details = {
        int(entry["num_sources"]): entry
        for entry in filt.last_mode_preserving_selected_cardinalities
    }
    assert protected.size > 0
    assert details[2]["externally_protected"] is True
    assert details[2]["target_protected_count"] == 3
    assert int(details[2]["protected_count"]) >= 1


def test_dynamic_cardinality_allocation_adds_entropy_quota() -> None:
    """High cardinality entropy should add dynamic fixed-budget protection."""
    config = PFConfig(
        num_particles=6,
        mode_preserving_resample=True,
        mode_preserving_max_modes=4,
        mode_preserving_particles_per_mode=1,
        mode_preserving_cardinality_strata=True,
        mode_preserving_min_particles_per_cardinality=1,
        mode_preserving_report_cardinality_strata=False,
        mode_preserving_dynamic_cardinality_allocation=True,
        mode_preserving_dynamic_cardinality_extra_particles=2,
        mode_preserving_dynamic_cardinality_min_mass=0.1,
        mode_preserving_dynamic_cardinality_entropy_min=0.1,
    )
    filt = IsotopeParticleFilter("Cs-137", kernel=None, config=config)
    particles = []
    for idx, count in enumerate((1, 1, 1, 3, 3, 3)):
        positions = np.column_stack(
            [
                np.linspace(float(idx), float(idx) + count - 1, count),
                np.zeros(count),
                np.zeros(count),
            ]
        )
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=count,
                    positions=positions,
                    strengths=np.full(count, 100.0, dtype=float),
                    background=0.0,
                ),
                log_weight=float(-np.log(6.0)),
            )
        )
    filt.continuous_particles = particles

    protected = filt._source_mode_preserving_indices(  # noqa: SLF001
        np.full(6, 1.0 / 6.0, dtype=float)
    )

    details = {
        int(entry["num_sources"]): entry
        for entry in filt.last_mode_preserving_selected_cardinalities
    }
    assert protected.size > 0
    assert details[1]["dynamically_protected"] is True
    assert details[3]["dynamically_protected"] is True
    assert details[1]["target_protected_count"] == 3
    assert details[3]["target_protected_count"] == 3


def test_dynamic_spatial_allocation_protects_cardinality_modes() -> None:
    """Dynamic spatial quota should protect source-count strata per mode."""
    config = PFConfig(
        num_particles=6,
        mode_preserving_resample=True,
        mode_preserving_max_modes=1,
        mode_preserving_particles_per_mode=1,
        mode_preserving_radius_m=2.0,
        mode_preserving_min_weight_fraction=0.0,
        mode_preserving_cardinality_strata=False,
        mode_preserving_dynamic_spatial_allocation=True,
        mode_preserving_dynamic_spatial_extra_particles=1,
        mode_preserving_dynamic_spatial_min_score_fraction=0.0,
    )
    filt = IsotopeParticleFilter("Cs-137", kernel=None, config=config)
    particles = []
    for count in (1, 1, 3, 3, 3, 3):
        positions = np.column_stack(
            [
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
            ]
        )
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=count,
                    positions=positions,
                    strengths=np.full(count, 100.0, dtype=float),
                    background=0.0,
                ),
                log_weight=float(-np.log(6.0)),
            )
        )
    filt.continuous_particles = particles

    protected = filt._source_mode_preserving_indices(  # noqa: SLF001
        np.full(6, 1.0 / 6.0, dtype=float)
    )

    selected = filt.last_mode_preserving_selected_strata
    dynamic = filt.last_mode_preserving_dynamic_spatial_summary
    assert protected.size >= 3
    assert selected
    assert selected[0]["dynamic_spatial_protected"] is True
    assert dynamic
    assert set(dynamic[0]["protected_by_cardinality"].keys()) == {"1", "3"}

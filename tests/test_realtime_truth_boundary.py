"""Structural guards for the truth-free online inference boundary."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from measurement.model import EnvironmentConfig, PointSource
from measurement.source_boundary import (
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from measurement.source_surfaces import generate_surface_sources
from measurement.surface_charts import (
    build_surface_chart_geometry,
    surface_chart_geometry_sha256,
)
import realtime_demo


def test_truth_overlay_is_constructed_only_after_log_finalization() -> None:
    """Live PF/planning must not receive truth before acquisition is final."""
    source = inspect.getsource(realtime_demo.run_live_pf)
    finalize_index = source.index("measurement_log_writer.finalize()")
    truth_overlay_index = source.index("_build_visualizer(include_truth=True)")

    assert finalize_index < truth_overlay_index
    assert "_build_visualizer(include_truth=True)" not in source[:finalize_index]
    assert '"true_sources": true_src if include_truth else {}' in source
    assert '"true_strengths": true_strengths if include_truth else {}' in source
    assert "true_sources=true_src" not in source
    assert "true_strengths=true_strengths" not in source


def test_truth_preflight_rejects_unidentifiable_colocated_sources() -> None:
    """Two identical same-isotope anchors cannot define an identifiable K=2 truth."""
    sources = [
        PointSource("Cs-137", (1.0, 2.0, 0.0), 500_000.0),
        PointSource("Cs-137", (1.0, 2.0, 0.0), 700_000.0),
    ]

    with pytest.raises(ValueError, match="cardinality is not identifiable"):
        realtime_demo._validate_truth_within_pf_state_support(
            sources,
            candidate_isotopes=("Cs-137",),
            max_sources_per_isotope=5,
            strength_prior_min_cps_1m=300_000.0,
            strength_prior_max_cps_1m=2_000_000.0,
        )


def test_truth_preflight_allows_colocated_different_isotopes() -> None:
    """Joint spectra can distinguish different isotopes at one physical anchor."""
    sources = [
        PointSource("Cs-137", (1.0, 2.0, 0.0), 500_000.0),
        PointSource("Co-60", (1.0, 2.0, 0.0), 700_000.0),
    ]

    realtime_demo._validate_truth_within_pf_state_support(
        sources,
        candidate_isotopes=("Cs-137", "Co-60"),
        max_sources_per_isotope=5,
        strength_prior_min_cps_1m=300_000.0,
        strength_prior_max_cps_1m=2_000_000.0,
    )


def _declared_surface_source_fixture() -> tuple[
    list[PointSource],
    object,
    dict[str, object],
]:
    """Return one schema-3 source file contract and its exact room atlas."""
    environment = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    geometry = build_surface_chart_geometry(
        environment,
        None,
        max_edge_m=1.0,
    )
    sources = generate_surface_sources(
        env=environment,
        obstacle_grid=None,
        isotopes=("Cs-137",),
        intensity_cps_1m=300_000.0,
        rng=np.random.default_rng(20260728),
        count=1,
        chart_max_edge_m=1.0,
    )
    payloads = [
        realtime_demo._source_runtime_payload(source) for source in sources
    ]
    metadata: dict[str, object] = {
        "source_surface_sampling_schema_version": 3,
        "obstacle_seed": 17,
        "sampling_measure": "continuous_area_uniform",
        "selection_conditioning": "none_physical_area_only",
        "surface_atlas_contract_sha256": (
            surface_chart_geometry_sha256(geometry)
        ),
        "surface_chart_max_edge_m": 1.0,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "surface_source_runtime_contract_sha256": (
            surface_source_runtime_contract_sha256(payloads)
        ),
    }
    return sources, geometry, metadata


def test_declared_surface_source_contract_matches_runtime_atlas() -> None:
    """Schema-3 truth should retain its generation atlas through preflight."""
    sources, geometry, metadata = _declared_surface_source_fixture()

    realtime_demo._validate_provided_surface_source_contract(
        {"provided_file_declared_metadata": metadata},
        sources,
        chart_geometry=geometry,
        obstacle_seed=17,
        chart_max_edge_m=1.0,
    )


def test_surface_binding_does_not_relocate_authoritative_chart_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-3 chart/UV truth must bypass the ambiguous XYZ resolver."""
    sources, geometry, _ = _declared_surface_source_fixture()

    def _unexpected_resolver_call(*args: object, **kwargs: object) -> object:
        """Fail if authoritative truth is sent through nearest-chart lookup."""
        raise AssertionError("authoritative chart/UV was re-located")

    monkeypatch.setattr(
        realtime_demo.ContinuousSurfaceAtlas,
        "locate_positions",
        _unexpected_resolver_call,
    )

    bound = realtime_demo._bind_sources_to_surface_transport(
        sources,
        geometry,
    )

    assert len(bound) == len(sources)
    assert bound[0].surface_chart_id == sources[0].surface_chart_id
    assert bound[0].surface_uv == sources[0].surface_uv
    assert bound[0].position == sources[0].position


def test_declared_surface_source_contract_rejects_xyz_only_entries() -> None:
    """Schema-3 truth must not reconstruct chart identity from ambiguous XYZ."""
    sources, geometry, metadata = _declared_surface_source_fixture()
    xyz_only_sources = [
        PointSource(
            isotope=source.isotope,
            position=source.position,
            intensity_cps_1m=source.intensity_cps_1m,
        )
        for source in sources
    ]

    with pytest.raises(ValueError, match="authoritative chart/UV"):
        realtime_demo._validate_provided_surface_source_contract(
            {"provided_file_declared_metadata": metadata},
            xyz_only_sources,
            chart_geometry=geometry,
            obstacle_seed=17,
            chart_max_edge_m=1.0,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "surface_atlas_contract_sha256",
            "0" * 64,
            "different continuous surface atlas",
        ),
        ("obstacle_seed", 18, "different obstacle seed"),
    ),
)
def test_declared_surface_source_contract_rejects_stale_geometry(
    field: str,
    replacement: object,
    message: str,
) -> None:
    """A stale generated layout must not be rebound onto another environment."""
    sources, geometry, metadata = _declared_surface_source_fixture()
    metadata[field] = replacement

    with pytest.raises(ValueError, match=message):
        realtime_demo._validate_provided_surface_source_contract(
            {"provided_file_declared_metadata": metadata},
            sources,
            chart_geometry=geometry,
            obstacle_seed=17,
            chart_max_edge_m=1.0,
        )

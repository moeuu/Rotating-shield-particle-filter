"""Complete physical-area quadrature for continuous source surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SurfaceAtlasQuadrature:
    """Represent a complete area-weighted chart-center surface quadrature."""

    positions_s3: NDArray[np.float64]
    area_weights_m2_s: NDArray[np.float64]
    chart_ids_s: NDArray[np.int64]
    chart_count: int
    total_area_m2: float
    maximum_hausdorff_bound_m: float
    kinds: tuple[str, ...]
    face_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate complete one-sample-per-chart quadrature semantics."""
        positions = np.asarray(self.positions_s3, dtype=np.float64)
        weights = np.asarray(self.area_weights_m2_s, dtype=np.float64).reshape(-1)
        chart_ids = np.asarray(self.chart_ids_s, dtype=np.int64).reshape(-1)
        count = int(self.chart_count)
        total_area = float(self.total_area_m2)
        hausdorff = float(self.maximum_hausdorff_bound_m)
        if (
            count <= 0
            or positions.shape != (count, 3)
            or weights.shape != (count,)
            or chart_ids.shape != (count,)
            or len(self.kinds) != count
            or len(self.face_ids) != count
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.array_equal(
                chart_ids,
                np.arange(count, dtype=np.int64),
            )
            or not np.isfinite(total_area)
            or total_area <= 0.0
            or not np.isclose(
                float(np.sum(weights, dtype=np.float64)),
                total_area,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            or not np.isfinite(hausdorff)
            or hausdorff < 0.0
        ):
            raise ValueError(
                "Surface quadrature must contain every chart exactly once "
                "with finite positive physical area."
            )
        object.__setattr__(
            self,
            "positions_s3",
            np.ascontiguousarray(positions),
        )
        object.__setattr__(
            self,
            "area_weights_m2_s",
            np.ascontiguousarray(weights),
        )
        object.__setattr__(
            self,
            "chart_ids_s",
            np.ascontiguousarray(chart_ids),
        )

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-safe completeness and spacing provenance."""
        unique_kinds, kind_counts = np.unique(
            np.asarray(self.kinds, dtype=object),
            return_counts=True,
        )
        return {
            "contract": "complete_chart_center_area_quadrature_v1",
            "sample_count": int(self.chart_count),
            "chart_count": int(self.chart_count),
            "total_area_m2": float(self.total_area_m2),
            "maximum_hausdorff_bound_m": float(self.maximum_hausdorff_bound_m),
            "physical_face_count": int(len(set(self.face_ids))),
            "surface_kind_chart_counts": {
                str(kind): int(count)
                for kind, count in zip(
                    unique_kinds,
                    kind_counts,
                    strict=True,
                )
            },
            "every_chart_represented": True,
            "area_weighted": True,
        }


def build_complete_surface_atlas_quadrature(
    atlas: object,
    *,
    max_points: int,
    maximum_hausdorff_bound_m: float,
) -> SurfaceAtlasQuadrature:
    """Build a fail-closed one-center-per-chart physical-area quadrature."""
    budget = int(max_points)
    requested_bound = float(maximum_hausdorff_bound_m)
    if budget <= 0:
        raise ValueError("Surface quadrature max_points must be positive.")
    if not np.isfinite(requested_bound) or requested_bound <= 0.0:
        raise ValueError(
            "Surface quadrature Hausdorff bound must be finite and positive."
        )
    chart_count = int(getattr(atlas, "chart_count"))
    if chart_count > budget:
        raise RuntimeError(
            "Surface coverage quadrature budget cannot represent every "
            f"chart ({chart_count} > {budget}). Increase the predeclared "
            "coverage_surface_quadrature_max_points budget."
        )
    geometry = getattr(atlas, "geometry")
    vertices = np.asarray(geometry.vertices_xyz, dtype=np.float64)
    centers = np.asarray(geometry.centers_xyz, dtype=np.float64)
    if (
        vertices.shape != (chart_count, 4, 3)
        or centers.shape != (chart_count, 3)
        or np.any(~np.isfinite(vertices))
        or np.any(~np.isfinite(centers))
    ):
        raise RuntimeError(
            "Surface atlas quadrature requires finite quadrilateral charts."
        )
    chart_center_vertex_radius = np.max(
        np.linalg.norm(
            vertices - centers[:, None, :],
            axis=2,
        ),
        axis=1,
    )
    maximum_bound = float(np.max(chart_center_vertex_radius))
    if maximum_bound > requested_bound + 1.0e-12:
        raise RuntimeError(
            "Surface chart-center quadrature exceeds the predeclared "
            "Hausdorff bound "
            f"({maximum_bound:.6g} m > {requested_bound:.6g} m). "
            "Refine the continuous surface atlas before planning."
        )
    chart_ids = np.arange(chart_count, dtype=np.int64)
    uv = np.full((chart_count, 2), 0.5, dtype=np.float64)
    positions = np.asarray(
        atlas.positions_xyz(chart_ids, uv),
        dtype=np.float64,
    )
    if not np.allclose(
        positions,
        centers,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Surface atlas center coordinates and chart mapping differ.")
    return SurfaceAtlasQuadrature(
        positions_s3=positions,
        area_weights_m2_s=np.asarray(
            geometry.areas_m2,
            dtype=np.float64,
        ).copy(),
        chart_ids_s=chart_ids,
        chart_count=chart_count,
        total_area_m2=float(getattr(atlas, "total_area_m2")),
        maximum_hausdorff_bound_m=maximum_bound,
        kinds=tuple(str(value) for value in geometry.kinds),
        face_ids=tuple(str(value) for value in geometry.face_ids),
    )

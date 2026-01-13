"""Model octant-shaped Pb/Fe shields and their attenuation/blocking logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:  # optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False

# Signed unit normals for the eight octants ((+,+,+), (+,+,-), ...).
OCTANT_NORMALS: NDArray[np.float64] = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=float,
)
OCTANT_NORMALS /= np.linalg.norm(OCTANT_NORMALS, axis=1, keepdims=True)

OCTANT_THETA_PHI_RANGES: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((0.0, np.pi / 2.0), (0.0, np.pi / 2.0)),
    ((np.pi / 2.0, np.pi), (0.0, np.pi / 2.0)),
    ((0.0, np.pi / 2.0), (3.0 * np.pi / 2.0, 2.0 * np.pi)),
    ((np.pi / 2.0, np.pi), (3.0 * np.pi / 2.0, 2.0 * np.pi)),
    ((0.0, np.pi / 2.0), (np.pi / 2.0, np.pi)),
    ((np.pi / 2.0, np.pi), (np.pi / 2.0, np.pi)),
    ((0.0, np.pi / 2.0), (np.pi, 3.0 * np.pi / 2.0)),
    ((np.pi / 2.0, np.pi), (np.pi, 3.0 * np.pi / 2.0)),
]

# Half-value layer (HVL) and tenth-value layer (TVL) in millimeters.
HVL_TVL_TABLE_MM: dict[str, dict[str, dict[str, float]]] = {
    "Cs-137": {"pb": {"hvl": 7.0, "tvl": 22.0}, "fe": {"hvl": 15.0, "tvl": 50.0}},
    "Co-60": {"pb": {"hvl": 12.0, "tvl": 40.0}, "fe": {"hvl": 20.0, "tvl": 67.0}},
    "Eu-154": {"pb": {"hvl": 7.4, "tvl": 24.6}, "fe": {"hvl": 13.8, "tvl": 45.8}},
}
CS137_TVL_PB_MM = float(HVL_TVL_TABLE_MM["Cs-137"]["pb"]["tvl"])
CS137_TVL_FE_MM = float(HVL_TVL_TABLE_MM["Cs-137"]["fe"]["tvl"])


def mu_from_tvl_mm(tvl_mm: float) -> float:
    """Return linear attenuation coefficient (1/cm) for a TVL given in millimeters."""
    if tvl_mm <= 0:
        raise ValueError("tvl_mm must be positive.")
    return float(np.log(10.0) / (tvl_mm / 10.0))


def mu_by_isotope_from_tvl_mm(
    table_mm: dict[str, dict[str, dict[str, float]]],
    isotopes: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Build per-isotope attenuation coefficients from TVL values (mm).

    Returns:
        dict: {isotope: {"fe": mu_fe, "pb": mu_pb}} with mu in 1/cm.
    """
    isotopes = list(isotopes) if isotopes is not None else list(table_mm.keys())
    mu_by_isotope: dict[str, dict[str, float]] = {}
    for iso in isotopes:
        entry = table_mm.get(iso)
        if entry is None:
            continue
        fe_tvl = entry.get("fe", {}).get("tvl")
        pb_tvl = entry.get("pb", {}).get("tvl")
        if fe_tvl is None or pb_tvl is None:
            continue
        mu_by_isotope[iso] = {
            "fe": mu_from_tvl_mm(float(fe_tvl)),
            "pb": mu_from_tvl_mm(float(pb_tvl)),
        }
    return mu_by_isotope


def cartesian_to_spherical(vec: NDArray[np.float64]) -> Tuple[float, float, float]:
    """
    Convert a Cartesian vector to spherical coordinates (r, theta, phi).

    theta: polar angle from the z-axis [0, pi]
    phi: azimuth from the x-axis in the xy-plane [0, 2pi)
    """
    x, y, z = vec
    r = float(np.linalg.norm(vec))
    if r == 0:
        return 0.0, 0.0, 0.0
    theta = float(np.arccos(z / r))
    phi = float(np.arctan2(y, x) % (2 * np.pi))
    return r, theta, phi


def blocks_ray_torch(direction: "torch.Tensor", octant_index: int, tol: float = 1e-6) -> "torch.Tensor":
    """
    Return a boolean mask indicating which directions fall in an octant (torch).

    direction: (N, 3) tensor of direction vectors.
    """
    if torch is None:
        raise RuntimeError("torch is not available")
    if octant_index < 0 or octant_index >= len(OCTANT_THETA_PHI_RANGES):
        raise ValueError("octant_index must be in [0, 7]")
    r = torch.linalg.norm(direction, dim=1)
    tol_t = torch.as_tensor(tol, device=direction.device, dtype=direction.dtype)
    r = torch.where(r <= tol_t, tol_t, r)
    dir_unit = direction / r.unsqueeze(-1)
    theta = torch.acos(torch.clamp(dir_unit[:, 2], -1.0, 1.0))
    phi = torch.remainder(torch.atan2(dir_unit[:, 1], dir_unit[:, 0]), 2 * np.pi)
    (theta_low, theta_high), (phi_low, phi_high) = OCTANT_THETA_PHI_RANGES[octant_index]
    return (
        (theta + tol_t >= theta_low)
        & (theta - tol_t < theta_high)
        & (phi + tol_t >= phi_low)
        & (phi - tol_t < phi_high)
    )


def path_length_cm_torch(
    direction: "torch.Tensor",
    shield_normal: "torch.Tensor",
    thickness_cm: float,
    blocked_mask: "torch.Tensor",
    use_angle_attenuation: bool = False,
    tol: float = 1e-9,
) -> "torch.Tensor":
    """
    Compute path length through the octant shell (torch, batched).

    direction: (N, 3) tensor, shield_normal: (3,) tensor.
    """
    if torch is None:
        raise RuntimeError("torch is not available")
    norm = torch.linalg.norm(direction, dim=1)
    tol_t = torch.as_tensor(tol, device=direction.device, dtype=direction.dtype)
    norm = torch.where(norm <= tol_t, tol_t, norm)
    dir_unit = direction / norm.unsqueeze(-1)
    cos_theta = torch.clamp(torch.sum(dir_unit * shield_normal, dim=1), 0.0, 1.0)
    thickness = torch.as_tensor(thickness_cm, device=direction.device, dtype=direction.dtype)
    if use_angle_attenuation:
        return torch.where(blocked_mask & (cos_theta > tol_t), thickness / cos_theta, torch.zeros_like(cos_theta))
    return torch.where(blocked_mask, thickness, torch.zeros_like(cos_theta))


def shield_blocks_radiation(direction: NDArray[np.float64], shield_normal: NDArray[np.float64], tol: float = 1e-6) -> bool:
    """
    Check whether a direction points into the selected octant shield.

    direction: source-to-detector (or detector-to-source) direction vector
    shield_normal: octant normal (one of OCTANT_NORMALS)
    """
    if np.linalg.norm(direction) == 0:
        return False
    dir_unit = direction / np.linalg.norm(direction)
    # Only block when direction signs match the octant (1/8 shell model).
    sign_dir = np.sign(np.where(np.abs(dir_unit) < tol, 0.0, dir_unit))
    sign_shield = np.sign(shield_normal)
    return bool(np.all(sign_dir == sign_shield))


def resolve_mu_values(
    mu_by_isotope: dict[str, object] | None,
    isotope: str,
    default_fe: float,
    default_pb: float,
) -> Tuple[float, float]:
    """
    Resolve per-isotope attenuation coefficients for Fe/Pb.

    Accepts one of:
        - float: use the same mu for Fe and Pb
        - (mu_fe, mu_pb) tuple/list
        - dict with "fe"/"pb" keys
    Falls back to defaults if isotope is missing or value is None.
    """
    if mu_by_isotope is None:
        return float(default_fe), float(default_pb)
    mu_val = mu_by_isotope.get(isotope)
    if mu_val is None:
        return float(default_fe), float(default_pb)
    if isinstance(mu_val, dict):
        mu_fe = float(mu_val.get("fe", default_fe))
        mu_pb = float(mu_val.get("pb", default_pb))
        return mu_fe, mu_pb
    if isinstance(mu_val, (tuple, list, np.ndarray)) and len(mu_val) == 2:
        return float(mu_val[0]), float(mu_val[1])
    if isinstance(mu_val, (int, float)):
        mu = float(mu_val)
        return mu, mu
    raise ValueError(f"Unsupported mu_by_isotope entry for {isotope}: {mu_val!r}")


def path_length_cm(
    direction: NDArray[np.float64],
    shield_normal: NDArray[np.float64],
    thickness_cm: float,
    blocked: bool | None = None,
    use_angle_attenuation: bool = False,
    tol: float = 1e-9,
) -> float:
    """
    Compute path length through the octant shell for a given direction and normal.

    When blocked is None, a sign-based octant check is used. Otherwise the caller can
    pass a precomputed blocked flag (e.g., OctantShield.blocks_ray).
    When use_angle_attenuation is False, the path length equals the nominal thickness.
    """
    if blocked is None:
        blocked = shield_blocks_radiation(direction, shield_normal)
    if not blocked:
        return 0.0
    if not use_angle_attenuation:
        return float(thickness_cm)
    norm = float(np.linalg.norm(direction))
    if norm <= tol:
        return 0.0
    dir_unit = direction / norm
    cos_theta = float(np.clip(np.dot(dir_unit, shield_normal), 0.0, 1.0))
    if cos_theta <= tol:
        return 0.0
    return float(thickness_cm / cos_theta)


@dataclass(frozen=True)
class SphericalOctantShield:
    """Simple 1/8 spherical shell shield model."""

    mu_cm_inv: float  # Material attenuation coefficient (1/cm).
    thickness_cm: float = 2.0  # Shell thickness.
    inner_radius_cm: float = 5.0  # Inner radius (detector center).

    def orientations(self) -> NDArray[np.float64]:
        """Return the octant normals."""
        return OCTANT_NORMALS.copy()

    def path_length_cm(self, direction: NDArray[np.float64], shield_normal: NDArray[np.float64]) -> float:
        """
        Return the path length through the shield for the given direction.

        Only directions into the octant return a thickness scaled by cos(theta).
        """
        if not shield_blocks_radiation(direction, shield_normal):
            return 0.0
        dir_unit = direction / (np.linalg.norm(direction) + 1e-9)
        cos_theta = float(np.clip(np.dot(dir_unit, shield_normal), 0.0, 1.0))
        if cos_theta <= 0.0:
            return 0.0
        return float(self.thickness_cm / cos_theta)

    def attenuation_factor(self, direction: NDArray[np.float64], shield_normal: NDArray[np.float64]) -> float:
        """
        Return the attenuation factor exp(-mu * L).

        direction: source-to-detector direction vector
        shield_normal: selected octant normal
        """
        path = self.path_length_cm(direction, shield_normal)
        if path <= 0.0:
            return 1.0
        return float(np.exp(-self.mu_cm_inv * path))


def lead_shield(thickness_cm: float = 2.0, inner_radius_cm: float = 5.0) -> SphericalOctantShield:
    """
    Construct a lead shield.

    Uses a representative mu_cm_inv of 0.7 1/cm.
    """
    return SphericalOctantShield(mu_cm_inv=0.7, thickness_cm=thickness_cm, inner_radius_cm=inner_radius_cm)


def iron_shield(thickness_cm: float = 2.0, inner_radius_cm: float = 5.0) -> SphericalOctantShield:
    """
    Construct an iron shield.

    Uses a representative mu_cm_inv of 0.5 1/cm.
    """
    return SphericalOctantShield(mu_cm_inv=0.5, thickness_cm=thickness_cm, inner_radius_cm=inner_radius_cm)


def _angle_in_range(angle: float, low: float, high: float, tol: float = 1e-6) -> bool:
    """Return True if angle is within [low, high) without wrap-around."""
    return (angle + tol) >= low and (angle - tol) < high


def generate_octant_orientations() -> NDArray[np.float64]:
    """
    Return the eight octant normals (shared by Pb and Fe).

    This is typically passed as the orientation matrix to KernelPrecomputer.
    """
    return OCTANT_NORMALS.copy()


def rotation_matrix_from_normal(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Generate a simple rotation matrix for an octant defined by a normal.

    The third column is aligned with the (normalized) octant normal. The first two
    columns are any orthonormal basis spanning the plane perpendicular to the normal.
    """
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("normal must be non-zero")
    n_unit = n / n_norm
    # choose a helper vector that is not collinear
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, n_unit)
    x_axis /= (np.linalg.norm(x_axis) + 1e-12)
    y_axis = np.cross(n_unit, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-12)
    R = np.stack([x_axis, y_axis, n_unit], axis=1)
    return R


def octant_index_from_rotation(R: NDArray[np.float64]) -> int:
    """
    Map a rotation matrix (diagonal sign matrix) back to its octant index.

    Uses the third column as the normal, consistent with expected_counts_single_isotope.
    """
    n = R[:, 2] if R.shape == (3, 3) else np.asarray(R, dtype=float)
    return octant_index_from_normal(n)


def generate_octant_rotation_matrices() -> NDArray[np.float64]:
    """Return 8 diagonal rotation matrices corresponding to the octant normals."""
    normals = generate_octant_orientations()
    mats = [rotation_matrix_from_normal(n) for n in normals]
    return np.stack(mats, axis=0)


def generate_fe_pb_orientation_pairs() -> list[dict]:
    """
    Generate candidate orientation pairs (RFe, RPb) per Sec. 3.4.1.

    Returns a list of dictionaries with keys:
        - id: integer orientation ID
        - fe_index, pb_index: octant indices for Fe/Pb
        - RFe, RPb: 3x3 rotation matrices (diagonal sign matrices)
    """
    fe_normals = generate_octant_orientations()
    pb_normals = generate_octant_orientations()
    fe_mats = generate_octant_rotation_matrices()
    pb_mats = generate_octant_rotation_matrices()
    pairs = []
    oid = 0
    for i in range(len(fe_normals)):
        for j in range(len(pb_normals)):
            pairs.append(
                {
                    "id": oid,
                    "fe_index": i,
                    "pb_index": j,
                    "RFe": fe_mats[i],
                    "RPb": pb_mats[j],
                }
            )
            oid += 1
    return pairs


def octant_index_from_normal(normal: NDArray[np.float64], tol: float = 1e-6) -> int:
    """
    Return the nearest octant index for a given normal.

    For axis-aligned vectors (e.g., [1, 0, 0]) choose the octant with maximum dot.
    """
    n = np.asarray(normal, dtype=float)
    if np.linalg.norm(n) == 0:
        raise ValueError("normal must be non-zero")
    n_unit = n / np.linalg.norm(n)
    dots = OCTANT_NORMALS @ n_unit
    return int(np.argmax(dots))


class OctantShield:
    """
    Geometric blocking test for a 1/8 spherical shell shield.

    Computes vector v = detector - source and uses spherical coordinates (theta, phi)
    to determine whether the ray intersects a given octant (0..7).
    - theta: polar angle [0, pi] from the z-axis
    - phi: azimuth [0, 2pi) from the x-axis in the xy-plane
    """

    def __init__(
        self,
        material: str | None = None,
        orientation_index: int = 0,
        octant_normals: NDArray[np.float64] | None = None,
    ) -> None:
        self.material = material or "generic"
        self.orientation_index = orientation_index
        self.octant_normals = octant_normals if octant_normals is not None else OCTANT_NORMALS
        # Define (theta, phi) ranges for each octant.
        self.theta_phi_ranges = [
            ((0.0, np.pi / 2.0), (0.0, np.pi / 2.0)),  # + + +
            ((np.pi / 2.0, np.pi), (0.0, np.pi / 2.0)),  # + + -
            ((0.0, np.pi / 2.0), (3.0 * np.pi / 2.0, 2.0 * np.pi)),  # + - +
            ((np.pi / 2.0, np.pi), (3.0 * np.pi / 2.0, 2.0 * np.pi)),  # + - -
            ((0.0, np.pi / 2.0), (np.pi / 2.0, np.pi)),  # - + +
            ((np.pi / 2.0, np.pi), (np.pi / 2.0, np.pi)),  # - + -
            ((0.0, np.pi / 2.0), (np.pi, 3.0 * np.pi / 2.0)),  # - - +
            ((np.pi / 2.0, np.pi), (np.pi, 3.0 * np.pi / 2.0)),  # - - -
        ]

    def blocks_ray(
        self,
        detector_position: NDArray[np.float64],
        source_position: NDArray[np.float64],
        octant_index: int | None = None,
    ) -> bool:
        """
        Determine if the ray from source to detector passes through the octant.

        - v = detector - source
        - compute (theta, phi) and check whether it falls in the octant range
        If octant_index is None, use the instance's orientation_index.
        """
        idx = self.orientation_index if octant_index is None else octant_index
        if idx < 0 or idx >= len(self.theta_phi_ranges):
            raise ValueError("octant_index must be in [0, 7]")
        v = np.asarray(detector_position, dtype=float) - np.asarray(source_position, dtype=float)
        r, theta, phi = cartesian_to_spherical(v)
        if r == 0.0:
            return False
        (theta_low, theta_high), (phi_low, phi_high) = self.theta_phi_ranges[idx]
        in_theta = _angle_in_range(theta, theta_low, theta_high)
        in_phi = _angle_in_range(phi, phi_low, phi_high)
        return bool(in_theta and in_phi)

"""Shield geometry and attenuation tests for 1/8 spherical shells."""

import numpy as np
import pytest

from measurement.shielding import (
    OCTANT_NORMALS,
    OctantShield,
    generate_octant_orientations,
    octant_index_from_normal,
    cartesian_to_spherical,
    iron_shield,
    lead_shield,
    shield_blocks_radiation,
)
from measurement.kernels import KernelPrecomputer, ShieldParams


def test_cartesian_to_spherical_roundtrip_axes() -> None:
    """Basic spherical conversion sanity on unit axes."""
    r, theta, phi = cartesian_to_spherical(np.array([0.0, 0.0, 1.0]))
    assert r == 1.0
    assert abs(theta - 0.0) < 1e-6
    assert abs(phi - 0.0) < 1e-6


def test_shield_blocks_matching_octant_only() -> None:
    """A direction is blocked only when signs match the octant."""
    direction = np.array([1.0, 1.0, 1.0])
    assert shield_blocks_radiation(direction, OCTANT_NORMALS[0])
    assert not shield_blocks_radiation(direction, OCTANT_NORMALS[7])


def test_attenuation_respects_thickness() -> None:
    """Attenuation reduces flux when heading into the shield, none otherwise."""
    direction = np.array([1.0, 1.0, 1.0])
    pb = lead_shield(thickness_cm=2.0)
    fe = iron_shield(thickness_cm=2.0)
    blocked_pb = pb.attenuation_factor(direction, OCTANT_NORMALS[0])
    blocked_fe = fe.attenuation_factor(direction, OCTANT_NORMALS[0])
    unblocked = pb.attenuation_factor(direction, OCTANT_NORMALS[7])
    assert blocked_pb < 1.0
    assert blocked_fe < 1.0
    assert unblocked == 1.0


def test_octant_shield_blocks_ray_by_angles() -> None:
    """blocks_ray should map directions to correct octant via (theta, phi)."""
    shield = OctantShield()
    src = np.array([0.0, 0.0, 0.0])
    det_ppp = np.array([1.0, 1.0, 1.0])
    det_mmp = np.array([-1.0, -1.0, 1.0])
    assert shield.blocks_ray(src, det_ppp, octant_index=0)
    assert not shield.blocks_ray(src, det_ppp, octant_index=7)
    assert shield.blocks_ray(src, det_mmp, octant_index=6)
    assert not shield.blocks_ray(src, det_mmp, octant_index=1)


def test_kernel_attenuation_factor_applies_tenth_when_blocked() -> None:
    """KernelPrecomputer should scale to 0.1 when octant blocks the ray."""
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    poses = np.array([[1.0, 1.0, 1.0]], dtype=float)
    orientations = OCTANT_NORMALS
    mu = {"Cs-137": 0.5}
    kernel = KernelPrecomputer(
        candidate_sources=candidate_sources,
        poses=poses,
        orientations=orientations,
        shield_params=ShieldParams(),
        mu_by_isotope=mu,
    )
    k_block = kernel.kernel("Cs-137", pose_idx=0, orient_idx=0)[0]
    k_free = kernel.kernel("Cs-137", pose_idx=0, orient_idx=7)[0]
    assert np.isclose(k_block, 0.1 * k_free, rtol=1e-6)


def test_generate_octant_orientations_and_index() -> None:
    """Orientation helper should return 8 normals and index mapping should be consistent."""
    normals = generate_octant_orientations()
    assert normals.shape == (8, 3)
    for i, n in enumerate(normals):
        idx = octant_index_from_normal(n)
        assert idx == i


def test_rotation_changes_counts_by_tenth_when_blocked() -> None:
    """
    Rotating from an unblocked to a blocked octant should reduce counts to ~1/10.

    This reproduces the qualitative behavior in IAS-19 Fig. 1: orientation
    modulates count rate and provides directional information.
    """
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    poses = np.array([[1.0, 1.0, 1.0]], dtype=float)
    orientations = generate_octant_orientations()
    mu = {"Cs-137": 0.5}
    kernel = KernelPrecomputer(
        candidate_sources=candidate_sources,
        poses=poses,
        orientations=orientations,
        shield_params=ShieldParams(),
        mu_by_isotope=mu,
    )
    strength = np.array([10.0])
    unblocked = kernel.expected_counts("Cs-137", pose_idx=0, orient_idx=7, source_strengths=strength, background=0.0)
    blocked = kernel.expected_counts("Cs-137", pose_idx=0, orient_idx=0, source_strengths=strength, background=0.0)
    assert blocked == pytest.approx(0.1 * unblocked, rel=1e-6)

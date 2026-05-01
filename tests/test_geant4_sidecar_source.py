"""Source-level regression checks for Geant4 sidecar geometry conventions."""

from pathlib import Path


def test_geant4_sidecar_uses_inverted_placement_rotation_for_placed_volumes() -> None:
    """Static volumes, shields, and detector housing should share placement rotation rules."""
    source = Path("native/geant4_sidecar/geant4_sidecar.cpp").read_text(encoding="utf-8")

    assert "QuaternionToShieldPlacementRotation" not in source
    assert "auto rotation = QuaternionToPlacementRotation(volume.qw, volume.qx, volume.qy, volume.qz);" in source
    assert "auto rotation = QuaternionToPlacementRotation(pose.qw, pose.qx, pose.qy, pose.qz);" in source
    assert "auto housing_rotation = QuaternionToPlacementRotation(" in source
    assert "rotation->invert();" in source

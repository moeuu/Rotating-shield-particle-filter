"""Material presets and attenuation helpers for the Isaac Sim bridge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MaterialPreset:
    """Describe a reusable material preset for attenuation fallback."""

    name: str
    density_g_cm3: float | None = None
    mass_att_by_isotope_cm2_g: dict[str, float] = field(default_factory=dict)
    composition_by_mass: dict[str, float] = field(default_factory=dict)


ELEMENTAL_MASS_ATT_CM2_G: dict[str, dict[str, float]] = {
    "H": {"Cs-137": 0.0840, "Co-60": 0.0610, "Eu-154": 0.0840},
    "C": {"Cs-137": 0.0770, "Co-60": 0.0530, "Eu-154": 0.0780},
    "N": {"Cs-137": 0.0760, "Co-60": 0.0520, "Eu-154": 0.0770},
    "O": {"Cs-137": 0.0770, "Co-60": 0.0530, "Eu-154": 0.0780},
    "Al": {"Cs-137": 0.0750, "Co-60": 0.0510, "Eu-154": 0.0760},
    "Si": {"Cs-137": 0.0740, "Co-60": 0.0500, "Eu-154": 0.0750},
    "Ca": {"Cs-137": 0.0745, "Co-60": 0.0505, "Eu-154": 0.0755},
    "Cr": {"Cs-137": 0.0720, "Co-60": 0.0480, "Eu-154": 0.0730},
    "Fe": {"Cs-137": 0.0710, "Co-60": 0.0460, "Eu-154": 0.0720},
    "Ni": {"Cs-137": 0.0715, "Co-60": 0.0465, "Eu-154": 0.0725},
    "Ar": {"Cs-137": 0.0760, "Co-60": 0.0520, "Eu-154": 0.0770},
    "Pb": {"Cs-137": 0.1100, "Co-60": 0.0575, "Eu-154": 0.1110},
}

ELEMENTAL_MASS_ATT_CURVES_CM2_G: dict[str, dict[float, float]] = {
    "H": {662.0: 0.0840, 1000.0: 0.0710, 1600.0: 0.0580},
    "C": {662.0: 0.0770, 1000.0: 0.0640, 1600.0: 0.0530},
    "N": {662.0: 0.0760, 1000.0: 0.0630, 1600.0: 0.0520},
    "O": {662.0: 0.0770, 1000.0: 0.0640, 1600.0: 0.0530},
    "Al": {662.0: 0.0750, 1000.0: 0.0620, 1600.0: 0.0510},
    "Si": {662.0: 0.0740, 1000.0: 0.0610, 1600.0: 0.0500},
    "Ca": {662.0: 0.0745, 1000.0: 0.0615, 1600.0: 0.0505},
    "Cr": {662.0: 0.0600, 1000.0: 0.0510, 1600.0: 0.0435},
    "Fe": {662.0: 0.0585, 1000.0: 0.0500, 1600.0: 0.0430},
    "Ni": {662.0: 0.0605, 1000.0: 0.0515, 1600.0: 0.0440},
    "Ar": {662.0: 0.0760, 1000.0: 0.0630, 1600.0: 0.0520},
    "Pb": {662.0: 0.0922, 1000.0: 0.0695, 1600.0: 0.0508},
}

MATERIAL_PRESETS: dict[str, MaterialPreset] = {
    "air": MaterialPreset(
        name="air",
        density_g_cm3=0.001225,
        composition_by_mass={"N": 0.755, "O": 0.232, "Ar": 0.013},
    ),
    "water": MaterialPreset(
        name="water",
        density_g_cm3=1.0,
        composition_by_mass={"H": 0.1119, "O": 0.8881},
    ),
    "concrete": MaterialPreset(
        name="concrete",
        density_g_cm3=2.3,
        composition_by_mass={"O": 0.525, "Si": 0.325, "Ca": 0.090, "Al": 0.060},
    ),
    "aluminum": MaterialPreset(
        name="aluminum",
        density_g_cm3=2.7,
        composition_by_mass={"Al": 1.0},
    ),
    "iron": MaterialPreset(
        name="iron",
        density_g_cm3=7.87,
        composition_by_mass={"Fe": 1.0},
    ),
    "steel": MaterialPreset(
        name="steel",
        density_g_cm3=7.85,
        composition_by_mass={"Fe": 0.98, "C": 0.02},
    ),
    "stainless_steel": MaterialPreset(
        name="stainless_steel",
        density_g_cm3=8.0,
        composition_by_mass={"Fe": 0.70, "Cr": 0.19, "Ni": 0.10, "C": 0.01},
    ),
    "lead": MaterialPreset(
        name="lead",
        density_g_cm3=11.34,
        composition_by_mass={"Pb": 1.0},
    ),
}

MATERIAL_ALIASES: dict[str, str] = {
    "al": "aluminum",
    "alu": "aluminum",
    "aluminium": "aluminum",
    "fe": "iron",
    "iron": "iron",
    "lead": "lead",
    "pb": "lead",
    "ss": "stainless_steel",
    "ss304": "stainless_steel",
    "stainless": "stainless_steel",
}

ELEMENT_ALIASES: dict[str, str] = {
    "al": "Al",
    "aluminum": "Al",
    "aluminium": "Al",
    "ar": "Ar",
    "argon": "Ar",
    "c": "C",
    "carbon": "C",
    "ca": "Ca",
    "calcium": "Ca",
    "cr": "Cr",
    "chromium": "Cr",
    "fe": "Fe",
    "iron": "Fe",
    "h": "H",
    "hydrogen": "H",
    "n": "N",
    "nitrogen": "N",
    "ni": "Ni",
    "nickel": "Ni",
    "o": "O",
    "oxygen": "O",
    "pb": "Pb",
    "lead": "Pb",
    "si": "Si",
    "silicon": "Si",
}


def normalize_material_name(name: str) -> str:
    """Normalize a material identifier into a preset lookup key."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")
    return MATERIAL_ALIASES.get(normalized, normalized)


def normalize_element_name(name: str) -> str | None:
    """Normalize an element token into its symbol."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")
    return ELEMENT_ALIASES.get(normalized)


def parse_composition_string(raw_value: str) -> dict[str, float]:
    """Parse a simple mass-fraction composition string."""
    composition: dict[str, float] = {}
    text = str(raw_value).strip()
    if not text:
        return composition
    normalized = text.replace(";", ",")
    for part in normalized.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(f"Invalid composition entry: {item!r}")
        element = normalize_element_name(key)
        if element is None:
            continue
        composition[element] = float(value)
    return composition


def normalize_composition_by_mass(raw_value: dict[str, float] | str | None) -> dict[str, float]:
    """Normalize a composition dictionary or composition string."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, str):
        return parse_composition_string(raw_value)
    composition: dict[str, float] = {}
    for key, value in raw_value.items():
        element = normalize_element_name(str(key))
        if element is None:
            continue
        composition[element] = float(value)
    return composition


def resolve_material_preset(name: str | None) -> MaterialPreset | None:
    """Resolve a normalized preset entry by material name."""
    if name in (None, ""):
        return None
    return MATERIAL_PRESETS.get(normalize_material_name(str(name)))


def composition_mass_attenuation(
    composition_by_mass: dict[str, float] | str | None,
    isotope: str,
) -> float | None:
    """Return a mixture mass attenuation coefficient from elemental fractions."""
    composition = normalize_composition_by_mass(composition_by_mass)
    if not composition:
        return None
    total_weight = 0.0
    weighted_mu = 0.0
    for element, weight in composition.items():
        table = ELEMENTAL_MASS_ATT_CM2_G.get(element)
        if table is None:
            continue
        mass_att = table.get(isotope)
        if mass_att is None:
            continue
        numeric_weight = max(0.0, float(weight))
        if numeric_weight <= 0.0:
            continue
        total_weight += numeric_weight
        weighted_mu += numeric_weight * float(mass_att)
    if total_weight <= 0.0:
        return None
    return weighted_mu / total_weight


def interpolate_mass_attenuation_curve(curve_by_energy_keV: dict[float, float], energy_keV: float) -> float | None:
    """Interpolate a mass attenuation coefficient from a discrete energy curve."""
    if not curve_by_energy_keV:
        return None
    energies = np.asarray(sorted(float(energy) for energy in curve_by_energy_keV.keys()), dtype=float)
    values = np.asarray([float(curve_by_energy_keV[float(energy)]) for energy in energies], dtype=float)
    if energies.size == 0:
        return None
    if energies.size == 1:
        return float(values[0])
    clamped_energy_keV = float(np.clip(float(energy_keV), float(energies[0]), float(energies[-1])))
    return float(np.interp(clamped_energy_keV, energies, values))


def composition_mass_attenuation_at_energy(
    composition_by_mass: dict[str, float] | str | None,
    energy_keV: float,
) -> float | None:
    """Return a mixture mass attenuation coefficient at a specific photon energy."""
    composition = normalize_composition_by_mass(composition_by_mass)
    if not composition:
        return None
    total_weight = 0.0
    weighted_mu = 0.0
    for element, weight in composition.items():
        curve = ELEMENTAL_MASS_ATT_CURVES_CM2_G.get(element)
        if curve is None:
            continue
        mass_att = interpolate_mass_attenuation_curve(curve, energy_keV)
        if mass_att is None:
            continue
        numeric_weight = max(0.0, float(weight))
        if numeric_weight <= 0.0:
            continue
        total_weight += numeric_weight
        weighted_mu += numeric_weight * float(mass_att)
    if total_weight <= 0.0:
        return None
    return weighted_mu / total_weight

"""核種ライブラリの定義と取得を扱うモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class NuclideLine:
    """単一ガンマ線のエネルギーと強度を表す。"""

    energy_keV: float
    intensity: float


@dataclass(frozen=True)
class Nuclide:
    """核種とその代表ピーク群を保持する。"""

    name: str
    lines: List[NuclideLine]
    representative_energy_keV: float


def default_library() -> Dict[str, Nuclide]:
    """Cs-137, Co-60, Eu-154の代表ピークを含むライブラリを返す。"""
    return {
        "Cs-137": Nuclide(
            name="Cs-137",
            lines=[NuclideLine(energy_keV=662.0, intensity=0.85)],
            representative_energy_keV=662.0,
        ),
        "Co-60": Nuclide(
            name="Co-60",
            lines=[
                NuclideLine(energy_keV=1173.0, intensity=0.5),
                NuclideLine(energy_keV=1332.0, intensity=0.5),
            ],
            representative_energy_keV=1250.0,
        ),
        "Eu-154": Nuclide(
            name="Eu-154",
            lines=[
                NuclideLine(energy_keV=723.3, intensity=0.25),
                NuclideLine(energy_keV=873.2, intensity=0.14),
                NuclideLine(energy_keV=996.3, intensity=0.14),
                NuclideLine(energy_keV=1274.5, intensity=0.45),
                NuclideLine(energy_keV=1494.0, intensity=0.01),
                NuclideLine(energy_keV=1596.5, intensity=0.02),
            ],
            representative_energy_keV=1274.5,
        ),
    }

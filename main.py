"""Replay a shared MeasurementLog v2 with the particle-filter estimator."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pf.replay import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

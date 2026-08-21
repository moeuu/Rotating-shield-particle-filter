"""Compatibility entry point for truth-free PF replay."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> int:
    """Run the truth-free PF replay command through the compatibility shim."""
    from pf.replay import main as replay_main

    return replay_main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Path-safe launcher for the existing online_match.py server.

online_match.py imports the engine from game_sim/bo_yi. This wrapper only adds
that existing package directory to sys.path, then calls online_match.main().
It leaves online_match.py unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SIM_ROOT = ROOT / "game_sim" / "bo_yi"

sim_path = str(SIM_ROOT)
if sim_path not in sys.path:
    sys.path.insert(0, sim_path)

from online_match import main  # noqa: E402


if __name__ == "__main__":
    main()

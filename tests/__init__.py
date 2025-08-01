"""Test package for the RL chess agent.

This file marks the directory as a Python package and adjusts sys.path so
that the project's src module is importable when running tests via
python -m unittest. Without this tweak the top-level src package would not
be discoverable by Python's import machinery.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path to import the `src` package.
current_dir = Path(__file__).resolve()
project_root = current_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

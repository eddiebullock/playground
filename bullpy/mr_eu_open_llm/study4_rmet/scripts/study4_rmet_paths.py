"""Path helpers for study4 scripts (no study3 imports)."""

from __future__ import annotations

import sys
from pathlib import Path

STUDY4_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent


def ensure_study4_on_path() -> None:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if str(STUDY4_ROOT) not in sys.path:
        sys.path.insert(0, str(STUDY4_ROOT))

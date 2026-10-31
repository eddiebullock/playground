#!/usr/bin/env python3
"""CLI wrapper so Phase 2 can be launched with PYTHONPATH=src."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paper_v2.probe import main  # noqa: E402

if __name__ == "__main__":
    main()

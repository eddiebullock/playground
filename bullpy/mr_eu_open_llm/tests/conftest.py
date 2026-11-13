"""Shared test isolation (added 2026-11-12).

`scripts/a7_recovery.py` replaces A2's `stimulus_interventions.signflip_maxt` with a cached fast version
inside recovery workers (`install_fast_statistics`, called by `_init`). `tests/test_a7.py` calls `_init`
in the test process from a module-scoped fixture, so the replacement outlived that module, and a later A2
test (`test_signflip_is_exact_and_has_the_right_floor`) ran against the fast version, which counts the
identity flip differently at very large t (t about 109 there). The real analyses never see the
replacement: they run as fresh processes and never import the recovery module.

A2's function is captured here, when pytest loads this file and before any test module is imported, and
put back when each test module finishes, so test order cannot change a result while A7's own tests keep
the fast version they install. No frozen file is changed.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scripts import stimulus_interventions as _SI
    _A2_SIGNFLIP = _SI.signflip_maxt
except Exception:  # an environment without the A2 dependencies: nothing to restore
    _SI = None
    _A2_SIGNFLIP = None


@pytest.fixture(autouse=True, scope="module")
def _restore_a2_signflip():
    yield
    if _SI is not None:
        _SI.signflip_maxt = _A2_SIGNFLIP

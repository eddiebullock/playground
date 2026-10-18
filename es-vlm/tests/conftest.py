from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure() -> None:
    (ROOT / "tests" / "fixtures" / "synthetic").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def synthetic_dir(tmp_path: Path) -> Path:
    from esvlm.tasks.synthetic import write_fixtures

    dest = tmp_path / "synthetic"
    write_fixtures(dest)
    return dest

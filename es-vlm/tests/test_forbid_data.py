from __future__ import annotations

import importlib.util
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "forbid_data.py"


def _mod():
    spec = importlib.util.spec_from_file_location("forbid_data", HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blocks_data_and_local_allows_example(tmp_path: Path) -> None:
    forbid = _mod()
    assert forbid.check("data/rmet/x.json")
    assert forbid.check("es-vlm/local/site.sh")
    assert forbid.check("local/HPC_RUNBOOK.md")
    assert forbid.check("es-vlm/local/site.example.sh") is None
    assert forbid.check("README.md") is None
    assert forbid.check("face.png")
    assert forbid.check("tests/fixtures/synthetic/a.png") is None
    big = tmp_path / "notes.txt"
    big.write_bytes(b"a" * (5 * 1024 * 1024 + 8))
    assert forbid.check(str(big))

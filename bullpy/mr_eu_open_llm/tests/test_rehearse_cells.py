"""The rehearsal-cell tool never writes where real cells are pulled (2026-11-13)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rehearse_cells_under_test", ROOT / "research/scripts/rehearse_cells.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


@pytest.mark.parametrize("target", ["results/a4", "results/a9", "results/a4_rehearsal"])
def test_refuses_any_folder_under_results(target, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("stim_id\n")
    with pytest.raises(SystemExit, match="may not be written under results"):
        rc.main(["a4", "--manifest", str(manifest), "--out", str(ROOT / target)])
    assert not (ROOT / "results" / "a4_rehearsal").exists()

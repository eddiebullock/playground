import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/scripts/make_figures.py"


def _run(out, *args):
    subprocess.run([sys.executable, str(SCRIPT), *args, "--out", str(out)], check=True,
                   capture_output=True, cwd=ROOT)


def check_all_figures_written(out):
    _run(out, "--all")
    for n in (1, 3):
        for ext in ("pdf", "png"):
            path = out / f"fig{n}.{ext}"
            assert path.is_file() and path.stat().st_size > 1000, path
    sources = (out / "SOURCES.md").read_text(encoding="utf-8")
    assert "## Figure 1" in sources and "## Figure 3" in sources
    assert "cfa_identifiability.json" in sources
    assert "prompt_confound_audit.json" in sources


def check_single_figure_keeps_other_sources(out):
    _run(out, "--all")
    _run(out, "--fig", "3")
    sources = (out / "SOURCES.md").read_text(encoding="utf-8")
    assert "## Figure 1" in sources and "## Figure 3" in sources


def test_all_figures_written(tmp_path):
    check_all_figures_written(tmp_path)


def test_single_figure_keeps_other_sources(tmp_path):
    check_single_figure_keeps_other_sources(tmp_path)


if __name__ == "__main__":
    # Runs without pytest, for environments that lack it.
    with tempfile.TemporaryDirectory() as d:
        check_all_figures_written(Path(d))
    with tempfile.TemporaryDirectory() as d:
        check_single_figure_keeps_other_sources(Path(d))
    print("ok")

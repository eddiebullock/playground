import csv
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/scripts/make_figures.py"
LIT = ROOT / "research/reproduction/literature_table.csv"
FIGURES_DIR = ROOT / "research/paper/v2/figures"


def _run(out, *args, check=True):
    return subprocess.run([sys.executable, str(SCRIPT), *args, "--out", str(out)], check=check,
                          capture_output=True, text=True, cwd=ROOT)


def _module():
    spec = importlib.util.spec_from_file_location("make_figures_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.style()
    return mod


# ---------------------------------------------------------------- committed figures


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
    assert "| e, eight-paper table | `research/reproduction/literature_table.csv` |" in sources


def check_single_figure_keeps_other_sources(out):
    _run(out, "--all")
    _run(out, "--fig", "3")
    sources = (out / "SOURCES.md").read_text(encoding="utf-8")
    assert "## Figure 1" in sources and "## Figure 3" in sources


def test_all_figures_written(tmp_path):
    check_all_figures_written(tmp_path)


def test_single_figure_keeps_other_sources(tmp_path):
    check_single_figure_keeps_other_sources(tmp_path)


# ---------------------------------------------------------------- literature table


def check_literature_table():
    rows = list(csv.DictReader(LIT.open(encoding="utf-8")))
    assert [int(r["ref"]) for r in rows] == [1, 2, 8, 9, 10, 11, 12, 13]
    codes = {"yes", "partial", "no", "na", "not_stated"}
    refs = (ROOT / "research/paper/v2/references.md").read_text(encoding="utf-8").splitlines()
    for r in rows:
        for key in ("target_position", "option_sets", "delivery_check", "face_dependence"):
            assert r[f"{key}_code"] in codes, (r["ref"], key)
        assert r["verification"].strip() and r["source_location"].strip()
        for loc in r["source_location"].split(";"):
            path, line = loc.strip().rsplit(":", 1)
            lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
            assert 1 <= int(line) <= len(lines), loc
        # The references.md row named in the sources is this paper's row.
        ref_rows = [loc for loc in r["source_location"].split(";") if "references.md" in loc]
        assert len(ref_rows) == 1
        line = int(ref_rows[0].rsplit(":", 1)[1])
        assert re.match(rf"\| {r['ref']} \|", refs[line - 1]), (r["ref"], refs[line - 1][:20])
        for name in filter(None, (c.strip() for c in r["inferred_cells"].split(";"))):
            assert name in ("target_position", "option_sets", "delivery_check", "face_dependence")
    # Nothing in the repo states a delivery check for any of the eight.
    assert {r["delivery_check_code"] for r in rows} == {"not_stated"}


def test_literature_table():
    check_literature_table()


# ---------------------------------------------------------------- unlocked figures


def check_synthetic_render(out, n):
    _run(out, "--fig", str(n), "--synthetic")
    for ext in ("pdf", "png"):
        path = out / f"fig{n}.{ext}"
        assert path.is_file() and path.stat().st_size > 1000, path
    sources = (out / "SOURCES.md").read_text(encoding="utf-8")
    assert f"## Figure {n}" in sources and "SYNTHETIC" in sources


def test_synthetic_fig2(tmp_path):
    check_synthetic_render(tmp_path, 2)


def test_synthetic_fig4(tmp_path):
    check_synthetic_render(tmp_path, 4)


def test_synthetic_fig5(tmp_path):
    check_synthetic_render(tmp_path, 5)


def test_synthetic_fig6(tmp_path):
    check_synthetic_render(tmp_path, 6)


def check_synthetic_refuses_paper_folder():
    result = _run(FIGURES_DIR, "--fig", "4", "--synthetic", check=False)
    assert result.returncode != 0 and "may not be written" in result.stderr
    assert not (FIGURES_DIR / "fig4.png").exists()


def test_synthetic_refuses_paper_folder():
    check_synthetic_refuses_paper_folder()


def check_unlocked_needs_mode(out):
    result = _run(out, "--fig", "5", check=False)
    assert result.returncode != 0 and "not locked" in result.stderr


def test_unlocked_needs_mode(tmp_path):
    check_unlocked_needs_mode(tmp_path)


def check_real_refuses_without_locked(out):
    mod = _module()
    for n in mod.UNLOCKED:
        mod.LOCKED[n] = [out / f"missing_{n}.json"]
        try:
            mod.main(["--fig", str(n), "--real", "--out", str(out)])
        except SystemExit as exc:
            assert "locked output missing" in str(exc)
        else:
            raise AssertionError(f"Figure {n} built without its locked output")
        assert not (out / f"fig{n}.png").exists()


def test_real_refuses_without_locked(tmp_path):
    check_real_refuses_without_locked(tmp_path)


def check_real_loaders_read_the_schema(out):
    """The --real code path, fed synthetic files at the locked paths. Nothing is saved."""
    mod = _module()
    a1, a3 = mod.synthetic_fig2()
    files = {2: {"rotation_summary.json": a1, "face_input_validity_summary.json": a3},
             4: {f"same_clip_family_model_{m}.json": v for m, v in mod.synthetic_fig4().items()},
             5: {"summary.json": mod.synthetic_fig5()},
             6: {f"stimulus_interventions_{m}.json": v
                 for m, v in zip(("gemma4", "qwen3vl", "molmo2"), mod.synthetic_fig6().values())}}
    for n, named in files.items():
        mod.LOCKED[n] = []
        for name, data in named.items():
            path = out / name
            path.write_text(json.dumps(data), encoding="utf-8")
            mod.LOCKED[n].append(path)
        fig = mod.REAL[n]()
        assert len(fig.axes) >= 1
        plt.close(fig)


def test_real_loaders_read_the_schema(tmp_path):
    check_real_loaders_read_the_schema(tmp_path)


def check_schema_errors_are_loud():
    mod = _module()
    a1, a3 = mod.synthetic_fig2()
    del a1["gemma4"]["delta_ci_actor_bootstrap"]
    try:
        mod.fig2_from(a1, a3, synthetic=True)
    except KeyError as exc:
        assert "delta_ci_actor_bootstrap" in str(exc)
    else:
        raise AssertionError("missing field not caught")


def test_schema_errors_are_loud():
    check_schema_errors_are_loud()


def check_synthetic_titles_and_constants():
    mod = _module()
    sys.path.insert(0, str(ROOT))
    from scripts import a4_design

    assert mod.SESOI_CFA == a4_design.SESOI_CFA
    fig = mod.fig5_from(mod.synthetic_fig5(), synthetic=True)
    assert fig._suptitle is not None and fig._suptitle.get_text().startswith("SYNTHETIC")
    assert all(ax.get_title().startswith("SYNTHETIC") for ax in fig.axes if ax.get_title())
    plt.close(fig)
    # Every verdict class the A2 decision table can return has a style.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "research" / "scripts"))
    source = (ROOT / "scripts/analyse_stimulus_interventions.py").read_text(encoding="utf-8")
    verdicts = set(re.findall(r'return "([a-z_]+)"', source[source.index("def verdict"):
                                                            source.index("def _actor_means")]))
    assert verdicts == set(mod.VERDICT_STYLE)


def test_synthetic_titles_and_constants():
    check_synthetic_titles_and_constants()


if __name__ == "__main__":
    # Runs without pytest, for environments that lack it.
    for check in (check_all_figures_written, check_single_figure_keeps_other_sources,
                  check_unlocked_needs_mode, check_real_refuses_without_locked,
                  check_real_loaders_read_the_schema):
        with tempfile.TemporaryDirectory() as d:
            check(Path(d))
    for n in (2, 4, 5, 6):
        with tempfile.TemporaryDirectory() as d:
            check_synthetic_render(Path(d), n)
    check_literature_table()
    check_synthetic_refuses_paper_folder()
    check_schema_errors_are_loud()
    check_synthetic_titles_and_constants()
    print("ok")

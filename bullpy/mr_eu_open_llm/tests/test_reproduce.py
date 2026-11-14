"""Tests for scripts/reproduce.py: step table, pins, pending isolation and the comparison rules.

These run in seconds and never run a reproduction step. The full check is
`python scripts/reproduce.py --quick` (see research/reproduction/REPRODUCE.md).
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce.py"


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("reproduce_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available here")


def _tracked(*dirs):
    out = _git("ls-files", *dirs).decode().split()
    return set(out)


# ----------------------------------------------------------------------------- step table

def test_step_names_unique(rp):
    names = [s.name for s in rp.STEPS]
    assert len(names) == len(set(names))


def test_every_rebuilt_output_listed_once(rp):
    paths = [o.path for s in rp.STEPS if s.cmd is not None and s.quick for o in s.outputs]
    assert len(paths) == len(set(paths))


def test_hash_steps_use_every_pin_once(rp):
    hashed = [o.path for s in rp.STEPS if s.cmd is None for o in s.outputs]
    assert len(hashed) == len(set(hashed))
    assert set(hashed) == set(rp.PINS)
    for s in rp.STEPS:
        if s.cmd is None:
            assert s.quick, f"{s.name} must run in --quick"
            for o in s.outputs:
                assert o.kind == "sha256" and o.sha256 == rp.PINS[o.path]
                assert o.note, f"{o.path} needs a stated reason for not being rerun"


def test_long_recoveries_are_hash_verified_in_quick(rp):
    hashed = {o.path for s in rp.STEPS if s.cmd is None for o in s.outputs}
    for s in rp.STEPS:
        if s.cmd is not None and not s.quick:
            for o in s.outputs:
                committed_target = o.ref or o.path
                assert committed_target in hashed, f"{s.name}: {committed_target} is neither rebuilt in quick nor hashed"


def test_drafts_only_a5_a6(rp):
    for s in rp.STEPS:
        for o in s.outputs:
            if o.draft:
                assert Path(o.path).name.startswith(("a5_", "a6_")), o.path


def test_pins_match_committed_files(rp):
    top = Path(_git("rev-parse", "--show-toplevel").decode().strip())
    prefix = ROOT.resolve().relative_to(top.resolve()).as_posix()
    for path, want in rp.PINS.items():
        data = _git("show", f"HEAD:{prefix}/{path}")
        assert hashlib.sha256(data).hexdigest() == want, f"{path}: committed file changed; update its pin on relock"


def test_inventory_covers_every_committed_output(rp):
    """Every committed output, table and figure is rebuilt, hash-verified, or deliberately not an output."""
    covered = {o.ref or o.path for s in rp.STEPS for o in s.outputs}
    not_outputs = {"research/reproduction/REPRODUCE.md", "research/reproduction/environment_pinned.yml",
                   "research/figures/.gitkeep"}
    tracked = _tracked("research/reproduction", "research/paper/v2/figures", "research/figures")
    assert not covered & set(rp.UNCOVERED), "a covered output is also listed as uncovered"
    missing = sorted(tracked - covered - not_outputs - set(rp.UNCOVERED))
    assert not missing, f"committed outputs not in reproduce.py: {missing}"


# ----------------------------------------------------------------------------- pending isolation

# A1 and the natural-clip model files left this list when they were locked (2026-11-09) and are now
# hash-verified records; A6 left it when it was closed as a limit without running (2026-11-10); A4's
# summary and Figure 5 left it when A4 was locked, and A9's summary when A9 was (both 2026-11-14).
PENDING_TOKENS = ("stimulus_interventions_{m}", "a5_analysis", "a7_analysis")


def test_pending_never_in_a_step(rp):
    pending = {p for _, paths, _ in rp.PENDING for p in paths}
    assert pending, "PENDING must list the not-yet-existing outputs"
    for s in rp.STEPS:
        text = json.dumps([s.cmd, [o.path for o in s.outputs], [o.ref for o in s.outputs]])
        for p in pending:
            assert p not in text, f"{s.name} touches pending output {p}"
        for token in PENDING_TOKENS:
            assert token not in text, (s.name, token)


def test_pending_paths_only_named_in_pending_block(rp):
    src = SCRIPT.read_text(encoding="utf-8")
    start = src.index("PENDING: List[")
    end = src.index("\n]\n", start)
    outside = src[:start] + src[end:]
    for token in PENDING_TOKENS:
        assert token in src[start:end], token
        assert token not in outside, f"{token} appears outside PENDING; pending results must not be read"


def test_list_shows_pending(rp, capsys):
    assert rp.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "[pending] A2 stimulus interventions" in out and "hash_long_recoveries" in out


# ----------------------------------------------------------------------------- comparison rules

def _pdf(lines, box=b"[0 0 100 100]"):
    body = zlib.compress(b"\n".join(lines))
    return (b"%PDF-1.4\n1 0 obj << /MediaBox " + box + b" >> endobj\n2 0 obj << /Filter /FlateDecode >>\nstream\n"
            + body + b"\nendstream\nendobj\n")


def test_pdf_reordered_operators_pass(rp, tmp_path):
    ref = _pdf([b"0 0 m", b"1 1 l", b"f", b"2 2 m"])
    new = tmp_path / "x.pdf"
    new.write_bytes(_pdf([b"2 2 m", b"0 0 m", b"f", b"1 1 l"]))
    verdict, _ = rp.compare(rp.Out("x.pdf"), ref, new)
    assert verdict == "within_tolerance"


def test_pdf_changed_number_fails(rp, tmp_path):
    ref = _pdf([b"0 0 m", b"1 1 l"])
    new = tmp_path / "x.pdf"
    new.write_bytes(_pdf([b"0 0 m", b"1 1.5 l"]))
    assert rp.compare(rp.Out("x.pdf"), ref, new)[0] == "differs"
    new.write_bytes(_pdf([b"0 0 m", b"1 1 l"], box=b"[0 0 100 200]"))
    assert rp.compare(rp.Out("x.pdf"), ref, new)[0] == "differs"


def test_text_kind_reports_line(rp, tmp_path):
    new = tmp_path / "SOURCES.md"
    new.write_text("a\nb\n")
    verdict, detail = rp.compare(rp.Out("SOURCES.md"), b"a\nc\n", new)
    assert verdict == "differs" and "line 2" in detail
    assert rp.compare(rp.Out("SOURCES.md"), b"a\nb\n", new)[0] == "exact"


def test_expected_kind_checks_only_pinned_keys(rp, tmp_path):
    new = tmp_path / "c.json"
    new.write_text(json.dumps({"a": {"n": 68, "extra": 1}, "b": 0.5, "unpinned": "x"}))
    out = rp.Out("c.json", kind="expected", expected={"a": {"n": 68}, "b": 0.5})
    assert rp.compare(out, None, new)[0] == "within_tolerance"
    out = rp.Out("c.json", kind="expected", expected={"a": {"n": 67}})
    assert rp.compare(out, None, new)[0] == "differs"


def test_ref_and_only_compare_one_key(rp, tmp_path):
    new = tmp_path / "s.json"
    new.write_text(json.dumps({"single_column_check": ["n=8 x"]}))
    ref = json.dumps({"single_column_check": ["n=8 x"], "results": {"big": 1}}).encode()
    out = rp.Out("s.json", ref="a2.json", only=("single_column_check",))
    assert rp.compare(out, ref, new)[0] == "within_tolerance"


def test_draft_mismatch_is_a_warning(rp):
    assert rp.finish_verdict(rp.Out("a6_recovery.json", draft=True), "differs") == "draft_changed"
    assert rp.finish_verdict(rp.Out("a4_recovery.json"), "differs") == "differs"
    assert rp.finish_verdict(rp.Out("a6_recovery.json", draft=True), "error") == "error"


def test_verify_by_hash_reads_committed_bytes(rp, monkeypatch):
    blobs = {"good.json": b"{}", "bad.json": b"[1]"}
    monkeypatch.setattr(rp, "committed", lambda rev, prefix, rel: blobs.get(rel))
    step = rp.Step("h", None, [
        rp.Out("good.json", kind="sha256", sha256=hashlib.sha256(b"{}").hexdigest(), note="n"),
        rp.Out("bad.json", kind="sha256", sha256=hashlib.sha256(b"[2]").hexdigest(), note="n"),
        rp.Out("gone.json", kind="sha256", sha256="0" * 64, note="n")])
    verdicts = [r["verdict"] for r in rp.verify_by_hash(step, "HEAD", "")]
    assert verdicts == ["hash_verified", "differs", "missing_output"]


def test_stdout_capture(rp, tmp_path):
    step = rp.Step("s", [sys.executable, "-c", "print('line one'); print(); print('line two')"], [],
                   stdout="out/s.json", stdout_key="k")
    logs = tmp_path / "logs"
    logs.mkdir()
    rc, _ = rp.run_step(step, tmp_path, 1, {"PATH": ""}, logs)
    assert rc == 0
    assert json.loads((tmp_path / "out/s.json").read_text()) == {"k": ["line one", "line two"]}

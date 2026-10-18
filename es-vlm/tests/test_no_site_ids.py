"""Fail if tracked files contain values from local/site.sh."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "local" / "site.sh"
PLACEHOLDER = re.compile(r"YOUR_|CHANGE_ME|example\.invalid|placeholder", re.I)
SENSITIVE_KEYS = {
    "HPC_USER",
    "HPC_HOST",
    "HPC_ROOT",
    "HPC_GPU_ACCOUNT",
    "HPC_CPU_ACCOUNT",
    "ESVLM_MODELS_DIR",
    "ESVLM_NEW_MODELS_DIR",
    "ESVLM_RMET_DIR",
    "ESVLM_EU_EMOTIONS_DIR",
    "ESVLM_MINDREADING_DIR",
    "ESVLM_CACHE_ROOT",
    "SITE_DENYLIST",
}


def _parse_site(path: Path) -> list[str]:
    values: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key == "SITE_DENYLIST":
            values.extend(v.strip() for v in val.split(",") if v.strip())
            continue
        if key in SENSITIVE_KEYS and val:
            values.append(val)
    out = []
    for v in values:
        if PLACEHOLDER.search(v):
            continue
        if v.startswith("${") or v.startswith("$"):
            continue
        if len(v) < 4:
            continue
        out.append(v)
    return out


def _tracked_files() -> list[Path]:
    repo = ROOT.parent
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "es-vlm"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [repo / line for line in proc.stdout.splitlines() if line]
    if files:
        return files
    # Before the first commit, scan the working tree except ignored dirs.
    skip = {".venv", "__pycache__", "outputs", "cache", "data", ".git"}
    found = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        if path.name == "site.sh":
            continue
        found.append(path)
    return found


@pytest.mark.skipif(not SITE.exists(), reason="local/site.sh absent")
def test_no_site_ids_in_tracked_files() -> None:
    needles = _parse_site(SITE)
    if not needles:
        return
    hits = []
    for path in _tracked_files():
        if path.name == "site.sh":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in needles:
            if needle in text:
                hits.append(f"{path}: {needle}")
    assert not hits, "site identifiers leaked into tracked files:\n" + "\n".join(hits)


def test_skip_or_run_is_deterministic() -> None:
    # The skip path is itself a test: absence of site.sh must not fail CI.
    if not SITE.exists():
        assert os.environ.get("ESVLM_REQUIRE_SITE", "0") != "1"

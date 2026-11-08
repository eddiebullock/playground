"""
Design block of amendment A7, read from its one place: research/PREREG_A7_DRAFT.md.

Standard library only, so that any environment can hash it. The block is parsed from the
draft, and the code refuses to run if its hash differs from the value pinned here or from
the current value the draft states. All constants live in the block and none are repeated
here. The pinned A2 code, gate code and evidence files named in the block are checked by
check_pins.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "research" / "PREREG_A7_DRAFT.md"
# Restated in the draft. Any edit to the block must change both, on purpose.
DESIGN_SHA256 = "9e40cfcfbe8761703da22cf2400c3af6cc7f05019df8b186f91e421dce50bb09"

_BLOCK = re.compile(r"^## Design block$.*?^```json\n(.*?)^```$", re.M | re.S)
_CODE = re.compile(r"^## Code hashes$.*?^```text\n(.*?)^```$", re.M | re.S)
# The draft states the pre-audit hash first and the current one on its own line.
_STATED = re.compile(r"Current `design_sha256` of the block below: `([0-9a-f]{64})`")


def block_bytes(path: Path = DRAFT) -> bytes:
    """The bytes between the fence lines, without the fences, ending in one newline."""
    found = _BLOCK.search(path.read_text(encoding="utf-8"))
    if found is None:
        raise SystemExit(f"No design block in {path}.")
    return found.group(1).encode("utf-8")


def block_hash(path: Path = DRAFT) -> str:
    return hashlib.sha256(block_bytes(path)).hexdigest()


def stated_hash(path: Path = DRAFT) -> str:
    found = _STATED.search(path.read_text(encoding="utf-8"))
    if found is None:
        raise SystemExit(f"{path} states no current design_sha256.")
    return found.group(1)


def load_design(path: Path = DRAFT, expected: str = DESIGN_SHA256) -> Dict[str, Any]:
    """The parsed block. Refuses a block that is not the pinned and stated one."""
    actual, stated = block_hash(path), stated_hash(path)
    if actual != expected:
        raise SystemExit(f"Design block hash {actual} is not the pinned {expected}.")
    if actual != stated:
        raise SystemExit(f"Design block hash {actual} is not the {stated} that the draft states.")
    return json.loads(block_bytes(path))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pins(design: Mapping[str, Any], root: Path = ROOT) -> None:
    """Refuses frozen A2 code, gate code or evidence that has changed. A7 edits none of it."""
    for group in ("a2_code_sha256", "other_code_sha256", "evidence_sha256"):
        for name, pinned in design[group].items():
            path = root / name
            if not path.exists():
                raise SystemExit(f"{name} is missing; it is pinned at {pinned}.")
            digest = file_sha256(path)
            if digest != pinned:
                raise SystemExit(f"{name} is {digest}, not the pinned {pinned}.")


def code_pins(path: Path = DRAFT) -> Dict[str, str]:
    """A7's own code, pinned in the draft outside the block at the freeze. Empty before it.

    They sit outside the block because a7_design.py holds the block's hash, so the block
    cannot hold this file's hash.
    """
    found = _CODE.search(path.read_text(encoding="utf-8"))
    if found is None:
        return {}
    pins = {}
    for line in found.group(1).splitlines():
        if line.strip():
            name, digest = line.split()
            pins[name] = digest
    return pins


def check_code_pins(root: Path = ROOT, path: Path = DRAFT, *, require: bool = True) -> Dict[str, str]:
    """Checks A7's own code against the draft. With require, no pins means not frozen."""
    pins = code_pins(path)
    if not pins and require:
        raise SystemExit(f"{path} pins no A7 code. A7 is not frozen.")
    for name, pinned in pins.items():
        digest = file_sha256(root / name)
        if digest != pinned:
            raise SystemExit(f"{name} is {digest}, not the pinned {pinned}.")
    return pins


A7_FILES = (
    "scripts/a7_design.py",
    "scripts/a7_cuts.py",
    "scripts/a7_stage0.py",
    "scripts/a7_run.py",
    "scripts/a7_analyse.py",
    "scripts/a7_recovery.py",
    "scripts/a7_toy.py",
)


def own_code_sha256(root: Path = ROOT) -> Dict[str, str]:
    """Hashes of the A7 files as they are now, recorded in every output."""
    return {name: file_sha256(root / name) for name in A7_FILES if (root / name).exists()}

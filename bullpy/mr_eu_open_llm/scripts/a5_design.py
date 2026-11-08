"""
Design block of amendment A5, read from its one place: research/PREREG_A5_DRAFT.md.

Standard library only, so that any environment can hash it. The block is parsed from the
draft, and the code refuses to run if its hash differs from the value pinned here or from
the value the draft states. All constants live in the block and none are repeated here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "research" / "PREREG_A5_DRAFT.md"
# Restated in the draft. Any edit to the block must change both, on purpose.
DESIGN_SHA256 = "fd20495bf411b97bb876a32b0d93994b98bcc73a05aa104c6976f3fe2988efd4"

_BLOCK = re.compile(r"^## Design block$.*?^```json\n(.*?)^```$", re.M | re.S)
_CODE = re.compile(r"^## Code hashes$.*?^```text\n(.*?)^```$", re.M | re.S)
_STATED = re.compile(r"`design_sha256` of the block below: `([0-9a-f]{64})`")


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
        raise SystemExit(f"{path} states no design_sha256.")
    return found.group(1)


def load_design(path: Path = DRAFT, expected: str = DESIGN_SHA256) -> Dict[str, Any]:
    """The parsed block. Refuses a block that is not the pinned and stated one."""
    actual, stated = block_hash(path), stated_hash(path)
    if actual != expected:
        raise SystemExit(f"Design block hash {actual} is not the pinned {expected}.")
    if actual != stated:
        raise SystemExit(f"Design block hash {actual} is not the {stated} that the draft states.")
    return json.loads(block_bytes(path))


def code_pins(path: Path = DRAFT) -> Dict[str, str]:
    """A5's own code and the helpers it imports, pinned in the draft outside the block.

    They sit outside the block because a5_design.py holds the block's hash, so the block
    cannot hold this file's hash.
    """
    found = _CODE.search(path.read_text(encoding="utf-8"))
    if found is None:
        raise SystemExit(f"No code hashes in {path}.")
    pins = {}
    for line in found.group(1).splitlines():
        if line.strip():
            name, digest = line.split()
            pins[name] = digest
    return pins


def check_code_pins(root: Path = ROOT, path: Path = DRAFT) -> None:
    for name, pinned in code_pins(path).items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != pinned:
            raise SystemExit(f"{name} is {digest}, not the pinned {pinned}.")


def check_a2_code(design: Mapping[str, Any], root: Path = ROOT) -> None:
    """Refuses frozen A2 code that has changed. A5 imports it and edits none of it."""
    for name, pinned in design["a2_code_sha256"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != pinned:
            raise SystemExit(f"{name} is {digest}, not the frozen {pinned}.")

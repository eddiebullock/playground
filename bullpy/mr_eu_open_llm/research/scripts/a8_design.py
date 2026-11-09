"""
Design block of amendment A8, read from its one place: research/PREREG_A8_DRAFT.md.

Standard library only. The block is hashed the way scripts/a7_design.py hashes A7's: the
bytes between the fence lines of "## Design block", without the fences. The code refuses a
block whose hash differs from the value pinned here or from the value the draft states.
Every frozen file A8 calls is checked against `frozen_code_sha256`; A8's own code is checked
against "## Code hashes" once that section holds pins.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "research" / "PREREG_A8_DRAFT.md"
# Restated in the draft. Any edit to the block must change both, on purpose.
DESIGN_SHA256 = "49665a159771043e50e4799d972e3deaa84b1f03a04009bdb458abf6fbea11eb"

_BLOCK = re.compile(r"^## Design block$.*?^```json\n(.*?)^```$", re.M | re.S)
_CODE = re.compile(r"^## Code hashes$.*?^```text\n(.*?)^```$", re.M | re.S)
_STATED = re.compile(r"Current `design_sha256` of the block below: `([0-9a-f]{64})`")


def block_bytes(path: Path = DRAFT) -> bytes:
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
        raise SystemExit(f"A8 design block hash {actual} is not the pinned {expected}.")
    if actual != stated:
        raise SystemExit(f"A8 design block hash {actual} is not the {stated} that the draft states.")
    return json.loads(block_bytes(path))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen_code(design: Mapping[str, Any], root: Path = ROOT) -> Dict[str, str]:
    """Every frozen file A8 calls, against its pin. Any difference stops the caller."""
    bad = {}
    for name, pinned in design["frozen_code_sha256"].items():
        path = root / name
        digest = file_sha256(path) if path.exists() else "missing"
        if digest != pinned:
            bad[name] = digest
    if bad:
        raise SystemExit(f"Frozen files differ from the A8 pins: {sorted(bad)}")
    return dict(design["frozen_code_sha256"])


def code_pins(path: Path = DRAFT) -> Dict[str, str]:
    """A8's own code pins, `<sha256>  <path>` per line. Empty while the section is pending."""
    found = _CODE.search(path.read_text(encoding="utf-8"))
    if found is None:
        raise SystemExit(f"No code hashes section in {path}.")
    pins = {}
    for line in found.group(1).splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            pins[parts[1]] = parts[0]
    return pins


def check_own_code(path: Path = DRAFT, root: Path = ROOT, require: bool = False) -> Dict[str, str]:
    """Check A8's code against its pins. With require=True, pending pins stop the caller."""
    pins = code_pins(path)
    if not pins:
        if require:
            raise SystemExit("A8 code is not pinned yet; scored A8 runs wait on the dated pin step.")
        return {}
    bad = {n: file_sha256(root / n) for n, h in pins.items() if not (root / n).exists() or file_sha256(root / n) != h}
    if bad:
        raise SystemExit(f"A8 code differs from its pins: {sorted(bad)}")
    return pins


_NOTES = re.compile(r"^## Dated notes$(.*?)(?=^## |\Z)", re.M | re.S)
_ADMIT = re.compile(r"^- (\d{4}-\d{2}-\d{2}): admit `([a-z0-9_]+)`", re.M)


def admitted(path: Path = DRAFT) -> Dict[str, str]:
    """Conditional models admitted by a dated note, `- YYYY-MM-DD: admit `id``, with the date."""
    found = _NOTES.search(path.read_text(encoding="utf-8"))
    return {} if found is None else {m: d for d, m in _ADMIT.findall(found.group(1))}


def model(design: Mapping[str, Any], a8_id: str, path: Path = DRAFT) -> Dict[str, Any]:
    """One A8 model. A conditional model is refused until a dated note admits it."""
    models = design["models"]
    if a8_id not in models:
        raise SystemExit(f"{a8_id} is not an A8 model.")
    entry = dict(models[a8_id])
    if entry["status"] != "confirmed" and a8_id not in admitted(path):
        raise SystemExit(f"{a8_id} is {entry['status']}; it enters A8 only by a dated note.")
    return entry


def active_models(design: Mapping[str, Any], path: Path = DRAFT) -> list:
    """Confirmed models plus admitted ones, in block order. This list is every A8 family."""
    notes = admitted(path)
    return [m for m, e in design["models"].items() if e["status"] == "confirmed" or m in notes]


if __name__ == "__main__":
    print(block_hash())

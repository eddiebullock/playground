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
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "research" / "PREREG_A8_DRAFT.md"
# Restated in the draft. Any edit to the block must change both, on purpose.
DESIGN_SHA256 = "95dd41b5d3c33b79ee9e54a20d34ea1e18b6c9156fcabd719ebfb084709b7b50"

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


def is_active(entry: Mapping[str, Any]) -> bool:
    """Membership lives inside the hashed block: confirmed, or admitted with a date."""
    return entry["status"] == "confirmed" or str(entry["status"]).startswith("admitted_")


def model(design: Mapping[str, Any], a8_id: str) -> Dict[str, Any]:
    """One active A8 model. Anything else is refused."""
    models = design["models"]
    if a8_id not in models:
        raise SystemExit(f"{a8_id} is not an A8 model.")
    entry = dict(models[a8_id])
    if not is_active(entry):
        raise SystemExit(f"{a8_id} is {entry['status']}, not an active A8 model.")
    return entry


def active_models(design: Mapping[str, Any]) -> List[str]:
    """Active models in block order. Families are subsets of this, computed by a8_family.py."""
    return [m for m, e in design["models"].items() if is_active(e)]


def check_data(design: Mapping[str, Any], names: Sequence[str], root: Path = ROOT) -> Dict[str, str]:
    """Pinned data inputs. A missing or changed file stops the caller."""
    out = {}
    for name in names:
        pinned = design["data_sha256"][name]
        path = root / name
        if not path.exists():
            raise SystemExit(f"{name} is missing; A8 never falls back to another input.")
        digest = file_sha256(path)
        if digest != pinned:
            raise SystemExit(f"{name} sha256 {digest} is not the pinned {pinned}.")
        out[name] = digest
    return out


# Import closure: every repo module reached from the frozen entry points A8 runs.

ENTRY_POINTS = (
    "scripts/evaluate.py", "scripts/vision_gates.py", "scripts/run_stimulus_interventions.py",
    "scripts/face_input_validity.py", "scripts/readout_ladder.py", "scripts/a4_run.py",
    "scripts/a4_analyse.py", "scripts/analyse_stimulus_interventions.py",
    "research/scripts/analyse_rotation.py", "research/scripts/analyse_rotation_run.py",
    "research/scripts/same_clip_family_analysis.py", "research/scripts/decoder_preflight.py",
    "src/paper_v2/option_score.py", "research/scripts/a4_detector.py",
)
# Loaded by file path at run time, not by an import statement.
DYNAMIC = {"research/scripts/same_clip_family_analysis.py": ("research/scripts/same_clip_cue_analysis.py",)}


def _module_files(name: str, here: Path, root: Path) -> List[Path]:
    parts = name.split(".")
    bases = []
    if parts[0] == "scripts" and len(parts) > 1:
        bases.append(root / "scripts" / Path(*parts[1:]))
    if parts[0] == "paper_v2":
        bases.append(root / "src" / Path(*parts))
    if name == "config":
        bases.append(root / "config")
    bases += [here.parent / Path(*parts), root / "research" / "scripts" / Path(*parts)]
    for base in bases:
        for cand in (base.with_suffix(".py"), base / "__init__.py"):
            if cand.exists():
                return [cand]
    return []


def import_closure(root: Path = ROOT, entries: Sequence[str] = ENTRY_POINTS) -> List[str]:
    """Repo files reachable by import (module or function level) from the entry points."""
    import ast

    seen, stack = set(), [(root / e).resolve() for e in entries]
    while stack:
        path = stack.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        rel = str(path.relative_to(root.resolve()))
        stack += [(root / d).resolve() for d in DYNAMIC.get(rel, ())]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                stack += [p.resolve() for p in _module_files(name, path, root)]
    return sorted(str(p.relative_to(root.resolve())) for p in seen)


if __name__ == "__main__":
    print(block_hash())

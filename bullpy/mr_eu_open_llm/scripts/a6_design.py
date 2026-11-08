"""
Design of draft amendment A6 (RQ5, shared errors on CREMA-D), read from its own draft.

Every fixed constant lives in the JSON design block of research/PREREG_A6_DRAFT.md and
nowhere else. This module parses that block, refuses a block that is not the pinned and
stated one, and refuses frozen code or a selection table that differs from its pin.
Standard library only, so that any environment can hash it. See the draft.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _draft() -> Path:
    """The amendment's text, under its draft name or a frozen name, which must be unique."""
    found = sorted((ROOT / "research").glob("PREREG_A6*.md"))
    if len(found) != 1:
        raise SystemExit(f"Expected one research/PREREG_A6*.md, found {[f.name for f in found]}.")
    return found[0]


DRAFT = _draft()
SELECTION = ROOT / "research" / "reproduction" / "crema_d_selection.csv"
HUMAN_TARGET = ROOT / "research" / "reproduction" / "a6_human_target.json"
RECOVERY_OUT = ROOT / "research" / "reproduction" / "a6_recovery.json"
OUTPUT_DIR = Path("results/a6")
GEOMETRY = OUTPUT_DIR / "geometry.json"

# The design block this code was written against. A change to the block needs a change here.
DESIGN_SHA256 = "1554afc0c44b7e51c7cbbbfddb7b658a74ac8e956c1c0b638d2779e67ab60847"

_BLOCK = re.compile(r"^```json\n(.*?)^```$", re.DOTALL | re.MULTILINE)
_STATED = re.compile(r"design_sha256` of the block below: `([0-9a-f]{64})`")
PENDING = "PENDING"


def block_bytes(path: Path = DRAFT) -> bytes:
    """The bytes between the fence lines, excluding the fences, with a final newline."""
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
    design = json.loads(block_bytes(path))
    validate(design)
    return design


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding_hash(path: Path = DRAFT) -> str:
    """The block's hash with the cluster-only pins blanked.

    Recovery binds to this, since it reads neither the geometry nor the weights, so pinning
    them at the freeze does not call for a rerun. Every run and analysis binds to the full hash.
    """
    design = json.loads(block_bytes(path))
    for key in design["unbound_keys"]:
        design[key] = "UNBOUND"
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode("utf-8")).hexdigest()


def pins(design: Mapping[str, Any]) -> Dict[str, str]:
    """Every file whose bytes the block pins, by path from the repository root.

    Not pinned, by design: scripts/a6_design.py, which holds the block's own hash, so it
    cannot be named in the block; tests/test_a6.py, which no result reads.
    """
    out = {**design["a2_code_sha256"], **design["other_code_sha256"], **design["a6_code_sha256"]}
    out["scripts/face_input_validity.py"] = design["a3_code_sha256"]
    out["research/reproduction/crema_d_selection.csv"] = design["selection_sha256"]
    return out


def check_pins(design: Mapping[str, Any], root: Path = ROOT) -> None:
    """Refuses frozen code or a selection table that has changed."""
    for name, pinned in pins(design).items():
        digest = file_digest(root / name)
        if digest != pinned:
            raise SystemExit(f"{name} is {digest}, not the pinned {pinned}.")


def pending_items(design: Mapping[str, Any]) -> List[str]:
    out = []
    for key, value in design.items():
        if value == PENDING or (isinstance(value, Mapping) and PENDING in value.values()):
            out.append(key)
    return out


def check_frozen(design: Mapping[str, Any]) -> None:
    """No real run or real analysis while any pin is pending."""
    open_items = pending_items(design)
    if open_items:
        raise SystemExit(f"A6 is not frozen: {open_items} are PENDING.")


def check_human_target(design: Mapping[str, Any], path: Path = HUMAN_TARGET) -> None:
    digest = file_digest(path)
    if digest != design["human_target_sha256"]:
        raise SystemExit(f"{path} is {digest}, not the pinned {design['human_target_sha256']}.")


def check_geometry(design: Mapping[str, Any], path: Path) -> None:
    digest = file_digest(path)
    if digest != design["geometry_sha256"]:
        raise SystemExit(f"Geometry hash {digest} is not the pinned {design['geometry_sha256']}.")


def adjacent(order: Sequence[str], pair: Sequence[str]) -> bool:
    return abs(list(order).index(pair[0]) - list(order).index(pair[1])) == 1


def adjacent_rows(design: Mapping[str, Any], pair: Sequence[str]) -> List[int]:
    return [i for i, order in enumerate(design["orders"]) if adjacent(order, pair)]


def far_rows(design: Mapping[str, Any], pair: Sequence[str]) -> List[int]:
    return [i for i, order in enumerate(design["orders"]) if not adjacent(order, pair)]


def edit_rows_by_rule(design: Mapping[str, Any]) -> List[int]:
    """Rows where the target pair is not adjacent and neither member sits at an end."""
    last = len(design["options"]) - 1
    rows = []
    for row in far_rows(design, design["target_pair"]):
        positions = [design["orders"][row].index(label) for label in design["target_pair"]]
        if all(0 < p < last for p in positions):
            rows.append(row)
    return rows


def condition_spec(name: str) -> Dict[str, str]:
    """Region and fill of one condition name, from the name's fixed form."""
    if name == "original":
        return {"kind": "none"}
    if name == "sham_self_fill":
        return {"kind": "mask", "region": "SHAM", "fill": "self"}
    _, region, fill = name.split("_")
    return {"kind": "mask", "region": region, "fill": fill}


def validate(design: Mapping[str, Any]) -> None:
    """Structural claims the draft makes about its own block. A failure stops everything."""
    options, orders = design["options"], design["orders"]
    n = len(options)
    if any(sorted(order) != sorted(options) for order in orders) or len(orders) != n:
        raise SystemExit("Each order must be a permutation of the options, one per option.")
    for position in range(n):
        if len({order[position] for order in orders}) != n:
            raise SystemExit("Each option must take each position once.")
    steps = [step for order in orders for step in zip(order, order[1:])]
    if len(set(steps)) != len(steps) or len(steps) != n * (n - 1):
        raise SystemExit("Each ordered pair of options must be adjacent exactly once.")
    for a, b in itertools.combinations(options, 2):
        if len(adjacent_rows(design, (a, b))) != 2:
            raise SystemExit(f"{a} and {b} must be adjacent in exactly 2 orders.")
    if edit_rows_by_rule(design) != design["edit_order_rows"]:
        raise SystemExit(f"edit_order_rows {design['edit_order_rows']} do not follow the rule.")
    if design["target_pair"] != design["shared_family"][0]:
        raise SystemExit("The target pair must be the first shared-family pair.")
    if sorted(design["family_map"]) != sorted(options) or sorted(design["prime_words"]) != sorted(options):
        raise SystemExit("family_map and prime_words must cover the options.")
    regions = set(design["regions"])
    for name in design["conditions"]:
        spec = condition_spec(name)
        if spec["kind"] == "mask" and spec["region"] not in regions | {"SHAM"}:
            raise SystemExit(f"{name} has no region in the block.")
    if not set(design["unbound_keys"]) <= set(design):
        raise SystemExit("unbound_keys must be keys of the block.")
    if design["sham_region"] not in regions:
        raise SystemExit("sham_region must be a region.")
    if design["stage3_null_claim"] != "none_reported_inconclusive":
        raise SystemExit("stage3_null_claim must be none_reported_inconclusive (decision of 2026-11-08).")
    if design["reference_check_sensitivity"] != "happy_inside_margin":
        raise SystemExit("reference_check_sensitivity must be happy_inside_margin (decision of 2026-11-08).")


def contrasts(design: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """C1 to C4 of the Predictions table. Direction is the predicted sign of the change."""
    a, b = design["target_pair"]
    return (
        {"id": "C1", "region": "mid", "acted": a, "partner": b, "direction": -1},
        {"id": "C2", "region": "mid", "acted": b, "partner": a, "direction": -1},
        {"id": "C3", "region": "midlower", "acted": b, "partner": a, "direction": 1},
        {"id": "C4", "region": "upper", "acted": a, "partner": b, "direction": 1},
    )


DESIGN = load_design()

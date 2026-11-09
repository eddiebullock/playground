"""
Design block of draft amendment A9, read from its one place: research/PREREG_A9_DRAFT.md.

Standard library only, so that any environment can hash it. A9's own constants live in the
block and are not repeated here. Values A9 shares with A4 are read from scripts/a4_design.py,
whose sha256 the block pins, so A9 cannot drift from A4 and A4 is never edited. Import refuses a
block whose hash differs from DESIGN_SHA256 or from the hash the draft states.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import a4_design as a4  # noqa: E402

DRAFT = ROOT / "research" / "PREREG_A9_DRAFT.md"
# Restated in the draft. Any edit to the block must change both, on purpose.
DESIGN_SHA256 = "21341eb47a4d5da81e7d672e21c56c83029cbf9ba2b928f98f10fcba35220192"

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
    actual, stated = block_hash(path), stated_hash(path)
    if actual != expected:
        raise SystemExit(f"A9 design block hash {actual} is not the pinned {expected}.")
    if actual != stated:
        raise SystemExit(f"A9 design block hash {actual} is not the {stated} that the draft states.")
    return json.loads(block_bytes(path))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


DESIGN = load_design()


def design_hash() -> str:
    """The block's hash. Carried by every A9 cell and output."""
    return DESIGN_SHA256


def check_pins(root: Path = ROOT, groups: Tuple[str, ...] = ("a4_pins", "input_pins")) -> None:
    """Refuses changed A4 code or outputs, or changed A9 inputs. A9 edits none of the A4 files."""
    for group in groups:
        for name, pinned in DESIGN[group].items():
            path = root / name
            if not path.exists():
                raise SystemExit(f"{name} is missing; it is pinned at {pinned}.")
            digest = file_sha256(path)
            if digest != pinned:
                raise SystemExit(f"{name} is {digest}, not the pinned {pinned}.")


def check_manifest(path: Path) -> None:
    digest = file_sha256(path)
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"Manifest hash {digest} is not the pinned {MANIFEST_SHA256}.")


def code_pins(path: Path = DRAFT) -> Dict[str, str]:
    """A9's own code, pinned in the draft outside the block at the freeze. Empty before it.

    Outside the block because this file holds the block's hash."""
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
    pins = code_pins(path)
    if not pins and require:
        raise SystemExit(f"{path} pins no A9 code. A9 is not frozen.")
    for name, pinned in pins.items():
        digest = file_sha256(root / name)
        if digest != pinned:
            raise SystemExit(f"{name} is {digest}, not the pinned {pinned}.")
    return pins


A9_FILES = (
    "scripts/a9_design.py",
    "scripts/a9_analyse.py",
    "scripts/a9_run.py",
    "scripts/a9_recovery.py",
    "research/scripts/a9_human_target.py",
    "research/scripts/a9_manifest.py",
)


def own_code_sha256(root: Path = ROOT) -> Dict[str, str]:
    return {name: file_sha256(root / name) for name in A9_FILES if (root / name).exists()}


# --------------------------------------------------------------------------
# Constants from the block
# --------------------------------------------------------------------------
LIU_DIR = DESIGN["liu"]["dir"]
LIU_FILES: Dict[str, str] = DESIGN["liu"]["files_sha256"]
LIU_SOURCE: Dict[str, str] = DESIGN["liu"]["source"]
N_STIMULI = DESIGN["liu"]["n_stimuli"]
N_SUBJECTS = DESIGN["liu"]["n_subjects"]
LIU_LABELS: Tuple[str, ...] = tuple(DESIGN["liu"]["au_labels"])
RESPONSE_CODES: Dict[int, str] = {int(k): v for k, v in DESIGN["liu"]["response_codes"].items()}

SUFFIXES: Tuple[str, ...] = tuple(DESIGN["au_map"]["suffixes_dropped"])
COLLISION_RULE = DESIGN["au_map"]["collision_rule"]
AMPLITUDE_SOURCE = DESIGN["au_map"]["amplitude_source"]
AMPLITUDE_DECIMALS = DESIGN["au_map"]["amplitude_decimals"]
UNCOVERED_RULE = DESIGN["au_map"]["uncovered_rule"]
UNCOVERED: Tuple[str, ...] = tuple(DESIGN["au_map"]["uncovered"])
AU_COLS: Tuple[str, ...] = tuple(a4_cols for a4_cols in DESIGN["au_map"]["columns"])

DATASET = DESIGN["dataset"]
DATASETS = (DATASET,)
TASK = DESIGN["task"]
SEED = DESIGN["seed"]
HUMAN_RIDGE = DESIGN["human"]["ridge_selected"]
LAMBDA_GRID: Tuple[float, ...] = tuple(DESIGN["human"]["lambda_grid"])
CEILING_LAMBDA = DESIGN["human"]["ceiling_ridge"]
IDENTITIES = a4.IDENTITIES[DESIGN["renders"]["identity_group"]]
REPLICATE_CONFIGS = DESIGN["renders"]["replicate_configs"]
FRAME_COUNT = DESIGN["renders"]["frame_count"]
SETS: Tuple[str, ...] = tuple(DESIGN["renders"]["sets"])
DESIGNS: Tuple[str, ...] = tuple(DESIGN["renders"]["designs"])
DESIGN_SELECTED = DESIGN["renders"]["design_selected"]
DESIGN_B_GENDERS: Tuple[float, ...] = tuple(DESIGN["renders"]["design_b_genders"])
MANIFEST_SHA256 = DESIGN["renders"]["manifest_sha256"]
RECOVERY_WITHDRAWN: Tuple[str, ...] = tuple(DESIGN["recovery"]["withdrawn_cells"])
CORRELATED_RHOS: Tuple[float, ...] = tuple(DESIGN["recovery"]["face_noise_correlations"])
RECOVERY_RUNS: Tuple[str, ...] = tuple(DESIGN["recovery"]["runs"])

# --------------------------------------------------------------------------
# Shared with A4, read from a4_design (pinned by sha256 in the block)
# --------------------------------------------------------------------------
_SHARED = DESIGN["shared_with_a4"]
OPTIONS = a4.OPTIONS[TASK]
HUMAN_LABELS = a4.HUMAN_LABELS[TASK]
ABSTAIN = a4.ABSTAIN
ROTATIONS = a4.ROTATIONS[TASK]
MODELS = a4.MODELS
DESCRIPTIVE_MODELS = a4.DESCRIPTIVE_MODELS
STATISTIC = a4.STATISTIC
SESOI_CFA = a4.SESOI_CFA
ALPHA = a4.ALPHA
R_S_LOWER_BOUND = a4.R_S_LOWER_BOUND
MAX_LOST_SHARE = a4.MAX_LOST_SHARE
CUE_PERMUTATIONS = a4.CUE_PERMUTATIONS
SIGNAL_PERMUTATIONS = a4.SIGNAL_PERMUTATIONS
BOOTSTRAP_DRAWS = a4.BOOTSTRAP_DRAWS
SPLIT_HALF_SPLITS = a4.SPLIT_HALF_SPLITS
REPLICATE_DRAWS = a4.REPLICATE_DRAWS
CEILING_SPLITS = a4.CEILING_SPLITS
BASELINE_STIMULI = a4.BASELINE_STIMULI
V_RULES = a4.V_RULES
RECOVERY = a4.RECOVERY
APPROXIMATED_FAMILIES = a4.APPROXIMATED_FAMILIES
CONFIRMED_FAMILIES = a4.CONFIRMED_FAMILIES
RENDER_TARS = a4.RENDER_TARS

# The block restates the shared values it relies on; a mismatch with a4_design stops import.
_ACTUAL = {"statistic": STATISTIC, "sesoi_cfa": SESOI_CFA, "alpha": ALPHA, "models": list(MODELS),
           "options": list(OPTIONS), "human_labels": list(HUMAN_LABELS), "rotations": ROTATIONS,
           "r_s_lower_bound": R_S_LOWER_BOUND, "max_lost_share": MAX_LOST_SHARE,
           "cue_permutations": CUE_PERMUTATIONS, "signal_permutations": SIGNAL_PERMUTATIONS,
           "bootstrap_draws": BOOTSTRAP_DRAWS, "split_half_splits": SPLIT_HALF_SPLITS,
           "ceiling_splits": CEILING_SPLITS, "baseline_stimuli": BASELINE_STIMULI}
for _key, _value in _ACTUAL.items():
    if _SHARED[_key] != _value:
        raise SystemExit(f"A9 block states {_key} = {_SHARED[_key]!r}; a4_design has {_value!r}.")


# --------------------------------------------------------------------------
# AU map
# --------------------------------------------------------------------------


def snoek_token(liu_label: str) -> str:
    out = liu_label
    for suffix in SUFFIXES:
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def token_columns(token: str) -> Tuple[str, ...]:
    """The 33-basis columns a raw token sets. Raises if any part has no column."""
    side = token[-1] if token[-1] in "LR" else None
    out: List[str] = []
    for number in re.findall(r"\d+", token):
        family = f"AU{int(number):02d}"
        if family in AU_COLS:
            out.append(family)
            continue
        sides = [family + s for s in ("L", "R")] if side is None else [family + side]
        missing = [c for c in sides if c not in AU_COLS]
        if missing:
            raise KeyError(f"{token}: no column for {missing}")
        out.extend(sides)
    return tuple(out)


AU_MAP: Dict[str, Tuple[str, ...]] = {label: token_columns(snoek_token(label)) for label in LIU_LABELS}
if {k: list(v) for k, v in AU_MAP.items()} != DESIGN["au_map"]["map"]:
    raise SystemExit("The AU map derived from the rule differs from the map the block states.")
SUFFIXED_LABELS = tuple(label for label in LIU_LABELS if snoek_token(label) != label)
COMPOUND_LABELS = tuple(label for label in LIU_LABELS if "-" in label)


def liu_to_columns(amplitudes: Mapping[str, float], decimals: int = AMPLITUDE_DECIMALS) -> Dict[str, float]:
    """{Liu label: amplitude} -> {column: amplitude}, by the collision rule, zeros dropped."""
    out: Dict[str, float] = {}
    for label, value in amplitudes.items():
        value = float(value)
        if label in UNCOVERED or value <= 0:
            continue
        for column in AU_MAP[label]:
            out[column] = max(out.get(column, 0.0), value)
    rounded = {k: round(v, decimals) for k, v in out.items()}
    return {k: v for k, v in sorted(rounded.items()) if v > 0}


def family_table() -> List[Dict[str, Any]]:
    families = sorted({c[:4] for c in AU_COLS})
    rows = []
    for family in families:
        members = [label for label in LIU_LABELS if any(c[:4] == family for c in AU_MAP[label])]
        rows.append({"family": family, "liu_labels": members,
                     "approximated_in_render": family in APPROXIMATED_FAMILIES})
    return rows


def check_liu_files(root: Path = ROOT) -> None:
    for name, digest in LIU_FILES.items():
        got = file_sha256(root / LIU_DIR / name)
        if got != digest:
            raise SystemExit(f"{name}: sha256 {got} is not the pinned {digest}")


def rotated_options(rotation: int) -> List[str]:
    return a4.rotated_options(TASK, rotation)


def shard_of(stim_id: str, n_shards: int) -> int:
    return a4.shard_of(stim_id, n_shards)

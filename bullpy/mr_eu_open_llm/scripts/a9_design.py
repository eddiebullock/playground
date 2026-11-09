"""
Fixed specification of draft amendment A9: A4's confirmatory test replicated on Liu et al. 2022,
Experiment II. See research/PREREG_A9_DRAFT.md.

Standard library only, so that any environment can hash it. Values that A9 shares with A4 are
read from scripts/a4_design.py, not retyped, so that A9 cannot drift from A4 by accident; A4's
file is never edited. Every A9 output carries `design_hash()`.
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

AMENDMENT = "A9"
STATUS = "DRAFT"

# --------------------------------------------------------------------------
# Source data. Gitignored, never copied into tracked files.
# --------------------------------------------------------------------------
LIU_DIR = "data/external/liu2022"
LIU_FILES = {
    "resp_threetask_20par.mat": "52602f372f39660fa6b8117598255f74cabaa54564774799e6155b5a28c275d5",
    "parameters_dynamic_precise.mat": "79e24df6b74d2dc76ad8ae95655042f8a2840c12538bacdd410dfd72106c143e",
}
LIU_SOURCE = {
    "article": "Liu M, Duan Y, Ince RAA, Chen C, Garrod OGB, Schyns PG, Jack RE (2022). Facial expressions "
               "elicit multiplexed perceptions of emotion categories and dimensions. Current Biology 32(1), "
               "200-209.e6. DOI 10.1016/j.cub.2021.10.035",
    "data_doi": "10.17632/dr853shk56.2",
    "licence": "CC BY 4.0",
}
N_STIMULI = 1200
N_SUBJECTS = 20

# The 42 AU labels, in the order of params.labels in parameters_dynamic_precise.mat. Read from the
# file; the tests check the file still says this.
LIU_LABELS = (
    "AU1", "AU1-2", "AU2", "AU2L", "AU4", "AU5", "AU6", "AU6L", "AU6R", "AU7L", "AU7R", "AU9",
    "AU10Open", "AU10LOpen", "AU10ROpen", "AU11L", "AU11R", "AU12", "AU25-12", "AU12L", "AU12R",
    "AU13", "AU14", "AU14L", "AU14R", "AU15", "AU16Open", "AU17", "AU20", "AU20L", "AU20R", "AU22",
    "AU23", "AU24", "AU25", "AU26", "AU27i", "AU38", "AU39", "AU43", "AU7", "AU12-6",
)

# Response codes 1 to 7 of resp_emo. Source: the released analysis code of Liu et al. (Mendeley Data
# 10.17632/dr853shk56.3, PMI_exp2.m), which declares
#   emotion_label = {'happy','surprise','fear','disgust','anger','sad','other'};
# and drops "other" with `idx_cond = emo_all<7`. The same lab's Chen et al. 2024 deposit
# (WE_resp_data.mat, label_responses) uses the same order. The variable is declared but not used
# for indexing in that script, so the order is corroborated against the data in the human target
# output (prototype AUs per code), never inferred from it.
RESPONSE_CODES = {1: "happy", 2: "surprise", 3: "fear", 4: "disgust", 5: "anger", 6: "sadness", 7: "other"}

# --------------------------------------------------------------------------
# AU map. Liu label -> Snoek raw token -> A4's 33 columns.
# --------------------------------------------------------------------------
# Snoek's row labels use the same 42 raw AUs with the suffixes "Open" and "i" dropped (AU10Open is
# written AU10, AU16Open AU16, AU27i AU27). The 42 stripped names equal Snoek's 42 tokens exactly
# (checked in the tests and in a9_human_target's output).
SUFFIXES = ("Open", "i")

# A4's 33 columns, as research/scripts/human_au_effects.AU_COLS (tests check they match).
AU_COLS = (
    "AU01", "AU02L", "AU02R", "AU04", "AU05", "AU06L", "AU06R", "AU07L", "AU07R",
    "AU09", "AU10L", "AU10R", "AU11L", "AU11R", "AU12L", "AU12R", "AU13", "AU14L",
    "AU14R", "AU15", "AU16", "AU17", "AU20L", "AU20R", "AU22", "AU23", "AU24",
    "AU25", "AU26", "AU27", "AU38", "AU39", "AU43",
)
# Snoek's recoding, as human_au_effects.DATA_CAVEATS describes it and as verified against 6,966
# Snoek configurations whose labels keep raw tokens (zero mismatches): a bilateral AU sets both its
# L and R columns, a one-sided AU sets its own side, a compound sets each part, and where two raw AUs
# set one column the column takes the maximum. Snoek then rounded columns to one decimal; A9 keeps
# AMPLITUDE_DECIMALS instead, because Liu releases continuous amplitudes.
COLLISION_RULE = "max"
AMPLITUDE_SOURCE = "p"  # peak amplitude per AU and stimulus, parameters_dynamic_precise.mat
AMPLITUDE_DECIMALS = 4  # as a4_stimuli.scaled rounds rendered amplitudes


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
# Rule fixed before any model data: a Liu AU whose parts have no column, or no renderer target, is
# dropped from both the human fit and the renders, and listed. With A4's renderer none is dropped.
UNCOVERED_RULE = "drop from human design and renders alike; list in outputs"
UNCOVERED: Tuple[str, ...] = ()
# Families A4 already treats as approximated in renders (research/notes_au_render.md).
APPROXIMATED_FAMILIES = a4.APPROXIMATED_FAMILIES
# Raw AUs whose GFG form has a component the A4 target for that column does not draw: the "Open"
# variants part the lips in GFG (UNVERIFIED: inferred from the name only), and AU27i is GFG's form of
# mouth stretch. Each is rendered with the A4 target of its column, as A4 does for Snoek, where the
# same raw AUs were merged by Snoek's recoding.
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
    """Liu AUs by A4 family, for the draft's table."""
    families = sorted({c[:4] for c in AU_COLS})
    rows = []
    for family in families:
        members = [label for label in LIU_LABELS if any(c[:4] == family for c in AU_MAP[label])]
        rows.append({"family": family, "liu_labels": members,
                     "approximated_in_render": family in APPROXIMATED_FAMILIES})
    return rows


# --------------------------------------------------------------------------
# What A9 reuses from A4 unchanged.
# --------------------------------------------------------------------------
DATASET = "liu_WC"
DATASETS = (DATASET,)
TASK = "basic"
OPTIONS = a4.OPTIONS["basic"]
HUMAN_LABELS = a4.HUMAN_LABELS["basic"]
ABSTAIN = a4.ABSTAIN
ROTATIONS = a4.ROTATIONS["basic"]
MODELS = a4.MODELS  # A9's own Holm family, separate from A4's
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
V_RULES = a4.V_RULES
RECOVERY = a4.RECOVERY  # same observers, strengths, realism and pass rule
# A4's withdrawn cells name A4 datasets only, so no A9 cell is withdrawn.
RECOVERY_WITHDRAWN: Tuple[str, ...] = ()

# Ridge selection: the human_au_effects rule (held-out log loss on unseen stimuli, two folds), on
# this grid, which equals human_au_effects.LAMBDA_GRID (tests check). Gender is not in Liu's data,
# so the human fit has an intercept as its only unpenalised covariate.
LAMBDA_GRID = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0)
CEILING_LAMBDA = 1.0  # a4_analyse.human_do_ceiling fits its halves at 1.0 for every dataset

# --------------------------------------------------------------------------
# Renders and manifest.
# --------------------------------------------------------------------------
SEED = 20261109
IDENTITIES = a4.IDENTITIES["WC"]  # four Western faces, as A4's WC sets
REPLICATE_CONFIGS = 200  # first 200 by rank on the other face of the same gender, for V3
BASELINE_STIMULI = a4.BASELINE_STIMULI  # V2, image-free, first 50 by rank
SETS = ("main", "replicate", "frame_max", "all_identities")
# Design A (primary): each configuration once, on one face. Design B (fallback, fixed before
# recovery ran): each configuration on all four faces, the model's responses averaged per
# configuration before the fit. B is used only if A fails recovery and B passes; see the draft.
DESIGNS = ("A", "B")
DESIGN_B_IDENTITIES = 4
# frame_max: the single animation frame (of 75) with the largest summed AU activation, from pseq.
# Secondary and descriptive: it is the only set in which every AU has the amplitude it had at one
# moment, whereas `main` sets every AU at its own peak, which may never coincide.
FRAME_COUNT = 75

DESIGN_FILES = (
    "research/reproduction/a9_human_target.json",
    "scripts/a4_design.py",
    "scripts/a4_analyse.py",
    "scripts/a4_recovery.py",
    "research/scripts/human_au_effects.py",
    "research/scripts/render_au_faces.py",
    "research/scripts/a4_stimuli.py",
    "scripts/prompts.py",
    "scripts/run_stimulus_interventions.py",
)


def constants() -> Dict[str, Any]:
    return {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (str, int, float, tuple, dict))}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_digests(root: Path = ROOT) -> Dict[str, str]:
    return {name: file_digest(root / name) for name in DESIGN_FILES if (root / name).exists()}


def design_hash() -> str:
    """Hash of every constant here (including those read from A4) and of DESIGN_FILES present."""
    payload = {"constants": constants(), "files": file_digests(), "missing": [
        n for n in DESIGN_FILES if not (ROOT / n).exists()]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()


def check_liu_files(root: Path = ROOT) -> None:
    for name, digest in LIU_FILES.items():
        got = file_digest(root / LIU_DIR / name)
        if got != digest:
            raise SystemExit(f"{name}: sha256 {got} is not the pinned {digest}")


def rotated_options(rotation: int) -> List[str]:
    return a4.rotated_options(TASK, rotation)

#!/usr/bin/env python3
"""Headless renderer for facial action unit configurations on a MakeHuman avatar.

Builds synthetic stimuli for the comparison of vision-language models with the
randomised human AU effects in the Snoek data. Nothing here calls a model of that
kind, and no render should be shown to one as part of building or checking stimuli.

Method: load MakeHuman's base mesh, shape it into an identity with MakeHuman's macro
targets, add amplitude-weighted FACSHuman offsets for the active AUs, fit eyes, teeth,
tongue, brows and lashes by their .mhclo bindings, then draw the result with a numpy
z-buffer rasteriser. There is no GUI, GPU or display dependency: numpy, scipy and
Pillow are all that is needed, and only for this file's main path.

Assets are not in the repo. Point AU_RENDER_ASSETS at a folder holding:
    FACSHuman/                 clone of github.com/montybot/FACSHuman
    FACSHuman-Py3-Enhanced/    clone of the same author's MakeHuman fork (data only)
    mh_assets_x/               skins, eyes, teeth, tongue, eyebrows and eyelashes from
                               makehuman_system_assets_cc0.zip

Limits, stated so they travel with the output:
  * Static peak frame. Humans saw 1.25 s animations.
  * Amplitude maps linearly to target weight. That is an assumption: a GFG amplitude
    of 0.5 has no known match here.
  * AU22 and AU27 are approximated. See COVERAGE.
  * Laterality is anatomical: AU..L moves the avatar's left side, on the right of the
    image. Whether GFG's L means the same is unverified.
  * Avatars are bald, with no hair asset. GFG faces and identities are not reproduced.
  * Column amplitudes are rounded to one decimal, as in the Snoek columns.

Usage:
    python research/scripts/render_au_faces.py render --au AU12L=1,AU12R=1 \
        --identity female_a --seed 0 --out face.png
    python research/scripts/render_au_faces.py coverage
    python research/scripts/render_au_faces.py validate --out-json report.json
    python research/scripts/render_au_faces.py timing --zip data/external/snoek_au/au_emotion_data.zip
    python research/scripts/render_au_faces.py batch --zip data/external/snoek_au/au_emotion_data.zip \
        --out-dir renders --identity neutral --shard 0/10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = Path(
    "/private/tmp/claude-502/-Users-eb2007/cf56ec16-1588-406d-9db5-6872ba6bedc3/scratchpad/au_render/src"
)
RESOLUTION = 768
SUPERSAMPLE = 2
AMBIENT, DIFFUSE = 0.62, 0.38  # even frontal light; shape stays legible, no cast shadows
MOUTH_INTERIOR = np.array([0.09, 0.035, 0.035], np.float32)
HEAD_MIN_Y = 5.3  # base-mesh units are decimetres; AU offsets below this are not face
NECK_CUT_Y = 5.9  # skin below this is not drawn, so no shoulders enter the frame
FRAME_MARGIN = 0.70  # crown-to-chin height as a fraction of image height
CAMERA_DISTANCE = 10.0  # 1 m, a mild portrait perspective
VERTEX_COUNT = 19158

# Same order and names as research/scripts/human_au_effects.py. Copied so this script
# has no pandas dependency; the tests check the two lists match.
AU_COLS = [
    "AU01", "AU02L", "AU02R", "AU04", "AU05", "AU06L", "AU06R", "AU07L", "AU07R",
    "AU09", "AU10L", "AU10R", "AU11L", "AU11R", "AU12L", "AU12R", "AU13", "AU14L",
    "AU14R", "AU15", "AU16", "AU17", "AU20L", "AU20R", "AU22", "AU23", "AU24",
    "AU25", "AU26", "AU27", "AU38", "AU39", "AU43",
]

UPPER = "01 Upper Face AUs"
LOWER = "02 Lower Face AUs"
JAW = "03 Lip Parting and Jaw Opening"
MISC = "06 Miscellaneous AUs"

# FACSHuman target for each non-lateral column.
SINGLE_TARGETS = {
    "AU01": (UPPER, "1"), "AU04": (UPPER, "4"), "AU05": (UPPER, "5"), "AU09": (LOWER, "9"),
    "AU13": (LOWER, "13"), "AU15": (LOWER, "15"), "AU16": (LOWER, "16"), "AU17": (LOWER, "17"),
    "AU23": (LOWER, "23"), "AU24": (LOWER, "24"), "AU25": (JAW, "25"), "AU26": (JAW, "26"),
    "AU38": (MISC, "38"), "AU39": (MISC, "39"), "AU43": (UPPER, "43"),
}
# Lateral families: folder, stem. Uses the L, R and bilateral target files.
LATERAL_TARGETS = {
    "AU02": (UPPER, "2"), "AU06": (UPPER, "6"), "AU07": (UPPER, "7"), "AU10": (LOWER, "10"),
    "AU11": (LOWER, "11"), "AU12": (LOWER, "12"), "AU14": (LOWER, "14"), "AU20": (LOWER, "20"),
}

# Composite recipes. See COVERAGE for why these two are not plain target files.
AU22_VERTICAL_KEEP = 0.6  # share of the 22_25 target's vertical motion kept for AU22
AU22_PURSE = 0.8  # weight of MakeHuman's mouth-pursing, which adds horizontal narrowing
AU22_EVERT = 1.0  # weight of MakeHuman's mouth-eversion, which rolls the lips outward
AU27_JAW = 1.3  # jaw travel relative to AU26 at amplitude 1
AU27_NARROW = 1.0  # weight of the horizontal part of MakeHuman's mouth-open
AU27_LIP_RAISE = 0.5  # weight of AU25, which lifts the upper lip off the teeth

COVERAGE: Dict[str, Tuple[str, str]] = {}
for _col in AU_COLS:
    _family = _col[:4]
    if _col in SINGLE_TARGETS:
        COVERAGE[_col] = ("direct", f"FACSHuman {SINGLE_TARGETS[_col][1]}.target")
    elif _family in LATERAL_TARGETS:
        _stem = LATERAL_TARGETS[_family][1]
        COVERAGE[_col] = ("direct", f"FACSHuman {_col[-1]}{_stem}.target; shared L and R part uses {_stem}.target")
COVERAGE["AU38"] = ("direct", "FACSHuman 38.target with one stray pelvis vertex removed")
COVERAGE["AU39"] = ("direct", "FACSHuman 39.target with one stray pelvis vertex removed")
COVERAGE["AU22"] = (
    "approximated",
    f"FACSHuman 22_25_up_down.target, which bakes in lip parting, with vertical motion scaled by "
    f"{AU22_VERTICAL_KEEP}, plus {AU22_PURSE} MakeHuman mouth-pursing and {AU22_EVERT} mouth-eversion. "
    f"Some lip parting remains, and MediaPipe scores it more as pucker than funnel",
)
COVERAGE["AU27"] = (
    "approximated",
    f"No target file. Built as {AU27_JAW} x AU26 jaw travel, {AU27_NARROW} x horizontal part of MakeHuman "
    f"mouth-open, {AU27_LIP_RAISE} x AU25. Shares the jaw with AU26 by maximum, not sum",
)


# --------------------------------------------------------------------------
# Paths and identity definitions
# --------------------------------------------------------------------------


def assets_root() -> Path:
    root = Path(os.environ.get("AU_RENDER_ASSETS", DEFAULT_ASSETS))
    if not (root / "FACSHuman" / "FACSHuman" / "au.json").exists():
        raise FileNotFoundError(
            f"FACSHuman assets not found under {root}. Set AU_RENDER_ASSETS; see this file's docstring."
        )
    return root


def mh_data() -> Path:
    return assets_root() / "FACSHuman-Py3-Enhanced" / "makehuman" / "data"


def system_assets() -> Path:
    return assets_root() / "mh_assets_x"


def facs_targets() -> Path:
    return assets_root() / "FACSHuman" / "FACSHuman"


@dataclass(frozen=True)
class Identity:
    name: str
    gender: float  # 0 female, 1 male
    races: Tuple[Tuple[str, float], ...]
    skins: Tuple[str, ...]
    muscle: float = 0.5
    weight: float = 0.5
    height: float = 0.5
    proportions: float = 0.5
    details: Tuple[Tuple[str, float], ...] = ()
    eyebrow: str = "eyebrow001"
    eye_colour: str = "brown"


_EQUAL_RACE = (("african", 1 / 3), ("asian", 1 / 3), ("caucasian", 1 / 3))
IDENTITIES: Dict[str, Identity] = {
    i.name: i
    for i in [
        Identity("neutral", 0.5, _EQUAL_RACE,
                 ("young_caucasian_female", "young_caucasian_male", "young_asian_female", "young_asian_male")),
        Identity("female_a", 0.0, (("caucasian", 1.0),), ("young_caucasian_female",), weight=0.42, proportions=0.8,
                 details=(("head/head-oval", 0.6), ("nose/nose-scale-horiz-decr", 0.3), ("chin/chin-width-decr", 0.3)),
                 eyebrow="eyebrow001", eye_colour="blue"),
        Identity("female_b", 0.0, (("caucasian", 1.0),), ("young_caucasian_female2",), weight=0.68, muscle=0.4,
                 proportions=0.35,
                 details=(("head/head-round", 0.6), ("nose/nose-scale-horiz-incr", 0.25), ("chin/chin-height-decr", 0.3)),
                 eyebrow="eyebrow002", eye_colour="brown"),
        Identity("male_a", 1.0, (("caucasian", 1.0),), ("young_caucasian_male",), muscle=0.68, proportions=0.7,
                 details=(("head/head-square", 0.6), ("chin/chin-width-incr", 0.4), ("nose/nose-scale-depth-incr", 0.25)),
                 eyebrow="eyebrow009", eye_colour="brown"),
        Identity("male_b", 1.0, (("caucasian", 1.0),), ("young_caucasian_male2",), weight=0.36, muscle=0.45,
                 proportions=0.3,
                 details=(("head/head-rectangular", 0.6), ("nose/nose-scale-vert-incr", 0.3), ("chin/chin-height-incr", 0.3)),
                 eyebrow="eyebrow010", eye_colour="green"),
        # East Asian faces, since the East Asian sets hold most shared configurations.
        Identity("ea_female", 0.0, (("asian", 1.0),), ("young_asian_female",), proportions=0.65,
                 details=(("head/head-oval", 0.4),), eyebrow="eyebrow002", eye_colour="brown"),
        Identity("ea_male", 1.0, (("asian", 1.0),), ("young_asian_male",), proportions=0.6,
                 details=(("head/head-square", 0.4),), eyebrow="eyebrow009", eye_colour="brown"),
    ]
}


# --------------------------------------------------------------------------
# File parsers
# --------------------------------------------------------------------------


@dataclass
class Obj:
    vertices: np.ndarray
    uvs: np.ndarray
    faces: List[Tuple[np.ndarray, np.ndarray, Optional[str]]]  # vertex ids, uv ids, group


def read_obj(path: Path) -> Obj:
    vertices, uvs, faces = [], [], []
    group = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tag = line[:2]
            if tag == "v ":
                vertices.append(line.split()[1:4])
            elif tag == "vt":
                uvs.append(line.split()[1:3])
            elif tag == "g ":
                group = line.split()[1] if len(line.split()) > 1 else None
            elif tag == "f ":
                parts = [p.split("/") for p in line.split()[1:]]
                faces.append(
                    (
                        np.array([int(p[0]) - 1 for p in parts]),
                        np.array([int(p[1]) - 1 if len(p) > 1 and p[1] else 0 for p in parts]),
                        group,
                    )
                )
    return Obj(np.array(vertices, np.float32), np.array(uvs, np.float32).reshape(-1, 2), faces)


@lru_cache(maxsize=None)
def read_target(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """A .target file: rows of vertex index and x, y, z offset."""
    ids, offsets = [], []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            words = line.split()
            if len(words) >= 4 and not line.startswith("#"):
                ids.append(int(words[0]))
                offsets.append(words[1:4])
    return np.array(ids, np.int64), np.array(offsets, np.float32).reshape(-1, 3)


def dense_target(path: Path) -> np.ndarray:
    ids, offsets = read_target(str(path))
    out = np.zeros((VERTEX_COUNT, 3), np.float32)
    out[ids] = offsets
    return out


@dataclass
class Clothes:
    """A proxy mesh bound to base-mesh vertices."""

    refs: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    scale_data: Tuple[Tuple[int, int, float], ...]
    material: str


def read_mhclo(path: Path) -> Clothes:
    refs, weights, offsets = [], [], []
    scales: Dict[int, Tuple[int, int, float]] = {}
    material = ""
    in_verts = False
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            words = line.split()
            if not words or line.startswith("#"):
                continue
            if in_verts:
                if len(words) == 1 and words[0].lstrip("-").isdigit():
                    refs.append((int(words[0]), 0, 1))
                    weights.append((1.0, 0.0, 0.0))
                    offsets.append((0.0, 0.0, 0.0))
                    continue
                if len(words) >= 6 and words[0].lstrip("-").isdigit():
                    refs.append(tuple(int(w) for w in words[:3]))
                    weights.append(tuple(float(w) for w in words[3:6]))
                    offsets.append(tuple(float(w) for w in words[6:9]) if len(words) >= 9 else (0.0, 0.0, 0.0))
                    continue
                in_verts = False
            key = words[0]
            if key == "verts":
                in_verts = True
            elif key in ("x_scale", "y_scale", "z_scale"):
                scales["xyz".index(key[0])] = (int(words[1]), int(words[2]), float(words[3]))
            elif key == "material":
                material = words[1]
    scale_data = tuple(scales[i] for i in range(3)) if len(scales) == 3 else ()
    return Clothes(
        np.array(refs, np.int64),
        np.array(weights, np.float32),
        np.array(offsets, np.float32),
        scale_data,
        material,
    )


def read_diffuse_name(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("diffuseTexture"):
                return line.split()[1]
    raise ValueError(f"no diffuseTexture in {path}")


def load_texture(paths: Sequence[Path]) -> np.ndarray:
    """RGBA uint8 texture; several paths are averaged (same UV layout)."""
    stack = [np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) for p in paths]
    return np.rint(np.mean(stack, axis=0)).astype(np.uint8)


# --------------------------------------------------------------------------
# Identity shaping: MakeHuman's macro target weights
# --------------------------------------------------------------------------


def _tri_weights(value: float, low: str, mid: str, high: str) -> Dict[str, float]:
    top = max(0.0, value * 2 - 1)
    bottom = max(0.0, 1 - value * 2)
    return {low: bottom, mid: 1 - top - bottom, high: top}


def macro_target_weights(identity: Identity) -> Dict[str, float]:
    """Relative target path to weight, following MakeHuman's apps/human.py formulas.

    Adults at age 25 only: the 'young' age bin has weight 1.
    """
    genders = {"female": 1 - identity.gender, "male": identity.gender}
    muscles = _tri_weights(identity.muscle, "minmuscle", "averagemuscle", "maxmuscle")
    weights = _tri_weights(identity.weight, "minweight", "averageweight", "maxweight")
    heights = {"minheight": max(0.0, 1 - identity.height * 2), "maxheight": max(0.0, identity.height * 2 - 1)}
    props = {
        "uncommonproportions": max(0.0, 1 - identity.proportions * 2),
        "idealproportions": max(0.0, identity.proportions * 2 - 1),
    }
    out: Dict[str, float] = {}
    total_race = sum(w for _, w in identity.races)
    for gender, g in genders.items():
        for race, r in identity.races:
            out[f"macrodetails/{race}-{gender}-young"] = g * r / total_race
        for muscle, m in muscles.items():
            for weight, w in weights.items():
                base = g * m * w
                out[f"macrodetails/universal-{gender}-young-{muscle}-{weight}"] = base
                stem = f"{gender}-young-{muscle}-{weight}"
                for height, h in heights.items():
                    out[f"macrodetails/height/{stem}-{height}"] = base * h
                for prop, p in props.items():
                    out[f"macrodetails/proportions/{stem}-{prop}"] = base * p
    for name, weight in identity.details:
        out[name] = weight
    return {k: v for k, v in out.items() if v > 1e-6}


def jitter_identity(identity: Identity, seed: int) -> Tuple[Identity, float]:
    """Seed 0 is the canonical identity. Others get small body-shape and skin-gain changes."""
    if seed == 0:
        return identity, 1.0
    rng = np.random.default_rng([20261107, seed, sum(identity.name.encode())])
    jitter = rng.uniform(-1, 1, 4)
    fields = {
        "muscle": float(np.clip(identity.muscle + 0.05 * jitter[0], 0, 1)),
        "weight": float(np.clip(identity.weight + 0.05 * jitter[1], 0, 1)),
        "proportions": float(np.clip(identity.proportions + 0.1 * jitter[2], 0, 1)),
    }
    changed = Identity(**{**identity.__dict__, **fields})
    return changed, float(1 + 0.03 * jitter[3])


# --------------------------------------------------------------------------
# Action units to vertex offsets
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def base_mesh() -> Obj:
    return read_obj(mh_data() / "3dobjs" / "base.obj")


def _head_only(offsets: np.ndarray) -> np.ndarray:
    """Zero offsets below the neck. 38 and 39 carry one stray vertex at the pelvis."""
    out = offsets.copy()
    out[base_mesh().vertices[:, 1] < HEAD_MIN_Y] = 0
    return out


@lru_cache(maxsize=None)
def au_channel(key: str) -> np.ndarray:
    """Offsets of one channel at full amplitude, vertices by 3."""
    root = facs_targets()
    expression = mh_data() / "targets" / "expression" / "units" / "caucasian"
    if key == "AU22":
        funnel = dense_target(root / LOWER / "22_25_up_down.target")
        funnel[:, 1] *= AU22_VERTICAL_KEEP
        funnel += AU22_PURSE * dense_target(expression / "mouth-pursing.target")
        return _head_only(funnel + AU22_EVERT * dense_target(expression / "mouth-eversion.target"))
    if key == "AU27_shape":  # the part of AU27 that is not jaw travel
        narrow = dense_target(expression / "mouth-open.target")
        narrow[:, 1:] = 0
        return _head_only(AU27_NARROW * narrow + AU27_LIP_RAISE * au_channel("AU25"))
    if key in SINGLE_TARGETS:
        folder, stem = SINGLE_TARGETS[key]
        return _head_only(dense_target(root / folder / f"{stem}.target"))
    family, side = key.split(":")  # for example AU12:L, AU12:R, AU12:both
    folder, stem = LATERAL_TARGETS[family]
    name = stem if side == "both" else f"{side}{stem}"
    return _head_only(dense_target(root / folder / f"{name}.target"))


def check_config(au: Mapping[str, float]) -> Dict[str, float]:
    unknown = sorted(set(au) - set(AU_COLS))
    if unknown:
        raise ValueError(f"unknown AU columns: {unknown}")
    out = {}
    for key, value in au.items():
        value = float(value)
        if not 0 <= value <= 1 or not np.isfinite(value):
            raise ValueError(f"{key} amplitude {value} is outside 0 to 1")
        out[key] = value
    return out


def au_offsets(au: Mapping[str, float]) -> np.ndarray:
    """Amplitude-weighted sum of offsets for an AU configuration, vertices by 3.

    L and R amplitudes split into a shared part, drawn with the bilateral target, and
    a one-sided remainder. The two meet continuously and neither needs a threshold.
    AU26 and AU27 share one jaw by maximum, so both at full amplitude cannot overshoot.
    """
    au = check_config(au)
    total = np.zeros((VERTEX_COUNT, 3), np.float32)
    for key, value in au.items():
        if value and (key in SINGLE_TARGETS and key != "AU26" or key == "AU22"):
            total += value * au_channel(key)
    for family in LATERAL_TARGETS:
        left, right = au.get(family + "L", 0.0), au.get(family + "R", 0.0)
        shared = min(left, right)
        if shared:
            total += shared * au_channel(f"{family}:both")
        if left > shared:
            total += (left - shared) * au_channel(f"{family}:L")
        if right > shared:
            total += (right - shared) * au_channel(f"{family}:R")
    a26, a27 = au.get("AU26", 0.0), au.get("AU27", 0.0)
    jaw = max(a26, AU27_JAW * a27)
    if jaw:
        total += jaw * au_channel("AU26")
    if a27:
        total += a27 * au_channel("AU27_shape")
    return total


# --------------------------------------------------------------------------
# Mesh helpers
# --------------------------------------------------------------------------


def catmull_clark(quads: np.ndarray, n_vertices: int) -> Tuple[sp.csr_matrix, np.ndarray]:
    """One Catmull-Clark level for an all-quad mesh.

    Returns (S, new_quads) where S maps old to new vertex positions. New vertices are
    ordered: old vertices, edge points, face points. Boundary vertices keep the usual
    boundary rule, so the cut at the neck stays clean.
    """
    n_faces = len(quads)
    corners = np.arange(4)
    face_ids = np.repeat(np.arange(n_faces), 4)
    a = quads[:, corners].ravel()
    b = quads[:, (corners + 1) % 4].ravel()
    low, high = np.minimum(a, b), np.maximum(a, b)
    pairs = low.astype(np.int64) * n_vertices + high
    unique, edge_of_side = np.unique(pairs, return_inverse=True)
    n_edges = len(unique)
    edge_low, edge_high = unique // n_vertices, unique % n_vertices
    edge_faces = np.bincount(edge_of_side, minlength=n_edges)
    boundary_edge = edge_faces == 1

    face_mat = sp.csr_matrix(
        (np.full(4 * n_faces, 0.25), (face_ids, quads.ravel())), shape=(n_faces, n_vertices)
    )
    # Edge points: midpoints on the boundary, else mean of ends and adjacent face points.
    ends = sp.csr_matrix(
        (np.ones(2 * n_edges), (np.r_[np.arange(n_edges), np.arange(n_edges)], np.r_[edge_low, edge_high])),
        shape=(n_edges, n_vertices),
    )
    side_faces = sp.csr_matrix(
        (np.ones(4 * n_faces), (edge_of_side, face_ids)), shape=(n_edges, n_faces)
    )
    interior = sp.diags((~boundary_edge).astype(float))
    border = sp.diags(boundary_edge.astype(float))
    edge_mat = interior @ (0.25 * ends + 0.25 * (side_faces @ face_mat)) + border @ (0.5 * ends)

    # Vertex points.
    vertex_faces = sp.csr_matrix(
        (np.ones(4 * n_faces), (quads.ravel(), face_ids)), shape=(n_vertices, n_faces)
    )
    vertex_edges = sp.csr_matrix(
        (np.ones(2 * n_edges), (np.r_[edge_low, edge_high], np.r_[np.arange(n_edges), np.arange(n_edges)])),
        shape=(n_vertices, n_edges),
    )
    valence = np.asarray(vertex_edges.sum(1)).ravel()
    boundary_edges_at = np.asarray((vertex_edges @ sp.diags(boundary_edge.astype(float))).sum(1)).ravel()
    is_boundary = boundary_edges_at > 0
    n = np.maximum(valence, 3)
    inv_faces = sp.diags(1 / np.maximum(np.asarray(vertex_faces.sum(1)).ravel(), 1))
    mean_face = inv_faces @ (vertex_faces @ face_mat)
    mean_edge = sp.diags(1 / np.maximum(valence, 1)) @ (vertex_edges @ (0.5 * ends))
    smooth = (
        sp.diags(1 / n) @ mean_face
        + sp.diags(2 / n) @ mean_edge
        + sp.diags((n - 3) / n)
    )
    border_neighbours = vertex_edges @ sp.diags(boundary_edge.astype(float)) @ ends  # includes v itself
    border_rule = 0.125 * border_neighbours + sp.diags(0.75 - 0.125 * boundary_edges_at)
    vertex_mat = sp.diags((~is_boundary).astype(float)) @ smooth + sp.diags(is_boundary.astype(float)) @ border_rule

    S = sp.vstack([vertex_mat, edge_mat, face_mat]).tocsr()
    e = edge_of_side.reshape(n_faces, 4) + n_vertices
    f = np.arange(n_faces) + n_vertices + n_edges
    new_quads = np.empty((n_faces * 4, 4), np.int64)
    for i in range(4):
        new_quads[i::4] = np.column_stack([quads[:, i], e[:, i], f, e[:, (i + 3) % 4]])
    return S, new_quads


def subdivide_uvs(uv: np.ndarray) -> np.ndarray:
    """Bilinear split of per-corner UVs, faces by 4 by 2, matching catmull_clark's order."""
    n = len(uv)
    centre = uv.mean(1)
    out = np.empty((n * 4, 4, 2), uv.dtype)
    for i in range(4):
        following = 0.5 * (uv[:, i] + uv[:, (i + 1) % 4])
        previous = 0.5 * (uv[:, i] + uv[:, (i + 3) % 4])
        out[i::4] = np.stack([uv[:, i], following, centre, previous], axis=1)
    return out


def vertex_normals(vertices: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = vertices[tris[:, 0]], vertices[tris[:, 1]], vertices[tris[:, 2]]
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for i in range(3):
        for axis in range(3):
            normals[:, axis] += np.bincount(tris[:, i], face[:, axis], minlength=len(vertices))
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)


def triangles_from_quads(quads: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    tri_uv = np.concatenate([uv[:, [0, 1, 2]], uv[:, [0, 2, 3]]])
    return tris, tri_uv


@dataclass
class Part:
    """A drawable mesh: triangles, per-corner UVs and an RGBA texture."""

    name: str
    tris: np.ndarray
    tri_uv: np.ndarray
    texture: np.ndarray
    clothes: Optional[Clothes] = None
    bias: float = 0.0  # depth bias, for thin layers lying on skin
    alpha_test: bool = False
    backface: str = "cull"  # 'cull' or 'interior'


def proxy_part(name: str, folder: Path, texture_path: Optional[Path] = None, **kwargs) -> Part:
    mhclo = next(folder.glob("*.mhclo"))
    clothes = read_mhclo(mhclo)
    obj = read_obj(next(p for p in sorted(folder.glob("*.obj"))))
    tris, tri_uv = [], []
    for ids, uv_ids, _ in obj.faces:
        for i in range(1, len(ids) - 1):
            tris.append((ids[0], ids[i], ids[i + 1]))
            tri_uv.append((obj.uvs[uv_ids[0]], obj.uvs[uv_ids[i]], obj.uvs[uv_ids[i + 1]]))
    if texture_path is None:
        texture_path = folder / read_diffuse_name(folder / clothes.material)
    return Part(
        name,
        np.array(tris, np.int64),
        np.array(tri_uv, np.float32),
        load_texture([texture_path]),
        clothes=clothes,
        **kwargs,
    )


def fit_clothes(clothes: Clothes, deformed: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Fit a proxy to the mesh.

    MakeHuman scales the proxy's local offsets by mesh spans. Those spans are taken from
    the expression-free mesh here, because a span across an open jaw would stretch teeth.
    """
    if clothes.scale_data:
        scale = np.array([abs(rest[a, i] - rest[b, i]) / den for i, (a, b, den) in enumerate(clothes.scale_data)])
    else:
        scale = np.ones(3)
    refs, weights = clothes.refs, clothes.weights
    return (
        deformed[refs[:, 0]] * weights[:, 0, None]
        + deformed[refs[:, 1]] * weights[:, 1, None]
        + deformed[refs[:, 2]] * weights[:, 2, None]
        + clothes.offsets * scale
    )


# Base-mesh vertices used for framing and measurement.
CHIN_VERTEX = 5225
UPPER_LIP_INNER = 520
LOWER_LIP_INNER = 523
LIP_CORNER_LEFT = 7188
LIP_CORNER_RIGHT = 428

# Expected face regions in base-mesh coordinates: half-width, then y range.
REGIONS = {
    "brow": (1.05, 7.25, 7.95),
    "eye": (1.05, 6.98, 7.55),
    "nose": (0.48, 6.72, 7.45),
    "cheek": (1.15, 6.5, 7.35),
    "mouth": (0.85, 6.15, 6.95),
    "chin": (0.75, 5.75, 6.6),
}
EXPECTED_REGION = {
    "AU01": ("brow",), "AU02": ("brow",), "AU04": ("brow", "nose"), "AU05": ("eye",), "AU06": ("eye", "cheek"),
    "AU07": ("eye",), "AU09": ("nose", "cheek"), "AU10": ("mouth", "nose"), "AU11": ("mouth", "cheek"),
    "AU12": ("mouth", "cheek"), "AU13": ("mouth", "cheek"), "AU14": ("mouth", "cheek"), "AU15": ("mouth", "chin"),
    "AU16": ("mouth", "chin"), "AU17": ("mouth", "chin"), "AU20": ("mouth", "cheek"), "AU22": ("mouth",),
    "AU23": ("mouth",), "AU24": ("mouth",), "AU25": ("mouth",), "AU26": ("mouth", "chin"),
    "AU27": ("mouth", "chin"), "AU38": ("nose",), "AU39": ("nose",), "AU43": ("eye",),
}


# --------------------------------------------------------------------------
# Avatar assembly
# --------------------------------------------------------------------------


@dataclass
class Avatar:
    identity: Identity
    gain: float
    rest: np.ndarray  # shaped mesh, no expression
    keep: np.ndarray  # head vertices, old ids
    subdivide: sp.csr_matrix
    skin: Part
    parts: List[Part]
    centre: Tuple[float, float, float]
    pixels_per_unit: float
    extras: Dict[str, np.ndarray] = field(default_factory=dict)


@lru_cache(maxsize=1)
def head_topology() -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray]:
    """Head skin mesh after one subdivision: old vertex ids, S, triangles, corner UVs."""
    obj = base_mesh()
    y = obj.vertices[:, 1]
    quads, uvs = [], []
    for ids, uv_ids, group in obj.faces:
        if group == "body" and len(ids) == 4 and (y[ids] > NECK_CUT_Y).all():
            quads.append(ids)
            uvs.append(obj.uvs[uv_ids])
    quads = np.array(quads)
    keep, local = np.unique(quads, return_inverse=True)
    local = local.reshape(quads.shape)
    S, new_quads = catmull_clark(local, len(keep))
    tris, tri_uv = triangles_from_quads(new_quads, subdivide_uvs(np.array(uvs, np.float32)))
    return keep, S.astype(np.float32), tris, tri_uv.astype(np.float32)


@lru_cache(maxsize=32)
def build_avatar(name: str, seed: int) -> Avatar:
    if name not in IDENTITIES:
        raise ValueError(f"unknown identity {name!r}; choose from {sorted(IDENTITIES)}")
    identity, gain = jitter_identity(IDENTITIES[name], seed)
    obj = base_mesh()
    rest = obj.vertices.copy()
    targets = mh_data() / "targets"
    for rel, weight in macro_target_weights(identity).items():
        ids, offsets = read_target(str(targets / f"{rel}.target"))
        rest[ids] += weight * offsets

    # FACSHuman's close_lips target seals the small gap the macro blends leave.
    close = dense_target(assets_root() / "FACSHuman" / "custom" / "close_lips.target")
    closing = float(close[LOWER_LIP_INNER, 1] - close[UPPER_LIP_INNER, 1])
    rest += close * float(np.clip(lip_aperture(rest) / closing, 1.0, 1.5))

    keep, S, tris, tri_uv = head_topology()
    assets = system_assets()
    skin_paths = []
    for skin in identity.skins:
        folder = assets / "skins" / skin
        skin_paths.append(folder / read_diffuse_name(folder / f"{skin}.mhmat"))
    skin = Part("skin", tris, tri_uv, load_texture(skin_paths), backface="interior")

    brow = assets / "eyebrows" / identity.eyebrow
    parts = [
        proxy_part("teeth", assets / "teeth" / "teeth_base"),
        proxy_part("tongue", assets / "tongue" / "tongue01"),
        proxy_part("eyes", assets / "eyes" / "high-poly",
                   texture_path=assets / "eyes" / "materials" / f"{identity.eye_colour}_eye.png", alpha_test=True),
        proxy_part("eyebrows", brow, alpha_test=True, bias=0.012),
        proxy_part("eyelashes", assets / "eyelashes" / "eyelashes01", alpha_test=True, bias=0.004),
    ]
    head_points = rest[keep]
    top = float(head_points[:, 1].max())
    chin = float(rest[CHIN_VERTEX, 1])
    # Fix the frame on the expression-free head so AUs can never move the camera.
    centre = (0.0, (top + chin) / 2 - 0.02 * (top - chin), float(head_points[:, 2].max()))
    ppu = FRAME_MARGIN * RESOLUTION * SUPERSAMPLE / (top - chin)
    return Avatar(identity, gain, rest, keep, S, skin, parts, centre, ppu)


def project(points: np.ndarray, avatar: Avatar, size: int) -> np.ndarray:
    """World to pixel x, pixel y and depth from the camera, for a size-square image."""
    cx, cy, cz = avatar.centre
    ppu = avatar.pixels_per_unit * size / (RESOLUTION * SUPERSAMPLE)
    depth = CAMERA_DISTANCE + cz - points[:, 2]
    factor = ppu * CAMERA_DISTANCE / depth
    x = size / 2 + (points[:, 0] - cx) * factor
    y = 0.46 * size - (points[:, 1] - cy) * factor
    return np.column_stack([x, y, depth])


# --------------------------------------------------------------------------
# Rasteriser
# --------------------------------------------------------------------------


def sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup, RGBA floats in 0 to 1. Image rows run opposite to OBJ's v."""
    height, width = texture.shape[:2]
    x = np.mod(uv[:, 0], 1.0) * (width - 1)
    y = (1 - np.mod(uv[:, 1], 1.0)) * (height - 1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx
    bottom = texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx
    return ((top * (1 - fy) + bottom * fy) / 255).astype(np.float32)


def fragments(points: np.ndarray, tris: np.ndarray, size: int):
    """All pixel centres inside each triangle, as flat arrays.

    Returns triangle index, pixel x, pixel y, barycentric weights (n by 3), and each
    triangle's signed area. Triangles are expanded together, so no Python loop runs
    per triangle. Depth and attributes are interpolated affinely: triangles are small
    and the camera is 1 m away, so the error is below a pixel.
    """
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    corners_x = np.stack([a[:, 0], b[:, 0], c[:, 0]], axis=1)
    corners_y = np.stack([a[:, 1], b[:, 1], c[:, 1]], axis=1)
    x_low = np.maximum(np.floor(corners_x.min(1)).astype(np.int64), 0)
    x_high = np.minimum(np.ceil(corners_x.max(1)).astype(np.int64), size - 1)
    y_low = np.maximum(np.floor(corners_y.min(1)).astype(np.int64), 0)
    y_high = np.minimum(np.ceil(corners_y.max(1)).astype(np.int64), size - 1)
    width, height = x_high - x_low + 1, y_high - y_low + 1
    live = np.flatnonzero((width > 0) & (height > 0) & (np.abs(area) > 1e-9))
    counts = width[live] * height[live]
    tri_index = np.repeat(live, counts)
    start = np.cumsum(counts) - counts
    local = np.arange(counts.sum()) - np.repeat(start, counts)
    rows = width[tri_index]
    px = x_low[tri_index] + local % rows
    py = y_low[tri_index] + local // rows
    sx, sy = px + 0.5, py + 0.5
    ax, ay, bx, by, cx, cy = (
        a[tri_index, 0], a[tri_index, 1], b[tri_index, 0], b[tri_index, 1], c[tri_index, 0], c[tri_index, 1],
    )
    inv = 1.0 / area[tri_index]
    w0 = ((bx - sx) * (cy - sy) - (by - sy) * (cx - sx)) * inv
    w1 = ((cx - sx) * (ay - sy) - (cy - sy) * (ax - sx)) * inv
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    weights = np.column_stack([w0[inside], w1[inside], w2[inside]]).astype(np.float32)
    return tri_index[inside], px[inside], py[inside], weights, area


def draw_part(part: Part, points: np.ndarray, normals: np.ndarray, frame: np.ndarray, zbuffer: np.ndarray, gain: float) -> None:
    size = frame.shape[0]
    tris = part.tris
    tri, px, py, bary, area = fragments(points, tris, size)
    # The y flip in projection turns world counter-clockwise into negative image area.
    front = area[tri] < 0
    if part.backface == "cull":
        tri, px, py, bary, front = tri[front], px[front], py[front], bary[front], front[front]
    uv = (part.tri_uv[tri] * bary[:, :, None]).sum(1)
    depth = (points[tris[tri], 2] * bary).sum(1) - part.bias
    texels = sample_texture(part.texture, uv)
    if part.alpha_test:
        solid = texels[:, 3] >= 0.5
        tri, px, py, bary, front, depth, texels = (
            tri[solid], px[solid], py[solid], bary[solid], front[solid], depth[solid], texels[solid],
        )
    pixel = py * size + px
    best = zbuffer.ravel()
    candidate = np.full(size * size, np.inf, np.float32)
    np.minimum.at(candidate, pixel, depth.astype(np.float32))
    winners = (depth.astype(np.float32) == candidate[pixel]) & (depth.astype(np.float32) < best[pixel])
    tri, px, py, bary, front, depth, texels = (
        tri[winners], px[winners], py[winners], bary[winners], front[winners], depth[winners], texels[winners],
    )
    pixel = pixel[winners]
    normal = (normals[tris[tri]] * bary[:, :, None]).sum(1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    light = AMBIENT + DIFFUSE * np.clip(normal[:, 2], 0, 1)
    colour = texels[:, :3] * light[:, None] * gain
    colour[~front] = MOUTH_INTERIOR
    best[pixel] = depth.astype(np.float32)
    frame.reshape(-1, 3)[pixel] = colour


def render_mesh(deformed: np.ndarray, avatar: Avatar) -> np.ndarray:
    size = RESOLUTION * SUPERSAMPLE
    frame = np.zeros((size, size, 3), np.float32)
    zbuffer = np.full((size, size), np.inf, np.float32)
    skin_vertices = np.asarray(avatar.subdivide @ deformed[avatar.keep], np.float32)
    points = project(skin_vertices, avatar, size)
    draw_part(avatar.skin, points, vertex_normals(skin_vertices, avatar.skin.tris), frame, zbuffer, avatar.gain)
    for part in avatar.parts:
        fitted = fit_clothes(part.clothes, deformed, avatar.rest)
        draw_part(part, project(fitted, avatar, size), vertex_normals(fitted, part.tris), frame, zbuffer, avatar.gain)
    small = frame.reshape(RESOLUTION, SUPERSAMPLE, RESOLUTION, SUPERSAMPLE, 3).mean((1, 3))
    return np.rint(np.clip(small, 0, 1) * 255).astype(np.uint8)


def deformed_mesh(au: Mapping[str, float], avatar: Avatar) -> np.ndarray:
    return avatar.rest + au_offsets(au)


def render_face(au: Mapping[str, float], identity: str = "neutral", seed: int = 0, out: Optional[Path] = None) -> np.ndarray:
    """Render an AU configuration, returning RGB uint8 of shape 768 by 768 by 3.

    Amplitudes must lie in 0 to 1, and unknown AU names raise an error. Seed 0 gives
    the canonical identity; other seeds add small seeded identity variation. The same
    arguments give the same pixels on the same platform.
    """
    avatar = build_avatar(identity, int(seed))
    image = render_mesh(deformed_mesh(au, avatar), avatar)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, "RGB").save(out, format="PNG", optimize=False, compress_level=6)
    return image


# --------------------------------------------------------------------------
# Validity check, with no vision-language model
# --------------------------------------------------------------------------


def region_masks(avatar: Avatar) -> Dict[str, np.ndarray]:
    """Boolean masks of the anatomical regions, drawn from the base-mesh boxes.

    Regions come from the expression-free head and fixed anatomy only, never from an
    AU's own offsets, so a target cannot pass by defining its own region.
    """
    base = base_mesh().vertices
    body = np.unique(np.concatenate([ids for ids, _, group in base_mesh().faces if group == "body"]))
    masks = {}
    for name, (half_width, y_low, y_high) in REGIONS.items():
        inside = body[
            (np.abs(base[body, 0]) <= half_width)
            & (base[body, 1] >= y_low)
            & (base[body, 1] <= y_high)
            & (base[body, 2] > 0.2)
        ]
        xy = project(avatar.rest[inside], avatar, RESOLUTION)[:, :2]
        canvas = Image.new("L", (RESOLUTION, RESOLUTION), 0)
        ImageDraw.Draw(canvas).polygon(_convex_hull(xy), fill=1)
        masks[name] = np.asarray(canvas, dtype=bool)
    return masks


def _convex_hull(points: np.ndarray) -> List[Tuple[float, float]]:
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    return [tuple(map(float, points[i])) for i in hull.vertices]


def expected_mask(column: str, masks: Mapping[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros((RESOLUTION, RESOLUTION), bool)
    for region in EXPECTED_REGION[column[:4]]:
        mask |= masks[region]
    return mask


def lip_aperture(deformed: np.ndarray) -> float:
    """Inner lip gap on the midline, in the mesh's units (decimetres)."""
    return float(deformed[UPPER_LIP_INNER, 1] - deformed[LOWER_LIP_INNER, 1])


def mouth_width(deformed: np.ndarray) -> float:
    return float(deformed[LIP_CORNER_LEFT, 0] - deformed[LIP_CORNER_RIGHT, 0])


def validate_au(column: str, identity: str, neutral: np.ndarray, masks, seed: int = 0,
                amplitudes: Sequence[float] = (0.25, 0.5, 1.0)) -> Dict[str, object]:
    avatar = build_avatar(identity, seed)
    expected = expected_mask(column, masks)
    outside = ~expected
    side = column[-1] if column[-1] in "LR" else None
    mesh_mean, mesh_max, pixel_in, pixel_out, share, side_share, images = [], [], [], [], [], [], {}
    for amplitude in amplitudes:
        config = {column: amplitude}
        moved = np.linalg.norm(deformed_mesh(config, avatar) - avatar.rest, axis=1)
        mesh_mean.append(float(moved.mean()))
        mesh_max.append(float(moved.max()))
        image = render_face(config, identity, seed)
        images[amplitude] = image
        change = np.abs(image.astype(np.float32) - neutral.astype(np.float32)).mean(2)
        pixel_in.append(float(change[expected].mean()))
        pixel_out.append(float(change[outside].mean()))
        total = float(change.sum())
        share.append(float(change[expected].sum() / total) if total else 0.0)
        if side:
            half = RESOLUTION // 2
            # The avatar's left is on the image right.
            near = change[:, half:] if side == "L" else change[:, :half]
            side_share.append(float(near.sum() / total) if total else 0.0)
    mesh_ok = bool(np.all(np.diff(mesh_max) > 0) and np.all(np.diff(mesh_mean) > 0))
    pixel_ok = bool(np.all(np.diff(pixel_in) > 0) and pixel_in[0] > 0)
    local_ok = bool(share[-1] >= 0.6)
    side_ok = bool(side_share[-1] >= 0.6) if side else None
    return {
        "column": column,
        "identity": identity,
        "amplitudes": list(amplitudes),
        "mesh_mean_displacement": mesh_mean,
        "mesh_max_displacement": mesh_max,
        "pixel_change_in_region": pixel_in,
        "pixel_change_outside": pixel_out,
        "share_of_change_in_region": share,
        "share_of_change_on_expected_side": side_share,
        "mesh_monotonic": mesh_ok,
        "pixel_monotonic": pixel_ok,
        "localised": local_ok,
        "lateralised": side_ok,
        "passed": bool(mesh_ok and pixel_ok and local_ok and side_ok is not False),
    }, images


def validate(identities: Sequence[str], columns: Sequence[str] = AU_COLS, seed: int = 0,
             detector=None) -> Dict[str, object]:
    rows, detector_rows = [], []
    for identity in identities:
        neutral = render_face({}, identity, seed)
        masks = region_masks(build_avatar(identity, seed))
        neutral_scores = detector(neutral) if detector else None
        for column in columns:
            row, images = validate_au(column, identity, neutral, masks, seed)
            rows.append(row)
            if detector:
                detector_rows.append(detector_row(column, identity, images, neutral_scores, detector))
    failures = [(r["identity"], r["column"]) for r in rows if not r["passed"]]
    return {
        "n_checks": len(rows),
        "n_failed": len(failures),
        "failures": failures,
        "rows": rows,
        "detector": detector_rows,
    }


# MediaPipe blendshapes that should answer to each AU family. MediaPipe gives ARKit-style
# blendshapes, not FACS AUs, so this is a coarse check and not an AU validation.
DETECTOR_MAP = {
    "AU01": ("browInnerUp",), "AU02": ("browOuterUpLeft", "browOuterUpRight"),
    "AU04": ("browDownLeft", "browDownRight"), "AU05": ("eyeWideLeft", "eyeWideRight"),
    "AU06": ("cheekSquintLeft", "cheekSquintRight"), "AU07": ("eyeSquintLeft", "eyeSquintRight"),
    "AU09": ("noseSneerLeft", "noseSneerRight"), "AU10": ("mouthUpperUpLeft", "mouthUpperUpRight"),
    "AU12": ("mouthSmileLeft", "mouthSmileRight"), "AU14": ("mouthDimpleLeft", "mouthDimpleRight"),
    "AU15": ("mouthFrownLeft", "mouthFrownRight"), "AU16": ("mouthLowerDownLeft", "mouthLowerDownRight"),
    "AU17": ("mouthShrugLower",), "AU20": ("mouthStretchLeft", "mouthStretchRight"),
    "AU22": ("mouthFunnel",), "AU23": ("mouthPressLeft", "mouthPressRight"),
    "AU24": ("mouthPressLeft", "mouthPressRight"), "AU25": ("mouthUpperUpLeft", "mouthLowerDownLeft"),
    "AU26": ("jawOpen",), "AU27": ("jawOpen",), "AU43": ("eyeBlinkLeft", "eyeBlinkRight"),
}


def mediapipe_detector(model_path: Path):
    """Return a function from RGB image to {blendshape: score}, or raise ImportError."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
        output_face_blendshapes=True,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(image: np.ndarray) -> Optional[Dict[str, float]]:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
        if not result.face_blendshapes:
            return None
        return {b.category_name: float(b.score) for b in result.face_blendshapes[0]}

    return detect


def detector_row(column, identity, images, neutral_scores, detect) -> Dict[str, object]:
    names = DETECTOR_MAP.get(column[:4])
    scores = {a: detect(img) for a, img in images.items()}
    row: Dict[str, object] = {"column": column, "identity": identity, "blendshapes": list(names or [])}
    if names is None:
        row["status"] = "no mapped blendshape"
        return row
    if neutral_scores is None or any(s is None for s in scores.values()):
        row["status"] = "face not found"
        return row
    base = float(np.mean([neutral_scores[n] for n in names]))
    series = [float(np.mean([scores[a][n] for n in names])) - base for a in sorted(scores)]
    row["score_change_by_amplitude"] = series
    row["responds"] = bool(series[-1] > 0.02)
    row["monotonic"] = bool(np.all(np.diff(series) >= -0.01))
    row["status"] = "ok"
    return row


def libreface_hook(image_dir: Path, out_csv: Path) -> None:
    """HOOK FOR THE CLUSTER: run LibreFace over rendered PNGs. Not implemented here.

    Expected use on the cluster: run LibreFace per image for AU intensities, write one
    row per image, then compare each AU's estimated intensity with the rendered amplitude
    for the renders from validate(). Package libreface 0.2.0 reports intensity for AUs
    1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26 and presence for 1, 2, 4, 6, 7, 10, 12, 14,
    15, 17, 23, 24, so it cannot confirm AU22 or AU27. It pins torch 2.0.0, dlib and
    mediapipe 0.10.5, and its weights come from Google Drive, so use a separate env.
    Its licence is a USC research licence.
    """
    raise NotImplementedError("LibreFace hook: run on the cluster, see this docstring")


# --------------------------------------------------------------------------
# Snoek configurations
# --------------------------------------------------------------------------


def snoek_configs(zip_path: Path, datasets: Sequence[str] = ("basic_EA", "conversational_EA")) -> List[Dict[str, float]]:
    """Distinct AU configurations in the named datasets, each as {column: amplitude}."""
    sys.path.insert(0, str(ROOT / "research" / "scripts"))
    import human_au_effects as hae

    frames, _ = hae.load_zip(zip_path)
    seen: Dict[Tuple[float, ...], None] = {}
    for key in datasets:
        frame = frames[key][0]
        for row in frame.drop_duplicates("cfg")[AU_COLS].to_numpy(float):
            seen[tuple(row)] = None
    return [{c: float(v) for c, v in zip(AU_COLS, row) if v > 0} for row in seen]


def shared_configs(zip_path: Path, a: str = "basic_EA", b: str = "conversational_EA") -> List[Dict[str, float]]:
    """Configurations present in both datasets, by label, as in the shared counts in the JSON."""
    sys.path.insert(0, str(ROOT / "research" / "scripts"))
    import human_au_effects as hae

    frames, _ = hae.load_zip(zip_path)
    first = frames[a][0].drop_duplicates("cfg").set_index("cfg")
    second = set(frames[b][0]["cfg"])
    rows = first[first.index.isin(second)][AU_COLS].to_numpy(float)
    return [{c: float(v) for c, v in zip(AU_COLS, row) if v > 0} for row in rows]


def timing(configs: Sequence[Mapping[str, float]], n: int, identity: str, out_dir: Path, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(configs), size=min(n, len(configs)), replace=False)
    build_start = time.perf_counter()
    build_avatar(identity, seed)
    build = time.perf_counter() - build_start
    start = time.perf_counter()
    sizes = []
    for k, i in enumerate(picks):
        path = out_dir / f"timing_{k:03d}.png"
        render_face(configs[int(i)], identity, seed, path)
        sizes.append(path.stat().st_size)
        path.unlink()
    per_image = (time.perf_counter() - start) / len(picks)
    return {
        "n_images": len(picks),
        "setup_seconds_per_process": round(build, 2),
        "seconds_per_image": round(per_image, 3),
        "mean_png_kb": round(float(np.mean(sizes)) / 1024, 1),
        "hours_for_2300_one_core": round(per_image * 2300 / 3600, 2),
        "gb_for_2300": round(float(np.mean(sizes)) * 2300 / 1024**3, 3),
    }


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_au(text: str) -> Dict[str, float]:
    out = {}
    for item in filter(None, text.split(",")):
        key, value = item.split("=")
        out[key.strip()] = float(value)
    return out


def coverage_rows() -> Iterable[Tuple[str, str, str]]:
    for column in AU_COLS:
        status, note = COVERAGE[column]
        yield column, status, note


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    one = commands.add_parser("render", help="render one configuration")
    one.add_argument("--au", default="", help="for example AU12L=1,AU12R=1")
    one.add_argument("--identity", default="neutral", choices=sorted(IDENTITIES))
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--out", type=Path, required=True)

    commands.add_parser("coverage", help="print coverage of the 33 Snoek columns")

    check = commands.add_parser("validate", help="single-AU checks, with no vision-language model")
    check.add_argument("--identity", nargs="*", default=sorted(IDENTITIES))
    check.add_argument("--out-json", type=Path)
    check.add_argument("--face-landmarker", type=Path, help="MediaPipe face_landmarker.task for the detector check")

    clock = commands.add_parser("timing", help="seconds per image and storage from real configurations")
    clock.add_argument("--zip", type=Path, required=True)
    clock.add_argument("--n", type=int, default=30)
    clock.add_argument("--identity", default="neutral", choices=sorted(IDENTITIES))
    clock.add_argument("--work-dir", type=Path, default=Path("."))

    batch = commands.add_parser("batch", help="render configurations shared by two datasets, with a manifest")
    batch.add_argument("--zip", type=Path, required=True)
    batch.add_argument("--out-dir", type=Path, required=True)
    batch.add_argument("--identity", default="neutral", choices=sorted(IDENTITIES))
    batch.add_argument("--seed", type=int, default=0)
    batch.add_argument("--shard", default="0/1", help="i/n, for an array job")

    args = parser.parse_args(argv)
    if args.command == "render":
        render_face(parse_au(args.au), args.identity, args.seed, args.out)
    elif args.command == "coverage":
        for row in coverage_rows():
            print("\t".join(row))
    elif args.command == "validate":
        detector = mediapipe_detector(args.face_landmarker) if args.face_landmarker else None
        result = validate(args.identity, detector=detector)
        print(json.dumps({k: result[k] for k in ("n_checks", "n_failed", "failures")}))
        if args.out_json:
            args.out_json.write_text(json.dumps(result, indent=1))
    elif args.command == "timing":
        print(json.dumps(timing(shared_configs(args.zip), args.n, args.identity, args.work_dir)))
    elif args.command == "batch":
        index, total = (int(x) for x in args.shard.split("/"))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        configs = shared_configs(args.zip)
        with open(args.out_dir / f"manifest_{args.identity}_{index}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["config_index", "identity", "seed", "file", "au_json"])
            for k in range(index, len(configs), total):
                name = f"{args.identity}_{args.seed}_{k:05d}.png"
                render_face(configs[k], args.identity, args.seed, args.out_dir / name)
                writer.writerow([k, args.identity, args.seed, name, json.dumps(configs[k], sort_keys=True)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

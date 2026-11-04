"""
Deterministic stimulus interventions for RQ1 and RQ2, applied in memory at load time.

Nothing here writes a frame, crop or edited clip. Every function takes arrays and returns
arrays. No generator is used anywhere: edits are crops, masks, filters, reorderings and
one local image warp.

Conventions
- A frame is an RGB uint8 array of shape (height, width, 3).
- Landmarks are a (478, 2) float array of pixel (x, y) from MediaPipe FaceMesh with
  refine_landmarks=True and static_image_mode=True. Each frame is detected on its own,
  with no tracking, so a frame's landmarks do not depend on which frames came before.

How regions are localised
- Mouth: the outer lip ring. Eyes: the two lid rings. Brows: the two brow outlines.
- Each region is the convex hull of its landmark ring on that frame, dilated by a
  fraction of the eye span (distance between outer eye corners), then feathered with a
  Gaussian. Sizes therefore follow the face and not the image.
- Donor images are registered to the target frame by a closed-form similarity transform
  on points that move little with expression (eye corners, nose bridge, face edge). The
  fit is exact arithmetic, with no random sampling, and its residual is logged.
- A frame with no detected face raises EditFailure. The caller logs the failure and the
  item stays in the output with its reason. There is no fallback to another frame's
  landmarks and no item is dropped silently.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SEED = 42
N_SENT_FRAMES = 8  # composite grids hold at most 8 cells, see scripts/multi_frame.py
N_LANDMARKS = 478

# MediaPipe FaceMesh indices. "Left" is the subject's left.
MOUTH_RING = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LEFT_EYE_RING = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
RIGHT_EYE_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_BROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
RIGHT_BROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
OUTER_EYE_CORNERS = (33, 263)
FACE_EDGES = (234, 454)
# Points that move little with expression, used to register one face to another.
STABLE_POINTS = (33, 133, 362, 263, 168, 6, 197, 195, 5, 234, 454, 10)
# (outer corner, inner corner, upper lid midpoint, lower lid midpoint) per eye.
EYE_POINTS = {"right": (33, 133, 159, 145), "left": (263, 362, 386, 374)}
EXPRESSIVE_POINTS = MOUTH_RING + LEFT_BROW + RIGHT_BROW + LEFT_EYE_RING + RIGHT_EYE_RING

REGIONS: Dict[str, Tuple[Tuple[int, ...], ...]] = {
    "mouth": (MOUTH_RING,),
    "eyes": (LEFT_EYE_RING, RIGHT_EYE_RING),
    "brows": (LEFT_BROW, RIGHT_BROW),
    "eyesmouth": (MOUTH_RING, LEFT_EYE_RING, RIGHT_EYE_RING),
}
# Dilation and feather as fractions of eye span. Fixed before any model saw an edit.
REGION_DILATE = {"mouth": 0.12, "eyes": 0.10, "brows": 0.08, "eyesmouth": 0.12}
FEATHER_FRACTION = 0.03
BLUR_FILL_SIGMA_FRACTION = 0.35
MAX_ALIGN_RESIDUAL = 0.05  # of eye span; above this the row is flagged, not removed
MIN_EYE_OPENING = 0.08  # lid gap over eye width; below this the eye counts as closed

FACE_BOX_MARGIN = 1.5
FACE_SIZE = 768  # fixed side of a face crop once it is sent; 16 by 16 tokens on Gemma 4
LOW_SF_CPF = 8.0  # conventional low-pass cutoff, cycles per face width
HIGH_SF_CPF = 32.0  # conventional high-pass cutoff, cycles per face width
APERTURE_SCALES = {"wide": 1.25, "narrow": 0.75}

FILL_TYPES = ("neutral", "first", "blur")
SPATIAL_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "original": {"kind": "none"},
    "sham_self_fill": {"kind": "mask", "region": "mouth", "fill": "self"},
    **{
        f"mask_{region}_{fill}": {"kind": "mask", "region": region, "fill": fill}
        for region in REGIONS
        for fill in FILL_TYPES
    },
    "eye_crop": {"kind": "eye_crop"},
    "eye_window": {"kind": "eye_window"},
    "sf_low": {"kind": "sf", "band": "low", "cpf": LOW_SF_CPF},
    "sf_high": {"kind": "sf", "band": "high", "cpf": HIGH_SF_CPF},
    "aperture_wide": {"kind": "aperture", "scale": APERTURE_SCALES["wide"]},
    "aperture_narrow": {"kind": "aperture", "scale": APERTURE_SCALES["narrow"]},
}
TEMPORAL_CONDITIONS = (
    "original",
    "shuffle_a",
    "shuffle_b",
    "reverse",
    "trunc_first_third",
    "trunc_last_third",
    "static_apex",
)


class EditFailure(RuntimeError):
    """An edit cannot be applied to this frame or clip. Log it; do not drop the item."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def array_sha256(array: np.ndarray) -> str:
    """Hash for logging that a frame or tensor is the same, which stores no pixels."""
    head = f"{array.shape}|{array.dtype}|".encode("utf-8")
    return hashlib.sha256(head + np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Frame selection and temporal conditions
# ---------------------------------------------------------------------------


def uniform_indices(start: int, stop: int, k: int = N_SENT_FRAMES) -> List[int]:
    """k positions spread evenly over frames start..stop-1, with rounding as in frame_sampling."""
    if stop <= start or k <= 0:
        raise ValueError(f"Empty frame range {start}..{stop} for k={k}")
    if k == 1:
        return [start]
    return [start + int(round(i * (stop - 1 - start) / (k - 1))) for i in range(k)]


def derangement(k: int, key: Any) -> List[int]:
    """Permutation of range(k) with no fixed point, fixed by key."""
    if k < 2:
        raise ValueError("A derangement needs at least two frames")
    rng = np.random.default_rng(stable_seed(SEED, "derangement", key) % 2**32)
    while True:
        perm = rng.permutation(k).tolist()
        if all(i != p for i, p in enumerate(perm)):
            return [int(p) for p in perm]


def temporal_plan(n_frames: int, trial_id: str, apex: Optional[int], k: int = N_SENT_FRAMES) -> Dict[str, List[int]]:
    """Frame indices in sent order, per temporal condition.

    Every condition sends exactly k frames, so grid layout and token count never change
    with the condition. Truncation resamples k frames inside the kept third. It therefore
    raises local frame density as well as removing content, which is why it is reported
    as a separate contrast. static_apex is absent when no apex exists.
    """
    base = uniform_indices(0, n_frames, k)
    third = max(1, n_frames // 3)
    plan = {
        "original": base,
        "shuffle_a": [base[p] for p in derangement(k, (trial_id, "a"))],
        "shuffle_b": [base[p] for p in derangement(k, (trial_id, "b"))],
        "reverse": base[::-1],
        "trunc_first_third": uniform_indices(0, third, k),
        "trunc_last_third": uniform_indices(n_frames - third, n_frames, k),
    }
    if apex is not None:
        plan["static_apex"] = [int(apex)] * k
    return plan


def read_frames(path: Path, indices: Sequence[int]) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Decode the requested frames into memory by reading forward from frame 0.

    Sequential decoding is used because seeking in H.264 can land on a neighbouring frame,
    and a stable index is the point of this function.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise EditFailure("video_unreadable", str(path))
    meta = {
        "n_frames_header": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    wanted = sorted(set(int(i) for i in indices))
    frames: Dict[int, np.ndarray] = {}
    position = 0
    for index in wanted:
        while position < index:
            if not capture.grab():
                break
            position += 1
        ok, bgr = capture.read()
        if not ok or bgr is None:
            break
        position += 1
        frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    capture.release()
    missing = [i for i in wanted if i not in frames]
    if missing:
        raise EditFailure("frames_unavailable", f"{path.name}: {missing[:5]}")
    return frames, meta


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def eye_span(landmarks: np.ndarray) -> float:
    a, b = OUTER_EYE_CORNERS
    return float(np.linalg.norm(landmarks[a] - landmarks[b]))


def face_width(landmarks: np.ndarray) -> float:
    a, b = FACE_EDGES
    return float(np.linalg.norm(landmarks[a] - landmarks[b]))


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares scale, rotation and shift from source to target, as a 2x3 matrix (Umeyama)."""
    src_mean, dst_mean = source.mean(0), target.mean(0)
    src, dst = source - src_mean, target - dst_mean
    u, singular, vt = np.linalg.svd(dst.T @ src / len(src))
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    variance = float((src**2).sum() / len(src))
    if variance <= 0:
        raise EditFailure("degenerate_alignment")
    scale = float((singular * np.diag(sign)).sum() / variance)
    shift = dst_mean - scale * rotation @ src_mean
    return np.hstack([scale * rotation, shift[:, None]])


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def align_to(donor: np.ndarray, donor_lm: np.ndarray, target_lm: np.ndarray, size_hw: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Warp a donor image onto a target face. Returns the image and residual over eye span."""
    stable = list(STABLE_POINTS)
    matrix = similarity_transform(donor_lm[stable], target_lm[stable])
    residual = apply_transform(matrix, donor_lm[stable]) - target_lm[stable]
    rms = float(np.sqrt((residual**2).sum(1).mean()))
    height, width = size_hw
    warped = cv2.warpAffine(donor, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, rms / max(eye_span(target_lm), 1e-6)


def region_mask(size_hw: Tuple[int, int], landmarks: np.ndarray, region: str) -> np.ndarray:
    """Soft mask in 0..1 over a landmark region, sized by the face."""
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region}")
    height, width = size_hw
    span = eye_span(landmarks)
    hard = np.zeros((height, width), dtype=np.uint8)
    for ring in REGIONS[region]:
        hull = cv2.convexHull(np.round(landmarks[list(ring)]).astype(np.int32))
        cv2.fillConvexPoly(hard, hull, 255)
    radius = max(1, int(round(REGION_DILATE[region] * span)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    hard = cv2.dilate(hard, kernel)
    sigma = max(0.5, FEATHER_FRACTION * span)
    return cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (0, 0), sigma)


def blend(frame: np.ndarray, fill: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight = alpha[..., None]
    out = frame.astype(np.float32) * (1.0 - weight) + fill.astype(np.float32) * weight
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def square_box(landmarks: Sequence[np.ndarray], size_hw: Tuple[int, int], margin: float = FACE_BOX_MARGIN) -> Tuple[int, int, int, int]:
    """One square crop for the whole clip, so head motion stays visible inside it."""
    if not landmarks:
        raise EditFailure("no_landmarks_for_box")
    height, width = size_hw
    lows = np.array([lm.min(0) for lm in landmarks])
    highs = np.array([lm.max(0) for lm in landmarks])
    centre = np.median((lows + highs) / 2.0, axis=0)
    side = int(round(margin * float((highs - lows).max())))
    side = min(side, height, width)
    x0 = int(np.clip(round(centre[0] - side / 2), 0, width - side))
    y0 = int(np.clip(round(centre[1] - side / 2), 0, height - side))
    return x0, y0, x0 + side, y0 + side


def eye_band_box(landmarks: Sequence[np.ndarray], size_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """RMET-style band: brow top to below the lower lid, across both eyes, fixed for the clip."""
    if not landmarks:
        raise EditFailure("no_landmarks_for_box")
    height, width = size_hw
    points = list(LEFT_EYE_RING + RIGHT_EYE_RING + LEFT_BROW + RIGHT_BROW)
    stack = np.concatenate([lm[points] for lm in landmarks])
    span = float(np.median([eye_span(lm) for lm in landmarks]))
    low = stack.min(0) - 0.12 * span
    high = stack.max(0) + 0.12 * span
    x0, y0 = int(max(0, math.floor(low[0]))), int(max(0, math.floor(low[1])))
    x1, y1 = int(min(width, math.ceil(high[0]))), int(min(height, math.ceil(high[1])))
    return x0, y0, x1, y1


def crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1].copy()


def fixed_size(image: np.ndarray, side: int = FACE_SIZE) -> np.ndarray:
    """Scale so the width is side, keeping aspect, so one crop always sends one size."""
    height, width = image.shape[:2]
    target = (side, max(1, int(round(height * side / width))))
    flags = cv2.INTER_AREA if width > side else cv2.INTER_CUBIC
    return cv2.resize(image, target, interpolation=flags)


def apex_index(frame_landmarks: Dict[int, Optional[np.ndarray]], order: Sequence[int]) -> Optional[int]:
    """Frame furthest from the first usable frame in expressive points, after registration.

    Comes from landmarks alone and uses no model output, so it can be fixed in advance.
    Ties go to the earliest frame. None when fewer than two frames have landmarks.
    """
    usable = [i for i in order if frame_landmarks.get(i) is not None]
    if len(usable) < 2:
        return None
    reference = frame_landmarks[usable[0]]
    stable, moving = list(STABLE_POINTS), list(EXPRESSIVE_POINTS)
    best, best_score = usable[0], -1.0
    for index in usable:
        current = frame_landmarks[index]
        matrix = similarity_transform(current[stable], reference[stable])
        moved = apply_transform(matrix, current[moving]) - reference[moving]
        score = float(np.sqrt((moved**2).sum(1)).mean())
        if score > best_score + 1e-9:
            best, best_score = index, score
    return int(best)


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------


def spatial_frequency_filter(frame: np.ndarray, band: str, cutoff_cpf: float, face_width_px: float) -> np.ndarray:
    """Gaussian low-pass or its complement. cutoff is the half-amplitude point in cycles per face.

    High-pass output keeps each channel's mean. Reflective padding stops image edges from
    wrapping around in the FFT.
    """
    if band not in {"low", "high"}:
        raise ValueError(f"Unknown band {band}")
    if face_width_px <= 0:
        raise EditFailure("no_face_width")
    pad = int(math.ceil(face_width_px / 2))
    padded = np.pad(frame.astype(np.float64), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    height, width = padded.shape[:2]
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    cutoff = float(cutoff_cpf) / float(face_width_px)  # cycles per pixel
    gain = np.exp(-math.log(2.0) * (fx**2 + fy**2) / cutoff**2)
    if band == "high":
        gain = 1.0 - gain
    out = np.empty(frame.shape, dtype=np.float64)
    for channel in range(frame.shape[2]):
        spectrum = np.fft.rfft2(padded[..., channel]) * gain
        restored = np.fft.irfft2(spectrum, s=(height, width))[pad:-pad, pad:-pad]
        if band == "high":
            restored = restored + frame[..., channel].mean()
        out[..., channel] = restored
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def eye_opening(landmarks: np.ndarray, eye: str) -> Tuple[float, float]:
    """Lid gap in pixels and lid gap over eye width."""
    outer, inner, upper, lower = EYE_POINTS[eye]
    gap = float(np.linalg.norm(landmarks[upper] - landmarks[lower]))
    return gap, gap / max(float(np.linalg.norm(landmarks[outer] - landmarks[inner])), 1e-6)


def _warp_one_eye(frame: np.ndarray, landmarks: np.ndarray, eye: str, scale: float) -> np.ndarray:
    outer, inner, upper, lower = (landmarks[i] for i in EYE_POINTS[eye])
    half_width = float(np.linalg.norm(outer - inner)) / 2.0
    half_gap = max(float(np.linalg.norm(upper - lower)) / 2.0, 1.0)
    centre = (upper + lower) / 2.0
    axis_u = (outer - inner) / max(np.linalg.norm(outer - inner), 1e-6)
    axis_v = np.array([-axis_u[1], axis_u[0]])
    reach_u, reach_v = 1.5 * half_width, 3.0 * half_gap
    plateau = 0.45
    extent = int(math.ceil(max(reach_u, reach_v))) + 2
    height, width = frame.shape[:2]
    x0, x1 = max(0, int(centre[0]) - extent), min(width, int(centre[0]) + extent + 1)
    y0, y1 = max(0, int(centre[1]) - extent), min(height, int(centre[1]) + extent + 1)
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    du = (xs - centre[0]) * axis_u[0] + (ys - centre[1]) * axis_u[1]
    dv = (xs - centre[0]) * axis_v[0] + (ys - centre[1]) * axis_v[1]
    radius = np.sqrt((du / reach_u) ** 2 + (dv / reach_v) ** 2)
    t = np.clip((1.0 - radius) / (1.0 - plateau), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    source_v = dv / (1.0 + (scale - 1.0) * weight)
    shift = source_v - dv
    map_x = (xs + shift * axis_v[0]).astype(np.float32)
    map_y = (ys + shift * axis_v[1]).astype(np.float32)
    out = frame.copy()
    out[y0:y1, x0:x1] = cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out


def scale_eye_aperture(frame: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    """Open or close both eyes by a local vertical warp about the eye centre, with no generator.

    Lids move by the full factor. The warp fades to nothing inside a bounded patch, so
    brows and everything outside it keep their exact pixels. The iris is stretched with
    the lids, which is a known artefact: the prereg treats wide and narrow as opposed
    doses, since an artefact should push both the same way and a cue should push them
    apart. Scale 1 returns an exact copy. Closed eyes raise EditFailure.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    for eye in EYE_POINTS:
        if eye_opening(landmarks, eye)[1] < MIN_EYE_OPENING:
            raise EditFailure("eye_closed", eye)
    if scale == 1.0:
        return frame.copy()
    out = frame
    for eye in EYE_POINTS:
        out = _warp_one_eye(out, landmarks, eye, scale)
    return out


# ---------------------------------------------------------------------------
# Applying a named condition
# ---------------------------------------------------------------------------


@dataclass
class ClipGeometry:
    """Everything an edit needs for one clip, fixed before any model is run."""

    face_box: Optional[Tuple[int, int, int, int]] = None
    eye_box: Optional[Tuple[int, int, int, int]] = None
    face_width_px: float = 0.0
    first_frame: Optional[np.ndarray] = None
    first_landmarks: Optional[np.ndarray] = None
    neutral_frame: Optional[np.ndarray] = None
    neutral_landmarks: Optional[np.ndarray] = None
    notes: Dict[str, Any] = field(default_factory=dict)


def _fill_image(frame: np.ndarray, landmarks: np.ndarray, fill: str, geometry: ClipGeometry) -> Tuple[np.ndarray, Dict[str, Any]]:
    size = frame.shape[:2]
    if fill == "self":
        return frame, {"fill": "self"}
    if fill == "blur":
        sigma = BLUR_FILL_SIGMA_FRACTION * eye_span(landmarks)
        return cv2.GaussianBlur(frame, (0, 0), sigma), {"fill": "blur", "sigma_px": round(sigma, 2)}
    if fill == "first":
        donor, donor_lm = geometry.first_frame, geometry.first_landmarks
        reason = "first_frame_no_landmarks"
    elif fill == "neutral":
        donor, donor_lm = geometry.neutral_frame, geometry.neutral_landmarks
        reason = "neutral_clip_no_landmarks"
    else:
        raise ValueError(f"Unknown fill {fill}")
    if donor is None or donor_lm is None:
        raise EditFailure(reason)
    warped, residual = align_to(donor, donor_lm, landmarks, size)
    return warped, {
        "fill": fill,
        "align_residual": round(residual, 4),
        "align_flagged": bool(residual > MAX_ALIGN_RESIDUAL),
    }


def apply_condition(
    name: str,
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    geometry: ClipGeometry,
    view: str = "face",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One spatial condition on one frame, then the view crop. Returns image and log."""
    if name not in SPATIAL_CONDITIONS:
        raise ValueError(f"Unknown spatial condition {name}")
    spec = SPATIAL_CONDITIONS[name]
    kind = spec["kind"]
    log: Dict[str, Any] = {"kind": kind}
    needs_landmarks = kind != "none" or view == "face"
    if needs_landmarks and landmarks is None:
        raise EditFailure("no_face_detected")
    image = frame
    if kind == "mask":
        alpha = region_mask(frame.shape[:2], landmarks, spec["region"])
        fill, fill_log = _fill_image(frame, landmarks, spec["fill"], geometry)
        image = blend(frame, fill, alpha)
        log.update(fill_log)
        log.update({"region": spec["region"], "mask_fraction": round(float(alpha.mean()), 5)})
    elif kind == "aperture":
        image = scale_eye_aperture(frame, landmarks, spec["scale"])
        log["scale"] = spec["scale"]
    elif kind == "eye_window":
        if geometry.eye_box is None:
            raise EditFailure("no_eye_box")
        grey = np.full_like(frame, 128)
        x0, y0, x1, y1 = geometry.eye_box
        grey[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        image = grey
    elif kind == "eye_crop":
        if geometry.eye_box is None:
            raise EditFailure("no_eye_box")
        return fixed_size(crop(frame, geometry.eye_box)), {**log, "view": "eye_band", "box": list(geometry.eye_box)}

    if view == "face":
        if geometry.face_box is None:
            raise EditFailure("no_face_box")
        image = crop(image, geometry.face_box)
        log["box"] = list(geometry.face_box)
    elif view != "full":
        raise ValueError(f"Unknown view {view}")
    log["view"] = view

    if kind == "sf":
        image = spatial_frequency_filter(image, spec["band"], spec["cpf"], geometry.face_width_px)
        log.update({"band": spec["band"], "cpf": spec["cpf"], "face_width_px": round(geometry.face_width_px, 1)})
    if view == "face":
        image = fixed_size(image)
    return image.copy() if image is frame else image, log


# ---------------------------------------------------------------------------
# Outcomes and inference, as specified in research/PREREG_A2_DRAFT.md
# ---------------------------------------------------------------------------


def clr(log_values: Sequence[float]) -> np.ndarray:
    """Centred log-ratio: a label's log score against the mean over labels."""
    values = np.asarray(log_values, dtype=np.float64)
    return values - values.mean()


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in nats between two distributions on the same support."""
    a, b = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    a, b = a / a.sum(), b / b.sum()
    mid = (a + b) / 2.0

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        keep = x > 0
        return float((x[keep] * np.log(x[keep] / y[keep])).sum())

    return 0.5 * kl(a, mid) + 0.5 * kl(b, mid)


def _t_stats(matrix: np.ndarray) -> np.ndarray:
    """One-sample t over the actor axis (-2), skipping NaN for absent actors."""
    present = ~np.isnan(matrix)
    n = present.sum(-2).astype(np.float64)
    filled = np.where(present, matrix, 0.0)
    mean = filled.sum(-2) / np.maximum(n, 1.0)
    residual = np.where(present, matrix - mean[..., None, :], 0.0)
    sd = np.sqrt((residual**2).sum(-2) / np.maximum(n - 1.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / np.sqrt(n))
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def signflip_maxt(
    actor_effects: np.ndarray,
    *,
    alternative: Sequence[int] = (),
    max_exact_actors: int = 17,
    draws: int = 100_000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Actor-level sign-flip test with Westfall-Young step-down maxT.

    actor_effects has one row per actor and one column per contrast, holding the actor's
    mean within-item change, NaN where the actor has no item for that contrast. The same
    flip applies to a whole row, so dependence across contrasts is kept. All 2^n flips are
    used when n <= max_exact_actors, otherwise a seeded Monte Carlo sample that includes
    the identity. alternative gives a sign per contrast: +1, -1, or 0 for two-sided.
    """
    effects = np.asarray(actor_effects, dtype=np.float64)
    n_actors, n_tests = effects.shape
    signs_wanted = np.asarray(alternative if len(alternative) else [0] * n_tests, dtype=np.float64)
    if signs_wanted.shape[0] != n_tests:
        raise ValueError("alternative must have one entry per contrast")
    exact = n_actors <= max_exact_actors
    if exact:
        flips = np.array(list(itertools.product((1.0, -1.0), repeat=n_actors)))
    else:
        rng = np.random.default_rng(seed)
        flips = rng.choice((1.0, -1.0), size=(draws, n_actors))
        flips[0] = 1.0

    def signed(t: np.ndarray) -> np.ndarray:
        return np.where(signs_wanted == 0, np.abs(t), t * signs_wanted)

    observed = signed(_t_stats(effects))
    null = np.empty((flips.shape[0], n_tests))
    for start in range(0, flips.shape[0], 4096):
        block = flips[start : start + 4096]
        null[start : start + len(block)] = signed(_t_stats(block[:, :, None] * effects))

    order = np.argsort(-observed, kind="stable")
    running = np.full(flips.shape[0], -np.inf)
    adjusted = np.empty(n_tests)
    for test in order[::-1]:
        running = np.maximum(running, null[:, test])
        adjusted[test] = float((running >= observed[test] - 1e-12).mean())
    previous = 0.0
    for test in order:
        previous = max(previous, adjusted[test])
        adjusted[test] = previous
    raw = (null >= observed - 1e-12).mean(0)
    return {
        "t": observed.tolist(),
        "p_raw": raw.tolist(),
        "p_maxt": adjusted.tolist(),
        "n_flips": int(flips.shape[0]),
        "exact": bool(exact),
    }

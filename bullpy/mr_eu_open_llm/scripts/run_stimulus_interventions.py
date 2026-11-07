"""
Stimulus interventions for RQ1 and RQ2: blind prefill readout plus rotated-option MCQ.

Steps
  landmarks        FaceMesh on the frames each clip can send. Writes numeric geometry only.
                   Needs MediaPipe, so it runs in venvs/au_tools. Fixing geometry once
                   means every model gets the same edits, and the Molmo2 venv needs no
                   MediaPipe.
  run              For each item, input and condition, score both readouts and append one
                   JSON line. Failed cells are written with their reason. Resumable.
  probe_prompt     Lexicon coverage of candidate blind prompts on unedited smoke items and a
                   grey image. Reads coverage only: no label, edit or contrast is involved.
  probe_processor  Synthetic frames through a model's processor, with no weights and no
                   clips, to record what each input route delivers.

The primary contrast is within item: y(condition) minus y(original) on the same input.
Frames are decoded and edited in memory. No frame, crop or edited clip is written; output
holds indices, hashes and scores. See research/PREREG_A2_DRAFT.md. Full runs wait on
gates G2 to G4 and RQ0, so `run` refuses to go past --max_items without
--confirm_gates_passed.

Inputs
  single_face  apex frame, fixed face crop. Spatial conditions live here, because a lip
               corner is sub-token on a grid.
  single_full  apex frame, whole frame, unedited. Bridges crop to full frame.
  video_face   k frames through the model's native video processor, face crop. Temporal
               conditions. Timestamps come from a synthetic 1 fps clock, so the prompt
               text is the same in every condition and every clip.
  legacy_full  k frames through prepare_images_for_model, whole frame: a composite grid on
               Gemma 4 and Molmo2, a native image list on Qwen. Bridges to earlier runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scripts import stimulus_design as design
from scripts import stimulus_interventions as si

ABSTAIN = "None of the above"
WEAK_LABELS = ("Kind", "Unfriendly", "Jealous")
# The geometry of record, as pinned in scripts/face_input_validity.py, which is frozen by
# A3 and so is not imported or edited here. A test holds the two equal.
GEOMETRY_SHA256 = "2bc76dd60be233ee6072ba5b2049721899c5febcd953ec49bff1dcc2f2c72383"

MCQ_PREFILL = "EMOTION:"
# Candidates in ladder order. The chosen pair comes from these by the ladder's rule, on
# calibration clips only; see research/DECISIONS.md.
BLIND_CANDIDATES = (
    ("Describe what this person is feeling in one word.", "They feel"),
    ("Describe what this person is feeling in one plain word, with no formatting.", "They feel"),
    ("In one plain word, with no formatting or punctuation, what is this person feeling?", "They feel"),
    ("In one plain word, with no formatting or punctuation, what is this person feeling?", "This person is feeling"),
    ("Finish the sentence with one plain word and no formatting: They feel", "They feel"),
    ("Finish the sentence with one plain word and no formatting: They look", "They look"),
)
BLIND_PROMPT, BLIND_PREFILL = BLIND_CANDIDATES[2]

# Preregistered lexicon: single-word forms from data/eu_emotion_synonyms.json that fit
# after "They feel". Removed: multi-word entries, forms listed under two labels (anxious,
# hostile), and forms with a mainly non-mental sense (blocked, dull, cold, warm, funny).
# The neutral family is new here, since the synonym file has none.
LEXICON: Dict[str, Tuple[str, ...]] = {
    "Afraid": ("afraid", "fear", "fearful", "frightened", "scared", "terrified", "apprehensive"),
    "Angry": ("angry", "anger", "mad", "furious", "irritated", "annoyed", "enraged"),
    "Ashamed": ("ashamed", "shame", "embarrassed", "humiliated", "mortified"),
    "Bored": ("bored", "boredom", "uninterested", "apathetic"),
    "Disappointed": ("disappointed", "disappointment", "disheartened"),
    "Disgusted": ("disgusted", "disgust", "revolted", "repulsed", "sickened"),
    "Excited": ("excited", "excitement", "enthusiastic", "thrilled", "elated"),
    "Frustrated": ("frustrated", "frustration", "thwarted"),
    "Happy": ("happy", "happiness", "joyful", "joy", "pleased", "cheerful", "delighted"),
    "Hurt": ("hurt", "wounded", "upset", "pained"),
    "Interested": ("interested", "interest", "curious", "engaged", "attentive"),
    "Jealous": ("jealous", "jealousy", "envious", "envy", "covetous"),
    "Joking": ("joking", "playful", "teasing", "humorous", "amused"),
    "Kind": ("kind", "kindness", "friendly", "caring", "compassionate"),
    "Neutral": ("neutral", "calm", "impassive", "unemotional"),
    "Proud": ("proud", "pride", "satisfied", "accomplished"),
    "Sad": ("sad", "sadness", "unhappy", "sorrowful", "melancholy", "downcast", "gloomy"),
    "Sneaky": ("sneaky", "sly", "devious", "cunning", "deceptive", "secretive"),
    "Surprised": ("surprised", "surprise", "shocked", "astonished", "amazed", "startled"),
    "Unfriendly": ("unfriendly", "unkind", "disdainful", "aloof"),
    "Worried": ("worried", "worry", "concerned", "uneasy", "nervous"),
}
OPTION_TO_FAMILY = {label: label for label in LEXICON}
OPTION_TO_FAMILY["Excitement"] = "Excited"

SPATIAL = tuple(si.SPATIAL_CONDITIONS)
TEMPORAL = si.TEMPORAL_CONDITIONS
INPUTS: Dict[str, Dict[str, Any]] = {
    "single_face": {"kind": "image", "view": "face", "conditions": SPATIAL},
    "single_full": {"kind": "image", "view": "full", "conditions": ("original",)},
    "video_face": {"kind": "video", "view": "face", "conditions": TEMPORAL},
    "legacy_full": {"kind": "legacy", "view": "full", "conditions": ("original",)},
}
PIXEL_KEYS = (
    "pixel_values",
    "image_position_ids",
    "pixel_values_videos",
    "video_position_ids",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "video_token_pooling",
    "video_grids",
)
# Gemma's default of 70 tokens per video frame gives under 5 tokens across the face.
VIDEO_KWARGS: Dict[str, Dict[str, Any]] = {"gemma4": {"max_soft_tokens": 280}}
MEDIA_TOKEN_ATTRS = ("image_token_id", "video_token_id", "image_patch_id")


def lexicon_words() -> List[Tuple[str, str]]:
    """(family, word) in fixed order."""
    return [(family, word) for family in sorted(LEXICON) for word in LEXICON[family]]


def text_sha256(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def readout_fields() -> Dict[str, Any]:
    """The fields of the design hash used up to smoke_v2, a1924d80."""
    return {
        "prompt": BLIND_PROMPT,
        "prefill": BLIND_PREFILL,
        "lexicon": LEXICON,
        "spatial": si.SPATIAL_CONDITIONS,
        "temporal": list(TEMPORAL),
        "k": si.N_SENT_FRAMES,
    }


def edit_fields() -> Dict[str, Any]:
    """Constants of the edits themselves, which a change of conditions would not show."""
    return {
        "seed": si.SEED,
        "regions": {
            "mouth_ring": si.MOUTH_RING,
            "left_eye_ring": si.LEFT_EYE_RING,
            "right_eye_ring": si.RIGHT_EYE_RING,
            "left_brow": si.LEFT_BROW,
            "right_brow": si.RIGHT_BROW,
            "stable_points": si.STABLE_POINTS,
            "outer_eye_corners": si.OUTER_EYE_CORNERS,
            "face_edges": si.FACE_EDGES,
            "eye_points": si.EYE_POINTS,
            "eye_band_margin": si.EYE_BAND_MARGIN,
            "dilate": si.REGION_DILATE,
        },
        "feather": si.FEATHER_FRACTION,
        "blur_sigma": si.BLUR_FILL_SIGMA_FRACTION,
        "max_align_residual": si.MAX_ALIGN_RESIDUAL,
        "min_eye_opening": si.MIN_EYE_OPENING,
        "face_box_margin": si.FACE_BOX_MARGIN,
        "face_size": si.FACE_SIZE,
    }


def design_hash() -> str:
    """Ties a result to the readout, edits, MCQ, geometry and analysis of the prereg."""
    from scripts.prompts import build_4afc_prompt

    placeholders = tuple(f"<option {n}>" for n in range(6))
    payload = {
        **readout_fields(),
        "edits": edit_fields(),
        "mcq": {
            "prefill": MCQ_PREFILL,
            "prompt": build_4afc_prompt(placeholders, condition="video_only"),
            "rotations": [mcq_rotations(position) for position in range(4)],
        },
        "video_kwargs": VIDEO_KWARGS,
        "geometry_sha256": GEOMETRY_SHA256,
        "analysis": design.design_spec(),
    }
    return text_sha256(json.dumps(payload, sort_keys=True))


def check_geometry(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GEOMETRY_SHA256:
        raise SystemExit(f"Geometry hash {digest} is not the pinned one.")


def resolve_shards(shard: int, n_shards: Optional[int], full_run: bool) -> int:
    """A full run must name its shard count, so a resubmitted task cannot re-shard."""
    if n_shards is None:
        if full_run:
            raise SystemExit("A full run needs --n_shards, stated explicitly.")
        n_shards = 1
    if n_shards < 1 or not 0 <= shard < n_shards:
        raise SystemExit(f"Shard {shard} is not valid for {n_shards} shards.")
    return n_shards


# ---------------------------------------------------------------------------
# Scoring logic with no model in it
# ---------------------------------------------------------------------------


def internal_prefixes(sequences: Iterable[Sequence[int]]) -> List[Tuple[int, ...]]:
    """Token prefixes needing a forward, shortest first. A word of n tokens needs n."""
    found = {()}
    for ids in sequences:
        for end in range(1, len(ids)):
            found.add(tuple(ids[:end]))
    return sorted(found, key=lambda p: (len(p), p))


def joint_logprob(ids: Sequence[int], table: Dict[Tuple[int, ...], Dict[int, float]]) -> float:
    total = 0.0
    for end, token in enumerate(ids):
        total += table[tuple(ids[:end])][token]
    return total


def needed_tokens(sequences: Iterable[Sequence[int]]) -> Dict[Tuple[int, ...], List[int]]:
    needed: Dict[Tuple[int, ...], set] = defaultdict(set)
    for ids in sequences:
        for end, token in enumerate(ids):
            needed[tuple(ids[:end])].add(token)
    return {k: sorted(v) for k, v in needed.items()}


def logsumexp(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    top = arr.max()
    return float(top + np.log(np.exp(arr - top).sum()))


def family_scores(words: Sequence[Tuple[str, str]], logprobs: Sequence[float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (family, _word), value in zip(words, logprobs):
        grouped[family].append(value)
    return {family: logsumexp(grouped[family]) for family in sorted(grouped)}


def human_comparable(family: Dict[str, float], human: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """Blind readout and human votes on the item's named options, each renormalised.

    None of the above has no lexicon family, so it leaves both sides.
    """
    names = [o for o in sorted(human) if o != ABSTAIN]
    scores = np.array([family[OPTION_TO_FAMILY[o]] for o in names])
    model = np.exp(scores - scores.max())
    votes = np.array([human[o] for o in names], dtype=np.float64)
    return (model / model.sum()).tolist(), (votes / votes.sum()).tolist()


def mcq_rotations(position: int) -> Tuple[int, int]:
    """Two rotations per item, never 0 (target first); position balances them across items."""
    return 1 + position % 4, 1 + (position + 2) % 4


def select_items(items: Sequence[Dict[str, Any]], max_items: Optional[int]) -> List[Dict[str, Any]]:
    """Fixed order with no model input: round-robin over actors, so a prefix is actor-diverse."""
    by_actor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=lambda i: si.stable_seed(si.SEED, "item", i["trial_id"])):
        by_actor[item["actor"]].append(item)
    actors = sorted(by_actor, key=lambda a: si.stable_seed(si.SEED, "actor", a))
    ordered: List[Dict[str, Any]] = []
    depth = 0
    while len(ordered) < len(items):
        for actor in actors:
            if depth < len(by_actor[actor]):
                ordered.append(by_actor[actor][depth])
        depth += 1
    for position, item in enumerate(ordered):
        item["position"] = position
    return ordered if max_items is None else ordered[:max_items]


def load_items(manifest: Path, lookup: Path, data_root: Path) -> List[Dict[str, Any]]:
    from paper_v2.schema import actor_id_from_eu_filename

    trials = json.loads(manifest.read_text(encoding="utf-8"))["trials"]
    human = json.loads(lookup.read_text(encoding="utf-8"))["trials"]
    items = []
    for trial in trials:
        trial_id = str(trial["trial_id"])
        entry = human.get(trial_id)
        if not entry or len(entry.get("human_options", [])) != 6:
            continue
        distribution = entry["human_response_distribution"]
        named = {k: v for k, v in distribution.items() if k != ABSTAIN}
        items.append(
            {
                "trial_id": trial_id,
                "label": entry["human_target_label"],
                "actor": actor_id_from_eu_filename(trial_id),
                "options": list(entry["human_options"]),
                "human_distribution": distribution,
                "human_modal_label": max(sorted(named), key=lambda k: named[k]),
                "weak_label": entry["human_target_label"] in WEAK_LABELS,
                "path": str(data_root / trial["stimulus_path"]),
            }
        )
    return items


# ---------------------------------------------------------------------------
# Step 1: geometry
# ---------------------------------------------------------------------------


class FaceMesh:
    """MediaPipe FaceMesh, one independent detection per frame."""

    def __init__(self) -> None:
        import mediapipe as mp

        self.version = mp.__version__
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        result = self.mesh.process(frame)
        if not result.multi_face_landmarks:
            return None
        height, width = frame.shape[:2]
        points = result.multi_face_landmarks[0].landmark
        return np.array([[p.x * width, p.y * height] for p in points], dtype=np.float64)


def count_frames(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise si.EditFailure("video_unreadable", path.name)
    count = 0
    while capture.grab():
        count += 1
    capture.release()
    return count


def clip_geometry_record(item: Dict[str, Any], detect: Callable[[np.ndarray], Optional[np.ndarray]]) -> Dict[str, Any]:
    """Landmarks and fixed boxes for a clip. Failures are recorded, never skipped."""
    path = Path(item["path"])
    record: Dict[str, Any] = {"trial_id": item["trial_id"], "actor": item["actor"], "label": item["label"]}
    n_frames = count_frames(path)
    base = si.uniform_indices(0, n_frames, si.N_SENT_FRAMES)
    frames, meta = si.read_frames(path, base)
    landmarks: Dict[int, Optional[np.ndarray]] = {i: detect(frames[i]) for i in base}
    size = (meta["height"], meta["width"])
    good = [landmarks[i] for i in base if landmarks[i] is not None]
    record.update(
        {
            "n_frames": n_frames,
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "base_indices": base,
            "frame_sha256": {str(i): si.array_sha256(frames[i]) for i in base},
            "landmarks": {
                str(i): None if v is None else np.round(v, 1).tolist() for i, v in landmarks.items()
            },
            "failed_frames": [i for i in base if landmarks[i] is None],
            "apex": si.apex_index(landmarks, base),
        }
    )
    if len(good) * 2 < len(base):
        record["box_failure"] = "fewer_than_half_frames_have_a_face"
        return record
    record["face_box"] = list(si.square_box(good, size))
    record["eye_box"] = list(si.eye_band_box(good, size))
    record["face_width_px"] = round(float(np.median([si.face_width(lm) for lm in good])), 2)
    apex = record["apex"]
    if apex is not None and landmarks[apex] is not None:
        # Dose check: re-detect the lids on the warped apex frame. Pixels stay in memory.
        before = {e: si.eye_opening(landmarks[apex], e)[0] for e in si.EYE_POINTS}
        check: Dict[str, Any] = {}
        for name, scale in si.APERTURE_SCALES.items():
            try:
                warped = si.scale_eye_aperture(frames[apex], landmarks[apex], scale)
            except si.EditFailure as exc:
                check[name] = {"failed": exc.reason}
                continue
            found = detect(warped)
            check[name] = (
                {"failed": "no_face_after_warp"}
                if found is None
                else {
                    "nominal": scale,
                    "measured": {e: round(si.eye_opening(found, e)[0] / before[e], 3) for e in si.EYE_POINTS},
                }
            )
        record["aperture_check"] = check
    return record


def step_landmarks(args: argparse.Namespace) -> None:
    if args.geometry.exists():
        raise SystemExit(f"{args.geometry} is fixed for all models and is not overwritten.")
    items = select_items(load_items(args.manifest, args.lookup, args.data_root), None)
    chosen = items if args.max_items is None else items[: args.max_items]
    actors = {i["actor"] for i in chosen}
    ids = {i["trial_id"] for i in chosen}
    needed = chosen + [
        i for i in items if i["label"] == "Neutral" and i["actor"] in actors and i["trial_id"] not in ids
    ]
    detect = FaceMesh()
    records: Dict[str, Any] = {}
    failures: List[Dict[str, Any]] = []
    started = time.time()
    for item in needed:
        try:
            records[item["trial_id"]] = clip_geometry_record(item, detect)
        except si.EditFailure as exc:
            records[item["trial_id"]] = {"trial_id": item["trial_id"], "actor": item["actor"], "clip_failure": exc.reason}
        record = records[item["trial_id"]]
        if record.get("clip_failure") or record.get("box_failure") or record.get("failed_frames"):
            failures.append(
                {
                    "trial_id": item["trial_id"],
                    "clip_failure": record.get("clip_failure"),
                    "box_failure": record.get("box_failure"),
                    "failed_frames": record.get("failed_frames", []),
                }
            )
    output = {
        "detector": f"mediapipe.face_mesh {detect.version} static_image_mode refine_landmarks",
        "k": si.N_SENT_FRAMES,
        "n_clips": len(records),
        "n_clips_with_failures": len(failures),
        "seconds": round(time.time() - started, 1),
        "failures": failures,
        "items": records,
    }
    args.geometry.parent.mkdir(parents=True, exist_ok=True)
    args.geometry.write_text(json.dumps(output) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("n_clips", "n_clips_with_failures", "seconds", "failures")}, indent=1))


# ---------------------------------------------------------------------------
# Building media from geometry
# ---------------------------------------------------------------------------


def _landmark_table(record: Dict[str, Any]) -> Dict[int, Optional[np.ndarray]]:
    return {int(k): None if v is None else np.asarray(v, dtype=np.float64) for k, v in record["landmarks"].items()}


@dataclass
class Clip:
    """One clip's decoded frames and fixed geometry, in memory for one item only."""

    item: Dict[str, Any]
    record: Dict[str, Any]
    frames: Dict[int, np.ndarray]
    landmarks: Dict[int, Optional[np.ndarray]]
    geometry: si.ClipGeometry
    plan: Dict[str, List[int]]
    apex: Optional[int]


def check_decoder(record: Dict[str, Any], frames: Dict[int, np.ndarray]) -> None:
    """Frames must hash as they did when geometry was fixed, or the edits sit on other pixels.

    cv2 4.13 in the Molmo venv fails colour conversion on clips flagged interlaced and
    returns a junk buffer, which is how gate 3 found this.
    """
    for index, expected in record.get("frame_sha256", {}).items():
        if int(index) in frames and si.array_sha256(frames[int(index)]) != expected:
            raise si.EditFailure("decoder_mismatch", f"frame {index}")


def load_clip(item: Dict[str, Any], geometry_items: Dict[str, Any], neutral_by_actor: Dict[str, str]) -> Clip:
    record = geometry_items.get(item["trial_id"])
    if record is None:
        raise si.EditFailure("no_geometry_record")
    if record.get("clip_failure"):
        raise si.EditFailure(record["clip_failure"])
    apex = record.get("apex")
    plan = si.temporal_plan(record["n_frames"], item["trial_id"], apex)
    wanted = sorted({i for row in plan.values() for i in row})
    frames, _ = si.read_frames(Path(item["path"]), wanted)
    check_decoder(record, frames)
    landmarks = _landmark_table(record)
    geometry = si.ClipGeometry()
    if not record.get("box_failure"):
        geometry.face_box = tuple(record["face_box"])
        geometry.eye_box = tuple(record["eye_box"])
        geometry.face_width_px = float(record["face_width_px"])
    first = record["base_indices"][0]
    geometry.first_frame = frames[first]
    geometry.first_landmarks = landmarks.get(first)

    neutral_id = neutral_by_actor.get(item["actor"])
    neutral_record = geometry_items.get(neutral_id) if neutral_id else None
    if neutral_record is not None and not neutral_record.get("clip_failure"):
        middle = neutral_record["base_indices"][len(neutral_record["base_indices"]) // 2]
        neutral_landmarks = _landmark_table(neutral_record).get(middle)
        if neutral_landmarks is not None:
            donor, _ = si.read_frames(Path(neutral_record["path"]), [middle])
            geometry.neutral_frame = donor[middle]
            geometry.neutral_landmarks = neutral_landmarks
            geometry.notes["neutral_donor"] = {
                "trial_id": neutral_id,
                "frame": middle,
                "self": neutral_id == item["trial_id"],
            }
    return Clip(item, record, frames, landmarks, geometry, plan, apex)


@dataclass
class Media:
    kind: str
    images: List[np.ndarray]
    sent_indices: List[int]
    log: Dict[str, Any]


def build_media(clip: Clip, input_name: str, condition: str) -> Media:
    spec = INPUTS[input_name]
    view = spec["view"]
    if spec["kind"] == "image":
        if clip.apex is None:
            raise si.EditFailure("no_apex_frame")
        image, log = si.apply_condition(condition, clip.frames[clip.apex], clip.landmarks.get(clip.apex), clip.geometry, view)
        if log.get("fill") == "neutral":
            log["neutral_donor"] = clip.geometry.notes.get("neutral_donor")
        if log.get("fill") == "first":
            log["donor_is_sent_frame"] = clip.apex == clip.record["base_indices"][0]
        return Media("image", [image], [clip.apex], log)
    if condition not in clip.plan:
        raise si.EditFailure("no_apex_frame" if condition == "static_apex" else "condition_has_no_plan")
    indices = clip.plan[condition]
    frames = [clip.frames[i] for i in indices]
    log: Dict[str, Any] = {"kind": "temporal", "view": view}
    if view == "face":
        if clip.geometry.face_box is None:
            raise si.EditFailure("no_face_box")
        frames = [si.fixed_size(si.crop(f, clip.geometry.face_box)) for f in frames]
        log["box"] = list(clip.geometry.face_box)
    return Media(spec["kind"], frames, list(indices), log)


# ---------------------------------------------------------------------------
# Encoding and scoring on a real model
# ---------------------------------------------------------------------------


def video_metadata(k: int, width: int, height: int) -> Any:
    """A fixed clock: timestamps depend only on position in the sent order."""
    from transformers.video_utils import VideoMetadata

    return VideoMetadata(
        total_num_frames=k,
        fps=1.0,
        width=width,
        height=height,
        duration=float(k),
        video_backend="in_memory",
        frames_indices=list(range(k)),
    )


def encode_video(model_key: str, processor: Any, images: Sequence[np.ndarray], prompt: str) -> Dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    height, width = images[0].shape[:2]
    clip = np.stack(images)
    return processor(
        text=text,
        videos=[clip],
        video_metadata=[video_metadata(len(images), width, height)],
        do_sample_frames=False,
        return_tensors="pt",
        **VIDEO_KWARGS.get(model_key, {}),
    )


def encode_media(handle: "ModelHandle", media: Media, prompt: str) -> Dict[str, Any]:
    from PIL import Image

    from scripts.activation_forward import build_forward_inputs, resolve_model_device
    from scripts.multi_frame import prepare_images_for_model

    if media.kind == "video":
        raw = encode_video(handle.key, handle.processor, media.images, prompt)
        device = resolve_model_device(handle.model)
        inputs: Dict[str, Any] = {}
        for key, value in dict(raw).items():
            if key.startswith("num_soft_tokens") or key == "video_metadata" or not hasattr(value, "to"):
                continue
            if handle.key == "molmo2" and key == "token_type_ids":
                continue  # as in build_forward_inputs: Molmo is causal-only, this switches that off
            is_float = value.is_floating_point()
            inputs[key] = value.to(device, dtype=handle.dtype) if is_float else value.to(device)
        return inputs
    images = [Image.fromarray(a) for a in media.images]
    for_processor: Any = images[0]
    if media.kind == "legacy":
        for_processor, meta = prepare_images_for_model(handle.key, images, enforce_multi_frame=True)
        media.log["multi_frame_strategy"] = meta["multi_frame_strategy"]
    return build_forward_inputs(
        handle.key,
        handle.model,
        handle.processor,
        prompt=prompt,
        images=images,
        images_for_processor=for_processor,
        device=handle.device,
        dtype=handle.dtype,
    )


@dataclass
class ModelHandle:
    key: str
    model: Any
    processor: Any
    tokenizer: Any
    device: str
    dtype: Any
    words: List[Tuple[str, str]]
    word_ids: List[List[int]]
    blind_prefill: List[int]
    mcq_prefill: List[int]
    forwards: int = 0
    cache_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_handle(model_key: str, dtype_name: str) -> ModelHandle:
    import torch
    from transformers import AutoProcessor

    from scripts.evaluate import load_hf_model_for_key, resolve_model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    path = resolve_model_path(model_key)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = load_hf_model_for_key(model_key, path, device, dtype)
    model.eval()
    words = lexicon_words()
    ids = [list(tokenizer(f" {word}", add_special_tokens=False).input_ids) for _family, word in words]
    return ModelHandle(
        model_key,
        model,
        processor,
        tokenizer,
        device,
        dtype,
        words,
        ids,
        list(tokenizer(BLIND_PREFILL, add_special_tokens=False).input_ids),
        list(tokenizer(MCQ_PREFILL, add_special_tokens=False).input_ids),
    )


def step_logprobs(handle: ModelHandle, inputs: Dict[str, Any], prefix: Sequence[int]) -> Any:
    import torch

    from scripts.vision_gates import append_tokens, last_logits

    handle.forwards += 1
    return torch.log_softmax(last_logits(handle.model, append_tokens(inputs, list(prefix))), dim=-1)


CACHE_TOLERANCE = 0.25  # nats; bfloat16 logits near 20 have a step of about 0.125
# Empty by decision, 2026-11-04 (research/DECISIONS.md). On Qwen the cached path reached 0.25
# nats on lexicon terms where repeated full forwards differ by 0.0, and on Gemma 4 it
# raised a CUDA index assert. All models use full forwards, so scoring is the same path.
CACHED_MODELS: frozenset = frozenset()


def continuation_inputs(
    handle: ModelHandle, step: Dict[str, Any], new_ids: Any, base_len: int, cache: Any
) -> Dict[str, Any]:
    """Text-only forward on top of a KV cache.

    Qwen gets explicit 3D positions from its stored rope delta: left to itself it builds
    them from the full attention mask, so new tokens get the wrong positions and the
    logits are off by about 21 nats. The other models go through their generation hook.
    """
    import torch

    if handle.key == "qwen3vl":
        count = new_ids.shape[1]
        text = torch.arange(base_len, base_len + count, device=new_ids.device).view(1, 1, -1)
        delta = handle.model.model.rope_deltas.to(new_ids.device).view(1, 1, 1)
        return {
            "input_ids": new_ids,
            "attention_mask": step["attention_mask"],
            "position_ids": (text + delta).expand(3, 1, -1),
            "past_key_values": cache,
            "use_cache": True,
        }
    position = torch.arange(base_len, step["input_ids"].shape[1], device=new_ids.device)
    extra = {k: step[k] for k in ("mm_token_type_ids", "token_type_ids") if k in step}
    return handle.model.prepare_inputs_for_generation(
        step["input_ids"],
        past_key_values=cache,
        attention_mask=step["attention_mask"],
        cache_position=position,
        use_cache=True,
        **extra,
    )


def cached_tables(
    handle: ModelHandle,
    inputs: Dict[str, Any],
    prefill: Sequence[int],
    prefixes: Sequence[Tuple[int, ...]],
    needed: Dict[Tuple[int, ...], List[int]],
) -> Dict[Tuple[int, ...], Dict[int, float]]:
    """One image forward, then text-only continuations from the cache.

    Gemma's sliding-window layers cannot be cropped once extended, so it copies the cache
    for each prefix. The others crop back after each one.
    """
    import copy

    import torch

    from scripts.vision_gates import append_tokens

    base = append_tokens(inputs, list(prefill))
    base_len = int(base["input_ids"].shape[1])
    with torch.no_grad():
        out = handle.model(**base, use_cache=True)
    handle.forwards += 1
    cache = out.past_key_values
    table = {(): {t: float(v) for t, v in zip(needed[()], torch.log_softmax(out.logits[0, -1].float(), -1)[needed[()]])}}
    keep = {k: v for k, v in base.items() if k in ("attention_mask", "mm_token_type_ids", "token_type_ids")}
    copies = handle.key == "gemma4"
    for prefix in prefixes:
        if not prefix:
            continue
        step = append_tokens({"input_ids": base["input_ids"], **keep}, list(prefix))
        new_ids = step["input_ids"][:, base_len:]
        active = copy.deepcopy(cache) if copies else cache
        with torch.no_grad():
            logits = handle.model(**continuation_inputs(handle, step, new_ids, base_len, active)).logits[0, -1].float()
        handle.forwards += 1
        if not copies:
            cache.crop(base_len)
        ids = needed[prefix]
        table[prefix] = {t: float(v) for t, v in zip(ids, torch.log_softmax(logits, -1)[ids])}
    return table


def full_tables(
    handle: ModelHandle,
    inputs: Dict[str, Any],
    prefill: Sequence[int],
    prefixes: Sequence[Tuple[int, ...]],
    needed: Dict[Tuple[int, ...], List[int]],
) -> Dict[Tuple[int, ...], Dict[int, float]]:
    table: Dict[Tuple[int, ...], Dict[int, float]] = {}
    for prefix in prefixes:
        logprobs = step_logprobs(handle, inputs, list(prefill) + list(prefix))
        table[prefix] = {t: float(logprobs[t]) for t in needed[prefix]}
    return table


def score_sequences(
    handle: ModelHandle, inputs: Dict[str, Any], prefill: Sequence[int], sequences: Sequence[Sequence[int]]
) -> List[float]:
    """Joint logprob of each token sequence after the prefill.

    The first call for each media kind checks the cached path against full forwards. If
    they disagree beyond tolerance, or the cached path raises, that kind keeps full
    forwards. The decision and worst gap per kind are recorded on the handle.
    """
    needed = needed_tokens(sequences)
    prefixes = internal_prefixes(sequences)
    kind = "video" if "pixel_values_videos" in inputs else "image"
    if handle.key not in CACHED_MODELS:
        handle.cache_checks.setdefault(kind, {"used": False, "gap": None, "error": "not tried on this model"})
        table = full_tables(handle, inputs, prefill, prefixes, needed)
    elif kind not in handle.cache_checks:
        table = full_tables(handle, inputs, prefill, prefixes, needed)
        try:
            cached = cached_tables(handle, inputs, prefill, prefixes, needed)
            gap = max(abs(cached[p][t] - table[p][t]) for p in table for t in table[p])
            handle.cache_checks[kind] = {"used": bool(gap <= CACHE_TOLERANCE), "gap": gap, "error": None}
        except Exception as exc:  # recorded: the fallback is the safe path
            handle.cache_checks[kind] = {"used": False, "gap": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    elif handle.cache_checks[kind]["used"]:
        table = cached_tables(handle, inputs, prefill, prefixes, needed)
    else:
        table = full_tables(handle, inputs, prefill, prefixes, needed)
    return [joint_logprob(ids, table) for ids in sequences]


def score_blind(handle: ModelHandle, media: Media) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inputs = encode_media(handle, media, BLIND_PROMPT)
    logprobs = score_sequences(handle, inputs, handle.blind_prefill, handle.word_ids)
    family = family_scores(handle.words, logprobs)
    blind = {
        "word_logprob": [round(v, 4) for v in logprobs],
        "family_logprob": {k: round(v, 4) for k, v in family.items()},
        "lexicon_mass": round(math.exp(logsumexp(logprobs)), 5),
    }
    return blind, inputs


def score_mcq(handle: ModelHandle, media: Media, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    from scripts.prompts import build_4afc_prompt
    from scripts.vision_gates import gate_options

    rows = []
    for rotation in mcq_rotations(item["position"]):
        options = gate_options(item["options"], rotation, item["label"])
        inputs = encode_media(handle, media, build_4afc_prompt(options, condition="video_only"))
        ids = [list(handle.tokenizer(f" {o}", add_special_tokens=False).input_ids) for o in options]
        joint = score_sequences(handle, inputs, handle.mcq_prefill, ids)
        rows.append(
            {
                "rotation": rotation,
                "options": options,
                "target_position": options.index(item["label"]),
                "joint_logprob": [round(v, 4) for v in joint],
            }
        )
    return rows


def media_token_counts(handle: ModelHandle, inputs: Dict[str, Any]) -> Dict[str, int]:
    ids = inputs["input_ids"][0]
    config = handle.model.config
    return {
        attr: int((ids == getattr(config, attr)).sum())
        for attr in MEDIA_TOKEN_ATTRS
        if getattr(config, attr, None) is not None
    }


def tensor_digest(value: Any) -> str:
    array = value.detach().float().cpu().numpy()
    return si.array_sha256(array)


def input_record(handle: ModelHandle, inputs: Dict[str, Any]) -> Dict[str, Any]:
    pixels = {k: list(v.shape) for k, v in inputs.items() if k in PIXEL_KEYS and hasattr(v, "shape")}
    digest = {k: tensor_digest(inputs[k]) for k in ("pixel_values", "pixel_values_videos") if k in inputs}
    ids = inputs["input_ids"][0].detach().cpu().numpy()
    return {
        "pixel_shapes": pixels,
        "pixel_sha256": digest,
        "media_tokens": media_token_counts(handle, inputs),
        "n_tokens": int(ids.shape[0]),
        "input_ids_sha256": si.array_sha256(ids),
    }


def score_cell(handle: ModelHandle, media: Media, item: Dict[str, Any]) -> Dict[str, Any]:
    start = handle.forwards
    tick = time.time()
    blind, inputs = score_blind(handle, media)
    record = input_record(handle, inputs)
    mcq = score_mcq(handle, media, item)
    return {
        "status": "ok",
        "sent_frame_indices": media.sent_indices,
        "edit": media.log,
        "media_sha256": [si.array_sha256(a) for a in media.images],
        "input": record,
        "blind": blind,
        "mcq": mcq,
        "forwards": handle.forwards - start,
        "seconds": round(time.time() - tick, 3),
    }


# ---------------------------------------------------------------------------
# Delivery checks: does each frame reach the answer?
# ---------------------------------------------------------------------------


MIN_TOKENS_ACROSS_FACE = 8.0


def view_tokens_wide(inputs: Dict[str, Any], media: Media, composite: bool) -> float:
    """Tokens across the width of one sent view, from the model's own grid if it has one."""
    n_cells = 4 if composite and media.kind == "legacy" else 1
    for key, column in (("image_grids", -1), ("video_grids", -1)):
        if key in inputs:
            return float(inputs[key].reshape(-1, inputs[key].shape[-1])[0, column]) / n_cells
    media_tokens = sum(v for k, v in inputs["_counts"].items())
    units = len(media.images)
    if media.kind == "video" and "video_grid_thw" in inputs:
        units = int(inputs["video_grid_thw"][0, 0])
    per_unit = media_tokens / max(units, 1)
    height, width = media.images[0].shape[:2]
    aspect = 1.0 if composite and media.kind == "legacy" else width / height
    return math.sqrt(per_unit * aspect)


def rebuilt_sha(clip: Clip, name: str, media: Media) -> List[str]:
    """Hash what the logged indices decode to now, by a second read of the file."""
    frames, _ = si.read_frames(Path(clip.item["path"]), media.sent_indices)
    images = [frames[i] for i in media.sent_indices]
    if INPUTS[name]["view"] == "face":
        images = [si.fixed_size(si.crop(f, clip.geometry.face_box)) for f in images]
    return [si.array_sha256(a) for a in images]


def delivery_for_item(handle: ModelHandle, clip: Clip, inputs_wanted: Sequence[str]) -> Dict[str, Any]:
    """Delivery gate checks for one item, on each route's unedited input."""
    import torch

    from config import FRAME_POLICY
    from scripts.frame_sampling import frame_indices_for_video
    from scripts.multi_frame import NATIVE_MULTI_IMAGE_MODELS

    report: Dict[str, Any] = {"trial_id": clip.item["trial_id"]}
    record = clip.record
    duration = record["n_frames"] / record["fps"] if record["fps"] else None
    policy = frame_indices_for_video(
        record["n_frames"], duration, fps=float(FRAME_POLICY["fps"]), max_frames=int(FRAME_POLICY["max_frames"])
    )
    composite = handle.key not in NATIVE_MULTI_IMAGE_MODELS
    report["frame_policy_check"] = {
        "sampled_by_policy": len(policy),
        "in_composite_grid": min(8, len(policy)) if composite else None,
        "frames_lost_to_grid_cap": max(0, len(policy) - 8) if composite else 0,
    }
    for name in inputs_wanted:
        row: Dict[str, Any] = {}
        try:
            media = build_media(clip, name, "original")
        except si.EditFailure as exc:
            report[name] = {"failed": exc.reason}
            continue
        base_inputs = encode_media(handle, media, BLIND_PROMPT)
        base = step_logprobs(handle, base_inputs, handle.blind_prefill)
        again = step_logprobs(handle, base_inputs, handle.blind_prefill)
        row["repeat_forward_max_abs"] = float((base - again).abs().max())
        row["input"] = input_record(handle, base_inputs)
        row["tokens_across_view"] = round(
            view_tokens_wide({**base_inputs, "_counts": row["input"]["media_tokens"]}, media, composite), 2
        )
        view_width = record["width"]
        if INPUTS[name]["view"] == "face":
            view_width = clip.geometry.face_box[2] - clip.geometry.face_box[0]
        row["view_width_source_px"] = view_width
        row["face_width_tokens"] = round(row["tokens_across_view"] * record.get("face_width_px", 0.0) / view_width, 2)

        changes = []
        same_ids = []
        for position in range(len(media.images)):
            blanked = list(media.images)
            blanked[position] = np.zeros_like(blanked[position])
            other_inputs = encode_media(handle, Media(media.kind, blanked, media.sent_indices, {}), BLIND_PROMPT)
            same_ids.append(bool(torch.equal(other_inputs["input_ids"], base_inputs["input_ids"])))
            delta = (step_logprobs(handle, other_inputs, handle.blind_prefill) - base).abs().max()
            changes.append(round(float(delta), 4))
        row["blank_frame_max_abs_logprob_change"] = changes
        row["gate_2_every_frame_changes_output"] = bool(all(c > 0 for c in changes))
        row["gate_3_logged_indices_are_the_frames_sent"] = bool(
            rebuilt_sha(clip, name, media) == [si.array_sha256(a) for a in media.images]
        )
        if len(media.images) > 1:
            shuffled = build_media(clip, name, "shuffle_a") if "shuffle_a" in INPUTS[name]["conditions"] else None
            if shuffled is not None:
                shuffled_inputs = encode_media(handle, shuffled, BLIND_PROMPT)
                same_ids.append(bool(torch.equal(base_inputs["input_ids"], shuffled_inputs["input_ids"])))
                key = "pixel_values_videos"
                if key in base_inputs and base_inputs[key].dim() >= 3 and base_inputs[key].shape[1] == len(media.images):
                    order = [media.sent_indices.index(i) for i in shuffled.sent_indices]
                    row["pixels_arrive_in_sent_order"] = bool(torch.equal(base_inputs[key][:, order], shuffled_inputs[key]))
                # Qwen has no frame axis, since it merges pairs, so this is all it can show.
                row["pixels_change_under_shuffle"] = input_record(handle, shuffled_inputs)["pixel_sha256"] != row["input"]["pixel_sha256"]
        row["gate_4_ids_unchanged_within_clip"] = bool(all(same_ids))
        row["input_ids_sha256"] = row["input"]["input_ids_sha256"]
        report[name] = row
    return report


def order_check(row: Dict[str, Any]) -> bool:
    """False if a shuffle fails to reach the pixels, or a frame axis shows the wrong order."""
    return row.get("pixels_arrive_in_sent_order") is not False and row.get("pixels_change_under_shuffle") is not False


def step_delivery(args: argparse.Namespace) -> None:
    """Delivery gate for one model on every route, research/PLAN_READOUT_AND_INPUT.md."""
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    records = geometry["items"]
    all_items = load_items(args.manifest, args.lookup, args.data_root)
    paths = {i["trial_id"]: i["path"] for i in all_items}
    for trial_id, record in records.items():
        record["path"] = paths.get(trial_id, "")
    neutral_by_actor = {i["actor"]: i["trial_id"] for i in all_items if i["label"] == "Neutral"}
    items = select_items(all_items, args.max_items or 3)
    handle = load_handle(args.model, args.dtype)
    inputs_wanted = [s.strip() for s in args.inputs.split(",") if s.strip()]
    rows = [delivery_for_item(handle, load_clip(i, records, neutral_by_actor), inputs_wanted) for i in items]

    usable = [r for r in records.values() if r.get("face_width_px") and r.get("face_box")]
    verdicts: Dict[str, Any] = {}
    for name in inputs_wanted:
        checked = [r[name] for r in rows if "failed" not in r.get(name, {"failed": 1})]
        if not checked:
            verdicts[name] = {"pass": False, "reason": "no item could be built"}
            continue
        across = float(np.median([c["tokens_across_view"] for c in checked]))
        faces = []
        for record in usable:
            view = record["face_box"][2] - record["face_box"][0] if INPUTS[name]["view"] == "face" else record["width"]
            faces.append(across * record["face_width_px"] / view)
        gates = {
            "gate_1_median_tokens_across_face": round(float(np.median(faces)), 2),
            "gate_1_min_tokens_across_face": round(float(np.min(faces)), 2),
            "gate_1_share_clips_at_least_8": round(float(np.mean(np.asarray(faces) >= MIN_TOKENS_ACROSS_FACE)), 4),
            "gate_2": all(c["gate_2_every_frame_changes_output"] for c in checked),
            "gate_3": all(c["gate_3_logged_indices_are_the_frames_sent"] for c in checked),
            "gate_4": all(c["gate_4_ids_unchanged_within_clip"] for c in checked)
            and len({c["input_ids_sha256"] for c in checked}) == 1,
            "gate_5_shuffle_order": all(order_check(c) for c in checked),
        }
        gate_1 = gates["gate_1_median_tokens_across_face"] >= MIN_TOKENS_ACROSS_FACE
        verdicts[name] = {
            **gates,
            "gate_1": bool(gate_1),
            "pass": bool(gate_1 and gates["gate_2"] and gates["gate_3"] and gates["gate_4"] and gates["gate_5_shuffle_order"]),
        }
    output = {
        "model": args.model,
        "n_items_checked": len(rows),
        "n_clips_for_face_width": len(usable),
        "landmark_clips_with_failures": geometry.get("n_clips_with_failures"),
        "gate_1_rule": "median over clips of tokens across the face at least 8",
        "verdicts": verdicts,
        "items": rows,
    }
    args.reproduction.mkdir(parents=True, exist_ok=True)
    path = args.reproduction / f"input_delivery_{args.model}.json"
    path.write_text(json.dumps(output, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "verdicts": verdicts}, indent=1))


def baseline_cells(handle: ModelHandle, clip: Clip, inputs_wanted: Sequence[str]) -> List[Dict[str, Any]]:
    """Prior of the blind readout on each input: pixels removed, and a uniform grey image."""
    from scripts.vision_gates import strip_pixels

    rows = []
    for name in inputs_wanted:
        try:
            media = build_media(clip, name, "original")
        except si.EditFailure as exc:
            rows.append({"input_name": name, "condition": "image_free", "status": "failed", "reason": exc.reason})
            continue
        inputs = encode_media(handle, media, BLIND_PROMPT)
        free = strip_pixels(inputs, PIXEL_KEYS)
        scores = score_sequences(handle, free, handle.blind_prefill, handle.word_ids)
        grey = Media(media.kind, [np.full_like(a, 128) for a in media.images], [], {})
        grey_scores = score_sequences(handle, encode_media(handle, grey, BLIND_PROMPT), handle.blind_prefill, handle.word_ids)
        for condition, values in (("image_free", scores), ("grey", grey_scores)):
            rows.append(
                {
                    "input_name": name,
                    "condition": condition,
                    "status": "ok",
                    "family_logprob": {k: round(v, 4) for k, v in family_scores(handle.words, values).items()},
                    "lexicon_mass": round(math.exp(logsumexp(values)), 5),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Step 2: run
# ---------------------------------------------------------------------------


def done_cells(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((row["trial_id"], row["input_name"], row["condition"]))
    return done


def step_run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    if args.max_items is None and not args.confirm_gates_passed:
        raise SystemExit("Full runs wait on G2 to G4 and RQ0. Pass --max_items for a smoke run.")
    args.n_shards = resolve_shards(args.shard, args.n_shards, args.max_items is None)
    check_geometry(args.geometry)
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    geometry_items = geometry["items"]
    all_items = load_items(args.manifest, args.lookup, args.data_root)
    paths = {i["trial_id"]: i["path"] for i in all_items}
    for trial_id, record in geometry_items.items():
        record["path"] = paths.get(trial_id, "")
    neutral_by_actor = {i["actor"]: i["trial_id"] for i in all_items if i["label"] == "Neutral"}
    items = select_items(all_items, args.max_items)
    if args.n_shards > 1:
        items = [i for n, i in enumerate(items) if n % args.n_shards == args.shard]

    inputs_wanted = [s.strip() for s in args.inputs.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}{'_smoke' if args.max_items is not None else ''}"
    suffix = f"_shard{args.shard}" if args.n_shards > 1 else ""
    cells_path = args.output_dir / f"cells_{stem}{suffix}.jsonl"
    report_path = args.output_dir / f"run_{stem}{suffix}.json"
    done = done_cells(cells_path)
    design_sha256 = design_hash()

    started = time.time()
    handle = load_handle(args.model, args.dtype)
    report: Dict[str, Any] = {
        "model": args.model,
        "smoke": args.max_items is not None,
        "design_sha256": design_sha256,
        "shard": args.shard,
        "n_shards": args.n_shards,
        "blind_prompt": BLIND_PROMPT,
        "blind_prefill": BLIND_PREFILL,
        "k": si.N_SENT_FRAMES,
        "lexicon_token_lengths": dict(Counter(len(i) for i in handle.word_ids)),
        "n_prefixes_blind": len(internal_prefixes(handle.word_ids)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_load_seconds": round(time.time() - started, 1),
        "delivery": [],
        "baselines": None,
        "failures": [],
    }

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    save()
    n_ok = 0
    timing: Dict[str, List[float]] = defaultdict(list)
    with cells_path.open("a", encoding="utf-8") as sink:
        for count, item in enumerate(items):
            base = {
                "trial_id": item["trial_id"],
                "actor": item["actor"],
                "label": item["label"],
                "human_modal_label": item["human_modal_label"],
                "weak_label": item["weak_label"],
                "position": item["position"],
                "design_sha256": design_sha256,
            }
            clip: Optional[Clip] = None
            clip_error = ""
            try:
                clip = load_clip(item, geometry_items, neutral_by_actor)
            except si.EditFailure as exc:
                clip_error = exc.reason
            for name in inputs_wanted:
                for condition in INPUTS[name]["conditions"]:
                    if (item["trial_id"], name, condition) in done:
                        continue
                    row = {**base, "input_name": name, "condition": condition}
                    try:
                        if clip is None:
                            raise si.EditFailure(clip_error)
                        cell = score_cell(handle, build_media(clip, name, condition), item)
                        row.update(cell)
                        n_ok += 1
                        timing[name].append(cell["seconds"])
                    except si.EditFailure as exc:
                        row.update({"status": "failed", "reason": exc.reason, "detail": exc.detail})
                        report["failures"].append(
                            {k: row[k] for k in ("trial_id", "input_name", "condition", "reason")}
                        )
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
            if clip is not None and count < args.n_delivery_items:
                report["delivery"].append(delivery_for_item(handle, clip, inputs_wanted))
                if report["baselines"] is None:
                    report["baselines"] = baseline_cells(handle, clip, inputs_wanted)
                save()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            report.update(
                {
                    "items_finished": count + 1,
                    "n_cells_ok": n_ok,
                    "n_cells_failed": len(report["failures"]),
                    "failure_reasons": dict(Counter(f["reason"] for f in report["failures"])),
                    "mean_seconds_per_cell": {k: round(float(np.mean(v)), 3) for k, v in timing.items()},
                    "forwards_total": handle.forwards,
                    "cached_scoring": handle.cache_checks,
                    "seconds": round(time.time() - started, 1),
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    if torch.cuda.is_available()
                    else None,
                }
            )
            save()
    print(f"Done: {cells_path} {report_path}")


# ---------------------------------------------------------------------------
# Step 3: prompt coverage
# ---------------------------------------------------------------------------


def step_probe_prompt(args: argparse.Namespace) -> None:
    import torch

    if args.max_items is None:
        raise SystemExit("probe_prompt runs on smoke items only.")
    geometry_items = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    all_items = load_items(args.manifest, args.lookup, args.data_root)
    paths = {i["trial_id"]: i["path"] for i in all_items}
    for trial_id, record in geometry_items.items():
        record["path"] = paths.get(trial_id, "")
    neutral_by_actor = {i["actor"]: i["trial_id"] for i in all_items if i["label"] == "Neutral"}
    clips = [load_clip(i, geometry_items, neutral_by_actor) for i in select_items(all_items, args.max_items)]
    handle = load_handle(args.model, args.dtype)
    media = [build_media(c, "single_face", "original") for c in clips]
    grey = Media("image", [np.full_like(media[0].images[0], 128)], [], {})
    rows = []
    for prompt, prefill in BLIND_CANDIDATES:
        ids = list(handle.tokenizer(prefill, add_special_tokens=False).input_ids)
        masses = []
        for sent in media:
            inputs = encode_media(handle, sent, prompt)
            masses.append(math.exp(logsumexp(score_sequences(handle, inputs, ids, handle.word_ids))))
        inputs = encode_media(handle, grey, prompt)
        grey_mass = math.exp(logsumexp(score_sequences(handle, inputs, ids, handle.word_ids)))
        top = torch.topk(step_logprobs(handle, encode_media(handle, media[0], prompt), ids), 6)
        rows.append(
            {
                "prompt": prompt,
                "prefill": prefill,
                "lexicon_mass_items": [round(m, 5) for m in masses],
                "lexicon_mass_grey": round(grey_mass, 5),
                "top_next_tokens_first_item": [
                    [handle.tokenizer.decode([int(i)]), round(float(v), 2)] for v, i in zip(top.values, top.indices)
                ],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"prompt_probe_{args.model}.json"
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# Cache parity: where does a gap between cached and full forwards come from?
# ---------------------------------------------------------------------------


def step_cache_probe(args: argparse.Namespace) -> None:
    """Token-level gap between cached and full scoring, one dtype per run. Scores, no labels."""
    from scripts.prompts import build_4afc_prompt

    geometry_items = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    all_items = load_items(args.manifest, args.lookup, args.data_root)
    paths = {i["trial_id"]: i["path"] for i in all_items}
    for trial_id, record in geometry_items.items():
        record["path"] = paths.get(trial_id, "")
    neutral_by_actor = {i["actor"]: i["trial_id"] for i in all_items if i["label"] == "Neutral"}
    handle = load_handle(args.model, args.dtype)
    rows: List[Dict[str, Any]] = []
    for item in select_items(all_items, args.max_items or 3):
        clip = load_clip(item, geometry_items, neutral_by_actor)
        for name in ("single_face", "video_face"):
            media = build_media(clip, name, "original")
            mcq = build_4afc_prompt(item["options"], condition="video_only")
            option_ids = [list(handle.tokenizer(f" {o}", add_special_tokens=False).input_ids) for o in item["options"]]
            for readout, prompt, prefill, sequences in (
                ("blind", BLIND_PROMPT, handle.blind_prefill, handle.word_ids),
                ("mcq", mcq, handle.mcq_prefill, option_ids),
            ):
                inputs = encode_media(handle, media, prompt)
                needed = needed_tokens(sequences)
                prefixes = internal_prefixes(sequences)
                full = full_tables(handle, inputs, prefill, prefixes, needed)
                again = full_tables(handle, inputs, prefill, prefixes, needed)
                cached = cached_tables(handle, inputs, prefill, prefixes, needed)
                gaps = sorted(
                    (
                        (abs(cached[p][t] - full[p][t]), len(p), handle.tokenizer.decode(list(p) + [t]))
                        for p in full
                        for t in full[p]
                    ),
                    reverse=True,
                )
                by_depth = {
                    str(d): round(max([g[0] for g in gaps if g[1] == d] or [0.0]), 6)
                    for d in sorted({g[1] for g in gaps})
                }
                values = np.array([g[0] for g in gaps])
                rows.append(
                    {
                        "trial_id": item["trial_id"],
                        "route": name,
                        "readout": readout,
                        "n_tokens_scored": len(gaps),
                        "max_gap": round(float(values.max()), 6),
                        "mean_gap": round(float(values.mean()), 6),
                        "max_gap_by_prefix_length": by_depth,
                        "repeat_full_max_gap": round(
                            max(abs(again[p][t] - full[p][t]) for p in full for t in full[p]), 6
                        ),
                        "worst_tokens": [[round(g[0], 5), g[1], g[2]] for g in gaps[:5]],
                    }
                )
                args.reproduction.mkdir(parents=True, exist_ok=True)
                path = args.reproduction / f"cache_parity_{args.model}_{args.dtype}.json"
                path.write_text(json.dumps({"model": args.model, "dtype": args.dtype, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("route", "readout", "max_gap", "mean_gap", "max_gap_by_prefix_length", "repeat_full_max_gap")} for r in rows], indent=0))


# ---------------------------------------------------------------------------
# Step 4: processor probe, no weights and no clips
# ---------------------------------------------------------------------------


def _noise_frames(k: int, height: int = 1080, width: int = 1920) -> List[np.ndarray]:
    rng = np.random.default_rng(si.SEED)
    return [rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) for _ in range(k)]


def probe_one(model_key: str, processor: Any, frames: List[np.ndarray], route: str) -> Dict[str, Any]:
    from PIL import Image

    from scripts.multi_frame import prepare_images_for_model

    prompt = BLIND_PROMPT
    if route == "video":
        out = dict(encode_video(model_key, processor, frames, prompt))
    else:
        images = [Image.fromarray(f) for f in frames]
        payload, meta = (images[0], {}) if route == "image" else prepare_images_for_model(model_key, images)
        n = len(payload) if isinstance(payload, list) else 1
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in range(n)]
        content.append({"type": "text", "text": prompt})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        out = dict(processor(images=payload, text=text, return_tensors="pt"))
        out["multi_frame_meta"] = meta
    ids = out["input_ids"][0]
    uniques, counts = np.unique(ids.numpy(), return_counts=True)
    frequent = {int(u): int(c) for u, c in zip(uniques, counts) if c > 3}
    return {
        "shapes": {k: list(v.shape) for k, v in out.items() if hasattr(v, "shape")},
        "n_tokens": int(ids.shape[0]),
        "repeated_token_ids": frequent,
        "repeated_token_names": {str(k): processor.tokenizer.decode([k]) for k in frequent},
        "multi_frame_meta": out.get("multi_frame_meta"),
    }


def step_probe(args: argparse.Namespace) -> None:
    from transformers import AutoProcessor

    from scripts.evaluate import resolve_model_path

    processor = AutoProcessor.from_pretrained(resolve_model_path(args.model), trust_remote_code=True)
    frames = _noise_frames(si.N_SENT_FRAMES)
    report: Dict[str, Any] = {"model": args.model, "k": si.N_SENT_FRAMES, "routes": {}}
    for route, sent in (("image", frames[:1]), ("legacy", frames), ("video", frames)):
        try:
            report["routes"][route] = probe_one(args.model, processor, sent, route)
        except Exception as exc:  # recorded: a route that fails is itself the finding
            report["routes"][route] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"processor_probe_{args.model}.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--step", required=True, choices=("landmarks", "run", "delivery", "cache_probe", "probe_prompt", "probe_processor"))
    parser.add_argument("--model", default="gemma4", choices=("gemma4", "qwen3vl", "molmo2"))
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--data_root", type=Path, default=Path("data/eu_emotions"))
    parser.add_argument("--geometry", type=Path, default=Path("results/stimulus_interventions/geometry.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/stimulus_interventions"))
    parser.add_argument("--reproduction", type=Path, default=Path("research/reproduction"))
    parser.add_argument("--inputs", default="single_face,single_full,video_face,legacy_full")
    parser.add_argument("--max_items", type=int, default=None, help="Smoke only: first N items in fixed order.")
    parser.add_argument("--n_delivery_items", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=None, help="Required for a full run.")
    parser.add_argument("--confirm_gates_passed", action="store_true")
    args = parser.parse_args()
    steps = {
        "landmarks": step_landmarks,
        "run": step_run,
        "delivery": step_delivery,
        "cache_probe": step_cache_probe,
        "probe_prompt": step_probe_prompt,
        "probe_processor": step_probe,
    }
    steps[args.step](args)


if __name__ == "__main__":
    main()

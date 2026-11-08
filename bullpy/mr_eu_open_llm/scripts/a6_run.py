"""
Run step of draft amendment A6 (RQ5, shared errors on CREMA-D). See research/PREREG_A6_DRAFT.md.

Steps
  landmarks    FaceMesh on the 819 clips. Numeric geometry only. Needs MediaPipe (au_tools).
  decoder      cv2 version and stable .flv frames, on three clips including a short h264 one.
  au_manifest  The clip list for research/scripts/extract_au_measures.py, unchanged, for the
               human cue check. No model.
  baselines    image_free, grey and text primes, and the delivery rows, for one model.
  run          One stage for one model. Stage 1: six orders on single_face, the blind readout
               on originals, and the video_face bridge. Stage 3: every condition at both edit
               orders, and the blind conditions, for a model whose locked Stage 2 reaches it.

Frames are decoded and edited in memory. No frame, crop or edited clip is written. A run is
refused until every PENDING pin is filled, --a6_frozen and --confirm_gates_passed are passed,
and --n_shards is stated. --mechanics_only N needs no freeze, and writes only timings and
reasons: it goes through the whole path and writes no score, as in A3.

Frozen A2 and A3 code is imported unchanged: read_frames, landmarks, align, fills, blend, crop,
fixed_size, load_clip, score_blind, encode_media and score_sequences from A2, and score_cell,
read_cells, partners_for and mechanics_row from A3. REGIONS is a frozen constant and cannot take
a new key, so region_mask here matches si.region_mask line for line, and a test holds them equal.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import a6_design as d  # noqa: E402
from scripts import face_input_validity as fiv  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402

D = d.DESIGN
OPTIONS: Tuple[str, ...] = tuple(D["options"])
CODE_TO_OPTION = {"A": "Anger", "D": "Disgust", "F": "Fear", "H": "Happy", "N": "No Emotion", "S": "Sad"}
PRIMARY, BRIDGE = D["primary_route"], D["bridge_route"]
EDIT_ROWS = tuple(D["edit_order_rows"])
SHORT_DECODE_CLIPS = tuple(D["runtime"]["short_decode_clips"])  # CREMA_D_PLAN.md
NAMED_RINGS = {"A2 eyes rings": si.REGIONS["eyes"], "A2 brows rings": si.REGIONS["brows"], "A2 mouth ring": si.REGIONS["mouth"]}
OUTCOME_DIR = d.OUTPUT_DIR
MECHANICS_DIR = Path("results/a6_mechanics")
REQUIRED_CV2 = D["runtime"]["cv2_prefix"]
DTYPE = D["runtime"]["dtype"]


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------


def load_items(selection: Path, data_root: Path) -> List[Dict[str, Any]]:
    """The 819 selected clips. A clip's mode counts as intended if the intended label ties for top."""
    with selection.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    items = []
    for row in rows:
        votes = {CODE_TO_OPTION[c]: int(row[f"votes_{c}"]) for c in CODE_TO_OPTION}
        label = CODE_TO_OPTION[row["emotion"]]
        top = max(votes.values())
        items.append({
            "trial_id": row["clip_id"],
            "actor": row["actor_id"],
            "label": label,
            "options": list(OPTIONS),
            "human_votes": votes,
            "human_p_target": votes[label] / sum(votes.values()),
            "human_modal_label": max(sorted(votes), key=lambda k: votes[k]),
            "human_modal_correct": bool(votes[label] == top),
            "sex": row["sex"],
            "weak_label": False,
            "path": str(data_root / f"{row['clip_id']}.flv"),
        })
    if len(items) != D["n_items"] or len({i["actor"] for i in items}) != D["n_actors"]:
        raise SystemExit(f"Expected {D['n_items']} clips of {D['n_actors']} actors in {selection}.")
    return items


def partners_for(items: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """A3's rule, seed 42, on these clips. A clip's mismatch cell is its partner's real cell."""
    return fiv.partners_for(sorted(items, key=lambda i: i["trial_id"]))


def neutral_donors(items: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Each actor's donor: the first Neutral clip by clip id."""
    out: Dict[str, str] = {}
    for item in sorted(items, key=lambda i: i["trial_id"]):
        if item["label"] == "No Emotion":
            out.setdefault(item["actor"], item["trial_id"])
    return out


# ---------------------------------------------------------------------------
# Regions: the same steps as si.region_mask, on rings chosen here
# ---------------------------------------------------------------------------


def region_rings(region: str, regions: Optional[Mapping[str, Any]] = None) -> Tuple[Tuple[int, ...], ...]:
    """The landmark rings of a design-block region, resolving nested names."""
    table = D["regions"] if regions is None else regions
    entry = table[region]
    if isinstance(entry, Mapping):
        return (tuple(entry["hull"]),)
    rings: List[Tuple[int, ...]] = []
    for name in entry:
        rings.extend(NAMED_RINGS[name] if name in NAMED_RINGS else region_rings(name, table))
    return tuple(rings)


def hard_region(size_hw: Tuple[int, int], landmarks: np.ndarray, rings: Sequence[Sequence[int]], dilate: float) -> np.ndarray:
    height, width = size_hw
    span = si.eye_span(landmarks)
    hard = np.zeros((height, width), dtype=np.uint8)
    for ring in rings:
        hull = cv2.convexHull(np.round(landmarks[list(ring)]).astype(np.int32))
        cv2.fillConvexPoly(hard, hull, 255)
    radius = max(1, int(round(dilate * span)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(hard, kernel)


def region_mask(size_hw: Tuple[int, int], landmarks: np.ndarray, rings: Sequence[Sequence[int]], dilate: float) -> np.ndarray:
    """Soft mask in 0..1 over landmark rings, sized by the face. As si.region_mask."""
    hard = hard_region(size_hw, landmarks, rings, dilate)
    sigma = max(0.5, si.FEATHER_FRACTION * si.eye_span(landmarks))
    return cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (0, 0), sigma)


def narrowest_side(hard: np.ndarray) -> float:
    """Short side in pixels of the tightest box around the thinnest connected part of a region."""
    count, labels = cv2.connectedComponents((hard > 0).astype(np.uint8))
    sides = []
    for part in range(1, count):
        ys, xs = np.nonzero(labels == part)
        points = np.column_stack([xs, ys]).astype(np.float32)
        (_, (width, height), _) = cv2.minAreaRect(points)
        sides.append(min(width, height) + 1.0)
    return float(min(sides)) if sides else 0.0


def edit_media(clip: run.Clip, condition: str) -> run.Media:
    """One A6 condition on the apex frame, then A2's crop and size. Nothing is written."""
    spec = d.condition_spec(condition)
    if spec["kind"] == "none":
        return run.build_media(clip, PRIMARY, "original")
    if clip.apex is None:
        raise si.EditFailure("no_apex_frame")
    frame, landmarks = clip.frames[clip.apex], clip.landmarks.get(clip.apex)
    if landmarks is None:
        raise si.EditFailure("no_face_detected")
    if clip.geometry.face_box is None:
        raise si.EditFailure("no_face_box")
    region = D["sham_region"] if spec["region"] == "SHAM" else spec["region"]
    rings = region_rings(region)
    dilate = float(D["region_dilate_fraction"])
    size = frame.shape[:2]
    alpha = region_mask(size, landmarks, rings, dilate)
    fill, log = si._fill_image(frame, landmarks, spec["fill"], clip.geometry)
    image = si.fixed_size(si.crop(si.blend(frame, fill, alpha), clip.geometry.face_box))
    span = narrowest_side(hard_region(size, landmarks, rings, dilate))
    box = clip.geometry.face_box
    log = {**log, "kind": "mask", "region": region, "view": "face", "box": list(box),
           "mask_fraction": round(float(alpha.mean()), 5), "span_px": round(span, 1),
           "span_view_fraction": round(span / (box[2] - box[0]), 5)}
    if log.get("fill") == "neutral":
        log["neutral_donor"] = clip.geometry.notes.get("neutral_donor")
    if log.get("fill") == "first":
        log["donor_is_sent_frame"] = clip.apex == clip.record["base_indices"][0]
    return run.Media("image", [image], [clip.apex], log)


def build(clip: run.Clip, route: str, condition: str) -> run.Media:
    return edit_media(clip, condition) if route == PRIMARY else run.build_media(clip, route, condition)


# ---------------------------------------------------------------------------
# What a stage scores, with no model in it
# ---------------------------------------------------------------------------


def cell_key(cell: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (cell["stage"], cell["trial_id"], cell["route"], cell["readout"], cell["order"], cell["condition"])


def stage_plan(item: Mapping[str, Any], stage: int) -> List[Dict[str, Any]]:
    """Every cell of one clip in one stage."""
    base = {"stage": stage, "trial_id": item["trial_id"], "actor": item["actor"], "label": item["label"],
            "human_modal_label": item["human_modal_label"], "face_item": item["trial_id"]}

    def mcq(route: str, row: int, condition: str) -> Dict[str, Any]:
        return {**base, "route": route, "readout": "mcq", "order": row, "condition": condition, "options": list(D["orders"][row])}

    def blind(condition: str) -> Dict[str, Any]:
        return {**base, "route": PRIMARY, "readout": "blind", "order": None, "condition": condition}

    if stage == 1:
        return ([mcq(PRIMARY, row, "original") for row in range(len(D["orders"]))] + [blind("original")]
                + [mcq(BRIDGE, row, "original") for row in EDIT_ROWS])
    if stage == 3:
        if item["label"] not in D["stage3_emotions"]:
            return []
        return [mcq(PRIMARY, row, c) for c in D["conditions"] for row in EDIT_ROWS]
    raise ValueError(f"Stage {stage} has no cells.")


def prime_prompt(word_label: str, order: Sequence[str]) -> str:
    """A text prime: the fixed sentence, then the MCQ text, with no image."""
    from scripts.prompts import build_4afc_prompt

    word = D["prime_words"][word_label]
    return D["prime_prompt_prefix"].format(word=word) + D["prime_separator"] + build_4afc_prompt(list(order), condition="video_only")


# ---------------------------------------------------------------------------
# Scoring on a real model
# ---------------------------------------------------------------------------


def score_one(handle: Any, cell: Dict[str, Any], media_for: Callable[[str, str], run.Media]) -> Dict[str, Any]:
    """Score one planned cell in place. A face that cannot be built fails it with its reason."""
    route, condition = cell["route"], cell["condition"]
    if cell["readout"] == "mcq":
        scored = fiv.score_cell(handle, cell, lambda _trial: media_for(route, condition))
        if scored["status"] == "ok":
            media = media_for(route, condition)
            scored.update(edit=media.log, media_sha256=[si.array_sha256(a) for a in media.images])
        return scored
    start, tick = handle.forwards, time.time()
    try:
        media = media_for(route, condition)
        scores, _inputs = run.score_blind(handle, media)
        cell.update(status="ok", blind=scores, sent_frame_indices=media.sent_indices, edit=media.log,
                    media_sha256=[si.array_sha256(a) for a in media.images])
    except si.EditFailure as exc:
        cell.update(status="failed", reason=exc.reason)
    cell.update(seconds=round(time.time() - tick, 3), forwards=handle.forwards - start)
    return cell


def media_cache(clip: Optional[run.Clip], error: str) -> Callable[[str, str], run.Media]:
    """Edits for one clip, built once each, held in memory until the clip is finished."""
    cache: Dict[Tuple[str, str], Any] = {}

    def media_for(route: str, condition: str) -> run.Media:
        if clip is None:
            raise si.EditFailure(error)
        key = (route, condition)
        if key not in cache:
            try:
                cache[key] = build(clip, route, condition)
            except si.EditFailure as exc:
                cache[key] = exc
        value = cache[key]
        if isinstance(value, si.EditFailure):
            raise value
        return value

    return media_for


def encode_text(handle: Any, prompt: str) -> Dict[str, Any]:
    """A prompt with no image at all, through the model's own chat template."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = handle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = handle.processor(text=text, return_tensors="pt")
    return {k: v.to(handle.device) for k, v in dict(raw).items() if hasattr(v, "to")}


def score_options(handle: Any, inputs: Dict[str, Any], options: Sequence[str]) -> List[float]:
    ids = [list(handle.tokenizer(f" {o}", add_special_tokens=False).input_ids) for o in options]
    return [round(v, 4) for v in run.score_sequences(handle, inputs, handle.mcq_prefill, ids)]


def baseline_rows(handle: Any, clip: run.Clip) -> List[Dict[str, Any]]:
    """image_free and grey on each order, text primes, and A2's two blind baselines."""
    from scripts.prompts import build_4afc_prompt
    from scripts.vision_gates import strip_pixels

    media = build(clip, PRIMARY, "original")
    grey = run.Media("image", [np.full_like(media.images[0], 128)], [], {})
    rows: List[Dict[str, Any]] = []
    for row, order in enumerate(D["orders"]):
        prompt = build_4afc_prompt(list(order), condition="video_only")
        inputs = run.encode_media(handle, media, prompt)
        free = strip_pixels(inputs, run.PIXEL_KEYS)
        rows.append({"readout": "mcq", "condition": "image_free", "order": row, "options": list(order),
                     "status": "ok", "joint_logprob": score_options(handle, free, order)})
        rows.append({"readout": "mcq", "condition": "grey", "order": row, "options": list(order),
                     "status": "ok", "joint_logprob": score_options(handle, run.encode_media(handle, grey, prompt), order)})
        for label in OPTIONS:
            text = encode_text(handle, prime_prompt(label, order))
            rows.append({"readout": "mcq", "condition": f"prime_{label}", "order": row, "options": list(order),
                         "status": "ok", "joint_logprob": score_options(handle, text, order)})
    for baseline in run.baseline_cells(handle, clip, [PRIMARY]):
        rows.append({**baseline, "readout": "blind", "order": None})
    return rows


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def geometry_items(path: Path, items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))["items"]
    paths = {i["trial_id"]: i["path"] for i in items}
    for trial, record in records.items():
        record["path"] = paths.get(trial, "")
    return records


def step_landmarks(args: argparse.Namespace) -> None:
    if args.geometry.exists():
        raise SystemExit(f"{args.geometry} is fixed for all models and is not overwritten.")
    items = load_items(args.selection, args.data_root)
    detect = run.FaceMesh()
    if detect.version != D["runtime"]["mediapipe_version"]:
        raise SystemExit(f"MediaPipe {detect.version}; the block pins {D['runtime']['mediapipe_version']}.")
    records: Dict[str, Any] = {}
    failures: List[Dict[str, Any]] = []
    started = time.time()
    for item in sorted(items, key=lambda i: i["trial_id"]):
        try:
            records[item["trial_id"]] = run.clip_geometry_record(item, detect)
        except si.EditFailure as exc:
            records[item["trial_id"]] = {"trial_id": item["trial_id"], "actor": item["actor"], "clip_failure": exc.reason}
        record = records[item["trial_id"]]
        if record.get("clip_failure") or record.get("box_failure") or record.get("failed_frames"):
            failures.append({"trial_id": item["trial_id"], "clip_failure": record.get("clip_failure"),
                             "box_failure": record.get("box_failure"), "failed_frames": record.get("failed_frames", [])})
    output = {"detector": f"mediapipe.face_mesh {detect.version} static_image_mode refine_landmarks",
              "k": si.N_SENT_FRAMES, "n_clips": len(records), "n_clips_with_failures": len(failures),
              "seconds": round(time.time() - started, 1), "failures": failures, "items": records}
    args.geometry.parent.mkdir(parents=True, exist_ok=True)
    unusable = [f for f in failures if f["clip_failure"] or f["box_failure"]
                or (records[f["trial_id"]].get("apex") is None)
                or (records[f["trial_id"]].get("landmarks") or {}).get(str(records[f["trial_id"]].get("apex"))) is None]
    if len(unusable) / len(items) > D["lost_item_limit"]:
        failed = args.geometry.with_name("geometry_failed.json")
        failed.write_text(json.dumps({k: output[k] for k in ("detector", "n_clips", "n_clips_with_failures", "failures")}) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(unusable)} clips have no usable apex, over lost_item_limit. Nothing to pin; see {failed}.")
    args.geometry.write_text(json.dumps(output) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("n_clips", "n_clips_with_failures", "seconds", "failures")}, indent=1))


def step_decoder(args: argparse.Namespace) -> None:
    """decoder_preflight.py reads EU-Emotion clips only, so this repeats its check on .flv."""
    sys.path.insert(0, str(d.ROOT / "research" / "scripts"))
    import decoder_preflight as preflight

    if not cv2.__version__.startswith(REQUIRED_CV2):
        raise SystemExit(f"cv2 {cv2.__version__}; need {REQUIRED_CV2}x")
    items = {i["trial_id"]: i for i in load_items(args.selection, args.data_root)}
    chosen = sorted(items)[:2] + [SHORT_DECODE_CLIPS[0]]
    for trial in chosen:
        path = Path(items[trial]["path"])
        indices = si.uniform_indices(0, run.count_frames(path), si.N_SENT_FRAMES)
        reads = []
        for _ in range(3):
            frames, _meta = si.read_frames(path, indices)
            reads.append([frames[i] for i in indices])
        problems = preflight.check_reads(reads)
        if problems:
            raise SystemExit(f"{trial}: {problems}")
    print(f"Decoder ok on {chosen}, cv2 {cv2.__version__}")


def step_au_manifest(args: argparse.Namespace) -> None:
    items = load_items(args.selection, args.data_root)
    trials = [{"trial_id": i["trial_id"], "stimulus_path": f"{i['trial_id']}.flv"} for i in sorted(items, key=lambda x: x["trial_id"])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "au_manifest.json").write_text(json.dumps({"trials": trials}, indent=0) + "\n", encoding="utf-8")


def weights_manifest(model: str) -> str:
    """sha256 over each weight and config file of the model, by relative path, in sorted order."""
    import hashlib

    from scripts.evaluate import resolve_model_path

    root = Path(resolve_model_path(model))
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                file_digest.update(block)
        digest.update(f"{path.relative_to(root)}\t{file_digest.hexdigest()}\n".encode("utf-8"))
    return digest.hexdigest()


def step_weights(args: argparse.Namespace) -> None:
    print(json.dumps({"model": args.model, "weights_sha256": weights_manifest(args.model)}))


def check_weights(model: str) -> str:
    found = weights_manifest(model)
    if found != D["weights_sha256"][model]:
        raise SystemExit(f"{model} weights are {found}, not the pinned {D['weights_sha256'][model]}.")
    return found


def check_run_allowed(args: argparse.Namespace) -> None:
    if not args.a6_frozen:
        raise SystemExit("A6 is not frozen. This step will not run until the amendment is frozen.")
    if not args.confirm_gates_passed:
        raise SystemExit("Full runs wait on G1 to G4 and the wait in the draft. Pass --confirm_gates_passed.")
    d.check_frozen(D)
    d.check_pins(D)
    d.check_human_target(D)
    d.check_geometry(D, args.geometry)
    if args.dtype != DTYPE:
        raise SystemExit(f"Runs use {DTYPE}, the block's dtype.")
    check_weights(args.model)


def stage3_allowed(args: argparse.Namespace) -> str:
    """Stage 3 needs this model's locked Stage 2, on this design, reaching it."""
    if args.stage2_output is None or not args.stage2_output.exists():
        raise SystemExit("Stage 3 needs --stage2_output, this model's locked Stage 2.")
    locked = json.loads(args.stage2_output.read_text(encoding="utf-8"))
    if locked.get("model") != args.model or locked.get("design_sha256") != d.block_hash():
        raise SystemExit("That Stage 2 output is for another model or design.")
    if locked.get("stage2", {}).get("reaches_stage3") is not True:
        raise SystemExit(f"{args.model} does not reach Stage 3.")
    import hashlib

    return hashlib.sha256(args.stage2_output.read_bytes()).hexdigest()


def resume_keys(path: Path, design_sha256: str) -> set:
    """Cut a truncated tail, as A3 does, and return the cells already done on this design."""
    state = fiv.read_cells(path)
    if state["n_partial_dropped"]:
        with path.open("r+b") as handle:
            handle.truncate(state["good_bytes"])
    if any(r.get("design_sha256") != design_sha256 for r in state["rows"]):
        raise SystemExit(f"{path} holds cells from another design.")
    return {cell_key(r) for r in state["rows"]}


def step_baselines(args: argparse.Namespace) -> None:
    items = load_items(args.selection, args.data_root)
    records = geometry_items(args.geometry, items)
    donors = neutral_donors(items)
    ordered = run.select_items(items, None)
    handle = run.load_handle(args.model, args.dtype)
    clip = run.load_clip(ordered[0], records, donors)
    report: Dict[str, Any] = {"model": args.model, "design_sha256": d.block_hash(), "dtype": args.dtype,
                              "weights_sha256": D["weights_sha256"][args.model], "versions": versions(),
                              "baselines": baseline_rows(handle, clip), "delivery": []}
    for item in ordered[: args.n_delivery_items]:
        report["delivery"].append(run.delivery_for_item(handle, run.load_clip(item, records, donors), [PRIMARY, BRIDGE]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"baselines_{args.model}.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")


def step_run(args: argparse.Namespace) -> None:
    import torch

    check_run_allowed(args)
    stage2_sha256 = stage3_allowed(args) if args.stage == 3 else None
    args.n_shards = run.resolve_shards(args.shard, args.n_shards, True)
    design_sha256 = d.block_hash()
    items = load_items(args.selection, args.data_root)
    records = geometry_items(args.geometry, items)
    donors = neutral_donors(items)
    ordered = [i for n, i in enumerate(run.select_items(items, None)) if n % args.n_shards == args.shard]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard}" if args.n_shards > 1 else ""
    cells_path = args.output_dir / f"cells_{args.model}_stage{args.stage}{suffix}.jsonl"
    report_path = args.output_dir / f"run_{args.model}_stage{args.stage}{suffix}.json"
    done = resume_keys(cells_path, design_sha256)
    started = time.time()
    handle = run.load_handle(args.model, args.dtype)
    report: Dict[str, Any] = {
        "model": args.model, "stage": args.stage, "mechanics": False, "design_sha256": design_sha256,
        "shard": args.shard, "n_shards": args.n_shards, "failures": [], "stage2_sha256": stage2_sha256,
        "dtype": args.dtype, "weights_sha256": D["weights_sha256"][args.model], "versions": versions(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    save()
    with cells_path.open("a", encoding="utf-8") as sink:
        for count, item in enumerate(ordered):
            plan = [c for c in stage_plan(item, args.stage) if cell_key(c) not in done]
            if plan:
                clip, error = None, ""
                try:
                    clip = run.load_clip(item, records, donors)
                except si.EditFailure as exc:
                    error = exc.reason
                media_for = media_cache(clip, error)
                for cell in plan:
                    scored = score_one(handle, {**cell, "model": args.model, "design_sha256": design_sha256}, media_for)
                    if scored["status"] != "ok":
                        report["failures"].append({k: scored.get(k) for k in ("trial_id", "route", "condition", "reason")})
                    sink.write(json.dumps(scored) + "\n")
                sink.flush()
            report.update(items_finished=count + 1, seconds=round(time.time() - started, 1),
                          failure_reasons=dict(Counter(f["reason"] for f in report["failures"])))
            save()
    print(f"Done: {cells_path}")


def versions() -> Dict[str, str]:
    import platform

    import torch
    import transformers

    return {"transformers": transformers.__version__, "torch": torch.__version__,
            "python": platform.python_version(), "cv2": cv2.__version__, "numpy": np.__version__}


def mechanics_output_dir(path: Path) -> Path:
    resolved, outcomes = path.resolve(), OUTCOME_DIR.resolve()
    if resolved == outcomes or outcomes in resolved.parents:
        raise SystemExit("Mechanics output must stay outside results/a6.")
    return path


def step_mechanics(args: argparse.Namespace) -> None:
    """Every arm on N clips of the Stage 3 emotions. Writes timings and reasons, and no score."""
    output_dir = mechanics_output_dir(args.mechanics_dir)
    items = load_items(args.selection, args.data_root)
    records = geometry_items(args.geometry, items)
    donors = neutral_donors(items)
    wanted = [i for i in run.select_items(items, None) if i["label"] in D["stage3_emotions"]][: args.mechanics_only]
    handle = run.load_handle(args.model, args.dtype)
    rows: List[Dict[str, Any]] = []
    for item in wanted:
        media_for = media_cache(run.load_clip(item, records, donors), "")
        for stage in (1, 3):
            for cell in stage_plan(item, stage):
                rows.append(fiv.mechanics_row(score_one(handle, dict(cell), media_for)))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"mechanics_{args.model}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ok = [r for r in rows if r["status"] == "ok"]
    print(json.dumps({"model": args.model, "n_cells": len(rows), "n_ok": len(ok),
                      "reasons": dict(Counter(r.get("reason") for r in rows if r["status"] != "ok")),
                      "mean_seconds_per_cell": round(float(np.mean([r["seconds"] for r in ok])), 3) if ok else None}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--step", required=True, choices=("landmarks", "decoder", "au_manifest", "weights", "baselines", "run"))
    parser.add_argument("--model", choices=tuple(D["models"]))
    parser.add_argument("--stage", type=int, choices=(1, 3))
    parser.add_argument("--selection", type=Path, default=d.SELECTION)
    parser.add_argument("--data_root", type=Path, default=Path("data/crema_d"))
    parser.add_argument("--geometry", type=Path, default=d.GEOMETRY)
    parser.add_argument("--output_dir", type=Path, default=OUTCOME_DIR)
    parser.add_argument("--stage2_output", type=Path, default=None)
    parser.add_argument("--dtype", default=DTYPE, choices=("bfloat16", "float32"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=None, help="Required for a run.")
    parser.add_argument("--n_delivery_items", type=int, default=2)
    parser.add_argument("--a6_frozen", action="store_true")
    parser.add_argument("--confirm_gates_passed", action="store_true")
    parser.add_argument("--mechanics_only", type=int, default=None, help="N clips, no scores written. No freeze needed.")
    parser.add_argument("--mechanics_dir", type=Path, default=MECHANICS_DIR)
    args = parser.parse_args()
    if args.step in ("landmarks", "decoder", "au_manifest"):
        {"landmarks": step_landmarks, "decoder": step_decoder, "au_manifest": step_au_manifest}[args.step](args)
        return
    if args.model is None:
        raise SystemExit("--model is required")
    if args.step == "weights":
        step_weights(args)
        return
    if args.step == "baselines":
        check_run_allowed(args)
        step_baselines(args)
        return
    if args.mechanics_only is not None:
        if args.a6_frozen or args.confirm_gates_passed:
            raise SystemExit("--mechanics_only cannot be combined with --a6_frozen or --confirm_gates_passed.")
        step_mechanics(args)
        return
    if args.stage is None:
        raise SystemExit("--stage is required")
    step_run(args)


if __name__ == "__main__":
    main()

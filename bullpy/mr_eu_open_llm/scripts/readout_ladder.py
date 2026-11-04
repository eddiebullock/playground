"""
Option-free readout ladder, research/PLAN_READOUT_AND_INPUT.md, problem 1.

Calibration clips are 24 CREMA-D videos outside the 819 analysis clips. Their emotion
code in the filename balances the set and is used for nothing else. No label, vote or
human response is read, and every pass rule below concerns coverage or image sensitivity.

Rules fixed before running:
- Template position. The next token at the generation prompt is logged. If it is a special
  token, the model expects structure there, and the prefill moves after an empty thought
  block on Gemma 4. Otherwise the position stands.
- Rung 1. A candidate passes coverage if lexicon mass >= 0.05 on >= 90 percent of clips.
- Rung 2. P(yes) + P(no) >= 0.90 on >= 90 percent of clips, over all clip-label readings
  taken per clip as the mean over labels.
- Rung 3. No coverage rule.
- Image sensitivity. A readout is a vector over 21 labels, centred within clip; distance
  is RMS difference. A pair passes if its distance exceeds both clips' distance to their
  own 1-pixel shifted copy, on >= 80 percent of all pairs. The plan gives no number for
  the grey comparison, so the same 80 percent is applied to clip against grey.
- If several rung 1 candidates pass, the one with the highest median lexicon mass is
  taken for that model. A shared prompt maximises the lowest median across models.
- Frames stay in memory. Output holds masses and distances, not readout vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts import run_stimulus_interventions as run
from scripts import stimulus_interventions as si

CALIBRATION_CSV = Path("research/reproduction/crema_calibration_clips.csv")
EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
SENTENCES = ("DFA", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI")
DEFECTS = frozenset({"1076_MTI_NEU_XX", "1076_MTI_SAD_XX", "1064_TIE_SAD_XX", "1064_IEO_DIS_MD"})
N_PER_EMOTION = 4
MASS_FLOOR, MASS_SHARE = 0.05, 0.90
YESNO_FLOOR, YESNO_SHARE = 0.90, 0.90
SENSITIVITY_SHARE = 0.80
YES_TOKENS = ("Yes", "yes", " Yes", " yes")
NO_TOKENS = ("No", "no", " No", " no")
YESNO_PROMPT = "Does this person look {label}? Answer yes or no."
LIKELIHOOD_PROMPT = "Describe this person's expression."
LIKELIHOOD_PREFILL = "This person looks"
GEMMA_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def select_calibration(excluded: Sequence[str]) -> List[str]:
    """Fixed ids: 4 per emotion, 24 different actors, none in the analysis set."""
    skip = set(excluded) | DEFECTS
    candidates = [
        f"{actor}_{sentence}_{emotion}_XX"
        for actor in range(1001, 1092)
        for sentence in SENTENCES
        for emotion in EMOTIONS
    ]
    candidates = sorted((c for c in candidates if c not in skip), key=lambda c: si.stable_seed("calibration", c))
    chosen: List[str] = []
    actors = set()
    counts = {e: 0 for e in EMOTIONS}
    for clip in candidates:
        actor, _sentence, emotion, _level = clip.split("_")
        if counts[emotion] < N_PER_EMOTION and actor not in actors:
            chosen.append(clip)
            actors.add(actor)
            counts[emotion] += 1
    return sorted(chosen)


def share_above(values: Sequence[float], floor: float) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64) >= floor))


def centred_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))


def image_sensitivity(readouts: np.ndarray, shifted: np.ndarray, grey: np.ndarray) -> Dict[str, Any]:
    """Both plan criteria from clip-by-label readouts. Exposes only distances."""
    n = readouts.shape[0]
    own = np.array([centred_distance(readouts[i], shifted[i]) for i in range(n)])
    wins = []
    between = []
    for i in range(n):
        for j in range(i + 1, n):
            distance = centred_distance(readouts[i], readouts[j])
            between.append(distance)
            wins.append(distance > max(own[i], own[j]))
    to_grey = np.array([centred_distance(readouts[i], grey) for i in range(n)])
    pair_share = float(np.mean(wins))
    grey_share = float(np.mean(to_grey > own))
    return {
        "n_clips": n,
        "n_pairs": len(wins),
        "median_distance_between_clips": round(float(np.median(between)), 4),
        "median_distance_to_shifted_copy": round(float(np.median(own)), 4),
        "median_distance_to_grey": round(float(np.median(to_grey)), 4),
        "share_pairs_above_shift": round(pair_share, 4),
        "share_clips_differ_from_grey": round(grey_share, 4),
        "pass": bool(pair_share >= SENSITIVITY_SHARE and grey_share >= SENSITIVITY_SHARE),
    }


def yes_no_mass(logprobs: Any, yes_ids: Sequence[int], no_ids: Sequence[int]) -> Tuple[float, float]:
    yes = sum(math.exp(float(logprobs[i])) for i in yes_ids)
    no = sum(math.exp(float(logprobs[i])) for i in no_ids)
    return yes, no


def choose_shared(medians: Dict[str, Dict[str, Optional[float]]]) -> Optional[str]:
    """Candidate passing on every model with the highest lowest median mass."""
    shared = None
    best = -1.0
    models = list(medians)
    for candidate in medians[models[0]]:
        values = [medians[m].get(candidate) for m in models]
        if all(v is not None for v in values) and min(values) > best:
            shared, best = candidate, min(values)
    return shared


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_select(args: argparse.Namespace) -> None:
    with args.selection.open(newline="", encoding="utf-8") as handle:
        excluded = [row["clip_id"] for row in csv.DictReader(handle)]
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    with args.calibration.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["clip_id"])
        writer.writerows([c] for c in select_calibration(excluded))


def calibration_items(path: Path, video_dir: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        ids = [row["clip_id"] for row in csv.DictReader(handle)]
    return [{"trial_id": c, "actor": c.split("_")[0], "label": "", "path": str(video_dir / f"{c}.flv")} for c in ids]


def step_geometry(args: argparse.Namespace) -> None:
    detect = run.FaceMesh()
    records = {}
    for item in calibration_items(args.calibration, args.video_dir):
        try:
            records[item["trial_id"]] = run.clip_geometry_record(item, detect)
        except si.EditFailure as exc:
            records[item["trial_id"]] = {"clip_failure": exc.reason}
        records[item["trial_id"]]["path"] = item["path"]
    args.geometry.parent.mkdir(parents=True, exist_ok=True)
    args.geometry.write_text(json.dumps({"items": records}) + "\n", encoding="utf-8")
    bad = [k for k, v in records.items() if v.get("clip_failure") or v.get("box_failure") or v.get("apex") is None]
    print(json.dumps({"n": len(records), "unusable": bad}, indent=1))


def face_views(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """The apex crop at the fixed size, and the same crop with its box moved by one pixel."""
    apex = record["apex"]
    frames, _ = si.read_frames(Path(record["path"]), [apex])
    run.check_decoder(record, frames)
    frame = frames[apex]
    box = tuple(record["face_box"])
    x0, y0, x1, y1 = box
    width = frame.shape[1]
    step = 1 if x1 < width else -1
    moved = (x0 + step, y0, x1 + step, y1)
    return si.fixed_size(si.crop(frame, box)), si.fixed_size(si.crop(frame, moved))


def lexicon_readout(handle: run.ModelHandle, media: run.Media, prompt: str, prefill_ids: Sequence[int]) -> Tuple[float, np.ndarray]:
    inputs = run.encode_media(handle, media, prompt)
    logprobs = run.score_sequences(handle, inputs, prefill_ids, handle.word_ids)
    family = run.family_scores(handle.words, logprobs)
    return math.exp(run.logsumexp(logprobs)), np.array([family[k] for k in sorted(family)])


def rung_one(handle: run.ModelHandle, views: Dict[str, Tuple[np.ndarray, np.ndarray]], extra: Sequence[int], smoke: Sequence[run.Media]) -> List[Dict[str, Any]]:
    rows = []
    grey_image = np.full_like(next(iter(views.values()))[0], 128)
    for prompt, prefill in run.BLIND_CANDIDATES:
        ids = list(extra) + list(handle.tokenizer(prefill, add_special_tokens=False).input_ids)
        masses, vectors = [], []
        for clip_id, (view, _shifted) in views.items():
            mass, vector = lexicon_readout(handle, run.Media("image", [view], [], {}), prompt, ids)
            masses.append(mass)
            vectors.append(vector)
        share = share_above(masses, MASS_FLOOR)
        row: Dict[str, Any] = {
            "prompt": prompt,
            "prefill": prefill,
            "lexicon_mass_by_clip": dict(zip(views, (round(m, 5) for m in masses))),
            "median_lexicon_mass": round(float(np.median(masses)), 5),
            "share_clips_mass_at_least_0.05": round(share, 4),
            "coverage_pass": bool(share >= MASS_SHARE),
            "smoke_item_mass_mechanics_only": [
                round(lexicon_readout(handle, m, prompt, ids)[0], 5) for m in smoke
            ],
        }
        if row["coverage_pass"]:
            shifted = [
                lexicon_readout(handle, run.Media("image", [view[1]], [], {}), prompt, ids)[1]
                for view in views.values()
            ]
            grey = lexicon_readout(handle, run.Media("image", [grey_image], [], {}), prompt, ids)
            row["grey_lexicon_mass"] = round(grey[0], 5)
            row["image_sensitivity"] = image_sensitivity(np.array(vectors), np.array(shifted), grey[1])
        row["pass"] = bool(row["coverage_pass"] and row.get("image_sensitivity", {}).get("pass"))
        rows.append(row)
    return rows


def rung_two(handle: run.ModelHandle, views: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    import torch  # noqa: F401

    tokenizer = handle.tokenizer
    yes_ids = sorted({tokenizer(t, add_special_tokens=False).input_ids[0] for t in YES_TOKENS})
    no_ids = sorted({tokenizer(t, add_special_tokens=False).input_ids[0] for t in NO_TOKENS})
    labels = sorted(run.LEXICON)

    def read(image: np.ndarray) -> Tuple[float, np.ndarray]:
        media = run.Media("image", [image], [], {})
        totals, logits = [], []
        for label in labels:
            prompt = YESNO_PROMPT.format(label=label.lower())
            logprobs = run.step_logprobs(handle, run.encode_media(handle, media, prompt), [])
            yes, no = yes_no_mass(logprobs, yes_ids, no_ids)
            totals.append(yes + no)
            logits.append(math.log(max(yes, 1e-12)) - math.log(max(no, 1e-12)))
        return float(np.mean(totals)), np.array(logits)

    masses, vectors = [], []
    for view, _shifted in views.values():
        mass, vector = read(view)
        masses.append(mass)
        vectors.append(vector)
    share = share_above(masses, YESNO_FLOOR)
    row: Dict[str, Any] = {
        "prompt": YESNO_PROMPT,
        "yes_no_mass_by_clip": dict(zip(views, (round(m, 5) for m in masses))),
        "median_yes_no_mass": round(float(np.median(masses)), 5),
        "share_clips_mass_at_least_0.90": round(share, 4),
        "coverage_pass": bool(share >= YESNO_SHARE),
    }
    if row["coverage_pass"]:
        shifted = [read(view[1])[1] for view in views.values()]
        grey = read(np.full_like(next(iter(views.values()))[0], 128))[1]
        row["image_sensitivity"] = image_sensitivity(np.array(vectors), np.array(shifted), grey)
    row["pass"] = bool(row["coverage_pass"] and row.get("image_sensitivity", {}).get("pass"))
    return row


def rung_three(handle: run.ModelHandle, views: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    labels = sorted(run.LEXICON)
    tokens = [list(handle.tokenizer(f" {label.lower()}.", add_special_tokens=False).input_ids) for label in labels]
    prefill = list(handle.tokenizer(LIKELIHOOD_PREFILL, add_special_tokens=False).input_ids)

    def read(image: np.ndarray) -> np.ndarray:
        inputs = run.encode_media(handle, run.Media("image", [image], [], {}), LIKELIHOOD_PROMPT)
        return np.array(run.score_sequences(handle, inputs, prefill, tokens))

    vectors = np.array([read(view[0]) for view in views.values()])
    shifted = np.array([read(view[1]) for view in views.values()])
    grey = read(np.full_like(next(iter(views.values()))[0], 128))
    sensitivity = image_sensitivity(vectors, shifted, grey)
    return {
        "note": "Weakest readout. Only differences within a label across images are valid.",
        "image_sensitivity": sensitivity,
        "pass": sensitivity["pass"],
    }


def template_position(handle: run.ModelHandle, media: run.Media) -> Dict[str, Any]:
    """Where should the prefill go? Decided from special-token behaviour alone."""
    import torch

    inputs = run.encode_media(handle, media, run.BLIND_PROMPT)
    logprobs = run.step_logprobs(handle, inputs, [])
    top = torch.topk(logprobs, 6)
    best = int(top.indices[0])
    special = best in set(handle.tokenizer.all_special_ids) or handle.tokenizer.decode([best]).startswith("<")
    report: Dict[str, Any] = {
        "prompt_tail": handle.tokenizer.decode(inputs["input_ids"][0][-16:], skip_special_tokens=False),
        "top_tokens_at_generation_position": [
            [handle.tokenizer.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values, top.indices)
        ],
        "next_token_is_special": bool(special),
        "extra_prefix": "",
    }
    if handle.key == "gemma4":
        thought = list(handle.tokenizer(GEMMA_EMPTY_THOUGHT, add_special_tokens=False).input_ids)
        ids = thought + list(handle.tokenizer(run.BLIND_PREFILL, add_special_tokens=False).input_ids)
        report["mass_candidate_1_after_empty_thought"] = round(lexicon_readout(handle, media, run.BLIND_PROMPT, ids)[0], 5)
        if special:
            report["extra_prefix"] = GEMMA_EMPTY_THOUGHT
    return report


def step_ladder(args: argparse.Namespace) -> None:
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    usable = {k: v for k, v in geometry.items() if not (v.get("clip_failure") or v.get("box_failure")) and v.get("apex") is not None}
    views = {k: face_views(v) for k, v in usable.items()}
    started = time.time()
    handle = run.load_handle(args.model, args.dtype)
    out_path = args.output / f"readout_ladder_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "model": args.model,
        "route": "single_face",
        "n_calibration_clips": len(geometry),
        "n_usable": len(views),
        "unusable": sorted(set(geometry) - set(views)),
        "labels_or_votes_read": False,
        "rules": __doc__.split("Rules fixed before running:")[1].split("- Frames stay")[0],
    }
    first = next(iter(views.values()))[0]
    report["template_position"] = template_position(handle, run.Media("image", [first], [], {}))
    extra = list(handle.tokenizer(report["template_position"]["extra_prefix"], add_special_tokens=False).input_ids) if report["template_position"]["extra_prefix"] else []

    smoke: List[run.Media] = []
    if args.smoke_geometry and args.smoke_geometry.exists():
        items = run.select_items(run.load_items(args.manifest, args.lookup, args.data_root), args.n_smoke)
        smoke_geometry = json.loads(args.smoke_geometry.read_text(encoding="utf-8"))["items"]
        paths = {i["trial_id"]: i["path"] for i in items}
        for trial_id, record in smoke_geometry.items():
            record["path"] = paths.get(trial_id, "")
        for item in items:
            record = smoke_geometry[item["trial_id"]]
            clip = run.Clip(item, record, {}, {}, si.ClipGeometry(), {}, record["apex"])
            clip.frames, _ = si.read_frames(Path(item["path"]), [record["apex"]])
            clip.landmarks = run._landmark_table(record)
            clip.geometry.face_box = tuple(record["face_box"])
            clip.geometry.face_width_px = float(record["face_width_px"])
            smoke.append(run.build_media(clip, "single_face", "original"))

    def save() -> None:
        report["seconds"] = round(time.time() - started, 1)
        report["forwards"] = handle.forwards
        report["cached_scoring"] = {"used": handle.use_cache, "gap": handle.cache_gap, "error": handle.cache_error}
        out_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    passed: Optional[str] = None
    if "1" in rungs:
        report["rung_1"] = rung_one(handle, views, extra, smoke)
        save()
        if any(r["pass"] for r in report["rung_1"]):
            passed = "1"
            best = max((r for r in report["rung_1"] if r["pass"]), key=lambda r: r["median_lexicon_mass"])
            report["rung_1_best_for_this_model"] = {"prompt": best["prompt"], "prefill": best["prefill"]}
    if "2" in rungs and (passed is None or args.force):
        report["rung_2"] = rung_two(handle, views)
        save()
        if report["rung_2"]["pass"] and passed is None:
            passed = "2"
    if "3" in rungs and (passed is None or args.force):
        report["rung_3"] = rung_three(handle, views)
        save()
        if report["rung_3"]["pass"] and passed is None:
            passed = "3"
    report["first_rung_passed"] = passed
    save()
    print(json.dumps({"model": args.model, "first_rung_passed": passed, "seconds": report["seconds"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--step", required=True, choices=("select", "geometry", "ladder"))
    parser.add_argument("--model", default="gemma4", choices=("gemma4", "qwen3vl", "molmo2"))
    parser.add_argument("--selection", type=Path, default=Path("research/reproduction/crema_d_selection.csv"))
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_CSV)
    parser.add_argument("--video_dir", type=Path, default=Path("data/crema_d_calibration"))
    parser.add_argument("--geometry", type=Path, default=Path("results/readout_ladder/geometry.json"))
    parser.add_argument("--smoke_geometry", type=Path, default=Path("results/stimulus_interventions/smoke/geometry.json"))
    parser.add_argument("--n_smoke", type=int, default=3)
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--data_root", type=Path, default=Path("data/eu_emotions"))
    parser.add_argument("--output", type=Path, default=Path("research/reproduction"))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--rungs", default="1,2,3")
    parser.add_argument("--force", action="store_true", help="Run later rungs even after a pass.")
    args = parser.parse_args()
    {"select": step_select, "geometry": step_geometry, "ladder": step_ladder}[args.step](args)


if __name__ == "__main__":
    main()

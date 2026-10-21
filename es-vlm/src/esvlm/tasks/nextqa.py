"""Paper 1 public same-scene pairing: NExT-QA CW vs descriptive MCQ.

Annotations are MIT-licensed CSVs. Videos are not fetched here; media
filenames are relative ``{video}.mp4`` and may be missing on disk.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from shutil import which
from typing import Any
from urllib.request import urlopen

from esvlm.tasks.prompts import format_options, render_prompt
from esvlm.types import Domain, MCQItem, MediaType

SOCIAL_TYPES = frozenset({"CW"})
PHYSICAL_TYPES = frozenset({"DO", "DC", "DL"})
PHYSICAL_TYPE_ORDER = ("DO", "DC", "DL")
OPTION_KEYS = ("a0", "a1", "a2", "a3", "a4")

VAL_CSV_URL = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/val.csv"
TRAIN_CSV_URL = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/train.csv"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    wget = which("wget")
    if wget:
        subprocess.check_call([wget, "-c", "-O", str(dest), url])
        return
    with urlopen(url) as src, dest.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _qid_key(trial: dict[str, Any]) -> tuple[int, int | str]:
    qid = trial.get("qid")
    if qid is None:
        return (1, "")
    try:
        return (0, int(qid))
    except (TypeError, ValueError):
        return (1, str(qid))


def _video_key(video: str) -> tuple[int, int | str]:
    try:
        return (0, int(video))
    except (TypeError, ValueError):
        return (1, str(video))


def parse_nextqa_row(row: dict) -> dict | None:
    """Map one NExT-QA CSV row to a trial dict, or None if options are missing."""

    options: list[str] = []
    for key in OPTION_KEYS:
        if key not in row or row[key] is None:
            return None
        text = str(row[key]).strip()
        if text == "":
            return None
        options.append(text)
    video = row.get("video")
    qid = row.get("qid")
    question = str(row.get("question") or "").strip()
    if video in (None, "") or qid in (None, "") or not question:
        return None
    video_s = str(video).strip()
    if not video_s:
        return None
    try:
        qid_val: int | str = int(qid)
    except (TypeError, ValueError):
        qid_val = str(qid).strip()
        if not qid_val:
            return None
    raw_answer = row.get("answer")
    if raw_answer is None or str(raw_answer).strip() == "":
        return None
    try:
        correct_idx = int(raw_answer)
    except (TypeError, ValueError):
        return None
    if not 0 <= correct_idx < len(options):
        return None
    return {
        "item_id": f"nextqa_{video_s}_{qid_val}",
        "question": question,
        "options": tuple(options),
        "correct_idx": correct_idx,
        "media": f"{video_s}.mp4",
        "media_type": "video",
        "video": video_s,
        "qid": qid_val,
        "question_type": str(row.get("type") or "").strip(),
    }


def load_nextqa_csv(path: Path) -> list[dict]:
    trials: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            trial = parse_nextqa_row(row)
            if trial is not None:
                trials.append(trial)
    return trials


def _pick_social(candidates: list[dict]) -> dict | None:
    social = [t for t in candidates if t.get("question_type") in SOCIAL_TYPES]
    if not social:
        return None
    return min(social, key=_qid_key)


def _pick_physical(candidates: list[dict]) -> dict | None:
    by_type: dict[str, list[dict]] = {name: [] for name in PHYSICAL_TYPE_ORDER}
    for trial in candidates:
        qtype = trial.get("question_type")
        if qtype in by_type:
            by_type[qtype].append(trial)
    for name in PHYSICAL_TYPE_ORDER:
        if by_type[name]:
            return min(by_type[name], key=_qid_key)
    return None


def pair_same_video(trials: list[dict]) -> list[dict]:
    """One CW vs descriptive pair per video when both sides exist.

    Social is CW only (CH is mixed and is skipped). Physical prefers DO,
    then DC, then DL. Lowest ``qid`` wins ties on each side.
    """

    by_video: dict[str, list[dict]] = defaultdict(list)
    for trial in trials:
        video = trial.get("video")
        if video in (None, ""):
            continue
        by_video[str(video)].append(trial)
    pairs: list[dict] = []
    for video in sorted(by_video, key=_video_key):
        social = _pick_social(by_video[video])
        physical = _pick_physical(by_video[video])
        if social is None or physical is None:
            continue
        pairs.append({"video": video, "social": social, "physical": physical})
    return pairs


def write_pair_manifest(path: Path, pairs: list[dict]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task": "nextqa_pairs", "n": len(pairs), "pairs": pairs}
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def nextqa_trials_from_pairs(pairs: list[dict], side: str) -> list[dict]:
    if side not in {"social", "physical"}:
        raise ValueError(f"side must be 'social' or 'physical', got {side!r}")
    return [pair[side] for pair in pairs]


def fetch_nextqa_annotations(root: Path, log) -> Path:
    """Download NExT-QA CSVs and write val same-scene pairs. No videos."""

    raw = Path(root) / "nextqa" / "raw"
    val_path = raw / "val.csv"
    log.info("nextqa val.csv %s", VAL_CSV_URL)
    _download(VAL_CSV_URL, val_path)
    try:
        log.info("nextqa train.csv %s", TRAIN_CSV_URL)
        _download(TRAIN_CSV_URL, raw / "train.csv")
    except Exception as exc:  # noqa: BLE001
        log.warning("nextqa train.csv failed: %s", exc)
    trials = load_nextqa_csv(val_path)
    pairs = pair_same_video(trials)
    dest = Path(root) / "nextqa" / "pairs_val.json"
    write_pair_manifest(dest, pairs)
    log.info("nextqa val pairs %s from %s trials", len(pairs), len(trials))
    return dest


def _render_nextqa_prompt(question: str, template_id: str, options: tuple[str, ...]) -> str:
    try:
        rendered = render_prompt("nextqa", template_id, options)
    except KeyError:
        rendered = format_options(options)
    if question:
        return f"{question}\n\n{rendered}"
    return rendered


def load_nextqa_side_items(
    manifest_path: Path,
    media_root: Path,
    *,
    side: str,
    template_id: str = "direct",
) -> list[MCQItem]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    pairs = data.get("pairs") or []
    trials = nextqa_trials_from_pairs(pairs, side)
    domain: Domain = "E" if side == "social" else "S"
    media_root = Path(media_root)
    items: list[MCQItem] = []
    for trial in trials:
        options = tuple(str(o) for o in trial["options"])
        prompt = _render_nextqa_prompt(str(trial.get("question") or "").strip(), template_id, options)
        media_rel = str(trial.get("media") or "")
        media_file = media_root / media_rel if media_rel else None
        if media_file is not None and media_file.is_file():
            media = str(media_file.resolve())
            media_type: MediaType = "video"
        else:
            media = ""
            media_type = "none"
        meta: dict[str, Any] = {
            "template_id": template_id,
            "video": trial.get("video"),
            "qid": trial.get("qid"),
            "question_type": trial.get("question_type"),
            "side": side,
        }
        if media_type == "none" and media_rel:
            meta["media_rel"] = media_rel
        items.append(
            MCQItem(
                item_id=str(trial["item_id"]),
                task="nextqa",
                domain=domain,
                media=media,
                media_type=media_type,
                prompt_text=prompt,
                options=options,
                correct_idx=int(trial["correct_idx"]),
                meta=meta,
            )
        )
    return items

#!/usr/bin/env python3
"""Deterministic face-only CREMA-D selection (prereg Phase 6; rule in research/CREMA_D_PLAN.md).

Eligibility and choice see only file names, the face-only rating count and actor sex.
Vote counts, agreement and all model outputs are withheld from `select_clips` on purpose:
human ambiguity is the signal, so it must not influence which clips are studied. Votes
are attached afterwards, for output only.

Standard library only, so it runs under `python -I` with no project imports:

    python -I research/scripts/crema_select.py --data-dir data/external/crema_d
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "research" / "reproduction" / "crema_d_selection.csv"

SEED = 42
MIN_FACE_RATINGS = 8  # README: 95% of clips have more than 7 ratings
EXTRA_PER_ACTOR = 3  # one clip per emotion + 3 = 9 per actor = 819 clips
FACE_QUERY_TYPE = "2"  # tabulatedVotes row name = 100000 * queryType + clipNum
N_CLIPS_UPSTREAM = 7442

EMOTION_CODES = ("A", "D", "F", "H", "N", "S")  # upstream vote columns, in order
EMOTION_FROM_FILENAME = {"ANG": "A", "DIS": "D", "FEA": "F", "HAP": "H", "NEU": "N", "SAD": "S"}

# README, "do not display correctly based on their filename". Named upstream as faulty
# files; excluded before any vote is read.
DOCUMENTED_DEFECTS = frozenset(
    {"1076_MTI_NEU_XX", "1076_MTI_SAD_XX", "1064_TIE_SAD_XX", "1064_IEO_DIS_MD"}
)

OUTPUT_FIELDS = [
    "clip_id", "actor_id", "sentence", "emotion", "level", "role", "n_face",
    "votes_A", "votes_D", "votes_F", "votes_H", "votes_N", "votes_S",
    "target_position", "sex", "age", "race", "ethnicity",
]  # fmt: skip


def stable_hash(purpose: str, key: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}|{purpose}|{key}".encode("utf-8")).hexdigest()


def parse_clip_id(clip_id: str) -> Dict[str, str]:
    actor, sentence, emotion, level = clip_id.split("_")
    return {
        "actor_id": actor,
        "sentence": sentence,
        "emotion": EMOTION_FROM_FILENAME[emotion],
        "level": level,
    }


def read_face_rows(path: Path) -> List[Dict[str, str]]:
    """Rows of tabulatedVotes.csv from the face-only query type (see CREMA_D_PLAN.md)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    face = [r for r in rows if r[""].startswith(FACE_QUERY_TYPE) and len(r[""]) == 6]
    if len(face) != N_CLIPS_UPSTREAM:
        raise ValueError(f"expected {N_CLIPS_UPSTREAM} face-only rows, found {len(face)}")
    return face


def read_demographics(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["ActorID"]: r for r in csv.DictReader(fh)}


def extra_emotions(sex_by_actor: Mapping[str, str]) -> Dict[str, Tuple[str, ...]]:
    """Cycle emotions over actors ordered by sex, so extras balance within sex too."""
    ordered = sorted(sex_by_actor, key=lambda a: (sex_by_actor[a], a))
    n = len(EMOTION_CODES)
    return {
        actor: tuple(
            EMOTION_CODES[(EXTRA_PER_ACTOR * i + j) % n] for j in range(EXTRA_PER_ACTOR)
        )
        for i, actor in enumerate(ordered)
    }


def select_clips(
    n_face_by_clip: Mapping[str, int],
    sex_by_actor: Mapping[str, str],
    *,
    min_ratings: int = MIN_FACE_RATINGS,
) -> List[Dict[str, str]]:
    """Choose clips. Takes rating counts and actor sex; deliberately takes no votes."""
    pool: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for clip_id, n_face in n_face_by_clip.items():
        if clip_id in DOCUMENTED_DEFECTS or n_face < min_ratings:
            continue
        meta = parse_clip_id(clip_id)
        pool[meta["actor_id"], meta["emotion"]].append(clip_id)

    extras = extra_emotions(sex_by_actor)
    by_emotion_sentence: Counter = Counter()
    by_actor_sentence: Counter = Counter()
    by_sentence: Counter = Counter()
    by_emotion_level: Counter = Counter()
    chosen: List[Dict[str, str]] = []
    used = set()

    for actor in sorted(sex_by_actor):
        plan = [(e, "base") for e in EMOTION_CODES] + [(e, "extra") for e in extras[actor]]
        for emotion, role in plan:
            candidates = [c for c in pool[actor, emotion] if c not in used]
            if not candidates:
                raise ValueError(f"no eligible clip left for actor {actor}, emotion {emotion}")

            def key(clip_id: str) -> Tuple[int, int, int, int, str]:
                meta = parse_clip_id(clip_id)
                return (
                    by_emotion_sentence[emotion, meta["sentence"]],
                    by_actor_sentence[actor, meta["sentence"]],
                    by_sentence[meta["sentence"]],
                    by_emotion_level[emotion, meta["level"]],
                    stable_hash("clip", clip_id),
                )

            clip_id = min(candidates, key=key)
            meta = parse_clip_id(clip_id)
            used.add(clip_id)
            by_emotion_sentence[emotion, meta["sentence"]] += 1
            by_actor_sentence[actor, meta["sentence"]] += 1
            by_sentence[meta["sentence"]] += 1
            by_emotion_level[emotion, meta["level"]] += 1
            chosen.append({"clip_id": clip_id, "role": role, **meta})
    return chosen


def assign_target_positions(clips: Iterable[Mapping[str, str]], n_options: int = 6) -> Dict[str, int]:
    """Position of the true label in the model prompt, balanced within each emotion.

    Hash order stops position from tracking actor, so no position is tied to a label
    or to a person. The adapter builds each prompt's option list from this value.
    """
    positions: Dict[str, int] = {}
    by_emotion: Dict[str, List[str]] = defaultdict(list)
    for clip in clips:
        by_emotion[clip["emotion"]].append(clip["clip_id"])
    for offset, emotion in enumerate(sorted(by_emotion)):
        ordered = sorted(by_emotion[emotion], key=lambda c: stable_hash("position", c))
        for i, clip_id in enumerate(ordered):
            positions[clip_id] = (i + offset) % n_options  # offset spreads remainders
    return positions


def build_rows(
    face_rows: Sequence[Mapping[str, str]], demographics: Mapping[str, Mapping[str, str]]
) -> List[Dict[str, object]]:
    n_face = {r["fileName"]: int(r["numResponses"]) for r in face_rows}
    sex = {actor: d["Sex"] for actor, d in demographics.items()}
    clips = select_clips(n_face, sex)  # votes are not passed in
    positions = assign_target_positions(clips)
    votes = {r["fileName"]: r for r in face_rows}
    rows: List[Dict[str, object]] = []
    for clip in sorted(clips, key=lambda c: c["clip_id"]):
        cid, demo, vote = clip["clip_id"], demographics[clip["actor_id"]], votes[clip["clip_id"]]
        rows.append(
            {
                **clip,
                "n_face": n_face[cid],
                **{f"votes_{e}": int(vote[e]) for e in EMOTION_CODES},
                "target_position": positions[cid],
                "sex": demo["Sex"],
                "age": demo["Age"],
                "race": demo["Race"],
                "ethnicity": demo["Ethnicity"],
            }
        )
    return rows


def write_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "external" / "crema_d")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(list(argv) or None)
    rows = build_rows(
        read_face_rows(args.data_dir / "tabulatedVotes.csv"),
        read_demographics(args.data_dir / "VideoDemographics.csv"),
    )
    write_rows(rows, args.out)
    print(f"{len(rows)} clips, {len({r['actor_id'] for r in rows})} actors -> {args.out}")
    for field in ("emotion", "sentence", "level", "sex", "target_position"):
        print(field, sorted(Counter(r[field] for r in rows).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

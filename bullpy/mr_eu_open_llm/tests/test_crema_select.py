from __future__ import annotations

import csv
import importlib.util
import inspect
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
path = ROOT / "research/scripts/crema_select.py"
spec = importlib.util.spec_from_file_location("crema_select", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SELECTION = ROOT / "research/reproduction/crema_d_selection.csv"
RAW = ROOT / "data/external/crema_d"
SENTENCES = ("DFA", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI")
EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")


def synthetic_counts(actors=("1001", "1002", "1003", "1004", "1064", "1076"), n=10):
    clips = {}
    for actor in actors:
        for emotion in EMOTIONS:
            for sentence in SENTENCES:
                clips[f"{actor}_{sentence}_{emotion}_XX"] = n
            levels = ("XX",) if emotion == "NEU" else ("LO", "MD", "HI")
            for level in levels:
                clips[f"{actor}_IEO_{emotion}_{level}"] = n
    return clips


def sex_for(clips):
    actors = sorted({c.split("_")[0] for c in clips})
    return {a: ("Female" if i % 2 else "Male") for i, a in enumerate(actors)}


def committed_rows():
    with open(SELECTION, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_selection_is_independent_of_input_order():
    clips = synthetic_counts()
    keys = list(clips)
    random.Random(1).shuffle(keys)
    shuffled = {k: clips[k] for k in keys}
    assert mod.select_clips(clips, sex_for(clips)) == mod.select_clips(shuffled, sex_for(clips))


def test_each_actor_gets_all_emotions_and_nine_clips():
    clips = synthetic_counts()
    chosen = mod.select_clips(clips, sex_for(clips))
    assert Counter(c["actor_id"] for c in chosen) == {a: 9 for a in sex_for(clips)}
    assert len({c["clip_id"] for c in chosen}) == len(chosen)
    for actor in sex_for(clips):
        assert {c["emotion"] for c in chosen if c["actor_id"] == actor} == set(mod.EMOTION_CODES)


def test_rating_floor_and_documented_defects_are_excluded():
    clips = synthetic_counts()
    low = {c: mod.MIN_FACE_RATINGS - 1 for c in clips if c.startswith("1001_") and "_ANG_" in c and "DFA" not in c}
    clips.update(low)
    chosen = {c["clip_id"] for c in mod.select_clips(clips, sex_for(clips))}
    assert not chosen & set(low)
    assert not chosen & mod.DOCUMENTED_DEFECTS


def test_empty_cell_stops_rather_than_silently_unbalancing():
    clips = synthetic_counts()
    clips = {c: (1 if c.startswith("1001_") and "_SAD_" in c else n) for c, n in clips.items()}
    with pytest.raises(ValueError, match="1001"):
        mod.select_clips(clips, sex_for(clips))


def test_selection_function_cannot_see_votes():
    assert list(inspect.signature(mod.select_clips).parameters) == [
        "n_face_by_clip", "sex_by_actor", "min_ratings",
    ]  # fmt: skip


def test_scrambling_votes_changes_nothing_but_the_votes():
    clips = synthetic_counts()
    rows = [
        {"fileName": c, "numResponses": str(n), "A": str(i % 7), "D": "1", "F": "0", "H": "2", "N": "3", "S": "4"}
        for i, (c, n) in enumerate(clips.items())
    ]
    demographics = {
        a: {"Sex": s, "Age": "30", "Race": "Unknown", "Ethnicity": "Not Hispanic"} for a, s in sex_for(clips).items()
    }
    scrambled = [dict(r) for r in rows]
    tallies = [r["A"] for r in scrambled]
    random.Random(2).shuffle(tallies)
    for row, tally in zip(scrambled, tallies):
        row["A"] = tally
    strip = lambda rs: [{k: v for k, v in r.items() if k != "votes_A"} for r in rs]  # noqa: E731
    assert strip(mod.build_rows(rows, demographics)) == strip(mod.build_rows(scrambled, demographics))


def test_target_positions_balance_within_each_emotion():
    clips = [{"clip_id": f"{i}_x", "emotion": "ADFHNS"[i % 6]} for i in range(600)]
    positions = mod.assign_target_positions(clips)
    for emotion in "ADFHNS":
        got = Counter(positions[c["clip_id"]] for c in clips if c["emotion"] == emotion)
        assert sorted(got) == list(range(6)) and set(got.values()) == {16, 17}


def test_committed_selection_is_balanced():
    rows = committed_rows()
    assert len(rows) == 819
    assert Counter(r["actor_id"] for r in rows) == {a: 9 for a in {r["actor_id"] for r in rows}}
    assert len({r["actor_id"] for r in rows}) == 91
    emotions = Counter(r["emotion"] for r in rows)
    assert set(emotions) == set("ADFHNS") and max(emotions.values()) - min(emotions.values()) <= 1
    sentences = Counter(r["sentence"] for r in rows)
    assert len(sentences) == 12 and max(sentences.values()) - min(sentences.values()) <= 1
    assert {(r["actor_id"], r["emotion"]) for r in rows} >= {
        (a, e) for a in {r["actor_id"] for r in rows} for e in "ADFHNS"
    }
    positions = Counter(r["target_position"] for r in rows)
    assert len(positions) == 6 and max(positions.values()) - min(positions.values()) <= 1


def test_committed_selection_obeys_eligibility_and_keeps_ambiguity():
    rows = committed_rows()
    assert all(int(r["n_face"]) >= mod.MIN_FACE_RATINGS for r in rows)
    assert not {r["clip_id"] for r in rows} & mod.DOCUMENTED_DEFECTS
    votes = [[int(r[f"votes_{e}"]) for e in "ADFHNS"] for r in rows]
    assert all(sum(v) == int(r["n_face"]) for v, r in zip(votes, rows))
    top_share = [max(v) / sum(v) for v in votes]
    assert sum(s == 1 for s in top_share) > 50  # unanimous clips stay in
    assert sum(s < 0.5 for s in top_share) > 100  # no-majority clips stay in


@pytest.mark.skipif(not (RAW / "tabulatedVotes.csv").exists(), reason="raw ratings not downloaded")
def test_committed_selection_reproduces_from_raw_ratings(tmp_path):
    out = tmp_path / "selection.csv"
    assert mod.main(["--data-dir", str(RAW), "--out", str(out)]) == 0
    assert out.read_bytes() == SELECTION.read_bytes()

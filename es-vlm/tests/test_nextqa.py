from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from esvlm.tasks.nextqa import (
    PHYSICAL_TYPES,
    SOCIAL_TYPES,
    TRAIN_CSV_URL,
    VAL_CSV_URL,
    fetch_nextqa_annotations,
    load_nextqa_csv,
    load_nextqa_side_items,
    nextqa_trials_from_pairs,
    pair_same_video,
    parse_nextqa_row,
    write_pair_manifest,
)

CSV_HEADER = [
    "video",
    "frame_count",
    "width",
    "height",
    "question",
    "answer",
    "qid",
    "type",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
]

OPTS = ("opt0", "opt1", "opt2", "opt3", "opt4")


def _row(video: str, qid: int, qtype: str, question: str, answer: int = 0) -> dict[str, str]:
    row = {
        "video": video,
        "frame_count": "10",
        "width": "64",
        "height": "64",
        "question": question,
        "answer": str(answer),
        "qid": str(qid),
        "type": qtype,
    }
    for i, opt in enumerate(OPTS):
        row[f"a{i}"] = opt
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _two_video_rows() -> list[dict[str, str]]:
    return [
        _row("111", 5, "CW", "why did they wave", 1),
        _row("111", 1, "CW", "why did they stop", 0),
        _row("111", 8, "DO", "what object is on the table", 2),
        _row("111", 3, "DC", "how many people are visible", 0),
        _row("111", 4, "DL", "where is the bag", 3),
        _row("222", 1, "CH", "how did they open the door", 0),
        _row("222", 2, "DO", "what is the person holding", 1),
    ]


def test_parse_nextqa_row_and_missing_options() -> None:
    trial = parse_nextqa_row(_row("111", 1, "CW", "why did they wave", 2))
    assert trial is not None
    assert trial["item_id"] == "nextqa_111_1"
    assert trial["question"] == "why did they wave"
    assert trial["options"] == OPTS
    assert trial["correct_idx"] == 2
    assert trial["media"] == "111.mp4"
    assert trial["media_type"] == "video"
    assert trial["video"] == "111"
    assert trial["qid"] == 1
    assert trial["question_type"] == "CW"

    missing = _row("111", 1, "CW", "why did they wave")
    missing["a4"] = ""
    assert parse_nextqa_row(missing) is None
    del missing["a3"]
    missing["a4"] = "opt4"
    assert parse_nextqa_row(missing) is None


def test_pair_same_video_skips_without_cw(tmp_path: Path) -> None:
    csv_path = tmp_path / "val.csv"
    _write_csv(csv_path, _two_video_rows())
    trials = load_nextqa_csv(csv_path)
    pairs = pair_same_video(trials)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["video"] == "111"
    assert pair["social"]["question_type"] == "CW"
    assert pair["social"]["qid"] == 1
    assert pair["social"]["item_id"] == "nextqa_111_1"
    assert pair["physical"]["question_type"] == "DO"
    assert pair["physical"]["qid"] == 8
    videos = {p["video"] for p in pairs}
    assert "222" not in videos
    assert "CW" in SOCIAL_TYPES
    assert "CH" not in SOCIAL_TYPES
    assert len(SOCIAL_TYPES) == 1
    assert "DO" in PHYSICAL_TYPES
    assert "DC" in PHYSICAL_TYPES
    assert "DL" in PHYSICAL_TYPES
    assert len(PHYSICAL_TYPES) == 3


def test_physical_prefers_do_then_dc_then_dl() -> None:
    dc_dl = [
        parse_nextqa_row(_row("9", 1, "CW", "why did they sit")),
        parse_nextqa_row(_row("9", 4, "DL", "where is the cup")),
        parse_nextqa_row(_row("9", 2, "DC", "how many cups")),
    ]
    pairs = pair_same_video([t for t in dc_dl if t is not None])
    assert len(pairs) == 1
    assert pairs[0]["physical"]["question_type"] == "DC"
    assert pairs[0]["physical"]["qid"] == 2

    dl_only = [
        parse_nextqa_row(_row("8", 7, "CW", "why did they sit")),
        parse_nextqa_row(_row("8", 9, "DL", "where is the cup")),
        parse_nextqa_row(_row("8", 3, "DL", "where is the chair")),
    ]
    pairs = pair_same_video([t for t in dl_only if t is not None])
    assert len(pairs) == 1
    assert pairs[0]["physical"]["question_type"] == "DL"
    assert pairs[0]["physical"]["qid"] == 3


def test_write_manifest_and_load_side_items_without_mp4(tmp_path: Path) -> None:
    trials = [t for t in (parse_nextqa_row(r) for r in _two_video_rows()) if t is not None]
    pairs = pair_same_video(trials)
    man = tmp_path / "pairs_val.json"
    write_pair_manifest(man, pairs)
    payload = json.loads(man.read_text(encoding="utf-8"))
    assert payload["task"] == "nextqa_pairs"
    assert payload["n"] == 1
    assert "social" in payload["pairs"][0]
    assert "physical" in payload["pairs"][0]
    assert payload["pairs"][0]["social"]["question"] == "why did they stop"
    assert payload["pairs"][0]["physical"]["options"] == list(OPTS)

    social_trials = nextqa_trials_from_pairs(pairs, "social")
    physical_trials = nextqa_trials_from_pairs(pairs, "physical")
    assert len(social_trials) == 1
    assert social_trials[0]["question_type"] == "CW"
    assert physical_trials[0]["question_type"] == "DO"

    social = load_nextqa_side_items(man, tmp_path, side="social")
    physical = load_nextqa_side_items(man, tmp_path, side="physical")
    assert len(social) == 1
    assert len(physical) == 1
    assert social[0].domain == "E"
    assert physical[0].domain == "S"
    assert social[0].task == "nextqa"
    assert social[0].media_type == "none"
    assert social[0].media == ""
    assert social[0].meta["media_rel"] == "111.mp4"
    assert physical[0].media_type == "none"
    assert physical[0].meta["media_rel"] == "111.mp4"
    assert "why did they stop" in social[0].prompt_text
    assert all(opt in social[0].prompt_text for opt in social[0].options)
    assert social[0].correct_idx == 0
    assert physical[0].correct_idx == 2


def test_fetch_nextqa_annotations_does_not_hit_network(tmp_path: Path, monkeypatch) -> None:
    rows = _two_video_rows()
    csv_path = tmp_path / "src.csv"
    _write_csv(csv_path, rows)
    csv_text = csv_path.read_text(encoding="utf-8")
    seen: list[str] = []

    def fake_download(url: str, dest: Path) -> None:
        seen.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(csv_text, encoding="utf-8")

    monkeypatch.setattr("esvlm.tasks.nextqa._download", fake_download)
    log = logging.getLogger("test_nextqa")
    dest = fetch_nextqa_annotations(tmp_path, log)
    assert dest == tmp_path / "nextqa" / "pairs_val.json"
    assert VAL_CSV_URL in seen
    assert TRAIN_CSV_URL in seen
    assert VAL_CSV_URL == "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/val.csv"
    assert TRAIN_CSV_URL == "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/train.csv"
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["n"] == 1
    assert payload["pairs"][0]["video"] == "111"

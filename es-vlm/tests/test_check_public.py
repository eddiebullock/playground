from __future__ import annotations

from pathlib import Path

import pytest

from esvlm.cli.check_public import EXPECTED_MMTOM, EXPECTED_NEXTQA_PAIRS, main
from esvlm.tasks.mmtom import mmtom_to_trial
from esvlm.tasks.nextqa import pair_same_video, parse_nextqa_row, write_pair_manifest
from esvlm.tasks.sdomain import piqa_to_trial, write_manifest


def _opts() -> dict[str, str]:
    return {f"a{i}": f"opt{i}" for i in range(5)}


def test_check_public_writes_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public = tmp_path / "public"
    nextqa = public / "nextqa"
    nextqa.mkdir(parents=True)
    trials = []
    for video in range(EXPECTED_NEXTQA_PAIRS):
        cw = parse_nextqa_row(
            {
                "video": str(video),
                "qid": "1",
                "question": "why did they sit down on the bench",
                "answer": "0",
                "type": "CW",
                **_opts(),
            }
        )
        do = parse_nextqa_row(
            {
                "video": str(video),
                "qid": "2",
                "question": "what is on the table",
                "answer": "1",
                "type": "DO",
                **_opts(),
            }
        )
        assert cw is not None and do is not None
        trials.extend([cw, do])
    write_pair_manifest(nextqa / "pairs_val.json", pair_same_video(trials))

    mm_trials = [
        mmtom_to_trial(
            {
                "question": (
                    f"Question: If agent {i} wants a plate, which is more likely? "
                    "(a) in the fridge. (b) not in the fridge."
                ),
                "answer": "a",
                "episode": i,
                "question_type": 1.1,
                "answer_list": ["a", "b"],
            },
            idx=i,
        )
        for i in range(EXPECTED_MMTOM)
    ]
    write_manifest(public / "mmtom" / "manifest.json", "mmtom", mm_trials)

    piqa_trials = []
    for i in range(8):
        trial = piqa_to_trial(
            {"goal": f"g{i}", "sol1": "a", "sol2": "b", "label": 0},
            split="valid",
            idx=i,
        )
        assert trial is not None
        piqa_trials.append(trial)
    write_manifest(public / "piqa" / "manifest.json", "piqa", piqa_trials)

    monkeypatch.setenv("ESVLM_PUBLIC_DIR", str(public))
    monkeypatch.setenv("ESVLM_NEXTQA_MANIFEST", "/missing/nextqa.json")
    monkeypatch.setenv("ESVLM_NEXTQA_DIR", str(nextqa / "videos"))
    monkeypatch.setenv("ESVLM_MMTOM_MANIFEST", "/missing/mmtom.json")
    monkeypatch.setenv("ESVLM_PIQA_MANIFEST", str(public / "piqa" / "manifest.json"))
    monkeypatch.setenv("ESVLM_PIQA_DIR", str(public / "piqa"))

    out = tmp_path / "lock.json"
    assert main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert f'"nextqa_pairs": {EXPECTED_NEXTQA_PAIRS}' in text
    assert f'"mmtom_n": {EXPECTED_MMTOM}' in text
    assert '"identical_question_text": 0' in text

"""Amendment A3 logic on made-up rows. No model, no clip, no rotation result."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper_v2.option_score import ABSTAIN_LABEL  # noqa: E402
from scripts import face_input_validity as fiv  # noqa: E402


def items():
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    return [
        {
            "trial_id": k,
            "label": v["human_target_label"],
            "actor": k.split("/")[-1][0].upper(),
            "options": v["human_options"],
        }
        for k, v in human.items()
    ]


def rows_for(n_actors, effect, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for actor in range(n_actors):
        for item in range(6):
            trial = f"{actor}_{item}"
            for rotation in fiv.ROTATIONS:
                rows.append({"trial_id": trial, "actor": str(actor), "rotation": rotation, "condition": "real", "status": "ok", "correct": rng.random() < 0.2 + effect})
                rows.append({"trial_id": trial, "actor": str(actor), "rotation": rotation, "condition": "mismatch", "status": "ok", "correct": rng.random() < 0.2})
    return rows


def test_statistics_are_the_a1_functions():
    assert fiv.holm.__module__ == "analyse_rotation"
    assert fiv.actor_sign_flip_p.__module__ == "analyse_rotation"
    assert fiv.actor_bootstrap_ci.__module__ == "analyse_rotation"


def test_partners_match_a1_seed_and_order_free():
    data = items()
    first = fiv.partners_for(data)
    assert first == fiv.partners_for(list(reversed(data)))
    assert len(first) == 243
    by = {i["trial_id"]: i for i in data}
    assert all(by[k]["label"] != by[v]["label"] and by[k]["actor"] != by[v]["actor"] for k, v in first.items())


def test_each_item_has_ten_cells_with_its_own_options():
    item, partner = items()[:2]
    cells = fiv.trial_cells(item, partner)
    assert len(cells) == 10
    assert {c["rotation"] for c in cells} == set(range(5))
    assert all(sorted(c["options"]) == sorted(item["options"]) for c in cells)
    assert all(c["options"][-1] == ABSTAIN_LABEL for c in cells)
    assert {c["face_item"] for c in cells if c["condition"] == "mismatch"} == {partner["trial_id"]}
    positions = {c["options"].index(item["label"]) for c in cells}
    assert positions == set(range(5))


def test_exact_scoring():
    options = ["A", "B", "C"]
    out = fiv.correct_from_scores(options, np.log([0.5, 0.3, 0.2]).tolist(), "A")
    assert out["correct"] and out["p_target"] == pytest.approx(0.5)
    assert not fiv.correct_from_scores(options, np.log([0.5, 0.3, 0.2]).tolist(), "B")["correct"]
    assert not fiv.correct_from_scores(options, [0.0, 0.0, -1.0], "A")["correct"]


def test_incomplete_items_are_left_out_and_failures_reported():
    rows = rows_for(4, 0.0)
    for row in rows:
        if row["trial_id"] == "0_0" and row["rotation"] == 2 and row["condition"] == "real":
            row.update(status="failed", reason="no_apex_frame")
    summary = fiv.summarise_model(rows)
    assert summary["n_items"] == 23 and summary["n_items_expected"] == 24
    assert summary["failure_reasons"] == {"no_apex_frame": 1}


def test_decision_rule_finds_a_planted_delta_only():
    summaries = {
        "a": fiv.summarise_model(rows_for(17, 0.5, 1)),
        "b": fiv.summarise_model(rows_for(17, 0.0, 2)),
        "c": fiv.summarise_model(rows_for(17, 0.0, 3)),
    }
    out = fiv.decide(summaries)
    assert out["a"]["vision_dependent"]
    assert not out["b"]["vision_dependent"] and not out["c"]["vision_dependent"]


def test_reading_table_is_the_plans():
    assert "grid limited" in fiv.reading(False, True)
    assert "No demonstrated" in fiv.reading(False, False)


def test_refuses_until_frozen(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["face_input_validity", "--model", "gemma4"])
    with pytest.raises(SystemExit, match="not frozen"):
        fiv.main()


def write_lines(path, *rows, tail=b""):
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows) + tail)


CELL = {"trial_id": "t", "rotation": 0, "condition": "real"}


def test_reader_drops_one_truncated_final_line(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_lines(path, CELL, tail=b'{"trial_id": "t", "rot')
    state = fiv.read_cells(path)
    assert len(state["rows"]) == 1 and state["n_partial_dropped"] == 1


def test_valid_json_with_no_newline_counts_as_truncated(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_lines(path, CELL, tail=json.dumps({**CELL, "rotation": 1}).encode())
    assert fiv.read_cells(path)["n_partial_dropped"] == 1


def test_bad_final_line_with_a_newline_is_also_dropped(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_lines(path, CELL, tail=b"{broken\n")
    assert fiv.read_cells(path)["n_partial_dropped"] == 1


def test_bad_line_before_the_end_raises(tmp_path):
    path = tmp_path / "cells.jsonl"
    path.write_bytes(b"{broken\n" + json.dumps(CELL).encode() + b"\n")
    with pytest.raises(ValueError, match="not the final line"):
        fiv.read_cells(path)


def test_resume_cuts_the_tail_and_rescores_the_cut_cell(tmp_path):
    path = tmp_path / "cells.jsonl"
    write_lines(path, CELL, tail=b'{"trial_id": "t", "rotation": 1, "cond')
    done = fiv.prepare_resume(path)
    assert done == {("t", 0, "real")}
    with path.open("a") as sink:
        sink.write(json.dumps({**CELL, "rotation": 1}) + "\n")
    state = fiv.read_cells(path)
    assert [r["rotation"] for r in state["rows"]] == [0, 1] and state["n_partial_dropped"] == 0


def test_resume_on_no_file_is_empty(tmp_path):
    assert fiv.prepare_resume(tmp_path / "none.jsonl") == set()


def test_mechanics_row_is_a_whitelist():
    row = fiv.mechanics_row(
        {**CELL, "status": "ok", "seconds": 1.0, "forwards": 7, "sent_frame_indices": [3], "joint_logprob": [0.0], "correct": True, "p_target": 0.5, "options": ["a"], "label": "a"}
    )
    assert set(row) == {"status", "seconds", "forwards", "sent_frame_indices"}


def test_mechanics_output_cannot_sit_in_the_outcome_folder():
    with pytest.raises(SystemExit, match="outside"):
        fiv.mechanics_output_dir(fiv.OUTCOME_DIR)
    with pytest.raises(SystemExit, match="outside"):
        fiv.mechanics_output_dir(fiv.OUTCOME_DIR / "sub")
    assert fiv.mechanics_output_dir(fiv.MECHANICS_DIR) == fiv.MECHANICS_DIR


def test_mechanics_refuses_the_freeze_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--model", "gemma4", "--mechanics_only", "2", "--a3_frozen"])
    with pytest.raises(SystemExit, match="cannot be combined"):
        fiv.main()


def test_geometry_hash_is_enforced(tmp_path):
    path = tmp_path / "g.json"
    path.write_text("{}")
    with pytest.raises(SystemExit, match="not the pinned one"):
        fiv.check_geometry(path)


def test_mechanics_run_writes_and_prints_no_outcome(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from scripts import run_stimulus_interventions as run

    data = []
    for index, (label, actor) in enumerate([("Sad", "A"), ("Happy", "B"), ("Angry", "C"), ("Happy", "A")]):
        data.append(
            {
                "trial_id": f"t{index}",
                "label": label,
                "actor": actor,
                "options": [label, "x1", "x2", "x3", "x4", ABSTAIN_LABEL],
                "path": "",
            }
        )
    geometry = tmp_path / "g.json"
    geometry.write_text('{"items": {}}')
    import hashlib

    monkeypatch.setattr(fiv, "GEOMETRY_SHA256", hashlib.sha256(geometry.read_bytes()).hexdigest())
    monkeypatch.setattr(fiv.run, "load_items", lambda *a: data)

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[len(text)])

    handle = SimpleNamespace(forwards=0, tokenizer=Tokenizer(), mcq_prefill=[1])
    monkeypatch.setattr(fiv.run, "load_handle", lambda *a: handle)
    monkeypatch.setattr(fiv.run, "load_clip", lambda item, *a: item)
    monkeypatch.setattr(fiv.run, "encode_media", lambda *a: {})

    def build(clip, *a):
        if clip["trial_id"] == "t2":
            raise fiv.si.EditFailure("no_apex_frame")
        return run.Media("image", [], [5], {})

    monkeypatch.setattr(fiv.run, "build_media", build)

    def score(_handle, _inputs, _prefill, sequences):
        _handle.forwards += 7
        return [-1.0 - i for i in range(len(sequences))]

    monkeypatch.setattr(fiv.run, "score_sequences", score)
    out_dir = tmp_path / "mech"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--model", "gemma4", "--mechanics_only", "2", "--mechanics_dir", str(out_dir), "--geometry", str(geometry)],
    )
    fiv.main()
    text = (out_dir / "mechanics_gemma4.jsonl").read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 20
    assert all(set(r) <= set(fiv.MECHANICS_FIELDS) for r in rows)
    printed = capsys.readouterr().out
    for word in ("joint_logprob", "correct", "p_target", "accuracy"):
        assert word not in text and word not in printed
    assert {r["forwards"] for r in rows if r["status"] == "ok"} == {7}


def test_every_target_is_among_its_options():
    assert all(i["label"] in i["options"] for i in items())


def test_secondary_rows_compute_on_made_up_cells():
    rows = []
    human = {}
    for actor in "ABCD":
        for item in range(3):
            trial = f"{actor}{item}"
            human[trial] = {
                "human_target_label": "Sad",
                "p_target": 0.5 + 0.01 * item,
                "human_response_distribution": {"Sad": 0.6, "Happy": 0.2 + 0.01 * item, "None of the above": 0.1},
            }
            for rotation in fiv.ROTATIONS:
                for condition, value in (("real", [-1.0, -2.0, -3.0]), ("mismatch", [-3.0, -2.0, -1.0])):
                    options = ["Sad", "Happy", ABSTAIN_LABEL]
                    rows.append(
                        {
                            "trial_id": trial,
                            "actor": actor,
                            "label": "Sad",
                            "rotation": rotation,
                            "condition": condition,
                            "status": "ok",
                            "options": options,
                            "joint_logprob": value,
                            **fiv.correct_from_scores(options, value, "Sad"),
                        }
                    )
    summary = fiv.summarise_model(rows, human)
    assert summary["secondary"]["delta_scored_against_human_modal_label"] == pytest.approx(1.0)
    assert summary["share_items_lost"] == 0 and not summary["over_lost_item_limit"]

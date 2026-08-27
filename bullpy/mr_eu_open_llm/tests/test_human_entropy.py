import math

from scripts.human_entropy import (
    base_emotion_label,
    index_human_items,
    match_trial,
    parse_stimulus_code,
    shannon_entropy,
)


def test_shannon_entropy_bounds():
    assert shannon_entropy([1.0, 0.0, 0.0]) == 0.0
    assert shannon_entropy([1 / 6] * 6) == math.pi * 0 + math.log(6)
    # unnormalised input is renormalised, matching the workbook's rounded proportions
    assert shannon_entropy([0.5, 0.5]) == math.log(2)
    assert shannon_entropy([50.0, 50.0]) == math.log(2)
    assert shannon_entropy([]) == 0.0


def test_base_emotion_label_strips_intensity():
    assert base_emotion_label("Angry Low Intensity") == "angry"
    assert base_emotion_label("Surprised High Intensity") == "surprised"
    assert base_emotion_label("Unfriendly") == "unfriendly"


def test_parse_stimulus_code_manifest_filenames():
    assert parse_stimulus_code("NF25") == ("n", 25, None)
    assert parse_stimulus_code("PF19v1") == ("p", 19, 1)
    assert parse_stimulus_code("SF9 v2") == ("s", 9, 2)
    assert parse_stimulus_code("CF8(1)") == ("c", 8, 1)


def test_parse_stimulus_code_tolerates_workbook_typos():
    """The workbook misspells emotions ('frutrated', 'Dissapoint') and versions with spaces."""
    assert parse_stimulus_code("frutratedHF23v2") == ("h", 23, 2)
    assert parse_stimulus_code("DissapointSF9 v1") == ("s", 9, 1)
    assert parse_stimulus_code("excitedOF7 1") == ("o", 7, 1)
    assert parse_stimulus_code("FrustratedLF23") == ("l", 23, None)
    assert parse_stimulus_code("no code here") is None


def _item(emotion: str, code, stimulus_id: str):
    return {
        "emotion": emotion,
        "code": code,
        "stimulus_id": stimulus_id,
        "human_entropy": 1.0,
    }


def test_index_reports_key_collisions_rather_than_dropping():
    items = [
        _item("Excited", ("o", 7, 1), "excitedOF7 1"),
        _item("Excited", ("o", 7, 1), "excitedOF7 2"),
    ]
    _, _, collisions = index_human_items(items)
    assert len(collisions) == 1
    assert "excitedOF7 1" in collisions[0] and "excitedOF7 2" in collisions[0]


def test_match_trial_prefers_versioned_then_falls_back():
    items = [
        _item("Surprised", ("p", 19, 2), "surprisedPF19v2"),
        _item("Bored", ("c", 8, None), "BoredCF8"),
    ]
    versioned, unversioned, _ = index_human_items(items)

    exact, rule = match_trial(
        {"trial_id": "emotions/EDITED/Surprised/PF19v2_cut.mp4", "label": "Surprised"},
        versioned,
        unversioned,
    )
    assert exact["stimulus_id"] == "surprisedPF19v2"
    assert rule == "emotion_code_version"

    # Manifest carries a "(1)" the workbook does not; version-insensitive match applies.
    loose, rule = match_trial(
        {"trial_id": "emotions/EDITED/Bored/CF8(1)_cut.mp4", "label": "Bored"},
        versioned,
        unversioned,
    )
    assert loose["stimulus_id"] == "BoredCF8"
    assert rule == "emotion_code"


def test_match_trial_uses_base_emotion_for_intensity_labels():
    items = [_item("Angry", ("n", 18, None), "angryNF18")]
    versioned, unversioned, _ = index_human_items(items)
    got, rule = match_trial(
        {"trial_id": "emotions/EDITED/Angry Low Intensity/NF18_cut.mp4", "label": "Angry Low Intensity"},
        versioned,
        unversioned,
    )
    assert got["stimulus_id"] == "angryNF18"
    assert rule == "emotion_code_version"


def test_match_trial_reports_unmatched_rather_than_guessing():
    items = [_item("Angry", ("n", 18, None), "angryNF18")]
    versioned, unversioned, _ = index_human_items(items)
    got, rule = match_trial(
        {"trial_id": "emotions/EDITED/Kind/ZF99_cut.mp4", "label": "Kind"}, versioned, unversioned
    )
    assert got is None
    assert rule == "unmatched"

from pathlib import Path

import pytest

from scripts.eu_audio_resolver import (
    build_uk_voice_index,
    find_eu_voices_root,
    normalize_emotion_label,
    resolve_eu_multimodal_audio,
    resolve_uk_voice_by_label,
)
from scripts.mindreading_audio_resolver import LeakageAudioPathError, _hard_guard_no_leakage
from scripts.trial_foils import (
    generate_candidate_labels,
    load_eu_emotion_label_pool,
    resolve_candidate_labels,
    resolve_eu_emotion_pool,
)


def test_normalize_emotion_label():
    assert normalize_emotion_label("Angry - Low Intensity") == "angry low intensity"


def test_find_eu_voices_root_accepts_short_EU_folder(tmp_path: Path):
    eu = tmp_path / "data" / "eu_emotions_118" / "EU"
    fixed = eu / "Fixed - amplified volume" / "Happy"
    fixed.mkdir(parents=True)
    (fixed / "clip.mp3").write_bytes(b"x")
    found = find_eu_voices_root(tmp_path / "data" / "eu_emotions_118")
    assert found == eu.resolve()


def test_build_uk_voice_index_prefers_fixed(tmp_path: Path):
    voices = tmp_path / "EU Emotion - UK Voices"
    fixed = voices / "Fixed - amplified volume" / "Afraid"
    orig = voices / "Original" / "Afraid"
    fixed.mkdir(parents=True)
    orig.mkdir(parents=True)
    (fixed / "fixed_a.mp3").write_bytes(b"a")
    (orig / "orig_b.mp3").write_bytes(b"b")
    index = build_uk_voice_index(voices)
    assert [p.name for p in index["afraid"]] == ["fixed_a.mp3", "orig_b.mp3"]


def test_resolve_uk_voice_deterministic(tmp_path: Path):
    folder = tmp_path / "EU Emotion - UK Voices" / "Original" / "Sneaky"
    folder.mkdir(parents=True)
    for name in ("a.mp3", "b.mp3", "c.mp3"):
        (folder / name).write_bytes(b"x")
    ap1, _ = resolve_uk_voice_by_label(
        emotion_label="sneaky", base_data_dir=tmp_path, trial_id="eu_trial_1", seed=42
    )
    ap2, _ = resolve_uk_voice_by_label(
        emotion_label="sneaky", base_data_dir=tmp_path, trial_id="eu_trial_1", seed=42
    )
    assert ap1 == ap2


def test_eu_sidecar_audio(tmp_path: Path):
    video = tmp_path / "Happy" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"v")
    sidecar = video.parent / "clip.wav"
    sidecar.write_bytes(b"w")
    ap, rule = resolve_eu_multimodal_audio(
        video, emotion_label="happy", base_data_dir=tmp_path, trial_id="t1", seed=0
    )
    assert ap == sidecar and rule == "same_stem"


def test_leakage_guard():
    with pytest.raises(LeakageAudioPathError):
        _hard_guard_no_leakage(Path("/data/MindReading/Emotions/Audio/leak.wav"))


def test_foil_determinism():
    pool = load_eu_emotion_label_pool(Path(__file__).resolve().parents[1] / "data" / "eu_emotion_states_list.txt")
    a = generate_candidate_labels("joking", pool, trial_id="train_10", seed=42)
    b = generate_candidate_labels("joking", pool, trial_id="train_10", seed=42)
    assert a == b and len(a) == 4


def test_eu_pool_survives_small_max_trials_slice():
    two_trials = [{"trial_id": "a", "correct_label": "happy"}, {"trial_id": "b", "correct_label": "sad"}]
    pool = resolve_eu_emotion_pool(label_paths=[Path("/nonexistent")], trials_fallback=two_trials)
    labels = generate_candidate_labels("joking", pool, trial_id="train_10", seed=42)
    assert len(labels) == 4


def test_resolve_skips_existing_foils():
    trial = {"trial_id": "t1", "correct_label": "sad", "candidate_labels": ["sad", "happy", "afraid", "bored"]}
    out = resolve_candidate_labels(trial, ["sad", "happy"], seed=42)
    assert out == ["sad", "happy", "afraid", "bored"]

"""Tests for parse_emotion (4AFC) — brittle parser regression guard."""

from scripts.emotion_parse import parse_emotion


def test_parse_emotion_exact_label():
    opts = ["Joking", "Proud", "Afraid", "Sad"]
    pred, _ = parse_emotion("EMOTION: Joking\nREASONING: smile\n", opts)
    assert pred == "Joking"


def test_parse_emotion_numbered_option():
    opts = ["Joking", "Proud", "Afraid", "Sad"]
    pred, _ = parse_emotion("EMOTION: 2) Proud\n", opts)
    assert pred == "Proud"


def test_parse_emotion_substring_longer_first():
    opts = ["Afraid", "Afraid Low Intensity", "Happy", "Sad"]
    pred, _ = parse_emotion("EMOTION: afraid low intensity\n", opts)
    assert pred == "Afraid Low Intensity"


def test_parse_emotion_skips_placeholder():
    opts = ["Joking", "Proud", "Afraid", "Sad"]
    pred, _ = parse_emotion("EMOTION: <one of the option labels exactly>\nEMOTION: Joking\n", opts)
    assert pred == "Joking"


def test_parse_emotion_unparsed_returns_none():
    opts = ["Joking", "Proud", "Afraid", "Sad"]
    pred, _ = parse_emotion("I have no idea what this is about.\n", opts)
    assert pred is None

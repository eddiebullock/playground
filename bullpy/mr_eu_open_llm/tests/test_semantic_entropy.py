"""Unit tests for semantic entropy (deterministic synthetic embeddings)."""

import numpy as np

from scripts.semantic_entropy import (
    base_emotion,
    collapse_probs_to_base,
    correct_label_metrics,
    cosine_probs,
    prepare_entropy_label_pool,
    rich_label_embedding_text,
    semantic_entropy,
)


def test_uniform_probs_max_entropy():
    n = 27
    probs = np.ones(n) / n
    h = semantic_entropy(probs, log_base="e")
    assert abs(h - np.log(n)) < 1e-5


def test_peaked_distribution_low_entropy():
    probs = np.zeros(27)
    probs[0] = 1.0
    h = semantic_entropy(probs)
    assert h < 0.01


def test_cosine_probs_sums_to_one():
    d = 8
    label_embs = np.eye(d, dtype=np.float32)
    text_emb = np.ones(d, dtype=np.float32) / np.sqrt(d)
    p = cosine_probs(text_emb, label_embs, temperature=0.1)
    assert abs(p.sum() - 1.0) < 1e-6


def test_prepare_entropy_label_pool_excludes_neutral():
    pool = ["happy", "neutral", "sad"]
    out = prepare_entropy_label_pool(pool, exclude=("neutral",))
    assert "neutral" not in out
    assert "happy" in out


def test_collapse_intensity_pairs():
    labels = ["happy", "happy low intensity", "sad"]
    probs = np.array([0.3, 0.2, 0.5])
    collapsed, bases = collapse_probs_to_base(probs, labels)
    assert bases == ["happy", "sad"]
    assert abs(collapsed.sum() - 1.0) < 1e-9
    assert abs(collapsed[0] - 0.5) < 1e-9


def test_base_emotion_strips_suffix():
    assert base_emotion("afraid low intensity") == "afraid"


def test_rich_label_embedding_text_mentions_intensity():
    t = rich_label_embedding_text("happy low intensity")
    assert "low emotional intensity" in t.casefold() or "low intensity" in t.casefold()
    assert "happy" in t.casefold()


def test_correct_label_metrics_margin():
    labels = ["happy", "sad", "angry"]
    probs = np.array([0.5, 0.3, 0.2])
    m = correct_label_metrics(probs, labels, "happy")
    assert m["p_correct"] == 0.5
    assert abs(m["margin_correct"] - 0.2) < 1e-9


def test_correct_label_metrics_neutral_not_in_pool():
    labels = ["happy", "sad"]
    probs = np.array([0.6, 0.4])
    m = correct_label_metrics(probs, labels, "neutral")
    assert m["correct_in_entropy_pool"] is False
    assert m["p_correct"] is None

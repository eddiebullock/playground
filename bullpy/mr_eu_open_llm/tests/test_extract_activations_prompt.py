"""Unit tests for 4AFC-aligned activation extraction prompt resolution."""

from __future__ import annotations

from scripts.extract_activations import (
    default_pooling_for_prompt_mode,
    resolve_trial_extraction_prompt,
)


def test_default_pooling_for_prompt_mode():
    assert default_pooling_for_prompt_mode("4afc") == "last_token"
    assert default_pooling_for_prompt_mode("free_response") == "mean"


def test_resolve_trial_extraction_prompt_free_response():
    prompt = resolve_trial_extraction_prompt(
        {"trial_id": "t1", "label": "happy"},
        prompt_mode="free_response",
        condition_modality="video_only",
        seed=42,
        trial_index=0,
    )
    assert "free text" in prompt.lower()
    assert "4-alternative" not in prompt.lower()


def test_resolve_trial_extraction_prompt_4afc():
    prompt = resolve_trial_extraction_prompt(
        {"trial_id": "t1", "label": "happy"},
        prompt_mode="4afc",
        condition_modality="video_only",
        seed=42,
        trial_index=0,
    )
    assert "4-alternative forced-choice" in prompt.lower()
    assert "OPTIONS:" in prompt
    assert "EMOTION:" in prompt


def test_global_prompt_overrides_mode():
    fixed = "CUSTOM PROMPT"
    prompt = resolve_trial_extraction_prompt(
        {"trial_id": "t1", "label": "happy"},
        prompt_mode="4afc",
        condition_modality="video_only",
        seed=42,
        trial_index=0,
        global_prompt=fixed,
    )
    assert prompt == fixed

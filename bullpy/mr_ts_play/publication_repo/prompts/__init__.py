from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _load_text(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


MULTIMODAL_PROMPT: str = _load_text("multimodal_prompt.txt")
VIDEO_ONLY_PROMPT: str = _load_text("video_only_prompt.txt")


def get_prompt(model_name: str) -> str:
    """
    Return the appropriate prompt template for a given model name.

    - Gemini models: multimodal (video + audio)
    - GPT-5 / Claude Opus 4.5 and others here: video-only
    """
    name = (model_name or "").strip().lower()
    if "gemini" in name:
        return MULTIMODAL_PROMPT
    return VIDEO_ONLY_PROMPT

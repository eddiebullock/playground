"""
Shared Gemini client setup and safe text generation helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from config import GEMINI_MODEL_NAME

logger = logging.getLogger(__name__)


def get_gemini_model():
    """
    Configure google-generativeai and return a GenerativeModel instance.

    Expects ``GEMINI_API_KEY`` in the environment (load via dotenv in entrypoints).

    Returns:
        A ``google.generativeai.GenerativeModel`` instance.

    Raises:
        RuntimeError: If the SDK is missing or the API key is not set.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("google-generativeai is not installed") from e

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    genai.configure(api_key=key)
    model_id = os.environ.get("GEMINI_MODEL_NAME", GEMINI_MODEL_NAME).strip()
    return genai.GenerativeModel(model_id)


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Run a single Gemini completion and return text, or None on failure.

    Args:
        prompt: User/content prompt.
        system_instruction: Optional system-style instruction (prepended if set).
        max_output_tokens: Generation cap.
        temperature: Sampling temperature.

    Returns:
        Model text or None if the call fails.
    """
    full = prompt
    if system_instruction:
        full = f"{system_instruction.strip()}\n\n---\n\n{prompt}"

    try:
        model = get_gemini_model()
        resp = model.generate_content(
            full,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        )
        if not resp.candidates:
            logger.warning("Gemini returned no candidates")
            return None
        text = getattr(resp, "text", None)
        if text:
            return str(text).strip()
        parts = []
        for part in resp.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
        return "".join(parts).strip() if parts else None
    except Exception as e:
        logger.exception("Gemini API error: %s", e)
        return None

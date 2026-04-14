"""
Shared Gemini client setup and safe text generation helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from google.api_core import exceptions as google_exceptions

from config import GEMINI_MODEL_FALLBACKS, GEMINI_MODEL_NAME_DEFAULT

logger = logging.getLogger(__name__)


def _ordered_model_ids() -> list[str]:
    """
    Primary model id first (env GEMINI_MODEL_NAME or default), then fallbacks, de-duplicated.

    Returns:
        Model id strings to try in order.
    """
    primary = os.environ.get("GEMINI_MODEL_NAME", "").strip() or GEMINI_MODEL_NAME_DEFAULT
    out: list[str] = []
    seen: set[str] = set()
    for m in [primary, *GEMINI_MODEL_FALLBACKS]:
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Run a single Gemini completion and return text, or None on failure.

    Tries ``GEMINI_MODEL_NAME`` (or default), then ``GEMINI_MODEL_FALLBACKS`` if the API
    returns 404 for a model (e.g. not enabled for new keys).

    Args:
        prompt: User/content prompt.
        system_instruction: Optional system-style instruction (prepended if set).
        max_output_tokens: Generation cap.
        temperature: Sampling temperature.

    Returns:
        Model text or None if the call fails.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("google-generativeai is not installed") from e

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        logger.error("GEMINI_API_KEY is not set")
        return None

    genai.configure(api_key=key)

    full = prompt
    if system_instruction:
        full = f"{system_instruction.strip()}\n\n---\n\n{prompt}"

    ids = _ordered_model_ids()
    last_error: Optional[BaseException] = None
    for idx, model_id in enumerate(ids):
        try:
            model = genai.GenerativeModel(model_id)
            resp = model.generate_content(
                full,
                generation_config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                },
            )
            if not resp.candidates:
                logger.warning("Gemini returned no candidates (model=%s)", model_id)
                return None
            text = getattr(resp, "text", None)
            if text:
                if idx > 0:
                    logger.info("Succeeded with fallback model: %s", model_id)
                return str(text).strip()
            parts = []
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
            out = "".join(parts).strip() if parts else None
            if out and idx > 0:
                logger.info("Succeeded with fallback model: %s", model_id)
            return out
        except google_exceptions.NotFound as e:
            logger.warning(
                "Gemini model not available: %s (%s). Trying next fallback.",
                model_id,
                e,
            )
            last_error = e
            continue
        except Exception as e:
            logger.exception("Gemini API error (model=%s): %s", model_id, e)
            return None

    if last_error:
        logger.error("All Gemini model candidates failed; last error: %s", last_error)
    return None

"""Build a short vLLM server job. The script itself is jobs/vllm_server.sh.

This module does not import vLLM. Eval jobs must set HF_HUB_OFFLINE=1 so a
missing local weight fails instead of downloading a different checkpoint.
"""

from __future__ import annotations

import hashlib


def multiple_choice_prompt(stem: str, options: list[str]) -> str:
    if len(options) < 2:
        raise ValueError("multiple choice needs at least two options")
    lines = [stem.rstrip(), ""]
    for i, option in enumerate(options):
        lines.append(f"{chr(ord('A') + i)}. {option}")
    lines.append("")
    lines.append("Reply with the single letter of the best option.")
    return "\n".join(lines)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def server_command(
    *,
    model_path: str,
    port: int = 8000,
    max_model_len: int = 4096,
) -> list[str]:
    if not model_path:
        raise ValueError("model path is empty")
    if port <= 0:
        raise ValueError("port must be positive")
    return [
        "vllm",
        "serve",
        model_path,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        "bfloat16",
    ]


def logprob_request(prompt: str, letters: list[str]) -> dict:
    """OpenAI-compatible request that asks for logprobs of the option letters."""
    return {
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": len(letters),
        "echo": False,
    }


def generation_request(prompt: str, *, seed: int, max_tokens: int = 64) -> dict:
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "seed": seed,
    }

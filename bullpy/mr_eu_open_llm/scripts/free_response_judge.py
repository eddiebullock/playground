"""Free-response mental-state judging (rule-based primary; optional LLM judge)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from scripts.eu_emotion_synonyms import judge_free_response_rule, load_synonym_config
from scripts.eu_emotion_definitions import definition_for_label


_LLM_JUDGE_TEMPLATE = """You grade whether a model's free-text description correctly identifies a mental state.

Gold label: {gold_label}
Definition: {definition}
Accepted synonyms (non-exhaustive): {synonyms}

Model response:
---
{response}
---

Reply with exactly one word: CORRECT or INCORRECT.
A response is CORRECT if it clearly identifies the gold mental state or an accepted synonym,
including appropriate intensity variants. Minor wording differences are fine.
"""


def _definition_for_label(label: str) -> str:
    rec = definition_for_label(label)
    return str(rec.get("definition", "")) if rec else ""


def _synonym_sample(label: str, max_terms: int = 8) -> str:
    cfg = load_synonym_config()
    base = label.strip().lower().replace(" low intensity", "")
    syns = cfg.get("synonyms", {}).get(base, [])
    return ", ".join(syns[:max_terms])


def judge_free_response_llm(
    text: str,
    gold_label: str,
    *,
    model: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """Optional LLM judge via Inspect-compatible model string or OpenAI env."""
    prompt = _LLM_JUDGE_TEMPLATE.format(
        gold_label=gold_label,
        definition=_definition_for_label(gold_label) or "(no definition loaded)",
        synonyms=_synonym_sample(gold_label),
        response=text.strip(),
    )
    try:
        from inspect_ai.model import get_model

        mdl = get_model(model)
        result = mdl.generate(prompt)
        completion = (result.completion or "").strip().upper()
        correct = completion.startswith("CORRECT")
        return {
            "correct": correct,
            "method": "llm_judge",
            "judge_model": model,
            "raw_judge_output": result.completion,
        }
    except Exception as e:
        # Fallback: OpenAI-compatible HTTP if inspect not installed
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM judge requires inspect-ai with model access or OPENAI_API_KEY"
            ) from e
        try:
            import urllib.request

            body = json.dumps(
                {
                    "model": model.split("/")[-1] if "/" in model else model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
            ).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            completion = data["choices"][0]["message"]["content"].strip().upper()
            correct = completion.startswith("CORRECT")
            return {
                "correct": correct,
                "method": "llm_judge_openai",
                "judge_model": model,
                "raw_judge_output": data["choices"][0]["message"]["content"],
            }
        except Exception as e2:
            raise RuntimeError(f"LLM judge failed: {e2}") from e2


def judge_free_response(
    text: str,
    gold_label: str,
    *,
    use_llm: bool = False,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Grade free-response text against gold EU-Emotions label.

    Primary metric: rule-based synonym match (auditable).
    Optional: LLM judge when use_llm=True.
    """
    rule = judge_free_response_rule(text, gold_label)
    if not use_llm:
        return rule
    llm = judge_free_response_llm(
        text, gold_label, model=llm_model or "openai/gpt-4o-mini"
    )
    return {
        **rule,
        "llm_correct": llm.get("correct"),
        "llm_method": llm.get("method"),
        "llm_judge_model": llm.get("judge_model"),
        "llm_raw": llm.get("raw_judge_output"),
        "correct": bool(rule.get("correct") or llm.get("correct")),
        "method": "rule_or_llm",
    }

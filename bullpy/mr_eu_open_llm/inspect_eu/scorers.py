"""Inspect AI scorers wrapping repo judge + 4AFC parser."""

from __future__ import annotations

from typing import Any, Optional

from scripts.free_response_judge import judge_free_response
from scripts.tolerant_parse import parse_emotion_tolerant
from scripts.trial_foils import resolve_eu_emotion_pool


def score_free_response(response: str, target: str, *, use_llm: bool = False) -> dict:
    return judge_free_response(response, target, use_llm=use_llm)


def score_4afc(response: str, metadata: dict) -> dict:
    options = metadata.get("candidate_labels") or []
    pool = resolve_eu_emotion_pool()
    pred, reasoning, method = parse_emotion_tolerant(
        response, options, full_label_pool=pool
    )
    target = metadata.get("label")
    correct = pred == target if pred is not None else False
    return {
        "correct": correct,
        "prediction": pred,
        "reasoning": reasoning,
        "parse_method": method,
        "metric": "4afc_secondary",
    }


def build_inspect_scorers(*, primary: str = "free_response", use_llm_judge: bool = False) -> Any:
    """
    Return Inspect scorer callables when inspect-ai is available.

    primary: 'free_response' | '4afc'
    """
    try:
        from inspect_ai.scorer import Score, Scorer, accuracy, scorer
        from inspect_ai.solver import TaskState
        from inspect_ai.scorer import Target
    except ImportError as e:
        raise ImportError(
            "inspect-ai is required for build_inspect_scorers; pip install inspect-ai"
        ) from e

    @scorer(metrics=[accuracy()], name="eu_free_response_judge")
    def free_response_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            text = (state.output.completion or "").strip()
            gold = str(target.text or target)
            meta = state.metadata or {}
            gold = meta.get("label") or gold
            result = score_free_response(text, gold, use_llm=use_llm_judge)
            return Score(
                value=1.0 if result.get("correct") else 0.0,
                answer=text,
                explanation=result.get("matched_term") or result.get("method"),
            )

        return score

    @scorer(metrics=[accuracy()], name="eu_4afc_secondary")
    def four_afc_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            text = (state.output.completion or "").strip()
            meta = state.metadata or {}
            result = score_4afc(text, meta)
            return Score(
                value=1.0 if result.get("correct") else 0.0,
                answer=result.get("prediction"),
                explanation=result.get("parse_method"),
            )

        return score

    return free_response_scorer() if primary == "free_response" else four_afc_scorer()

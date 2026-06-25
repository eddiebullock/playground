"""Tolerant label extraction for B0 fine-tune verification and rescoring."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from scripts.emotion_parse import parse_emotion
from scripts.eu_emotion_synonyms import accepted_terms_for_label, collapse_intensity_label, match_label_in_text
from scripts.trial_foils import resolve_eu_emotion_pool


def extract_free_text_label(raw: str, label_pool: Sequence[str]) -> Optional[str]:
    """Map arbitrary model text to the closest EU label in *label_pool*."""
    if not raw or not label_pool:
        return None
    text = str(raw).strip()
    # Finetune-style "LABEL: joking"
    for m in re.finditer(r"(?im)(?:label|emotion|mental state)\s*[:\-]\s*(.+?)(?:\n|$)", text):
        candidate = m.group(1).strip().strip(" .\"'`")
        hit = _match_pool(candidate, label_pool)
        if hit:
            return hit
    # Last non-empty line (common finetune completion)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        hit = _match_pool(lines[-1], label_pool)
        if hit:
            return hit
    # Synonym scan against each pool label
    best: Optional[str] = None
    for lbl in label_pool:
        if match_label_in_text(text, lbl):
            best = lbl
            break
    return best


def _match_pool(raw: str, label_pool: Sequence[str]) -> Optional[str]:
    raw0 = raw.strip().strip(" .\"'`")
    for opt in label_pool:
        if raw0.lower() == str(opt).lower():
            return str(opt)
    for opt in sorted(label_pool, key=lambda x: -len(str(x))):
        if str(opt).lower() in raw0.lower():
            return str(opt)
    base = collapse_intensity_label(raw0)
    for opt in label_pool:
        if collapse_intensity_label(str(opt)) == base:
            return str(opt)
    return None


def parse_emotion_tolerant(
    output_text: Any,
    options: Sequence[str],
    *,
    full_label_pool: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parse model output with strict 4AFC first, then tolerant pool match.

    Returns (prediction, reasoning, parse_method).
    """
    pred, reasoning = parse_emotion(output_text, options)
    if pred is not None:
        return pred, reasoning, "strict_4afc"

    pool = list(full_label_pool) if full_label_pool else list(options)
    if not isinstance(output_text, str):
        output_text = str(output_text)
    tolerant = extract_free_text_label(output_text, pool)
    if tolerant is not None:
        # Map back to 4AFC option if present
        for opt in options:
            if str(opt).lower() == tolerant.lower():
                return opt, reasoning, "tolerant_4afc_option"
            if collapse_intensity_label(str(opt)) == collapse_intensity_label(tolerant):
                return opt, reasoning, "tolerant_4afc_base"
        return tolerant, reasoning, "tolerant_full_pool"

    return None, reasoning, "unparsed"


def rescore_eval_json(trials: List[dict], *, seed: int = 42) -> dict:
    """Rescore stored trial records without re-running the model."""
    pool = resolve_eu_emotion_pool()
    strict_correct = 0
    tolerant_correct = 0
    stage1_correct = 0
    n_stage2 = 0
    n_stage1 = 0
    methods: dict = {}
    from scripts.free_response_judge import judge_free_response

    for t in trials:
        label = t.get("label") or ""
        s2 = t.get("stage2") or {}
        if s2:
            n_stage2 += 1
            opts = s2.get("options") or []
            raw = s2.get("raw_model_output") or ""
            pred, _, method = parse_emotion_tolerant(raw, opts, full_label_pool=pool)
            methods[method] = methods.get(method, 0) + 1
            if s2.get("prediction") == label:
                strict_correct += 1
            if pred == label or (
                pred is not None
                and collapse_intensity_label(str(pred)) == collapse_intensity_label(str(label))
            ):
                tolerant_correct += 1
        s1 = t.get("stage1") or {}
        if s1 and s1.get("free_response_text"):
            n_stage1 += 1
            jr = judge_free_response(s1["free_response_text"], label, use_llm=False)
            if jr.get("correct"):
                stage1_correct += 1

    return {
        "n_trials": len(trials),
        "n_stage2": n_stage2,
        "strict_4afc_accuracy": strict_correct / n_stage2 if n_stage2 else None,
        "tolerant_4afc_accuracy": tolerant_correct / n_stage2 if n_stage2 else None,
        "free_response_judge_accuracy": stage1_correct / n_stage1 if n_stage1 else None,
        "parse_methods": methods,
    }

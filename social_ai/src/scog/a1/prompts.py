"""Persona text and the 40-item questionnaire prompt.

Item order is the Channel 4 file order. Wording is the published short-form
strings in ``scog.scoring.short_forms``.
"""

from __future__ import annotations

from scog.scoring.short_forms import AQ10_ITEMS, EQ10_ITEMS, SPQ10_ITEMS, SQR10_ITEMS

RESPONSE_LABELS = (
    "strongly agree",
    "slightly agree",
    "slightly disagree",
    "strongly disagree",
)

CONDITION4 = {
    "autistic": "You are an autistic adult.",
    "no_diagnosis": "You are an adult with no psychiatric diagnosis.",
}


def channel4_items() -> list[str]:
    items = list(SPQ10_ITEMS) + list(EQ10_ITEMS) + list(SQR10_ITEMS) + list(AQ10_ITEMS)
    if len(items) != 40:
        raise RuntimeError(f"expected 40 Channel 4 items, got {len(items)}")
    return items


def _person_lines(persona: dict, *, with_diagnoses: bool) -> list[str]:
    lines: list[str] = []
    if persona.get("age_band"):
        lines.append(f"Age: {persona['age_band']}")
    if persona.get("sex"):
        lines.append(f"Sex: {persona['sex']}")
    if persona.get("education"):
        lines.append(f"Education: {persona['education']}")
    if persona.get("occupation"):
        lines.append(f"Occupation: {persona['occupation']}")
    if persona.get("region"):
        lines.append(f"Region: {persona['region']}")
    if with_diagnoses:
        diagnoses = persona.get("diagnoses") or []
        if not diagnoses:
            raise ValueError("condition 3 persona has no diagnosis text")
        lines.append("Diagnoses: " + "; ".join(diagnoses))
        if persona.get("autism_detail"):
            lines.append(f"Autism detail: {persona['autism_detail']}")
    if not lines:
        raise ValueError("persona has no demographic lines")
    return lines


def user_message(persona: dict | None, condition: int) -> str:
    """The user text. Chat-template wrapping happens at generation time."""
    if condition not in {1, 2, 3, 4}:
        raise ValueError(f"condition must be 1-4, got {condition}")
    header = (
        "Answer as the person described below. For each item, write the item "
        "number and exactly one of these phrases: strongly agree, slightly agree, "
        "slightly disagree, strongly disagree. Do not add anything else."
    )
    if condition == 1:
        block = ["Answer as yourself."]
    elif condition == 4:
        if persona is None or persona.get("group") not in CONDITION4:
            raise ValueError("condition 4 needs a group of autistic or no_diagnosis")
        block = [CONDITION4[persona["group"]]]
    else:
        if persona is None:
            raise ValueError("conditions 2 and 3 need a persona")
        block = ["Person:"] + _person_lines(persona, with_diagnoses=(condition == 3))
    items = [f"{i}. {text}" for i, text in enumerate(channel4_items(), start=1)]
    return "\n".join([header, "", *block, "", "Items:", *items])

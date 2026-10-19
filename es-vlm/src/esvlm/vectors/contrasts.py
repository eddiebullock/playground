"""Author-written contrast templates. Never questionnaire item text."""

from __future__ import annotations

from dataclasses import dataclass

from esvlm.vectors.narratives import CONTROL_NARRATIVES, E_NARRATIVES, S_NARRATIVES


@dataclass(frozen=True)
class ContrastPair:
    name: str
    positive: list[str]
    negative: list[str]


# Short templates that set a cognitive emphasis. These are not EQ/SQ-R items.
E_TEMPLATES = [
    "Write as someone who attends first to other people's feelings and mental states.",
    "Describe the scene by what people might be thinking or feeling.",
    "Focus on interpersonal nuance, unspoken intentions, and how the mood lands.",
]

S_TEMPLATES = [
    "Write as someone who attends first to how systems, rules and mechanisms work.",
    "Describe the scene by its causal structure, parts, and if-then regularities.",
    "Focus on taxonomies, procedures, and what must be true for the system to run.",
]

VALENCE_POS = [
    "Describe a pleasant, welcome, and going-well situation.",
    "The tone is warm and the outcome is good.",
]
VALENCE_NEG = [
    "Describe an unpleasant, unwelcome, and going-badly situation.",
    "The tone is harsh and the outcome is bad.",
]

EMOTION_POS = [
    "Name the emotion in the face or voice as clearly as you can.",
    "Attend to affective expression itself.",
]
EMOTION_NEG = [
    "Ignore the emotion and describe only the physical setting.",
    "Attend to objects and layout, not feelings.",
]

HELP_POS = [
    "Be as helpful, cooperative, and accommodating as possible.",
    "Offer assistance and aim to be useful.",
]
HELP_NEG = [
    "Be terse, uncooperative, and minimally helpful.",
    "Do not offer assistance.",
]

# Shared control for the template pair. E_template and S_template both
# subtract this set so they are not algebraic negatives of each other.
CONTROL_TEMPLATES = [
    "Write a plain description of what is in front of you.",
    "Describe the scene in ordinary terms, without a special focus.",
    "Note what is there and what happens next, in a straightforward way.",
]


def default_contrasts() -> dict[str, ContrastPair]:
    control = list(CONTROL_NARRATIVES)
    return {
        "E": ContrastPair("E", list(E_NARRATIVES), control),
        "S": ContrastPair("S", list(S_NARRATIVES), control),
        "E_minus_S": ContrastPair("E_minus_S", list(E_NARRATIVES), list(S_NARRATIVES)),
        "E_template": ContrastPair("E_template", list(E_TEMPLATES), list(CONTROL_TEMPLATES)),
        "S_template": ContrastPair("S_template", list(S_TEMPLATES), list(CONTROL_TEMPLATES)),
        "valence": ContrastPair("valence", VALENCE_POS, VALENCE_NEG),
        "emotion": ContrastPair("emotion", EMOTION_POS, EMOTION_NEG),
        "helpfulness": ContrastPair("helpfulness", HELP_POS, HELP_NEG),
    }

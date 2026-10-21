"""Prompt templates. At least three per task. No licensed item text."""

from __future__ import annotations

from collections.abc import Sequence

# RMET: one template is the classic wording from the published test booklet.
RMET_TEMPLATES: dict[str, str] = {
    "classic": (
        "Choose which word best describes what the person in the picture "
        "is thinking or feeling.\n\n{options}"
    ),
    "forced_choice": (
        "Look at the eye region. Choose exactly one of the following labels "
        "for the person's mental state.\n\n{options}\n\n"
        "Reply with the label only."
    ),
    "numbered": (
        "You are doing a {n}-alternative forced-choice mental-state task. "
        "Pick the single best label.\n\n{numbered}\n\n"
        "Answer with the label text, not the number."
    ),
    # Same images as classic RMET. Not a mental-state question.
    "appearance": (
        "Look at the picture. What kind of image is this?\n\n{options}\n\n"
        "Reply with the label only."
    ),
}

EU_EMOTIONS_TEMPLATES: dict[str, str] = {
    "direct": (
        "Which emotion is being expressed?\n\n{options}\n\nReply with the label only."
    ),
    "clip": (
        "Watch the clip. Choose the single emotion label that best matches "
        "what you see.\n\n{options}"
    ),
    "numbered": (
        "Forced-choice emotion recognition. {n} options:\n\n{numbered}\n\n"
        "Answer with the label text."
    ),
}

MINDREADING_TEMPLATES: dict[str, str] = {
    "direct": (
        "Which word best describes what this person is thinking or feeling?\n\n{options}"
    ),
    "classic_style": (
        "Choose which word best describes what the person is thinking or feeling.\n\n{options}"
    ),
    "numbered": (
        "Mental-state forced choice. {n} options:\n\n{numbered}\n\nAnswer with the label."
    ),
}

SYNTHETIC_TEMPLATES: dict[str, str] = {
    "color": "What is the dominant color of this image?\n\n{options}\n\nReply with the label only.",
    "shape": "Which shape is shown?\n\n{options}\n\nReply with the label only.",
    "numbered": "Forced-choice visual question. {n} options:\n\n{numbered}\n\nAnswer with the label.",
}

S_TASK_TEMPLATES: dict[str, str] = {
    "mechanism": (
        "Which description best matches the physical or mechanical process "
        "shown?\n\n{options}"
    ),
    "system": (
        "Attend to the rules and causal structure. Choose one.\n\n{options}"
    ),
    "numbered": (
        "Physical / mechanical forced choice. {n} options:\n\n{numbered}\n\n"
        "Answer with the label."
    ),
}

NEXTQA_TEMPLATES: dict[str, str] = {
    "direct": "{options}\n\nReply with the label only.",
    "video": (
        "Watch the clip. Choose the single best answer.\n\n{options}\n\n"
        "Reply with the label only."
    ),
    "numbered": (
        "Video question. {n} options:\n\n{numbered}\n\nAnswer with the label."
    ),
}

MMTOM_TEMPLATES: dict[str, str] = {
    "direct": (
        "Which statement is more likely to be true?\n\n{options}\n\n"
        "Reply with the label only."
    ),
    "belief": (
        "Choose the more likely belief or knowledge statement.\n\n{options}\n\n"
        "Reply with the label only."
    ),
    "numbered": (
        "Theory-of-mind forced choice. {n} options:\n\n{numbered}\n\n"
        "Answer with the label."
    ),
}

TEMPLATES: dict[str, dict[str, str]] = {
    "rmet": RMET_TEMPLATES,
    "eu_emotions": EU_EMOTIONS_TEMPLATES,
    "mindreading": MINDREADING_TEMPLATES,
    "synthetic": SYNTHETIC_TEMPLATES,
    "s_task": S_TASK_TEMPLATES,
    "physbench": S_TASK_TEMPLATES,
    "clevrer": S_TASK_TEMPLATES,
    "piqa": S_TASK_TEMPLATES,
    "nextqa": NEXTQA_TEMPLATES,
    "mmtom": MMTOM_TEMPLATES,
}


def format_options(options: Sequence[str]) -> str:
    return "\n".join(f"- {opt}" for opt in options)


def format_numbered(options: Sequence[str]) -> str:
    return "\n".join(f"{i + 1}) {opt}" for i, opt in enumerate(options))


def render_prompt(task: str, template_id: str, options: Sequence[str]) -> str:
    table = TEMPLATES.get(task)
    if table is None or template_id not in table:
        raise KeyError(f"no template {task}/{template_id}")
    return table[template_id].format(
        options=format_options(options),
        numbered=format_numbered(options),
        n=len(options),
    )


def template_ids(task: str) -> list[str]:
    return list(TEMPLATES[task])

"""Reading the Mind in the Eyes Test, revised adult form.

Option order is Baron-Cohen, Wheelwright, Hill, Raste and Plumb (2001),
JCPP 42:241-251, Appendix A. The paper prints the target in italic, which
the text extraction does not preserve. Two targets are explicit in the
paper's figure captions: item 29 reflective, item 32 serious. Those are
positions 4 and 1 in Appendix A.

The option index below is the key recovered as the modal CARD choice,
which reproduces the Eyes Test header "number correct" and matches those
two captions. It is stored so the audit can check the header, and it is
still marked as needing the italic targets checked off the printed page.
"""

from __future__ import annotations

from typing import Sequence

EYES_CITATION = (
    "Baron-Cohen, Wheelwright, Hill, Raste & Plumb (2001) JCPP 42:241-251"
)
EYES_KEY_STATUS = "option_order_published_index_matches_modal_key_italics_unchecked"

# correct_option is 1-based into `options`, Appendix A order.
EYES_ITEMS = (
    {"item": 1, "options": ("playful", "comforting", "irritated", "bored"), "correct_option": 1},
    {"item": 2, "options": ("terrified", "upset", "arrogant", "annoyed"), "correct_option": 2},
    {"item": 3, "options": ("joking", "flustered", "desire", "convinced"), "correct_option": 3},
    {"item": 4, "options": ("joking", "insisting", "amused", "relaxed"), "correct_option": 2},
    {"item": 5, "options": ("irritated", "sarcastic", "worried", "friendly"), "correct_option": 3},
    {"item": 6, "options": ("aghast", "fantasizing", "impatient", "alarmed"), "correct_option": 2},
    {"item": 7, "options": ("apologetic", "friendly", "uneasy", "dispirited"), "correct_option": 3},
    {"item": 8, "options": ("despondent", "relieved", "shy", "excited"), "correct_option": 1},
    {"item": 9, "options": ("annoyed", "hostile", "horrified", "preoccupied"), "correct_option": 4},
    {"item": 10, "options": ("cautious", "insisting", "bored", "aghast"), "correct_option": 1},
    {"item": 11, "options": ("terrified", "amused", "regretful", "flirtatious"), "correct_option": 3},
    {"item": 12, "options": ("indifferent", "embarrassed", "sceptical", "dispirited"), "correct_option": 3},
    {"item": 13, "options": ("decisive", "anticipating", "threatening", "shy"), "correct_option": 2},
    {"item": 14, "options": ("irritated", "disappointed", "depressed", "accusing"), "correct_option": 4},
    {"item": 15, "options": ("contemplative", "flustered", "encouraging", "amused"), "correct_option": 1},
    {"item": 16, "options": ("irritated", "thoughtful", "encouraging", "sympathetic"), "correct_option": 2},
    {"item": 17, "options": ("doubtful", "affectionate", "playful", "aghast"), "correct_option": 1},
    {"item": 18, "options": ("decisive", "amused", "aghast", "bored"), "correct_option": 1},
    {"item": 19, "options": ("arrogant", "grateful", "sarcastic", "tentative"), "correct_option": 4},
    {"item": 20, "options": ("dominant", "friendly", "guilty", "horrified"), "correct_option": 2},
    {"item": 21, "options": ("embarrassed", "fantasizing", "confused", "panicked"), "correct_option": 2},
    {"item": 22, "options": ("preoccupied", "grateful", "insisting", "imploring"), "correct_option": 1},
    {"item": 23, "options": ("contented", "apologetic", "defiant", "curious"), "correct_option": 3},
    {"item": 24, "options": ("pensive", "irritated", "excited", "hostile"), "correct_option": 1},
    {"item": 25, "options": ("panicked", "incredulous", "despondent", "interested"), "correct_option": 4},
    {"item": 26, "options": ("alarmed", "shy", "hostile", "anxious"), "correct_option": 3},
    {"item": 27, "options": ("joking", "cautious", "arrogant", "reassuring"), "correct_option": 2},
    {"item": 28, "options": ("interested", "joking", "affectionate", "contented"), "correct_option": 1},
    {"item": 29, "options": ("impatient", "aghast", "irritated", "reflective"), "correct_option": 4},
    {"item": 30, "options": ("grateful", "flirtatious", "hostile", "disappointed"), "correct_option": 2},
    {"item": 31, "options": ("ashamed", "confident", "joking", "dispirited"), "correct_option": 2},
    {"item": 32, "options": ("serious", "ashamed", "bewildered", "alarmed"), "correct_option": 1},
    {"item": 33, "options": ("embarrassed", "guilty", "fantasizing", "concerned"), "correct_option": 4},
    {"item": 34, "options": ("aghast", "baffled", "distrustful", "terrified"), "correct_option": 3},
    {"item": 35, "options": ("puzzled", "nervous", "insisting", "contemplative"), "correct_option": 2},
    {"item": 36, "options": ("ashamed", "nervous", "suspicious", "indecisive"), "correct_option": 3},
)

EYES_KEY = tuple(item["correct_option"] for item in EYES_ITEMS)
MAX_MISSING_ITEMS = 2


def score_choices(choices: Sequence[int | None], key: Sequence[int] = EYES_KEY) -> dict:
    """Score one respondent. None or -1 is missing. The pair's second value is ignored."""
    if len(choices) != len(key):
        raise ValueError(f"expected {len(key)} choices, got {len(choices)}")
    n_correct = 0
    n_missing = 0
    for choice, answer in zip(choices, key):
        if choice is None or choice == -1:
            n_missing += 1
            continue
        if int(choice) not in (1, 2, 3, 4):
            n_missing += 1
            continue
        n_correct += int(int(choice) == int(answer))
    return {
        "n_correct": n_correct,
        "n_missing": n_missing,
        "excluded_missing": n_missing > MAX_MISSING_ITEMS,
    }

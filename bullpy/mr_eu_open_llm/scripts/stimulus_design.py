"""
Fixed specification of the A2 analysis: contrasts, thresholds and the null rule's controls.

Standard library only, so that the run step can hash this on any environment, and the
hash ties cells to the analysis as well as to the readout. See research/PREREG_A2_DRAFT.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

ALPHA = 0.05
B_RULES = {"coverage": 0.05, "image_rms": 0.25, "sign_flip_p": 0.01, "repeat": 1e-3}
MARGIN_FRACTION = 0.10
N_ITEMS = 243
TOKEN_FLOOR = 8.0
MIN_REGION_TOKENS = 1.0
BOOTSTRAP_DRAWS = 10000
ORIGINAL = ("original",)
SHUFFLES = ("shuffle_a", "shuffle_b")
GATE_KEYS = ("rq0", "g2_g4", "gate_b", "delivery", "sham")
BASIC_LABELS = ("Afraid", "Angry", "Disgusted", "Happy", "Sad", "Surprised")
SMOKE_ITEMS = (
    "emotions_38/EDITED/Hurt/OF10_cut.mp4",
    "emotions_7/EDITED/Interested/JF21v1_cut.mp4",
    "emotions_4/EDITED/Sneaky/HF26_cut.mp4",
)


@dataclass(frozen=True)
class Contrast:
    id: str
    route: str
    labels: Tuple[str, ...]
    a: Tuple[str, ...]
    outcome: str = "acted"
    direction: int = -1
    b: Tuple[str, ...] = ORIGINAL
    versus_labels: Tuple[str, ...] = ()  # interaction: labels minus these, inside one actor set
    low_intensity: bool = False
    other_fills: Tuple[Tuple[str, ...], ...] = field(default=())
    other_fills_b: Tuple[Tuple[str, ...], ...] = field(default=())  # the b side, where b is a mask
    dose: Tuple[str, ...] = ()  # positive controls for the null rule, each against the original
    joint: Optional[str] = None  # joint ablation; None means this cue cannot reach Null


def _mask(region: str, fill: str = "neutral") -> Tuple[str, ...]:
    return (f"mask_{region}_{fill}",)


def _fills(region: str) -> Tuple[Tuple[str, ...], ...]:
    return (_mask(region, "first"), _mask(region, "blur"))


def _region(region: str, joint: Optional[str] = "mask_eyesmouth_neutral") -> Dict[str, Any]:
    return {"other_fills": _fills(region), "dose": _mask(region, "blur"), "joint": joint}


TEMPORAL_DOSE = ("static_apex",)

# P11 is out until the aperture dose is verified (see the plan). Brows are not in the joint
# mask, and P6 is an interaction, so neither has a joint ablation.
CONFIRMATORY: Tuple[Contrast, ...] = (
    Contrast("P1", "single_face", ("Happy",), _mask("mouth"), **_region("mouth")),
    Contrast("P2", "single_face", ("Disgusted",), _mask("mouth"), **_region("mouth")),
    Contrast("P3", "single_face", ("Afraid",), _mask("eyes"), **_region("eyes")),
    Contrast("P4", "single_face", ("Sad",), _mask("eyes"), **_region("eyes")),
    Contrast("P5", "single_face", ("Angry",), _mask("brows"), **_region("brows", None)),
    Contrast(
        "P6",
        "single_face",
        ("Happy", "Disgusted"),
        _mask("mouth"),
        b=_mask("eyes"),
        versus_labels=("Afraid",),
        other_fills=_fills("mouth"),
        other_fills_b=_fills("eyes"),
        dose=_mask("mouth", "blur") + _mask("eyes", "blur"),
    ),
    Contrast("P7", "single_face", ("Happy", "Sad", "Angry"), _mask("mouth"), "Neutral", 1, **_region("mouth")),
    Contrast("P8", "single_face", ("Disgusted",), _mask("mouth"), "Angry", 1, **_region("mouth")),
    Contrast("P9", "single_face", ("Happy",), ("sf_low",), direction=1, b=("sf_high",), dose=("sf_low", "sf_high")),
    Contrast("P10", "single_face", ("Sad",), ("sf_low",), direction=-1, b=("sf_high",), dose=("sf_low", "sf_high")),
    Contrast("P12", "video_face", ("Happy", "Surprised"), SHUFFLES, dose=TEMPORAL_DOSE),
    Contrast("P13", "video_face", (), SHUFFLES, low_intensity=True, dose=TEMPORAL_DOSE),
    Contrast("P14a", "video_face", ("Afraid",), ("trunc_first_third",), "Surprised", 1, ("trunc_last_third",), dose=TEMPORAL_DOSE),
    Contrast("P14b", "video_face", ("Surprised",), ("trunc_first_third",), "Afraid", 1, ("trunc_last_third",), dose=TEMPORAL_DOSE),
    Contrast("P15a", "video_face", ("Angry",), ("trunc_first_third",), "Disgusted", 1, ("trunc_last_third",), dose=TEMPORAL_DOSE),
    Contrast("P15b", "video_face", ("Disgusted",), ("trunc_first_third",), "Angry", 1, ("trunc_last_third",), dose=TEMPORAL_DOSE),
)


def design_spec() -> Dict[str, Any]:
    """Everything here that a result depends on, for the design hash."""
    return {
        "alpha": ALPHA,
        "b_rules": B_RULES,
        "margin_fraction": MARGIN_FRACTION,
        "n_items": N_ITEMS,
        "token_floor": TOKEN_FLOOR,
        "min_region_tokens": MIN_REGION_TOKENS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "gate_keys": GATE_KEYS,
        "basic_labels": BASIC_LABELS,
        "smoke_items": SMOKE_ITEMS,
        "confirmatory": [dataclasses.asdict(c) for c in CONFIRMATORY],
    }

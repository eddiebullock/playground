"""
Fixed specification of amendment A4: stimuli, options, thresholds and the pass rule.

Standard library only, so that any environment can hash it. Every cell and output carries
`design_hash()`. See research/PREREG_A4_DRAFT.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DESIGN_FILES = (
    "research/reproduction/human_au_effects.json",
    "scripts/prompts.py",
    "scripts/run_stimulus_interventions.py",
    "research/scripts/render_au_faces.py",
)

DATASETS = ("basic_WC", "basic_EA", "conversational_WC", "conversational_EA")
TASK_OF = {"basic_WC": "basic", "basic_EA": "basic",
           "conversational_WC": "conversational", "conversational_EA": "conversational"}
# Options shown to models, in the human column order of human_au_effects.LABEL_ORDER. "other"
# maps to the abstain label of A1, A2 and A3, and stays last under rotation.
ABSTAIN = "None of the above"
OPTIONS = {
    "basic": ("Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise", ABSTAIN),
    "conversational": ("Bored", "Confused", "Interested", "Thinking", ABSTAIN),
}
HUMAN_LABELS = {
    "basic": ("anger", "disgust", "fear", "happy", "sadness", "surprise", "other"),
    "conversational": ("bored", "confused", "interested", "thinking", "other"),
}
# A complete cyclic set: each named option takes each named position once. A1's five
# rotations were complete for five named labels; here the count follows the task.
ROTATIONS = {"basic": 6, "conversational": 4}
MODELS = ("gemma4", "qwen3vl")  # Holm family. Molmo2 fails G2 and G4, descriptive if run.
DESCRIPTIVE_MODELS = ("molmo2",)

SEED = 20261107
CONFIGS_PER_DATASET = 1200
MANIFEST_SHA256 = "0ae87d06d896542bd5dc944ad523c83cf8916463f9de39b3cb4a2f21badb8ac2"
BASELINE_STIMULI = 50
SPLIT_HALF_SPLITS = 20
REPLICATE_DRAWS = 9999
CEILING_SPLITS = 200
R_S_LOWER_BOUND = 0.85  # below this, raw CFA reads as a lower bound
IDENTITIES = {
    "WC": {"F": (("female_a", 0), ("female_b", 0)), "M": (("male_a", 0), ("male_b", 0))},
    "EA": {"F": (("ea_female", 0), ("ea_female", 1)), "M": (("ea_male", 0), ("ea_male", 1))},
}
ONE_AU_IDENTITIES = (("female_a", 0), ("female_b", 0), ("male_a", 0), ("male_b", 0),
                     ("ea_female", 0), ("ea_male", 0))
ONE_AU_AMPLITUDES = (0.25, 0.5, 1.0)
REPLICATE_CONFIGS = 200
RENDER_TARS = 40  # images live in this many tar files, since the cluster is near its file quota
SCALE_CONFIGS = 400
SCALES = (0.75, 1.25)
SCALE_DATASETS = ("basic_EA", "conversational_EA")
BLIND_CONFIGS = 600
BLIND_DATASET = "basic_EA"
BLIND_FAMILY = {"anger": "Angry", "disgust": "Disgusted", "fear": "Afraid", "happy": "Happy",
                "sadness": "Sad", "surprise": "Surprised"}

APPROXIMATED_FAMILIES = ("AU22", "AU27")
ALPHA = 0.05
SESOI_CFA = 0.30
MAX_LOST_SHARE = 0.05
CUE_PERMUTATIONS = 9999
SIGNAL_PERMUTATIONS = 999
BOOTSTRAP_DRAWS = 2000
V_RULES = {
    "detector_monotonic_share": 0.70,
    "image_rms_nats": 0.25,
    "replicate_p": 0.01,
    "max_abstain_mass": 0.90,
    "max_option_mass": 0.95,
}
B_RULES = {"coverage": 0.05, "image_rms": 0.25, "sign_flip_p": 0.01, "repeat": 1e-3}
RECOVERY = {
    "replicates": 100,
    "bootstrap_draws": 100,
    "split_half_splits": 6,
    "boundary_mix": 0.35,
    "cue_permutations": 1999,
    "signal_permutations": 199,
    "strengths": (0.25, 0.5, 1.0),  # human-like binds at 0.5 and 1, half-shared at 1, the rest at all
    "stimulus_logit_sd": 1.0,
    "gain_log_sd": 0.5,
    "pass": {"human_like_aligned_min": 0.80, "half_shared_alignment_min": 0.80,
             "half_shared_aligned_min_at_1x": 0.60, "false_aligned_max": 0.10,
             "prior_only_signal_max": 0.10, "not_aligned_different_cues_min": 0.50,
             "pair_cfa_error_max": 0.10, "pair_within_error_share_min": 0.90,
             "boundary_not_aligned_max": 0.10},
    # Withdrawn before any data, after the first run failed it: see the draft.
    "pair_error_withdrawn": ("conversational_WC|0.25", "conversational_EA|0.25"),
    "pairs": (("AU01", "AU02"), ("AU06", "AU12"), ("AU04", "AU07"), ("AU25", "AU26"),
              ("AU15", "AU17"), ("AU09", "AU10"), ("AU20", "AU14"), ("AU23", "AU24")),
    "pair_rho": 0.9,
}


def constants() -> Dict[str, Any]:
    return {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (str, int, float, tuple, dict))}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_digests(root: Path = ROOT) -> Dict[str, str]:
    """The inputs a result depends on beyond these constants: targets, prompts, renderer."""
    return {name: file_digest(root / name) for name in DESIGN_FILES}


def design_hash() -> str:
    """Hash of every constant here and of DESIGN_FILES."""
    payload = {"constants": constants(), "files": file_digests()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()


def check_manifest(path: Path) -> None:
    digest = file_digest(path)
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"Manifest hash {digest} is not the pinned one.")


def shard_of(stim_id: str, n_shards: int) -> int:
    return int(hashlib.sha256(stim_id.encode()).hexdigest(), 16) % n_shards


def rotations(task: str) -> Tuple[int, ...]:
    return tuple(range(ROTATIONS[task]))


def rotated_options(task: str, rotation: int) -> List[str]:
    named = [o for o in OPTIONS[task] if o != ABSTAIN]
    r = rotation % len(named)
    return named[r:] + named[:r] + [ABSTAIN]

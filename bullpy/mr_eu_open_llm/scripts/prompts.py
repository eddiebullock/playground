from __future__ import annotations

"""Condition-aware prompt templates for two-stage evaluation."""

from typing import Sequence

MODALITY_CONDITIONS = ("video_only", "audio_only", "multimodal")


def build_free_response_prompt(*, condition: str = "video_only") -> str:
    """Stage 1 free-response prompt; modality-aware (video, audio, or both)."""
    c = (condition or "video_only").strip().lower()
    if c == "audio_only":
        return (
            "You are listening to an audio clip of a person.\n"
            "Describe what mental state or states the person appears to be expressing.\n"
            "Do not choose from a fixed list of labels; respond in free text only.\n"
        )
    if c == "multimodal":
        return (
            "You are observing a person in a video with accompanying audio.\n"
            "Describe what mental state or states the person appears to be expressing.\n"
            "Do not choose from a fixed list of labels; respond in free text only.\n"
        )
    return (
        "You are observing a person in a video or image.\n"
        "Describe what mental state or states the person appears to be expressing.\n"
        "Do not choose from a fixed list of labels; respond in free text only.\n"
    )


def build_finetune_prompt(*, condition: str = "video_only") -> str:
    """Supervised fine-tuning: predict a single mental-state label."""
    c = (condition or "video_only").strip().lower()
    if c == "audio_only":
        modality = "Listen to the audio clip and identify the person's mental state.\n"
    elif c == "multimodal":
        modality = "Watch the video and listen to the audio. Identify the person's mental state.\n"
    else:
        modality = "Watch the video and identify the person's mental state.\n"
    return (
        "You are performing mental state recognition.\n"
        f"{modality}"
        "Respond with exactly one mental state label (single word or short phrase).\n"
        "LABEL:"
    )


def build_4afc_prompt(options: Sequence[str], *, condition: str = "video_only") -> str:
    opts = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
    c = (condition or "video_only").strip().lower()
    if c == "audio_only":
        modality = "Analyze audio only (voice tone, prosody, rhythm). Choose exactly one label.\n"
    elif c == "multimodal":
        modality = "Analyze video and audio together. Choose exactly one label.\n"
    else:
        modality = "Analyze video frames only. Choose exactly one label.\n"
    return (
        f"You are performing a 4-alternative forced-choice mental state recognition task.\n"
        f"{modality}\nOPTIONS:\n{opts}\n\n"
        "Respond with:\nEMOTION: <one of the option labels exactly>\nREASONING: <brief justification>\n"
    )

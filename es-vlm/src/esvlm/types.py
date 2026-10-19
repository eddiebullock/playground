"""Shared types. Keep this module free of torch so tests import cheaply."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Domain = Literal["E", "S", "control", "selfreport"]
MediaType = Literal["image", "video", "audio+video", "none"]
SteerMode = Literal["add", "ablate"]
SteerPosition = Literal["last_token", "post_media", "all"]
DeviceKind = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class MCQItem:
    """One forced-choice trial.

    ``media`` is a path on disk or empty when ``media_type`` is ``none``.
    Licensed stimuli are never copied into the repo; loaders resolve paths
    from config / ``local/site.sh``.
    """

    item_id: str
    task: str
    domain: Domain
    media: str
    media_type: MediaType
    prompt_text: str
    options: tuple[str, ...]
    correct_idx: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"{self.item_id}: options must be non-empty")
        if not 0 <= self.correct_idx < len(self.options):
            raise ValueError(
                f"{self.item_id}: correct_idx {self.correct_idx} "
                f"out of range for {len(self.options)} options"
            )
        if self.media_type != "none" and not self.media:
            raise ValueError(f"{self.item_id}: media path required for {self.media_type}")


@dataclass(frozen=True)
class ScoreRow:
    """One scored option-set. Written incrementally to sharded parquet."""

    model: str
    revision: str
    task: str
    item_id: str
    template_id: str
    condition: str
    vector: str
    layer: int | None
    alpha: float
    mode: str
    frames: int
    option_logprobs: tuple[float, ...]
    pred: int
    correct: bool
    seed: int
    tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceSpec:
    kind: DeviceKind
    dtype_name: str  # "bfloat16" | "float16" | "float32"
    device_str: str

    @property
    def is_gpu_like(self) -> bool:
        return self.kind in {"cuda", "mps"}


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser()

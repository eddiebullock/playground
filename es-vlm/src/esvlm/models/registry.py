"""Pinned model panel. IDs and revisions live in config, not in call sites."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EnvName = Literal["main", "molmo", "throwaway"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    env: EnvName
    adapter: str
    on_cluster: bool
    notes: str = ""


# Revisions verified via the Hugging Face models API on 2026-10-18.
PANEL: dict[str, ModelSpec] = {
    "qwen3vl_8b": ModelSpec(
        "qwen3vl_8b",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "main",
        "qwen_vl",
        True,
        "primary",
    ),
    "qwen3vl_2b": ModelSpec(
        "qwen3vl_2b",
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        "main",
        "qwen_vl",
        False,
        "Mac smoke / scale ladder",
    ),
    "qwen3vl_32b": ModelSpec(
        "qwen3vl_32b",
        "Qwen/Qwen3-VL-32B-Instruct",
        "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
        "main",
        "qwen_vl",
        False,
        "fits one A100 in bf16",
    ),
    "gemma4_e4b": ModelSpec(
        "gemma4_e4b",
        "google/gemma-4-E4B-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "main",
        "gemma4",
        True,
        "",
    ),
    "molmo2": ModelSpec(
        "molmo2",
        "allenai/Molmo2-O-7B",
        "784410650d12be9bc086118fdefa32d2c3bced86",
        "molmo",
        "molmo2",
        False,
        "excluded 2026-10-18: processor API; no E/S results",
    ),
    "qwen25vl_7b": ModelSpec(
        "qwen25vl_7b",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "main",
        "qwen_vl",
        False,
        "replication anchor",
    ),
    "qwen3_text": ModelSpec(
        "qwen3_text",
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        "main",
        "qwen_vl",
        False,
        "text-only control; Instruct id was HTTP 401",
    ),
    "internvl35_8b": ModelSpec(
        "internvl35_8b",
        "OpenGVLab/InternVL3_5-8B-HF",
        "741a7d03020411e666c6109218ab71e08151ef86",
        "throwaway",
        "internvl",
        False,
        "optional; drop after two load failures",
    ),
    "qwen2vl_7b": ModelSpec(
        "qwen2vl_7b",
        "Qwen/Qwen2-VL-7B-Instruct",
        "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        "main",
        "qwen_vl",
        True,
        "on cluster; not a primary panel model",
    ),
}

CLUSTER_KEY_ALIASES = {
    "qwen3vl": "qwen3vl_8b",
    "gemma4": "gemma4_e4b",
    "molmo2": "molmo2",
    "qwen2vl": "qwen2vl_7b",
}


def get_spec(key: str) -> ModelSpec:
    if key in PANEL:
        return PANEL[key]
    if key in CLUSTER_KEY_ALIASES:
        return PANEL[CLUSTER_KEY_ALIASES[key]]
    raise KeyError(f"unknown model key {key!r}; known: {sorted(PANEL)}")


def list_models() -> list[str]:
    return list(PANEL)


def resolve_local_dir(spec: ModelSpec, models_dir: Path | None) -> Path | None:
    dirs: list[Path] = []
    if models_dir is not None:
        dirs.append(Path(models_dir))
    for env_key in ("ESVLM_MODELS_DIR", "ESVLM_NEW_MODELS_DIR"):
        raw = os.environ.get(env_key)
        if raw:
            path = Path(raw)
            if path not in dirs:
                dirs.append(path)
    if not dirs:
        return None
    names = [spec.key, spec.hf_id.split("/")[-1]]
    for alias, key in CLUSTER_KEY_ALIASES.items():
        if key == spec.key:
            names.append(alias)
    unique_names = list(dict.fromkeys(names))
    for root in dirs:
        for name in unique_names:
            candidate = root / name
            if (candidate / "config.json").is_file():
                return candidate
    return None

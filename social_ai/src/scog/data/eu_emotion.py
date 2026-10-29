"""EU-Emotion item-level validation workbook.

This file is stimulus-level aggregates, not participant rows. Header is on
Excel row 4 (0-indexed row 3). Stimulus IDs are matched to video files
case-insensitively.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HEADER_ROW = 3  # 0-indexed
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


def load_eu_emotion(path: str | Path, header_row: int = HEADER_ROW) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"EU-Emotion workbook not found: {path}")
    df = pd.read_excel(path, header=header_row, engine="openpyxl")
    df = df.dropna(how="all")
    if len(df) == 0:
        raise RuntimeError(f"EU-Emotion workbook {path.name}: zero stimuli")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _find_column(columns: list[str], needles: list[str]) -> str | None:
    folded = {col.casefold(): col for col in columns}
    for needle in needles:
        if needle.casefold() in folded:
            return folded[needle.casefold()]
    for col in columns:
        low = col.casefold()
        if any(needle.casefold() in low for needle in needles):
            return col
    return None


def stimulus_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the columns the brief describes. Raises if a required one is absent."""
    columns = list(df.columns)
    mapping = {
        "modality": _find_column(columns, ["modality"]),
        "emotion": _find_column(columns, ["expressed emotion", "emotion"]),
        # "Intensity" is the mean rating. "Emotion intensity" is High/Low.
        "intensity": _find_column(columns, ["emotion intensity"]),
        "stimulus_id": _find_column(columns, ["stimulus id", "stimulus", "stimuli id"]),
        "n_raters": _find_column(
            columns,
            ["number of raters", "number of interviews", "raters", "n raters"],
        ),
    }
    missing = [name for name, col in mapping.items() if col is None]
    if missing:
        raise RuntimeError(
            "EU-Emotion workbook is missing expected columns "
            f"{missing}. Found: {columns}"
        )
    # Proportion columns: the six raw response proportions sit after n_raters.
    # Prefer names; otherwise take the next six numeric columns.
    prop_names = []
    for col in columns:
        # The sheet repeats the block three times. The first block is the raw
        # proportions; later repeats are named Target emotion.1 and .2.
        if ".1" in col or ".2" in col:
            continue
        folded = col.replace("\n", " ").casefold()
        if (
            folded.startswith("target emotion")
            or folded.startswith("control emotion")
            or "don't know" in folded
            or "do not know" in folded
        ):
            prop_names.append(col)
    work = pd.DataFrame({name: df[col] for name, col in mapping.items()})
    if len(prop_names) >= 6:
        for i, col in enumerate(prop_names[:6]):
            work[f"p{i}"] = pd.to_numeric(df[col], errors="coerce")
    else:
        start = columns.index(mapping["n_raters"]) + 1
        numeric_tail = []
        for col in columns[start:]:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().mean() > 0.8:
                numeric_tail.append(col)
            if len(numeric_tail) == 6:
                break
        if len(numeric_tail) < 6:
            raise RuntimeError("EU-Emotion workbook: could not find 6 response-proportion columns")
        for i, col in enumerate(numeric_tail):
            work[f"p{i}"] = pd.to_numeric(df[col], errors="coerce")
    work["n_raters"] = pd.to_numeric(work["n_raters"], errors="coerce")
    work["stimulus_id"] = work["stimulus_id"].astype(str).str.strip()
    if work["stimulus_id"].eq("").any() or work["stimulus_id"].eq("nan").any():
        raise RuntimeError("EU-Emotion workbook has a blank stimulus id")
    return work


def response_entropy_bits(props: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits of each row. Rows are renormalised if they sum to a positive value."""
    arr = np.asarray(props, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("entropy expects an (n, 6) array")
    totals = np.nansum(arr, axis=1)
    if np.any(totals <= 0):
        raise RuntimeError("EU-Emotion response proportions sum to zero for at least one stimulus")
    clean = np.nan_to_num(arr, nan=0.0)
    probs = clean / totals[:, None]
    logged = np.zeros_like(probs)
    positive = probs > 0
    logged[positive] = probs[positive] * np.log2(probs[positive])
    return -logged.sum(axis=1)


def summarise_stimuli(table: pd.DataFrame) -> dict:
    if len(table) == 0:
        raise RuntimeError("EU-Emotion summary: zero stimuli")
    props = table[[f"p{i}" for i in range(6)]].to_numpy(dtype=float)
    entropy = response_entropy_bits(props)
    target = props[:, 0]
    modal_is_target = np.nanargmax(np.nan_to_num(props, nan=-1), axis=1) == 0
    # Ties: if another option equals the target, argmax returns the first max, which is the target.
    return {
        "n_stimuli": int(len(table)),
        "n_modality": {str(k): int(v) for k, v in table["modality"].value_counts(dropna=False).items()},
        "n_emotion": int(table["emotion"].nunique(dropna=True)),
        "emotions": sorted(str(v) for v in table["emotion"].dropna().unique()),
        "n_intensity": {str(k): int(v) for k, v in table["intensity"].value_counts(dropna=False).items()},
        "raters_min": float(table["n_raters"].min()),
        "raters_max": float(table["n_raters"].max()),
        "raters_mean": float(table["n_raters"].mean()),
        "mean_target_proportion": float(np.nanmean(target)),
        "mean_entropy_bits": float(np.mean(entropy)),
        "sd_entropy_bits": float(np.std(entropy, ddof=1)),
        "n_modal_not_target": int((~modal_is_target).sum()),
    }


def match_videos(stimulus_ids: list[str], video_paths: list[Path]) -> dict:
    """Case-insensitive stem match. Reports missing stimuli, extra files, and collisions."""
    by_key: dict[str, list[str]] = {}
    for path in video_paths:
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        by_key.setdefault(path.stem.casefold(), []).append(path.name)
    collisions = {key: names for key, names in by_key.items() if len(names) > 1}
    missing = []
    matched = 0
    for stim in stimulus_ids:
        hits = by_key.get(str(stim).casefold(), [])
        if not hits:
            missing.append(str(stim))
        else:
            matched += 1
    used = {str(stim).casefold() for stim in stimulus_ids}
    extra = [names[0] for key, names in by_key.items() if key not in used]
    return {
        "n_video_files": sum(len(v) for v in by_key.values()),
        "n_matched": matched,
        "missing_stimuli": missing,
        "extra_files": extra,
        "collisions": collisions,
    }
